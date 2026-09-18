#include "llaccel/host.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <print>
#include <ranges>
#include <stdexcept>

namespace llaccel {

// ---- tokenizer ------------------------------------------------------------------------
Tokenizer Tokenizer::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open tokenizer " + path);
  auto j = nlohmann::json::parse(f);
  Tokenizer t;
  for (const auto& s : j.at("itos")) t.itos.push_back(s.get<std::string>());
  return t;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
  std::vector<uint32_t> ids;
  size_t i = 0;
  while (i < text.size()) {
    // UTF-8 aware: take the full code point.
    unsigned char c = text[i];
    size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
    std::string ch = text.substr(i, len);
    i += len;
    auto it = std::ranges::find(itos, ch);
    if (it == itos.end()) throw std::runtime_error("character not in vocabulary: '" + ch + "'");
    ids.push_back(uint32_t(it - itos.begin()));
  }
  return ids;
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
  std::string s;
  for (auto id : ids) s += id < itos.size() ? itos[id] : "?";
  return s;
}

// ---- host ------------------------------------------------------------------------------------
Host::Host(const Llbin& bin, Device& dev) : bin_(bin), dev_(dev) {
  const auto& m = bin.meta.at("model");
  uint64_t requiredHeadDim = 0;
  if (m.contains("head_dim")) {
    const auto& value = m.at("head_dim");
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<int64_t>() <= 0))
      throw std::runtime_error("invalid model head_dim");
    requiredHeadDim = value.get<uint64_t>();
    if (requiredHeadDim == 0) throw std::runtime_error("invalid model head_dim");
  }
  // check instructions before upload; metadata cannot bypass backend limits.
  for (const auto& program : bin.programs)
    for (const auto& in : program.instrs) {
      uint32_t D = 0;
      if (in.op() == Op::ATTN) D = in[9];
      else if (in.op() == Op::KV_WRITE || in.op() == Op::VEC_ROPE) D = in[6];
      requiredHeadDim = std::max<uint64_t>(requiredHeadDim, D);
    }
  if (requiredHeadDim > dev.maxAttentionHeadDim())
    throw std::runtime_error("model requires attention head dimension " + std::to_string(requiredHeadDim) +
      "; backend " + dev.name() + " supports at most " + std::to_string(dev.maxAttentionHeadDim()) +
      ". Rebuild or select a backend implementing the required head width.");
  vocab_ = m.at("vocab").get<uint32_t>();
  dim_ = m.at("dim").get<uint32_t>();
  maxSeq_ = m.at("max_seq").get<uint32_t>();
  const auto& d = bin.meta.at("dram");
  embAddr_ = d.at("embedding").at("addr").get<uint64_t>();
  embRowBytes_ = d.at("embedding").at("row_bytes").get<uint64_t>();
  inAddr_ = d.at("input").at("addr").get<uint64_t>();
  inRowBytes_ = d.at("input").at("row_bytes").get<uint64_t>();
  lgAddr_ = d.at("logits").at("addr").get<uint64_t>();
  lgRowBytes_ = d.at("logits").at("row_bytes").get<uint64_t>();
  if (bin.programs.empty()) throw std::runtime_error("no programs in llbin");
  if (!vocab_ || !dim_ || !maxSeq_) throw std::runtime_error("invalid zero model dimension");
  prefillM_ = std::ranges::max(bin.programs | std::views::transform(&Program::M));
  if (prefillM_ == 0) throw std::runtime_error("no programs in llbin");
  if (lgRowBytes_ < uint64_t(vocab_) * 2) throw std::runtime_error("logits row too small for vocab");
  if (embRowBytes_ < uint64_t(dim_) * 2 || inRowBytes_ < uint64_t(dim_) * 2) throw std::runtime_error("embedding/input row too small for dim");
  auto region = [&](uint64_t base, uint64_t stride, uint64_t rows) {
    if (base > bin.dramImage.size() || !stride || rows > (bin.dramImage.size() - base) / stride)
      throw std::runtime_error("model buffer metadata exceeds DRAM image");
  };
  if (maxSeq_ > kAttnTMax || maxSeq_ % prefillM_)
    throw std::runtime_error("invalid context/prefill dimensions");
  region(embAddr_, embRowBytes_, vocab_);
  region(inAddr_, inRowBytes_, prefillM_);
  region(lgAddr_, lgRowBytes_, prefillM_);
  if (d.contains("kv_cache")) {
    if (!d["kv_cache"].is_array()) throw std::runtime_error("DRAM KV metadata must be an array");
    for (const auto& cache : d["kv_cache"])
      region(cache.at("addr").get<uint64_t>(), cache.at("bytes").get<uint64_t>(), 1);
  }
  dev_.dramWrite(0, bin.dramImage.data(), bin.dramImage.size());
}

Host::Host(const Llbin& bin, Device& dev, const Tokenizer& tok) : Host(bin, dev) {
  if (tok.itos.size() != vocab_) throw std::runtime_error("tokenizer size != model vocab");
  tok_ = &tok;
}

void Host::writeInputRow(uint32_t row, uint32_t token) {
  if (token != UINT32_MAX && token >= vocab_) throw std::runtime_error("token ID exceeds model vocabulary");
  std::vector<uint8_t> buf(inRowBytes_, 0);
  if (token != UINT32_MAX) dev_.dramRead(embAddr_ + uint64_t(token) * embRowBytes_, buf.data(), dim_ * 2);
  dev_.dramWrite(inAddr_ + uint64_t(row) * inRowBytes_, buf.data(), inRowBytes_);
}

std::vector<int16_t> Host::readLogitsRow(uint32_t row) {
  std::vector<int16_t> v(vocab_);
  dev_.dramRead(lgAddr_ + uint64_t(row) * lgRowBytes_, v.data(), vocab_ * 2);
  return v;
}

StepResult Host::launch(const std::vector<uint32_t>& rowTokens, uint32_t M, uint32_t pos, uint32_t logitsRow) {
  if (M > maxSeq_ || pos > maxSeq_ - M) throw std::runtime_error("sequence exceeds max_seq");
  for (uint32_t r = 0; r < M; ++r) writeInputRow(r, r < rowTokens.size() ? rowTokens[r] : UINT32_MAX);
  const Program& prog = bin_.programForM(M);
  StepResult s{.pos = pos, .M = M, .argmax = 0, .logits = {}, .perf = dev_.run(prog.pc, pos)};
  s.logits = readLogitsRow(logitsRow);
  // greedy: first index of the maximum (ties -> lowest index, same as numpy argmax).
  s.argmax = uint32_t(std::ranges::max_element(s.logits) - s.logits.begin());
  return s;
}

GenerationResult Host::generate(const std::string& prompt, uint32_t nTokens, bool verbose) {
  if (!tok_) throw std::runtime_error("text generation requires a character tokenizer; use generateTokens");
  return generateTokens(tok_->encode(prompt), nTokens, verbose);
}

GenerationResult Host::generateTokens(const std::vector<uint32_t>& prompt, uint32_t nTokens, bool verbose) {
  GenerationResult r;
  r.promptTokens = prompt;
  if (r.promptTokens.empty()) throw std::runtime_error("empty prompt");
  if (r.promptTokens.size() > maxSeq_ || nTokens > maxSeq_ - r.promptTokens.size())
    throw std::runtime_error("prompt + tokens exceeds max_seq");
  for (auto id : r.promptTokens)
    if (id >= vocab_) throw std::runtime_error("prompt token ID exceeds model vocabulary");
  // each generation starts a session; clear KV DRAM while preserving weights
  // and program bytes. launches within the call share the cache.
  if (bin_.meta.at("dram").contains("kv_cache")) {
    const std::array<uint8_t, 4096> zeros{};
    for (const auto& cache : bin_.meta.at("dram").at("kv_cache")) {
      uint64_t addr = cache.at("addr").get<uint64_t>(), remaining = cache.at("bytes").get<uint64_t>();
      while (remaining) {
        const uint64_t count = std::min<uint64_t>(remaining, zeros.size());
        dev_.dramWrite(addr, zeros.data(), count);
        addr += count;
        remaining -= count;
      }
    }
  }
  uint32_t L = uint32_t(r.promptTokens.size());
  uint32_t nChunks = (L + prefillM_ - 1) / prefillM_;
  uint32_t next = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (uint32_t c = 0; c < nChunks; ++c) {
    std::vector<uint32_t> rows(r.promptTokens.begin() + c * prefillM_,
                               r.promptTokens.begin() + std::min<uint32_t>(L, (c + 1) * prefillM_));
    uint32_t lastRow = (c == nChunks - 1) ? (L - 1) % prefillM_ : prefillM_ - 1;
    StepResult s = launch(rows, prefillM_, c * prefillM_, lastRow);
    if (verbose) std::println("  prefill chunk {}/{}: pos={} cycles={}", c + 1, nChunks, c * prefillM_, s.perf[PERF_CYCLES]);
    next = s.argmax;
    r.steps.push_back(std::move(s));
  }
  for (uint32_t i = 0; i < nTokens; ++i) {
    r.generated.push_back(next);
    if (L + i + 1 >= maxSeq_) break;  // no room for another position
    StepResult s = launch({next}, 1, L + i, 0);
    if (verbose) {
      std::println("  decode {}/{}: pos={} cycles={} tok={}", i + 1, nTokens, L + i, s.perf[PERF_CYCLES], next);
      if (tok_) std::println("    character: '{}'", tok_->itos[next]);
    }
    next = s.argmax;
    r.steps.push_back(std::move(s));
  }
  auto t1 = std::chrono::steady_clock::now();
  for (const auto& s : r.steps)
    for (uint32_t i = 0; i < kNumPerf; ++i) r.total[i] += s.perf[i];
  if (verbose) std::println("  wall time: {:.2f} s", std::chrono::duration<double>(t1 - t0).count());
  return r;
}

nlohmann::json GenerationResult::toJson() const {
  nlohmann::json j;
  j["prompt_tokens"] = promptTokens;
  j["generated"] = generated;
  std::vector<uint32_t> am;
  std::vector<std::vector<int16_t>> lg;
  nlohmann::json stepsJ = nlohmann::json::array();
  for (const auto& s : steps) {
    am.push_back(s.argmax);
    lg.push_back(s.logits);
    nlohmann::json pj;
    for (uint32_t i = 0; i < kNumPerf; ++i) pj[std::string(kPerfNames[i])] = s.perf[i];
    stepsJ.push_back({{"pos", s.pos}, {"M", s.M}, {"argmax", s.argmax}, {"perf", pj}});
  }
  j["argmax_per_step"] = am;
  j["logits_last_rows"] = lg;
  j["steps"] = stepsJ;
  nlohmann::json tj;
  for (uint32_t i = 0; i < kNumPerf; ++i) tj[std::string(kPerfNames[i])] = total[i];
  j["perf_total"] = tj;
  return j;
}

namespace {
// JSON numeric conversions otherwise silently truncate fractions or wrap narrow integers.
template <typename T>
std::vector<T> goldenIntegerArray(const nlohmann::json& array, const char* field) {
  if (!array.is_array()) throw std::runtime_error(std::string("golden ") + field + " must be an array");
  std::vector<T> values;
  values.reserve(array.size());
  for (const auto& value : array) {
    bool valid = false;
    if (value.is_number_unsigned()) {
      valid = value.get<uint64_t>() <= uint64_t(std::numeric_limits<T>::max());
    } else if (value.is_number_integer()) {
      const auto integer = value.get<int64_t>();
      valid = integer >= int64_t(std::numeric_limits<T>::min()) &&
              integer <= int64_t(std::numeric_limits<T>::max());
    }
    if (!valid) throw std::runtime_error(std::string("golden ") + field + " contains a non-integer or out-of-range value");
    values.push_back(value.get<T>());
  }
  return values;
}
} // namespace

std::vector<uint32_t> loadPromptIds(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open prompt IDs " + path);
  const auto array = nlohmann::json::parse(f);
  // reuse the strict integer validation used for reference traces.
  return goldenIntegerArray<uint32_t>(array, "prompt IDs");
}

bool verifyAgainstGolden(const GenerationResult& r, const std::string& goldenPath, bool verbose) {
  std::ifstream f(goldenPath);
  if (!f) throw std::runtime_error("cannot open golden " + goldenPath);
  auto g = nlohmann::json::parse(f);
  auto gp = goldenIntegerArray<uint32_t>(g.at("prompt_tokens"), "prompt_tokens");
  auto gg = goldenIntegerArray<uint32_t>(g.at("generated"), "generated");
  auto ga = goldenIntegerArray<uint32_t>(g.at("argmax_per_step"), "argmax_per_step");
  if (!g.contains("logits_last_rows")) throw std::runtime_error("golden trace has no logits");
  const auto& rows = g.at("logits_last_rows");
  if (!rows.is_array()) throw std::runtime_error("golden logits_last_rows must be an array");
  std::vector<std::vector<int16_t>> gl;
  gl.reserve(rows.size());
  for (const auto& row : rows) gl.push_back(goldenIntegerArray<int16_t>(row, "logits_last_rows"));
  bool ok = true;
  if (gp != r.promptTokens) { std::println("VERIFY: prompt tokens differ from golden"); ok = false; }
  size_t n = std::min(ga.size(), r.steps.size());
  for (size_t i = 0; i < n; ++i)
    if (ga[i] != r.steps[i].argmax) {
      std::println("VERIFY: step {} argmax {} != golden {} (pos={} M={})", i, r.steps[i].argmax, ga[i], r.steps[i].pos, r.steps[i].M);
      ok = false;
      break;
    }
  if (ga.size() != r.steps.size()) {
    std::println("VERIFY: launch count {} != golden {}", r.steps.size(), ga.size());
    ok = false;
  }
  if (gg.size() != r.generated.size()) {
    std::println("VERIFY: generated count {} != golden {}", r.generated.size(), gg.size());
    ok = false;
  }
  if (ok) {
    if (gl.size() != r.steps.size()) {
      std::println("VERIFY: logits row count {} != launches {}", gl.size(), r.steps.size());
      ok = false;
    }
    for (size_t i = 0; i < std::min(gl.size(), r.steps.size()); ++i)
      if (gl[i] != r.steps[i].logits) {
        auto [a, b] = std::ranges::mismatch(gl[i], r.steps[i].logits);
        size_t k = size_t(a - gl[i].begin());
        std::println("VERIFY: step {} logits differ at index {}: {} vs golden {}", i, k,
                     b != r.steps[i].logits.end() ? *b : 0, a != gl[i].end() ? *a : 0);
        ok = false;
        break;
      }
  }
  size_t ng = std::min(gg.size(), r.generated.size());
  for (size_t i = 0; ok && i < ng; ++i)
    if (gg[i] != r.generated[i]) { std::println("VERIFY: generated token {} differs", i); ok = false; }
  if (verbose || !ok) std::println("VERIFY: {} ({} launches, {} generated tokens compared)", ok ? "MATCH" : "MISMATCH", n, ng);
  return ok;
}

void printPerf(const PerfCounters& p, uint32_t launches) {
  std::println("Performance ({} launches)\n-------------------------", launches);
  for (uint32_t i = 0; i < kNumPerf; ++i) std::println("  {:<22} {:>14}", kPerfNames[i], p[i]);
  if (p[PERF_CYCLES]) {
    auto pctOf = [&](uint64_t v) { return 100.0 * double(v) / double(p[PERF_CYCLES]); };
    std::println("  GEMM utilization       {:6.2f} %", pctOf(p[PERF_GEMM_MAC_CYCLES]));
    std::println("  ATTN MAC utilization   {:6.2f} %", pctOf(p[PERF_ATTN_MAC_CYCLES]));
    std::println("  CP wait stalls         {:6.2f} %", pctOf(p[PERF_CP_STALL_WAIT]));
    std::println("  DMA busy               {:6.2f} %", pctOf(p[PERF_DMA_BUSY]));
  }
}

}  // namespace llaccel
