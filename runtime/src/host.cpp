#include "llaccel/host.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
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
Host::Host(const Llbin& bin, Device& dev, const Tokenizer& tok) : bin_(bin), dev_(dev), tok_(tok) {
  const auto& m = bin.meta.at("model");
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
  prefillM_ = std::ranges::max(bin.programs | std::views::transform(&Program::M));
  if (prefillM_ == 0) throw std::runtime_error("no programs in llbin");
  if (tok.itos.size() != vocab_) throw std::runtime_error("tokenizer size != model vocab");
  if (embRowBytes_ < dim_ * 2 || inRowBytes_ < dim_ * 2) throw std::runtime_error("embedding/input row too small for dim");
  dev_.dramWrite(0, bin.dramImage.data(), bin.dramImage.size());
}

void Host::writeInputRow(uint32_t row, uint32_t token) {
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
  if (pos + M > maxSeq_) throw std::runtime_error("sequence exceeds max_seq");
  for (uint32_t r = 0; r < M; ++r) writeInputRow(r, r < rowTokens.size() ? rowTokens[r] : UINT32_MAX);
  const Program& prog = bin_.programForM(M);
  StepResult s{.pos = pos, .M = M, .argmax = 0, .logits = {}, .perf = dev_.run(prog.pc, pos)};
  s.logits = readLogitsRow(logitsRow);
  // Greedy: first index of the maximum (ties -> lowest index, same as numpy argmax).
  s.argmax = uint32_t(std::ranges::max_element(s.logits) - s.logits.begin());
  return s;
}

GenerationResult Host::generate(const std::string& prompt, uint32_t nTokens, bool verbose) {
  GenerationResult r;
  r.promptTokens = tok_.encode(prompt);
  if (r.promptTokens.empty()) throw std::runtime_error("empty prompt");
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
    if (verbose)
      std::println("  decode {}/{}: pos={} cycles={} tok={} '{}'", i + 1, nTokens, L + i, s.perf[PERF_CYCLES], next, tok_.itos[next]);
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

bool verifyAgainstGolden(const GenerationResult& r, const std::string& goldenPath, bool verbose) {
  std::ifstream f(goldenPath);
  if (!f) throw std::runtime_error("cannot open golden " + goldenPath);
  auto g = nlohmann::json::parse(f);
  auto gp = g.at("prompt_tokens").get<std::vector<uint32_t>>();
  auto gg = g.at("generated").get<std::vector<uint32_t>>();
  auto ga = g.at("argmax_per_step").get<std::vector<uint32_t>>();
  bool ok = true;
  if (gp != r.promptTokens) { std::println("VERIFY: prompt tokens differ from golden"); ok = false; }
  size_t n = std::min(ga.size(), r.steps.size());
  for (size_t i = 0; i < n; ++i)
    if (ga[i] != r.steps[i].argmax) {
      std::println("VERIFY: step {} argmax {} != golden {} (pos={} M={})", i, r.steps[i].argmax, ga[i], r.steps[i].pos, r.steps[i].M);
      ok = false;
      break;
    }
  if (ok && ga.size() != r.steps.size())
    std::println("VERIFY: note: {} steps run vs {} golden steps (compared the first {})", r.steps.size(), ga.size(), n);
  if (ok && g.contains("logits_last_rows")) {
    auto gl = g.at("logits_last_rows").get<std::vector<std::vector<int16_t>>>();
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
