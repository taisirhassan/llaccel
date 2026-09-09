#include "llaccel/host.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
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
    uint32_t id = UINT32_MAX;
    for (uint32_t k = 0; k < itos.size(); ++k)
      if (itos[k] == ch) { id = k; break; }
    if (id == UINT32_MAX) throw std::runtime_error("character not in vocabulary: '" + ch + "'");
    ids.push_back(id);
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
  prefillM_ = 0;
  for (const auto& p : bin.programs)
    if (p.M > prefillM_) prefillM_ = p.M;
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
  StepResult s;
  s.pos = pos;
  s.M = M;
  s.perf = dev_.run(prog.pc, pos);
  s.logits = readLogitsRow(logitsRow);
  uint32_t best = 0;
  for (uint32_t i = 1; i < vocab_; ++i)
    if (s.logits[i] > s.logits[best]) best = i;
  s.argmax = best;
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
    if (verbose) std::printf("  prefill chunk %u/%u: pos=%u cycles=%llu\n", c + 1, nChunks, c * prefillM_, (unsigned long long)s.perf[PERF_CYCLES]);
    next = s.argmax;
    r.steps.push_back(std::move(s));
  }
  for (uint32_t i = 0; i < nTokens; ++i) {
    r.generated.push_back(next);
    if (L + i + 1 >= maxSeq_) break;  // no room for another position
    StepResult s = launch({next}, 1, L + i, 0);
    if (verbose) std::printf("  decode %u/%u: pos=%u cycles=%llu tok=%u '%s'\n", i + 1, nTokens, L + i,
                             (unsigned long long)s.perf[PERF_CYCLES], next, tok_.itos[next].c_str());
    next = s.argmax;
    r.steps.push_back(std::move(s));
  }
  auto t1 = std::chrono::steady_clock::now();
  for (const auto& s : r.steps)
    for (uint32_t i = 0; i < kNumPerf; ++i) r.total[i] += s.perf[i];
  if (verbose) std::printf("  wall time: %.2f s\n", std::chrono::duration<double>(t1 - t0).count());
  return r;
}

nlohmann::json GenerationResult::toJson() const {
  nlohmann::json j;
  j["prompt_tokens"] = promptTokens;
  j["generated"] = generated;
  std::vector<uint32_t> am;
  std::vector<std::vector<int16_t>> lg;
  nlohmann::json steps = nlohmann::json::array();
  for (const auto& s : this->steps) {
    am.push_back(s.argmax);
    lg.push_back(s.logits);
    nlohmann::json sj;
    sj["pos"] = s.pos;
    sj["M"] = s.M;
    sj["argmax"] = s.argmax;
    nlohmann::json pj;
    for (uint32_t i = 0; i < kNumPerf; ++i) pj[std::string(kPerfNames[i])] = s.perf[i];
    sj["perf"] = pj;
    steps.push_back(sj);
  }
  j["argmax_per_step"] = am;
  j["logits_last_rows"] = lg;
  j["steps"] = steps;
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
  if (gp != r.promptTokens) { std::printf("VERIFY: prompt tokens differ from golden\n"); ok = false; }
  size_t n = std::min(ga.size(), r.steps.size());
  for (size_t i = 0; i < n; ++i)
    if (ga[i] != r.steps[i].argmax) {
      std::printf("VERIFY: step %zu argmax %u != golden %u (pos=%u M=%u)\n", i, r.steps[i].argmax, ga[i], r.steps[i].pos, r.steps[i].M);
      ok = false;
      break;
    }
  if (ok && ga.size() != r.steps.size())
    std::printf("VERIFY: note: %zu steps run vs %zu golden steps (compared the first %zu)\n", r.steps.size(), ga.size(), n);
  if (ok && g.contains("logits_last_rows")) {
    auto gl = g.at("logits_last_rows").get<std::vector<std::vector<int16_t>>>();
    for (size_t i = 0; i < std::min(gl.size(), r.steps.size()); ++i)
      if (gl[i] != r.steps[i].logits) {
        size_t k = 0;
        while (k < gl[i].size() && k < r.steps[i].logits.size() && gl[i][k] == r.steps[i].logits[k]) ++k;
        std::printf("VERIFY: step %zu logits differ at index %zu: %d vs golden %d\n", i, k, k < r.steps[i].logits.size() ? r.steps[i].logits[k] : 0,
                    k < gl[i].size() ? gl[i][k] : 0);
        ok = false;
        break;
      }
  }
  size_t ng = std::min(gg.size(), r.generated.size());
  for (size_t i = 0; ok && i < ng; ++i)
    if (gg[i] != r.generated[i]) { std::printf("VERIFY: generated token %zu differs\n", i); ok = false; }
  if (verbose || !ok) std::printf("VERIFY: %s (%zu launches, %zu generated tokens compared)\n", ok ? "MATCH" : "MISMATCH", n, ng);
  return ok;
}

void printPerf(const PerfCounters& p, uint32_t launches) {
  std::printf("Performance (%u launches)\n-------------------------\n", launches);
  for (uint32_t i = 0; i < kNumPerf; ++i) std::printf("  %-22s %14llu\n", std::string(kPerfNames[i]).c_str(), (unsigned long long)p[i]);
  if (p[PERF_CYCLES]) {
    std::printf("  GEMM utilization       %6.2f %%\n", 100.0 * double(p[PERF_GEMM_MAC_CYCLES]) / double(p[PERF_CYCLES]));
    std::printf("  ATTN MAC utilization   %6.2f %%\n", 100.0 * double(p[PERF_ATTN_MAC_CYCLES]) / double(p[PERF_CYCLES]));
    std::printf("  CP wait stalls         %6.2f %%\n", 100.0 * double(p[PERF_CP_STALL_WAIT]) / double(p[PERF_CYCLES]));
    std::printf("  DMA busy               %6.2f %%\n", 100.0 * double(p[PERF_DMA_BUSY]) / double(p[PERF_CYCLES]));
  }
}

}  // namespace llaccel
