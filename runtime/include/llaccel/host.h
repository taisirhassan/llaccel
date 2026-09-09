// Host-side model execution: tokenizer, embedding lookup, chunked prefill,
// greedy decode, verification against the golden trace, perf reporting.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "llaccel/device.h"
#include "llaccel/llbin.h"

namespace llaccel {

struct Tokenizer {
  std::vector<std::string> itos;  // UTF-8 strings per token id
  static Tokenizer load(const std::string& path);
  std::vector<uint32_t> encode(const std::string& text) const;  // char-level, longest-match not needed
  std::string decode(const std::vector<uint32_t>& ids) const;
};

struct StepResult {
  uint32_t pos;          // POS register used for the launch
  uint32_t M;            // program rows
  uint32_t argmax;       // greedy token from the relevant logits row
  std::vector<int16_t> logits;  // the relevant logits row (vocab entries, i16 at E_LOGIT)
  PerfCounters perf;
};

struct GenerationResult {
  std::vector<uint32_t> promptTokens, generated;
  std::vector<StepResult> steps;  // one per launch (prefill chunks first, then decode steps)
  PerfCounters total{};
  nlohmann::json toJson() const;
};

class Host {
 public:
  Host(const Llbin& bin, Device& dev, const Tokenizer& tok);
  // Runs chunked prefill on `prompt` followed by `nTokens` greedy decode steps.
  GenerationResult generate(const std::string& prompt, uint32_t nTokens, bool verbose);
  uint32_t vocab() const { return vocab_; }
  uint32_t dim() const { return dim_; }

 private:
  StepResult launch(const std::vector<uint32_t>& rowTokens, uint32_t M, uint32_t pos, uint32_t logitsRow);
  void writeInputRow(uint32_t row, uint32_t token);
  std::vector<int16_t> readLogitsRow(uint32_t row);

  const Llbin& bin_;
  Device& dev_;
  const Tokenizer& tok_;
  uint32_t vocab_, dim_, maxSeq_, prefillM_;
  uint64_t embAddr_, embRowBytes_, inAddr_, inRowBytes_, lgAddr_, lgRowBytes_;
};

// Compare a generation against golden.json (python -m llaccel.golden). Returns true on match; prints details.
bool verifyAgainstGolden(const GenerationResult& r, const std::string& goldenPath, bool verbose);

void printPerf(const PerfCounters& p, uint32_t launches);

}  // namespace llaccel
