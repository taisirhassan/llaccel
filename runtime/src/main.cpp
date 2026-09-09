// llaccel-sim: run a compiled .llbin on the functional simulator or the
// Verilated RTL, generate tokens, verify against the golden trace, report perf.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "llaccel/device.h"
#include "llaccel/host.h"
#include "llaccel/llbin.h"

using namespace llaccel;

static void usage() {
  std::fprintf(stderr,
               "usage: llaccel-sim <model.llbin> [options]\n"
               "  --backend func|rtl        (default func)\n"
               "  --prompt TEXT             (default \"ROMEO:\")\n"
               "  --tokens N                greedy tokens to generate (default 32)\n"
               "  --tokenizer PATH          (default: <llbin dir>/tokenizer.json)\n"
               "  --verify golden.json      compare tokens+logits with python -m llaccel.golden output\n"
               "  --out result.json         write tokens, logits and per-launch perf counters\n"
               "  --interleave SEED         func: random legal engine interleaving (hazard check)\n"
               "  --trace                   func: print every executed instruction\n"
               "  --dram-latency N          rtl: DRAM read latency in cycles (default 100)\n"
               "  --fst PATH                rtl: write an FST waveform\n"
               "  --quiet                   less output\n");
}

int main(int argc, char** argv) {
  if (argc < 2) { usage(); return 2; }
  std::string binPath = argv[1], backend = "func", prompt = "ROMEO:", tokPath, verifyPath, outPath;
  uint32_t nTokens = 32;
  DeviceOptions opt;
  bool quiet = false;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(2); }
      return argv[++i];
    };
    if (a == "--backend") backend = need("--backend");
    else if (a == "--prompt") prompt = need("--prompt");
    else if (a == "--tokens") nTokens = std::stoul(need("--tokens"));
    else if (a == "--tokenizer") tokPath = need("--tokenizer");
    else if (a == "--verify") verifyPath = need("--verify");
    else if (a == "--out") outPath = need("--out");
    else if (a == "--interleave") opt.interleaveSeed = std::stoul(need("--interleave"));
    else if (a == "--trace") opt.trace = true;
    else if (a == "--dram-latency") opt.dramLatency = std::stoul(need("--dram-latency"));
    else if (a == "--fst") opt.fstPath = need("--fst");
    else if (a == "--quiet") quiet = true;
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
  }
  try {
    Llbin bin = Llbin::load(binPath);
    if (tokPath.empty()) {
      auto slash = binPath.find_last_of('/');
      tokPath = (slash == std::string::npos ? std::string() : binPath.substr(0, slash + 1)) + "tokenizer.json";
    }
    Tokenizer tok = Tokenizer::load(tokPath);
    opt.epilogueFusion = bin.meta.value("fusion", false) || bin.meta.value("target", std::string()) == "llaccel-v2";
    opt.dramBytes = bin.dramImage.size() + (1u << 20);
    std::unique_ptr<Device> dev = backend == "rtl" ? makeRtlSim(opt) : makeFuncSim(opt);
    if (!quiet) {
      std::printf("Loading program...\n");
      const auto& st = bin.meta.value("stats", nlohmann::json::object());
      for (const auto& p : bin.programs)
        std::printf("  program M=%-2u  instructions: %zu\n", p.M, p.instrs.size());
      if (bin.meta.contains("sram"))
        std::printf("  SRAM peak use:     %.1f%% of %llu bytes\n",
                    100.0 * bin.meta["sram"].value("peak_used", 0.0) / bin.meta["sram"].value("bytes", 1.0),
                    (unsigned long long)bin.meta["sram"].value("bytes", 0ull));
      std::printf("  target: %s  fusion: %s  schedule: %s\n", bin.meta.value("target", "?").c_str(),
                  bin.meta.value("fusion", false) ? "on" : "off", bin.meta.value("schedule", "?").c_str());
      (void)st;
      std::printf("\nExecuting on %s...\n", dev->name().c_str());
    }
    Host host(bin, *dev, tok);
    GenerationResult r = host.generate(prompt, nTokens, !quiet);
    std::string text = tok.decode(r.generated);
    std::printf("\nprompt:  %s\noutput:  %s\n", prompt.c_str(), text.c_str());
    if (!quiet) {
      std::printf("\n");
      printPerf(r.total, uint32_t(r.steps.size()));
      // Per-token decode cost: average over decode launches (M == 1).
      uint64_t decCycles = 0, decN = 0, preCycles = 0, preN = 0;
      for (const auto& s : r.steps) {
        if (s.M == 1) { decCycles += s.perf[PERF_CYCLES]; decN++; }
        else { preCycles += s.perf[PERF_CYCLES]; preN++; }
      }
      if (decN) std::printf("  cycles/token (decode)  %14.1f  over %llu tokens\n", double(decCycles) / double(decN), (unsigned long long)decN);
      if (preN) std::printf("  cycles/prefill chunk   %14.1f  over %llu chunks\n", double(preCycles) / double(preN), (unsigned long long)preN);
    }
    int rc = 0;
    if (!verifyPath.empty()) {
      bool ok = verifyAgainstGolden(r, verifyPath, true);
      std::printf("PyTorch/golden reference:  %s\n", ok ? "MATCH" : "MISMATCH");
      rc = ok ? 0 : 1;
    }
    if (!outPath.empty()) {
      std::ofstream f(outPath);
      f << r.toJson().dump(1) << "\n";
    }
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
