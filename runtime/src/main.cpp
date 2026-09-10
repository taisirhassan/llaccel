// llaccel-sim: run a compiled .llbin on the functional simulator or the
// Verilated RTL, generate tokens, verify against the golden trace, report perf.
#include <fstream>
#include <print>
#include <span>
#include <string>
#include <string_view>

#include "llaccel/device.h"
#include "llaccel/host.h"
#include "llaccel/llbin.h"

using namespace llaccel;

namespace {

constexpr std::string_view kUsage =
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
    "  --quiet                   less output\n";

struct Args {
  std::string bin, backend = "func", prompt = "ROMEO:", tokenizer, verify, out;
  uint32_t tokens = 32;
  bool quiet = false;
  DeviceOptions dev;
};

Args parseArgs(std::span<char*> argv) {
  Args a;
  a.bin = argv[1];
  for (size_t i = 2; i < argv.size(); ++i) {
    std::string_view f = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argv.size()) { std::println(stderr, "{} needs a value", f); std::exit(2); }
      return argv[++i];
    };
    if (f == "--backend") a.backend = value();
    else if (f == "--prompt") a.prompt = value();
    else if (f == "--tokens") a.tokens = std::stoul(value());
    else if (f == "--tokenizer") a.tokenizer = value();
    else if (f == "--verify") a.verify = value();
    else if (f == "--out") a.out = value();
    else if (f == "--interleave") a.dev.interleaveSeed = std::stoul(value());
    else if (f == "--trace") a.dev.trace = true;
    else if (f == "--dram-latency") a.dev.dramLatency = std::stoul(value());
    else if (f == "--fst") a.dev.fstPath = value();
    else if (f == "--quiet") a.quiet = true;
    else if (f == "--help" || f == "-h") { std::print("{}", kUsage); std::exit(0); }
    else { std::println(stderr, "unknown option {}\n{}", f, kUsage); std::exit(2); }
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h") { std::print("{}", kUsage); return argc < 2 ? 2 : 0; }
  Args a = parseArgs({argv, size_t(argc)});
  try {
    Llbin bin = Llbin::load(a.bin);
    if (a.tokenizer.empty()) {
      auto slash = a.bin.find_last_of('/');
      a.tokenizer = (slash == std::string::npos ? std::string() : a.bin.substr(0, slash + 1)) + "tokenizer.json";
    }
    Tokenizer tok = Tokenizer::load(a.tokenizer);
    a.dev.epilogueFusion = bin.meta.value("fusion", false) || bin.meta.value("target", std::string()) == "llaccel-v2";
    a.dev.dramBytes = bin.dramImage.size() + (1u << 20);
    std::unique_ptr<Device> dev = a.backend == "rtl" ? makeRtlSim(a.dev) : makeFuncSim(a.dev);
    if (!a.quiet) {
      std::println("Loading program...");
      for (const auto& p : bin.programs) std::println("  program M={:<2}  instructions: {}", p.M, p.instrs.size());
      if (bin.meta.contains("sram"))
        std::println("  SRAM peak use:     {:.1f}% of {} bytes",
                     100.0 * bin.meta["sram"].value("peak_used", 0.0) / bin.meta["sram"].value("bytes", 1.0),
                     bin.meta["sram"].value("bytes", 0ull));
      std::println("  target: {}  fusion: {}  schedule: {}", bin.meta.value("target", "?"),
                   bin.meta.value("fusion", false) ? "on" : "off", bin.meta.value("schedule", "?"));
      std::println("\nExecuting on {}...", dev->name());
    }
    Host host(bin, *dev, tok);
    GenerationResult r = host.generate(a.prompt, a.tokens, !a.quiet);
    std::println("\nprompt:  {}\noutput:  {}", a.prompt, tok.decode(r.generated));
    if (!a.quiet) {
      std::println("");
      printPerf(r.total, uint32_t(r.steps.size()));
      uint64_t decCycles = 0, decN = 0, preCycles = 0, preN = 0;
      for (const auto& s : r.steps) {
        if (s.M == 1) { decCycles += s.perf[PERF_CYCLES]; decN++; }
        else { preCycles += s.perf[PERF_CYCLES]; preN++; }
      }
      if (decN) std::println("  cycles/token (decode)  {:14.1f}  over {} tokens", double(decCycles) / double(decN), decN);
      if (preN) std::println("  cycles/prefill chunk   {:14.1f}  over {} chunks", double(preCycles) / double(preN), preN);
    }
    int rc = 0;
    if (!a.verify.empty()) {
      bool ok = verifyAgainstGolden(r, a.verify, true);
      std::println("PyTorch/golden reference:  {}", ok ? "MATCH" : "MISMATCH");
      rc = ok ? 0 : 1;
    }
    if (!a.out.empty()) std::ofstream(a.out) << r.toJson().dump(1) << "\n";
    return rc;
  } catch (const std::exception& e) {
    std::println(stderr, "error: {}", e.what());
    return 1;
  }
}
