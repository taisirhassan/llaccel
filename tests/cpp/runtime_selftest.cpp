#include "llaccel/host.h"
#include "llaccel/dram_model.h"
#include <filesystem>
#include <fstream>
#include <print>
#include <stdexcept>
#include <limits>
#include <cstring>

using namespace llaccel;
void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
template<class F> void rejects(F f, const char* msg) {
  bool threw = false;
  try { f(); } catch (const std::exception&) { threw = true; }
  require(threw, msg);
}
class ConservativeDevice final : public Device {
 public:
  unsigned uploads = 0;
  std::string name() const override { return "conservative mock"; }
  void dramWrite(uint64_t, const void*, uint64_t) override { ++uploads; }
  void dramRead(uint64_t, void*, uint64_t) const override {}
  uint64_t dramSize() const override { return 1u << 20; }
  PerfCounters run(uint32_t, uint32_t) override { return {}; }
  std::vector<uint8_t> sramSnapshot() const override { return {}; }
};
int main(int argc, char** argv) {
  try {
    const auto dir = std::filesystem::path(argc > 1 ? argv[1] : ".") / "runtime-selftest";
    std::filesystem::create_directories(dir);
    auto golden = (dir / "golden.json").string();
    GenerationResult r;
    r.promptTokens = {0}; r.generated = {1};
    r.steps.push_back({.pos=0, .M=16, .argmax=1, .logits={-3, 4}, .perf={}});
    auto save = [&](const nlohmann::json& j) { std::ofstream(golden) << j; };
    auto good = r.toJson();
    save(good);
    require(verifyAgainstGolden(r, golden, false), "identical trace rejected");
    auto j=good; j["argmax_per_step"].push_back(1); save(j);
    require(!verifyAgainstGolden(r, golden, false), "extra golden launch accepted");
    j=good; j["generated"].push_back(1); save(j);
    require(!verifyAgainstGolden(r, golden, false), "extra golden token accepted");
    j=good; j["logits_last_rows"] = nlohmann::json::array(); save(j);
    require(!verifyAgainstGolden(r, golden, false), "missing logits accepted");
    j=good; j["logits_last_rows"][0][0] = -2; save(j);
    require(!verifyAgainstGolden(r, golden, false), "incorrect non-argmax logit accepted");
    j=good; j["argmax_per_step"] = nlohmann::json::array(); save(j);
    require(!verifyAgainstGolden(r, golden, false), "empty golden trace accepted");
    j=good; j.erase("logits_last_rows"); save(j);
    rejects([&]{verifyAgainstGolden(r, golden, false);}, "golden without logits accepted");
    unsigned malformedCases = 0;
    const auto malformedLogits = nlohmann::json::array({-3.5, -3.0, 65533, -65539,
        std::numeric_limits<uint64_t>::max(), true, nullptr, "-3"});
    for (const auto& value : malformedLogits) {
      j=good; j["logits_last_rows"][0][0]=value; save(j);
      rejects([&]{verifyAgainstGolden(r, golden, false);}, "malformed golden logit accepted");
      ++malformedCases;
    }
    for (const auto* field : {"prompt_tokens", "generated", "argmax_per_step"}) {
      for (const auto& value : nlohmann::json::array({1.5, 1.0, -1, uint64_t(1) << 32, true})) {
        j=good; j[field][0]=value; save(j);
        rejects([&]{verifyAgainstGolden(r, golden, false);}, "malformed golden token/index accepted");
        ++malformedCases;
      }
      j=good; j[field]=nullptr; save(j);
      rejects([&]{verifyAgainstGolden(r, golden, false);}, "non-array golden tokens accepted");
      ++malformedCases;
    }
    j=good; j["logits_last_rows"]=nullptr; save(j);
    rejects([&]{verifyAgainstGolden(r, golden, false);}, "non-array golden logits accepted");
    ++malformedCases;
    j=good; j["logits_last_rows"][0]=nullptr; save(j);
    rejects([&]{verifyAgainstGolden(r, golden, false);}, "non-array golden logit row accepted");
    ++malformedCases;
    GenerationResult extremes=r;
    extremes.steps[0].logits={std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()};
    save(extremes.toJson());
    require(verifyAgainstGolden(extremes, golden, false), "valid extreme logits rejected");
    DramModel dram(128);
    uint8_t byte=0;
    rejects([&]{dram.read(UINT64_MAX, &byte, 2);}, "wrapping DRAM read accepted");
    rejects([&]{dram.write(UINT64_MAX, &byte, 2);}, "wrapping DRAM write accepted");
    dram.read(128, &byte, 0); dram.write(128, &byte, 0);
    const auto binPath = (dir / "overflow.llbin").string();
    { std::ofstream f(binPath, std::ios::binary);
      uint32_t header[] = {kLlbinMagic,kLlbinVersion,1};
      SectionHeader h{}; h.kind=uint32_t(Section::DRAM_IMAGE); h.offset=UINT64_MAX; h.size=2;
      f.write(reinterpret_cast<const char*>(header),sizeof header);
      f.write(reinterpret_cast<const char*>(&h),sizeof h);
    }
    rejects([&]{Llbin::load(binPath);}, "wrapping llbin section accepted");
    auto dev = makeFuncSim({});
    rejects([&]{dev->dramRead(UINT64_MAX, &byte, 2);}, "wrapping func read accepted");
    rejects([&]{dev->dramWrite(UINT64_MAX, &byte, 2);}, "wrapping func write accepted");
    Instr invalid(Op::NOP); invalid.setWait(32,1);
    dev->dramWrite(0,invalid.w.data(),kInstrBytes);
    rejects([&]{dev->run(0,0);}, "invalid semaphore accepted");
    invalid = Instr(static_cast<Op>(2)); dev->dramWrite(0,invalid.w.data(),kInstrBytes);
    rejects([&]{dev->run(0,0);}, "invalid CP opcode accepted");
    Instr load(Op::DMA_LOAD); load.dmaSram()=0; load.dmaDram()=512+48;
    load.dmaRows()=2; load.dmaRowBytes()=32; load.dmaSrcStride()=64; load.dmaDstStride()=32;
    Instr halt(Op::HALT); dev->dramWrite(0,load.w.data(),kInstrBytes);
    dev->dramWrite(64,halt.w.data(),kInstrBytes);
    auto perf=dev->run(0,0);
    require(perf[PERF_DRAM_RD_BYTES]==384 && perf[PERF_SRAM_WR_BYTES]==64,
            "DMA counters fail to include both partially used beats per row plus instruction fetches");
    Instr badShift(Op::VEC_QUANT); badShift[4] = 16; badShift[6] = 64;
    dev->dramWrite(0,badShift.w.data(),kInstrBytes);
    rejects([&]{dev->run(0,0);}, "invalid arithmetic shift accepted");
    load = Instr(Op::DMA_STORE); load.dmaSram()=16; load.dmaDram()=512+48;
    load.dmaRows()=1; load.dmaRowBytes()=64; load.dmaSrcStride()=64; load.dmaDstStride()=64;
    dev->dramWrite(0,load.w.data(),kInstrBytes);
    perf=dev->run(0,0);
    require(perf[PERF_DRAM_WR_BYTES]==128 && perf[PERF_SRAM_RD_BYTES]==192,
            "DMA store bus traffic excludes partial/repeated SRAM reads");
    // Externally tokenized prompts never need a character vocabulary allocation.
    const auto idsPath = (dir / "prompt-ids.json").string();
    std::ofstream(idsPath) << "[0,1,4294967295]";
    require(loadPromptIds(idsPath) == std::vector<uint32_t>({0,1,UINT32_MAX}), "valid token IDs rejected");
    unsigned tokenCases = 0;
    for (const auto& invalidIds : {"null", "{}", "[1.0]", "[-1]", "[4294967296]", "[true]", "[null]", "[\"1\"]", "["}) {
      std::ofstream(idsPath) << invalidIds;
      rejects([&]{loadPromptIds(idsPath);}, "malformed prompt IDs accepted");
      ++tokenCases;
    }
    Llbin tiny;
    tiny.dramImage.resize(256);
    tiny.programs.push_back({.M=1, .pc=0, .instrs={halt}});
    std::memcpy(tiny.dramImage.data(), halt.w.data(), kInstrBytes);
    tiny.meta = {{"model", {{"vocab",2}, {"dim",1}, {"max_seq",16}}},
                 {"dram", {{"embedding", {{"addr",64}, {"row_bytes",2}}},
                           {"input", {{"addr",128}, {"row_bytes",2}}},
                           {"logits", {{"addr",192}, {"row_bytes",4}}}}}};
    ConservativeDevice limited;
    auto extended = tiny;
    extended.meta["model"]["head_dim"] = 128;
    rejects([&]{ Host unsupported(extended, limited); }, "wide-head hardware image accepted");
    require(limited.uploads == 0, "unsupported image uploaded before capability rejection");
    require(dev->maxAttentionHeadDim() == 256, "functional head capability missing");
    Host supported(extended, *dev);
    extended.meta["model"].erase("head_dim");
    for (auto op : {Op::ATTN, Op::KV_WRITE, Op::VEC_ROPE}) {
      Instr wide(op); wide[op == Op::ATTN ? 9 : 6] = 256;
      extended.programs[0].instrs = {wide, halt};
      rejects([&]{ Host unsupported(extended, limited); }, "program bypassed hardware head capability");
      require(limited.uploads == 0, "unsupported program uploaded before rejection");
    }
    Host tokenHost(tiny, *dev);
    require(tokenHost.generateTokens({1}, 1, false).promptTokens == std::vector<uint32_t>{1},
            "token-only host rejected valid IDs");
    rejects([&]{tokenHost.generateTokens({}, 1, false);}, "empty ID prompt accepted");
    rejects([&]{tokenHost.generateTokens({2}, 1, false);}, "out-of-vocabulary ID accepted");
    rejects([&]{tokenHost.generateTokens({UINT32_MAX}, 1, false);}, "padding sentinel accepted as prompt ID");
    rejects([&]{tokenHost.generate("a", 1, false);}, "text generation without tokenizer accepted");
    rejects([&]{tokenHost.generateTokens(std::vector<uint32_t>(17,0), 0, false);}, "oversized ID prompt accepted");
    Tokenizer chars{{"a", "b"}};
    Host charHost(tiny, *dev, chars);
    require(charHost.generate("b", 1, false).toJson() == tokenHost.generateTokens({1}, 1, false).toJson(),
            "character path differs from token-ID path");
    std::println("token-ID tests: PASS ({} parser rejection cases plus host bounds and compatibility)", tokenCases);
    std::println("runtime_selftest: PASS (19 verification/bounds/ISA cases + {} malformed golden cases)", malformedCases);
    return 0;
  } catch(const std::exception& e) { std::println(stderr,"FAIL: {}",e.what()); return 1; }
}
