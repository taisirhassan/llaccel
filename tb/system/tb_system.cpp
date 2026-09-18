// Full-core integration: queued DMA, semaphore dependencies, GEMM -> QUANT ->
// DMA store, HALT prefetch discard, and repeated launches without resetting RTL.
#include "Vllaccel_top.h"
#include "tb_common.h"
#include "verilated.h"
using namespace llaccel;
using namespace tb;
int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  Rng rng(seedFromArgs(argc, argv));
  Vllaccel_top top;
  DramModel dram(1 << 20, 37, 7);
  DramDriver<Vllaccel_top> drv(dram);
  auto tick = [&] {
    top.eval(); drv.preEdge(top); top.clk = 1; top.eval();
    drv.postEdge(top); top.clk = 0; top.eval();
  };
  top.clk = 0; top.rst_n = 0; top.start = 0; top.pos = 0;
  drv.init(top);
  for (int i = 0; i < 4; ++i) tick();
  top.rst_n = 1; tick();
  Report rep;
  for (unsigned test = 0; test < argValue(argc, argv, "cases", 30); ++test) {
    GemmParams p;
    p.M = rng.range(1, 16); p.N = 16 * rng.range(1, 4); p.K = 16 * rng.range(1, 8);
    std::vector<int8_t> a(p.M * p.K), w(p.N * p.K);
    for (auto& v : a) v = int8_t(rng.u64());
    for (auto& v : w) v = int8_t(rng.u64());
    std::vector<num::RqEntry> rq(p.N, {1 << 30, 36});
    auto tiled = tileWeights(w.data(), p.N, p.K);
    dram.write(0x10000, a.data(), a.size());
    dram.write(0x20000, tiled.data(), tiled.size());
    dram.write(0x30000, rq.data(), rq.size() * 8);
    std::memset(dram.data() + 0x80000, 0xa5, 0x10000);
    std::vector<Instr> program;
    // Fill the DMA queue, exercise qfull and CP/DMA DRAM response arbitration.
    for (int i = 0; i < 12; ++i) {
      auto in = mkDmaLoad(0x70000 + i * 64, 0x10000, 1, 64, 64, 64);
      in.setSignal(0); program.push_back(in);
    }
    auto addLoad = [&](uint32_t s, uint32_t d, uint32_t n) {
      auto in = mkDmaLoad(s, d, 1, n, n, n); in.setSignal(0); program.push_back(in);
    };
    addLoad(0x10000, 0x10000, a.size());
    addLoad(0x20000, 0x20000, tiled.size());
    addLoad(0x30010, 0x30000, rq.size() * 8);
    auto g = mkGemm(p, 0x10000, 0x20000, 0x400f0, 0x30010, 0, 0);
    g.setWait(0, 15); g.setSignal(1); program.push_back(g);
    auto q = mkVecQuant(0x400f0, 0x500f0, p.M * p.N, 1 << 30, 32);
    q.setWait(1, 1); q.setSignal(2); program.push_back(q);
    auto st = mkDmaStore(0x500f0, 0x80010, 1, p.M * p.N, p.M * p.N, p.M * p.N);
    st.setWait(2, 1); st.setSignal(3); program.push_back(st);
    for (uint32_t base : {0x600f0u, 0x610f0u}) {
      Instr kv(Op::KV_WRITE); kv.w[2]=0x500f0; kv.w[3]=base;
      kv.w[4]=1; kv.w[5]=1; kv.w[6]=16; kv.w[7]=256;
      kv.setWait(2,1); kv.setSignal(4); program.push_back(kv);
    }
    Instr at(Op::ATTN);
    at.w[2]=0x500f0; at.w[3]=0x620f0; at.w[4]=0x600f0; at.w[5]=0x610f0;
    at.w[6]=1; at.w[7]=1; at.w[8]=1; at.w[9]=16; at.w[10]=256;
    at.w[11]=1; at.w[12]=0; at.w[13]=1; at.w[14]=8;
    at.setWait(4,2); at.setSignal(5); program.push_back(at);
    auto ast=mkDmaStore(0x620f0,0x84010,1,16,16,16);
    ast.setWait(5,1); ast.setSignal(3); program.push_back(ast);
    auto halt = mkHalt(); halt.setWait(3, 2); program.push_back(halt);
    // Already-prefetched instructions beyond HALT must never issue.
    program.push_back(mkDmaStore(0x10000, 0x88000, 1, 64, 64, 64));
    dram.write(0, program.data(), program.size() * sizeof(Instr));
    auto expect = refGemm(p, a.data(), w.data(), rq.data(), nullptr, nullptr);
    std::vector<uint8_t> reference(dram.data(), dram.data() + dram.size());
    for (unsigned j = 0; j < p.M * p.N; ++j) {
      int16_t x; std::memcpy(&x, expect.data() + j * 2, 2);
      reference[0x80010 + j] = uint8_t(num::sat8(num::mulshift(x, 1 << 30, 32)));
    }
    std::array<int8_t,16> av, ao;
    std::memcpy(av.data(),reference.data()+0x80010,16);
    std::memcpy(reference.data()+0x600f0,av.data(),16);
    std::memcpy(reference.data()+0x610f0,av.data(),16);
    std::vector<int32_t> scores; std::vector<uint16_t> probs;
    auto atRow=[&](uint32_t){return av.data();};
    num::attention_head(av,16,1,atRow,atRow,1,0,1,8,ao,scores,probs);
    std::memcpy(reference.data()+0x84010,ao.data(),16);
    top.pc_start = 0; top.start = 1; tick(); top.start = 0;
    bool ok = !top.done;
    unsigned cycles = 0;
    while (!top.done && cycles++ < 200000) tick();
    ok &= top.done && firstDiff(dram.data(), reference.data(), reference.size()) < 0;
    ok &= top.perf[1] == program.size() - 1;
    ok &= top.perf[6] == (p.N / 16) * (p.K / 16) * p.M;
    char name[128]; std::snprintf(name, sizeof name, "launch %u M=%u N=%u K=%u cycles=%u", test, p.M, p.N, p.K, cycles);
    rep.tally(ok, name);
    if (!ok) {
      std::printf("done=%u issued=%llu mac=%llu first difference=%ld\n", top.done,
        (unsigned long long)top.perf[1], (unsigned long long)top.perf[6],
        firstDiff(dram.data(), reference.data(), reference.size()));
      break;
    }
    for (int i = 0; i < 5; ++i) tick();
    rep.tally(top.done, "HALT done remains asserted");
  }
  top.final();
  return rep.finish("tb_system (real engines)");
}
