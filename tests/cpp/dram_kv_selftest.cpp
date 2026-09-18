// DRAM KV contract: full-width addresses, tile boundaries, GQA and session reset.
#include "llaccel/host.h"
#include "llaccel/numerics.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace llaccel;
static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class F> static void rejects(F f) {
  bool threw=false; try { f(); } catch(const std::exception&) {threw=true;}
  require(threw,"invalid DRAM KV access accepted");
}
int main() {
 try {
  unsigned cases=0;
  std::mt19937 rng(193);
  constexpr uint32_t cacheBase=0x01010000, qDram=0x10000, qSram=256, outSram=4096;
  for (uint32_t seed : {0u,7u}) {
   DeviceOptions options; options.interleaveSeed=seed;
   auto dev=makeFuncSim(options);
   for (uint32_t D : {16u,32u,64u,128u,256u}) {
    const uint32_t stride=kAttnTMax*D, vb=cacheBase+stride;
    std::vector<int8_t> keys(stride), vals(stride);
    for(auto& x:keys)x=int8_t(rng()); for(auto& x:vals)x=int8_t(rng());
    dev->dramWrite(cacheBase,keys.data(),keys.size());
    dev->dramWrite(vb,vals.data(),vals.size());
    for (uint32_t T : {1u,255u,256u,257u,511u,512u,513u,1024u,4096u}) {
     constexpr uint32_t H=3;
     std::vector<int8_t> q(H*D), expected(H*D);
     for(auto& x:q)x=int8_t(rng());
     dev->dramWrite(qDram,q.data(),q.size());
     Instr load(Op::DMA_LOAD); load[2]=qSram;load[3]=qDram;load[4]=1;load[5]=q.size();
     load[6]=load[7]=q.size();load.setSignal(0);
     Instr attn(Op::ATTN,kFlagAttnWideProb);attn[2]=qSram;attn[3]=outSram;
     attn[4]=cacheBase;attn[5]=vb;attn[6]=1;attn[7]=H;attn[8]=1;attn[9]=D;
     attn[10]=stride;attn[11]=1u<<30;attn[12]=36;attn[13]=1u<<30;attn[14]=38;
     attn.setWait(0,1);
     Instr halt(Op::HALT);
     dev->dramWrite(0,load.w.data(),64);dev->dramWrite(64,attn.w.data(),64);dev->dramWrite(128,halt.w.data(),64);
     auto perf=dev->run(0,T-1);
     std::vector<int32_t> scores;std::vector<uint16_t> probs;
     for(uint32_t h=0;h<H;++h)
      num::attention_head(std::span<const int8_t>(q.data()+h*D,D),D,T,
       [&](uint32_t t){return keys.data()+t*D;},[&](uint32_t t){return vals.data()+t*D;},
       attn[11],attn[12],attn[13],attn[14],std::span<int8_t>(expected.data()+h*D,D),scores,probs,true);
     auto sram=dev->sramSnapshot();
     require(std::memcmp(sram.data()+outSram,expected.data(),expected.size())==0,"DRAM attention differs from untiled oracle");
     require(perf[PERF_ATTN_MAC_CYCLES]==4ull*T*H,"DRAM attention pass counter wrong");
     ++cases;
    }
    // Independent uniform-value oracle at1024/4096, unaffected by tile boundaries.
    std::fill(keys.begin(),keys.end(),0);std::fill(vals.begin(),vals.end(),127);
    dev->dramWrite(cacheBase,keys.data(),keys.size());dev->dramWrite(vb,vals.data(),vals.size());
    for(uint32_t T:{1024u,4096u}) {
     dev->run(0,T-1);
     auto sram=dev->sramSnapshot();
     require(std::all_of(sram.begin()+outSram,sram.begin()+outSram+3*D,
                        [](uint8_t value){return value==127;}),"uniform long-context attention changed constant");
     ++cases;
    }
    rejects([&]{dev->run(0,kAttnTMax);});
    ++cases;
   }
   // Four rows straddle the1024 boundary and preserve untouched DRAM bytes.
   constexpr uint32_t M=4,Hkv=2,D=32,stride=kAttnTMax*D,pos=1022;
   std::vector<uint8_t> cache(Hkv*stride,0xA5),src(M*Hkv*D),want=cache;
   for(auto& x:src)x=uint8_t(rng());
   dev->dramWrite(cacheBase,cache.data(),cache.size());dev->dramWrite(qDram,src.data(),src.size());
   for(uint32_t m=0;m<M;++m)for(uint32_t h=0;h<Hkv;++h)
    std::memcpy(want.data()+h*stride+(pos+m)*D,src.data()+(m*Hkv+h)*D,D);
   Instr load(Op::DMA_LOAD);load[2]=qSram;load[3]=qDram;load[4]=1;load[5]=src.size();load[6]=load[7]=src.size();load.setSignal(0);
   Instr write(Op::KV_WRITE);write[2]=qSram;write[3]=cacheBase;write[4]=M;write[5]=Hkv;write[6]=D;write[7]=stride;write.setWait(0,1);
   Instr halt(Op::HALT);
   dev->dramWrite(0,load.w.data(),64);dev->dramWrite(64,write.w.data(),64);dev->dramWrite(128,halt.w.data(),64);
   dev->run(0,pos);dev->dramRead(cacheBase,cache.data(),cache.size());
   require(cache==want,"DRAM KV write overwrote neighboring heads/positions");
   rejects([&]{dev->run(0,kAttnTMax-1);});
   write[3]=UINT32_MAX-15;dev->dramWrite(64,write.w.data(),64);
   rejects([&]{dev->run(0,0);});cases+=3;
  }
  // A fresh generation resets mutable KV only; model weights remain intact.
  auto dev=makeFuncSim({});
  Llbin tiny;Instr halt(Op::HALT);tiny.dramImage.resize(1024);halt.toBytes(tiny.dramImage.data());
  tiny.programs.push_back({.M=1,.pc=0,.instrs={halt}});
  tiny.meta={{"model",{{"vocab",2},{"dim",1},{"max_seq",1024}}},
             {"dram",{{"embedding",{{"addr",64},{"row_bytes",2}}},
                      {"input",{{"addr",128},{"row_bytes",2}}},
                      {"logits",{{"addr",192},{"row_bytes",4}}},
                      {"kv_cache",nlohmann::json::array({{{"addr",256},{"bytes",512}}})}}}};
  Host host(tiny,*dev);
  std::vector<uint8_t> dirty(512,0xFF),cache(512);
  for(unsigned repeat=0;repeat<2;++repeat) {
   dev->dramWrite(256,dirty.data(),dirty.size());
   host.generateTokens({0},0,false);dev->dramRead(256,cache.data(),cache.size());
   require(std::all_of(cache.begin(),cache.end(),[](uint8_t x){return x==0;}),"generation retained stale DRAM KV");
   ++cases;
  }
  std::cout<<"dram_kv_selftest: PASS ("<<cases<<" cases)\n";
  return 0;
 } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
