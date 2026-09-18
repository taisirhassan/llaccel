#include "llaccel/llbin.h"
#include <filesystem>
#include <fstream>
#include <functional>
#include <print>
#include <stdexcept>
#include <cstring>
using namespace llaccel;

int main(int argc, char** argv) {
  const auto path = std::filesystem::path(argc > 1 ? argv[1] : ".") / "llbin-selftest.bin";
  Instr halt(Op::HALT);
  std::vector<uint8_t> dram(128), program(64);
  halt.toBytes(dram.data()); halt.toBytes(program.data());
  auto meta = nlohmann::json::object();
  meta["programs"] = nlohmann::json::array();
  meta["programs"].push_back({{"M",1},{"pc",0}});
  auto make = [&](nlohmann::json j) {
    const auto text=j.dump();
    const size_t base=12+3*sizeof(SectionHeader);
    std::vector<uint8_t> bytes(base+dram.size()+program.size()+text.size());
    const uint32_t header[]={kLlbinMagic,kLlbinVersion,3};
    SectionHeader sections[]={{uint32_t(Section::DRAM_IMAGE),0,base,dram.size()},
      {uint32_t(Section::PROGRAM),1,base+dram.size(),program.size()},
      {uint32_t(Section::META_JSON),0,base+dram.size()+program.size(),text.size()}};
    std::memcpy(bytes.data(),header,sizeof header);
    std::memcpy(bytes.data()+12,sections,sizeof sections);
    std::memcpy(bytes.data()+base,dram.data(),dram.size());
    std::memcpy(bytes.data()+base+dram.size(),program.data(),program.size());
    std::memcpy(bytes.data()+base+dram.size()+program.size(),text.data(),text.size());
    return bytes;
  };
  auto check = [&](const auto& bytes, bool valid) {
    {std::ofstream file(path,std::ios::binary); file.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());}
    bool accepted=true;
    try { auto bin=Llbin::load(path.string()); if(bin.programForM(1).pc!=0) accepted=false; }
    catch(const std::exception&) {accepted=false;}
    if(accepted!=valid) throw std::runtime_error("llbin validation result differs from expectation");
  };
  try {
    check(make(meta),true);
    auto j=meta; j["programs"]=nlohmann::json::array(); check(make(j),false);
    j=meta; j["programs"][0]["M"]=2; check(make(j),false);
    j=meta; j["programs"][0]["pc"]=1; check(make(j),false);
    j=meta; j["programs"][0]["pc"]=64; check(make(j),false);
    j=meta; j["programs"][0]["pc"]=128; check(make(j),false);
    j=meta; j["programs"].push_back(j["programs"][0]); check(make(j),false);
    auto bytes=make(meta); bytes.back()='!'; check(bytes,false);
    bytes=make(meta); bytes.resize(14); check(bytes,false);
    bytes=make(meta); bytes[12+3*sizeof(SectionHeader)+128]^=1; check(bytes,false);
    bytes=make(meta); uint32_t zero=0; std::memcpy(bytes.data()+12+sizeof(SectionHeader)+4,&zero,4); check(bytes,false);
    bytes=make(meta); uint64_t overflow=UINT64_MAX; std::memcpy(bytes.data()+20,&overflow,8); check(bytes,false);
    bytes=make(meta);uint32_t oldVersion=1;std::memcpy(bytes.data()+4,&oldVersion,4);check(bytes,false);
    j=meta;j["dram"]={{"kv_cache",nlohmann::json::array({{{"addr",64},{"bytes",64}}})},
                      {"kv_cache_bytes",64}};
    check(make(j),true);
    auto invalid=j;invalid["dram"]["kv_cache"][0]["addr"]=0;check(make(invalid),false);
    invalid=j;invalid["dram"]["kv_cache"][0]["addr"]=65;check(make(invalid),false);
    invalid=j;invalid["dram"]["kv_cache"][0]["bytes"]=128;check(make(invalid),false);
    invalid=j;invalid["dram"]["kv_cache"][0]["bytes"]=0;check(make(invalid),false);
    invalid=j;invalid["dram"]["kv_cache"][0]["bytes"]=64.0;check(make(invalid),false);
    invalid=j;invalid["dram"]["kv_cache"][0]["addr"]=-1;check(make(invalid),false);
    invalid=j;invalid["dram"]["kv_cache_bytes"]=63;check(make(invalid),false);
    invalid=j;invalid["dram"]["kv_cache"].push_back(invalid["dram"]["kv_cache"][0]);check(make(invalid),false);
    dram[64]=1;check(make(j),false);dram[64]=0;
    auto geometry = meta;
    geometry["model"] = {{"dim",64},{"head_dim",256},{"n_heads",2},{"n_kv_heads",1},{"max_seq",16}};
    check(make(geometry),true);
    unsigned geometryCases = 1;
    for (const char* field : {"dim", "head_dim", "n_heads", "n_kv_heads", "max_seq"}) {
      for (const auto& value : nlohmann::json::array({0, -1, 1.0, true, nullptr, uint64_t(UINT32_MAX)+1})) {
        auto bad = geometry; bad["model"][field] = value; check(make(bad),false); ++geometryCases;
      }
    }
    auto bad = geometry; bad["model"]["n_kv_heads"] = 3; check(make(bad),false); ++geometryCases;
    bad = geometry; bad["model"]["head_dim"] = 512; check(make(bad),false); ++geometryCases;
    std::println("llbin geometry validation: PASS ({} cases)", geometryCases);
    std::println("llbin_selftest: PASS (23 valid/corrupted image cases)");
    return 0;
  } catch(const std::exception& e) {std::println(stderr,"FAIL: {}",e.what());return 1;}
}
