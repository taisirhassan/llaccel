// Isolated device launch from independently initialized KV state; not RTL prefill.
#include "llaccel/device.h"
#include "llaccel/llbin.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
using namespace llaccel;
using json = nlohmann::json;
std::vector<uint8_t> readFile(const std::filesystem::path& p, uint64_t expected) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f || uint64_t(f.tellg()) != expected) throw std::runtime_error("wrong fixture size: " + p.string());
  std::vector<uint8_t> data(expected); f.seekg(0); f.read(reinterpret_cast<char*>(data.data()), data.size());
  if (!f) throw std::runtime_error("fixture read failed");
  return data;
}
int main(int argc, char** argv) try {
  if (argc != 5) throw std::runtime_error("usage: boundary_launch IMAGE FIXTURE.json func|rtl RESULT.json");
  auto bin = Llbin::load(argv[1]);
  const auto manifestPath = std::filesystem::path(argv[2]);
  std::ifstream input(manifestPath); const auto fixtures = json::parse(input);
  if (!fixtures.contains("cases") || !fixtures.at("cases").is_array() || fixtures.at("cases").empty())
    throw std::runtime_error("fixture cases must be a nonempty array");
  const std::string backend = argv[3];
  if (backend != "func" && backend != "rtl") throw std::runtime_error("invalid backend");
  DeviceOptions opt; opt.dramBytes = bin.dramImage.size() + (1u<<20);
  opt.epilogueFusion = bin.meta.value("fusion",false) || bin.meta.value("target",std::string()) == "llaccel-v2";
  auto dev = backend == "rtl" ? makeRtlSim(opt) : makeFuncSim(opt);
  dev->dramWrite(0,bin.dramImage.data(),bin.dramImage.size());
  // The synchronous upload copied the bytes into device-owned storage.
  std::vector<uint8_t>{}.swap(bin.dramImage);
  const auto& dm = bin.meta.at("dram"); const auto& model = bin.meta.at("model");
  const uint64_t dim = model.at("dim"), vocab = model.at("vocab");
  json results = json::array();
  for (const auto& fixture : fixtures.at("cases")) {
    const uint32_t pos = fixture.at("pos"), token = fixture.at("token");
    if (pos >= model.at("max_seq").get<uint32_t>() || token >= vocab) throw std::runtime_error("invalid fixture position/token");
    const auto dir = manifestPath.parent_path() / fixture.at("directory").get<std::string>();
    for (const auto& region : dm.at("kv_cache")) {
      auto bytes = readFile(dir/(region.at("name").get<std::string>()+".before.bin"),region.at("bytes"));
      dev->dramWrite(region.at("addr"),bytes.data(),bytes.size());
    }
    std::vector<uint8_t> embedding(dm.at("input").at("row_bytes").get<uint64_t>(),0);
    dev->dramRead(dm.at("embedding").at("addr").get<uint64_t>()+token*dm.at("embedding").at("row_bytes").get<uint64_t>(),embedding.data(),dim*2);
    dev->dramWrite(dm.at("input").at("addr"),embedding.data(),embedding.size());
    auto perf = dev->run(bin.programForM(1).pc,pos);
    auto expected = readFile(dir/"logits.bin",vocab*2); std::vector<uint8_t> actual(expected.size());
    dev->dramRead(dm.at("logits").at("addr"),actual.data(),actual.size());
    if (actual != expected) throw std::runtime_error("all-vocabulary logits mismatch at pos " + std::to_string(pos));
    for (const auto& region : dm.at("kv_cache")) {
      auto want = readFile(dir/(region.at("name").get<std::string>()+".after.bin"),region.at("bytes"));
      std::vector<uint8_t> got(want.size()); dev->dramRead(region.at("addr"),got.data(),got.size());
      if (got != want) throw std::runtime_error("full KV region mismatch at pos " + std::to_string(pos));
    }
    json counters; for (unsigned i=0;i<kNumPerf;++i) counters[kPerfNames[i]]=perf[i];
    results.push_back({{"context",pos+1},{"backend",backend},{"result","MATCH"},{"perf",counters}});
    std::cout << "MATCH preinitialized-state boundary T=" << pos+1 << std::endl;
  }
  std::ofstream output(argv[4]); output << json{{"validation_kind","preinitialized-state boundary test"},{"status","PASS"},{"cases",results}}.dump(2) << '\n';
  if (!output) throw std::runtime_error("result write failed");
  return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
