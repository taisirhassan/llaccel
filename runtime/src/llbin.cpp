#include "llaccel/llbin.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <fstream>
#include <stdexcept>

namespace llaccel {

Llbin Llbin::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (bytes.size() < 12) throw std::runtime_error("llbin too small");
  uint32_t magic, version, nsec;
  std::memcpy(&magic, bytes.data(), 4);
  std::memcpy(&version, bytes.data() + 4, 4);
  std::memcpy(&nsec, bytes.data() + 8, 4);
  if (magic != kLlbinMagic) throw std::runtime_error("bad llbin magic");
  if (version != kLlbinVersion) throw std::runtime_error("unsupported llbin version " + std::to_string(version));
  if (nsec > (bytes.size() - 12) / sizeof(SectionHeader))
    throw std::runtime_error("truncated section table");
  Llbin out;
  bool haveDram = false, haveMeta = false;
  std::set<uint32_t> shapes;
  size_t off = 12;
  for (uint32_t i = 0; i < nsec; ++i) {
    if (off + sizeof(SectionHeader) > bytes.size()) throw std::runtime_error("truncated section table");
    SectionHeader h;
    std::memcpy(&h, bytes.data() + off, sizeof h);
    off += sizeof h;
    if (h.offset > bytes.size() || h.size > bytes.size() - h.offset) throw std::runtime_error("section out of range");
    const uint8_t* p = bytes.data() + h.offset;
    switch (static_cast<Section>(h.kind)) {
      case Section::DRAM_IMAGE:
        if (haveDram) throw std::runtime_error("duplicate DRAM section");
        haveDram = true;
        out.dramImage.assign(p, p + h.size);
        break;
      case Section::PROGRAM: {
        Program prog;
        prog.M = h.flags;
        if (prog.M == 0 || prog.M > kGemmTM || !shapes.insert(prog.M).second || h.size == 0)
          throw std::runtime_error("invalid or duplicate program shape");
        if (h.size % kInstrBytes) throw std::runtime_error("program size not a multiple of 64");
        for (uint64_t k = 0; k < h.size; k += kInstrBytes) prog.instrs.push_back(Instr::fromBytes(p + k));
        out.programs.push_back(std::move(prog));
        break;
      }
      case Section::META_JSON:
        if (haveMeta) throw std::runtime_error("duplicate META section");
        haveMeta = true;
        out.meta = nlohmann::json::parse(std::string(reinterpret_cast<const char*>(p), h.size));
        break;
      default:
        throw std::runtime_error("unknown section kind " + std::to_string(h.kind));
    }
  }
  if (out.dramImage.empty()) throw std::runtime_error("llbin has no DRAM image");
  if (!out.meta.contains("programs")) throw std::runtime_error("llbin META has no programs");
  if (out.programs.empty() || !out.meta["programs"].is_array() ||
      out.meta["programs"].size() != out.programs.size())
    throw std::runtime_error("program metadata count mismatch");
  // Full model containers carry explicit attention geometry. Query width is
  // independent of the residual hidden width (for example dense Qwen3).
  if (out.meta.contains("model") && out.meta["model"].contains("head_dim")) {
    const auto& model = out.meta["model"];
    auto positive = [&](const char* field) -> uint64_t {
      const auto& value = model.at(field);
      if (!value.is_number_unsigned() || value.get<uint64_t>() == 0)
        throw std::runtime_error("invalid model geometry field " + std::string(field));
      return value.get<uint64_t>();
    };
    const uint64_t D = positive("head_dim"), H = positive("n_heads"), Hkv = positive("n_kv_heads");
    if ((D != 16 && D != 32 && D != 64 && D != 128 && D != 256) ||
        H > 255 || Hkv > 255 || H % Hkv || positive("max_seq") > kAttnTMax)
      throw std::runtime_error("model attention geometry exceeds ISA limits");
    const uint64_t dim = positive("dim");
    if (dim % 16 || dim > UINT32_MAX / 32)
      throw std::runtime_error("model hidden width exceeds ISA limits");
  }
  std::set<uint32_t> mapped;
  for (const auto& pj : out.meta["programs"]) {
    const uint64_t shape = pj.at("M").get<uint64_t>();
    const uint64_t pc = pj.at("pc").get<uint64_t>();
    if (shape > kGemmTM || !shapes.contains(uint32_t(shape)) || !mapped.insert(uint32_t(shape)).second ||
        pc > UINT32_MAX || pc % kInstrBytes)
      throw std::runtime_error("invalid program entry metadata");
    auto& prog = *std::ranges::find(out.programs, uint32_t(shape), &Program::M);
    const uint64_t size = prog.instrs.size() * kInstrBytes;
    if (pc > out.dramImage.size() || size > out.dramImage.size() - pc)
      throw std::runtime_error("program entry outside DRAM image");
    if (std::memcmp(out.dramImage.data() + pc, prog.instrs.data(), size))
      throw std::runtime_error("program section differs from executable DRAM image");
    prog.pc = uint32_t(pc);
  }
  if (out.dramImage.size() > uint64_t(UINT32_MAX) + 1)
    throw std::runtime_error("DRAM image exceeds 32-bit device address space");
  if (out.meta.contains("dram") && out.meta["dram"].contains("kv_cache")) {
    const auto& cache = out.meta["dram"]["kv_cache"];
    if (!cache.is_array()) throw std::runtime_error("DRAM KV metadata must be an array");
    std::vector<std::pair<uint64_t,uint64_t>> regions;
    uint64_t total = 0;
    for (const auto& item : cache) {
      if (!item.at("addr").is_number_unsigned() || !item.at("bytes").is_number_unsigned())
        throw std::runtime_error("DRAM KV addresses/sizes must be unsigned integers");
      const uint64_t base = item.at("addr").get<uint64_t>(), size = item.at("bytes").get<uint64_t>();
      if (base % kDramBeat || !size || size % 16 || base > out.dramImage.size() ||
          size > out.dramImage.size() - base)
        throw std::runtime_error("DRAM KV region out of range or misaligned");
      for (auto [start, end] : regions)
        if (base < end && start < base + size) throw std::runtime_error("overlapping DRAM KV regions");
      for (const auto& prog : out.programs)
        if (base < prog.pc + uint64_t(prog.instrs.size()) * kInstrBytes && prog.pc < base + size)
          throw std::runtime_error("DRAM KV region overlaps executable program");
      if (!std::all_of(out.dramImage.begin() + base, out.dramImage.begin() + base + size,
                       [](uint8_t value) { return value == 0; }))
        throw std::runtime_error("DRAM KV image must be zero-initialized");
      regions.emplace_back(base, base + size);
      total += size;
    }
    if (out.meta["dram"].at("kv_cache_bytes").get<uint64_t>() != total)
      throw std::runtime_error("DRAM KV metadata byte count mismatch");
  }
  return out;
}

const Program& Llbin::programForM(uint32_t M) const {
  for (const auto& p : programs)
    if (p.M == M) return p;
  throw std::runtime_error("no program compiled for M=" + std::to_string(M));
}

}  // namespace llaccel
