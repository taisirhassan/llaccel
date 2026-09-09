#include "llaccel/llbin.h"

#include <cstring>
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
  Llbin out;
  size_t off = 12;
  for (uint32_t i = 0; i < nsec; ++i) {
    if (off + sizeof(SectionHeader) > bytes.size()) throw std::runtime_error("truncated section table");
    SectionHeader h;
    std::memcpy(&h, bytes.data() + off, sizeof h);
    off += sizeof h;
    if (h.offset + h.size > bytes.size()) throw std::runtime_error("section out of range");
    const uint8_t* p = bytes.data() + h.offset;
    switch (static_cast<Section>(h.kind)) {
      case Section::DRAM_IMAGE:
        out.dramImage.assign(p, p + h.size);
        break;
      case Section::PROGRAM: {
        Program prog;
        prog.M = h.flags;
        if (h.size % kInstrBytes) throw std::runtime_error("program size not a multiple of 64");
        for (uint64_t k = 0; k < h.size; k += kInstrBytes) prog.instrs.push_back(Instr::fromBytes(p + k));
        out.programs.push_back(std::move(prog));
        break;
      }
      case Section::META_JSON:
        out.meta = nlohmann::json::parse(std::string(reinterpret_cast<const char*>(p), h.size));
        break;
      default:
        throw std::runtime_error("unknown section kind " + std::to_string(h.kind));
    }
  }
  if (out.dramImage.empty()) throw std::runtime_error("llbin has no DRAM image");
  if (!out.meta.contains("programs")) throw std::runtime_error("llbin META has no programs");
  // Attach the entry PCs from META (programs are matched by M).
  for (const auto& pj : out.meta["programs"]) {
    uint32_t M = pj.at("M").get<uint32_t>();
    for (auto& prog : out.programs)
      if (prog.M == M) prog.pc = pj.at("pc").get<uint32_t>();
  }
  return out;
}

const Program& Llbin::programForM(uint32_t M) const {
  for (const auto& p : programs)
    if (p.M == M) return p;
  throw std::runtime_error("no program compiled for M=" + std::to_string(M));
}

}  // namespace llaccel
