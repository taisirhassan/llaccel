// Checked container reader and instruction disassembler.
#include "llaccel/isa.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
namespace cl = llvm::cl;
static cl::opt<std::string> input(cl::Positional, cl::Required);
int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "llaccel disassembler\n");
  auto file = llvm::MemoryBuffer::getFile(input);
  if (!file) { llvm::errs() << file.getError().message() << '\n'; return 1; }
  auto bytes = (*file)->getBuffer();
  auto error = [] { llvm::errs() << "invalid or truncated llbin container\n"; return 1; };
  if (bytes.size() < 12) return error();
  uint32_t h[3]; std::memcpy(h, bytes.data(), 12);
  if (h[0] != llaccel::kLlbinMagic || h[1] != llaccel::kLlbinVersion ||
      h[2] > (bytes.size() - 12) / sizeof(llaccel::SectionHeader)) return error();
  for (uint32_t i = 0; i < h[2]; ++i) {
    llaccel::SectionHeader s;
    std::memcpy(&s, bytes.data() + 12 + i * sizeof(s), sizeof(s));
    if (s.offset > bytes.size() || s.size > bytes.size() - s.offset) return error();
    if (s.kind != uint32_t(llaccel::Section::PROGRAM)) continue;
    if (s.size % llaccel::kInstrBytes) return error();
    llvm::outs() << "program M=" << s.flags << '\n';
    for (uint64_t p = 0; p < s.size; p += llaccel::kInstrBytes) {
      llaccel::Instr inst;
      std::memcpy(inst.w.data(), bytes.data() + s.offset + p, llaccel::kInstrBytes);
      auto name = llaccel::opName(inst.op());
      llvm::outs() << llvm::format_hex(p, 8) << " " << llvm::StringRef(name.data(), name.size())
                   << " wait=" << unsigned(inst.waitSem()) << ':' << inst.waitVal()
                   << " signal=" << unsigned(inst.signalSem());
      for (unsigned j = 2; j < 16; ++j) llvm::outs() << " " << llvm::format_hex(inst[j], 10);
      llvm::outs() << '\n';
    }
  }
}
