// Compile an exported llaccel graph through the complete MLIR pass pipeline.
#include "llaccel/Dialect/LLAccelDialect.h"
#include "llaccel/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
using namespace mlir::llaccel;
namespace cl = llvm::cl;
static cl::opt<std::string> input(cl::Positional, cl::Required, cl::desc("model.mlir"));
static cl::opt<std::string> output("o", cl::Required, cl::desc("Output .llbin"));
static cl::opt<std::string> weights("weights", cl::Required);
static cl::opt<std::string> weightsJson("weights-json", cl::Required);
static cl::opt<std::string> calib("calib", cl::Required);
static cl::opt<std::string> target("target", cl::init("llaccel-v1"));
static cl::opt<std::string> schedule("schedule", cl::init("inorder"));
static cl::opt<std::string> qgraph("dump-qgraph", cl::init(""));
static cl::opt<bool> fusion("enable-fusion", cl::init(false));
static cl::opt<bool> stats("print-stats", cl::init(false));
static cl::opt<int64_t> prefillM("prefill-m", cl::init(16));
static cl::opt<int64_t> chunkBytes("weight-chunk-bytes", cl::init(65536));
static cl::opt<int64_t> weightBuffers("weight-buffers", cl::init(2));
static cl::opt<int64_t> sramBytes("sram-bytes", cl::init(1048576));
int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "llaccel MLIR compiler\n");
  if ((target != "llaccel-v1" && target != "llaccel-v2") ||
      (fusion && target != "llaccel-v2") ||
      (schedule != "inorder" && schedule != "overlap")) {
    llvm::errs() << "invalid target/schedule, or fusion requested for v1\n";
    return 1;
  }
  MLIRContext context;
  context.loadDialect<LLAccelDialect, func::FuncDialect>();
  auto module = parseSourceFile<ModuleOp>(input, &context);
  if (!module) return 1;
  PassManager quant(&context);
  LLAccelQuantizeOptions qo;
  qo.weights = weights; qo.weightsJson = weightsJson; qo.calib = calib;
  quant.addPass(createLLAccelQuantize(qo));
  LLAccelFuseOptions fo; fo.enable = fusion;
  quant.addPass(createLLAccelFuse(fo));
  if (failed(quant.run(*module))) return 1;
  if (!qgraph.empty()) {
    if (failed(dumpQGraph(*module, qgraph, prefillM))) return 1;
    llvm::SmallString<256> tokenizer(input.getValue());
    llvm::sys::path::remove_filename(tokenizer);
    llvm::sys::path::append(tokenizer, "tokenizer.json");
    if (llvm::sys::fs::exists(tokenizer)) {
      llvm::SmallString<256> destination(qgraph.getValue());
      llvm::sys::path::append(destination, "tokenizer.json");
      if (auto ec = llvm::sys::fs::copy_file(tokenizer, destination)) {
        llvm::errs() << "cannot copy tokenizer: " << ec.message() << '\n';
        return 1;
      }
    }
  }
  PassManager lower(&context);
  LLAccelTileOptions to; to.weightChunkBytes = chunkBytes;
  lower.addPass(createLLAccelTile(to));
  LLAccelLowerToISAOptions lo; lo.prefillM = prefillM; lo.weightBuffers = weightBuffers;
  lower.addPass(createLLAccelLowerToISA(lo));
  LLAccelAllocSramOptions ao; ao.sramBytes = sramBytes;
  lower.addPass(createLLAccelAllocSram(ao));
  LLAccelScheduleOptions so; so.mode = schedule;
  lower.addPass(createLLAccelSchedule(so));
  LLAccelEmitOptions eo; eo.output = output; eo.target = target;
  eo.fusion = fusion; eo.schedule = schedule; eo.printStats = stats;
  lower.addPass(createLLAccelEmit(eo));
  return failed(lower.run(*module));
}
