//===- LLAccelOps.cpp - llaccel op implementations ------------------------===//
#include "llaccel/Dialect/LLAccelOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::llaccel;

//===----------------------------------------------------------------------===//
// custom<Off>: prints `[N]` for a byte-offset attribute (absent -> `[0]`).
//===----------------------------------------------------------------------===//
static ParseResult parseOff(OpAsmParser &parser, IntegerAttr &attr) {
  int64_t v = 0;
  if (parser.parseLSquare() || parser.parseInteger(v) || parser.parseRSquare())
    return failure();
  attr = parser.getBuilder().getI64IntegerAttr(v);
  return success();
}

static void printOff(OpAsmPrinter &p, Operation *, IntegerAttr attr) {
  p << '[' << (attr ? attr.getInt() : 0) << ']';
}

#include "llaccel/Dialect/LLAccelInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "llaccel/Dialect/LLAccelOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Value annotations
//===----------------------------------------------------------------------===//
namespace {
/// Returns the attribute `name` attached to the value: op attribute for op
/// results, function argument attribute for block arguments.
Attribute getValueAttr(Value v, StringRef name) {
  if (auto *op = v.getDefiningOp())
    return op->getAttr(name);
  auto arg = cast<BlockArgument>(v);
  auto fn = dyn_cast<FunctionOpInterface>(arg.getOwner()->getParentOp());
  if (!fn)
    return nullptr;
  return fn.getArgAttr(arg.getArgNumber(), name);
}

void setValueAttr(Value v, StringRef name, Attribute attr) {
  if (auto *op = v.getDefiningOp()) {
    op->setAttr(name, attr);
    return;
  }
  auto arg = cast<BlockArgument>(v);
  auto fn = cast<FunctionOpInterface>(arg.getOwner()->getParentOp());
  fn.setArgAttr(arg.getArgNumber(), name, attr);
}
} // namespace

std::optional<StringRef> mlir::llaccel::getValueName(Value v) {
  if (auto s = dyn_cast_or_null<StringAttr>(getValueAttr(v, LLAccelDialect::kNameAttr)))
    return s.getValue();
  return std::nullopt;
}
std::optional<int64_t> mlir::llaccel::getValueExp(Value v) {
  if (auto a = dyn_cast_or_null<IntegerAttr>(getValueAttr(v, LLAccelDialect::kExpAttr)))
    return a.getInt();
  return std::nullopt;
}
std::optional<double> mlir::llaccel::getValueScale(Value v) {
  if (auto a = dyn_cast_or_null<FloatAttr>(getValueAttr(v, LLAccelDialect::kScaleAttr)))
    return a.getValueAsDouble();
  return std::nullopt;
}
void mlir::llaccel::setValueName(Value v, StringRef name) {
  setValueAttr(v, LLAccelDialect::kNameAttr, StringAttr::get(v.getContext(), name));
}
void mlir::llaccel::setValueExp(Value v, int64_t exp) {
  Builder b(v.getContext());
  setValueAttr(v, LLAccelDialect::kExpAttr, b.getI64IntegerAttr(exp));
}
void mlir::llaccel::setValueScale(Value v, double scale) {
  Builder b(v.getContext());
  setValueAttr(v, LLAccelDialect::kScaleAttr, b.getF64FloatAttr(scale));
}

int64_t mlir::llaccel::tensorWidth(Value v) {
  return cast<RankedTensorType>(v.getType()).getShape().back();
}
int64_t mlir::llaccel::tensorElemBytes(Value v) {
  return cast<RankedTensorType>(v.getType()).getElementTypeBitWidth() / 8;
}

WeightOp mlir::llaccel::lookupWeight(Operation *from, StringRef name) {
  auto module = from->getParentOfType<ModuleOp>();
  if (!module)
    module = dyn_cast<ModuleOp>(from);
  if (!module)
    return nullptr;
  return dyn_cast_or_null<WeightOp>(SymbolTable::lookupSymbolIn(module, name));
}
WeightOp mlir::llaccel::lookupWeight(Operation *from, FlatSymbolRefAttr ref) {
  return ref ? lookupWeight(from, ref.getValue()) : nullptr;
}

//===----------------------------------------------------------------------===//
// WeightOp
//===----------------------------------------------------------------------===//
ArrayRef<char> WeightOp::getRawData() {
  Attribute data = getDataAttr();
  if (!data)
    return {};
  if (auto res = dyn_cast<DenseResourceElementsAttr>(data)) {
    if (const AsmResourceBlob *blob = res.getRawHandle().getBlob())
      return blob->getData();
    return {};
  }
  if (auto dense = dyn_cast<DenseElementsAttr>(data))
    return dense.getRawData();
  return {};
}

int64_t WeightOp::getByteSize() {
  if (auto n = getDramBytesAttr())
    return n.getInt();
  auto t = getTensorType();
  return t.getNumElements() * (t.getElementTypeBitWidth() / 8);
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//
LogicalResult LinearOp::verify() {
  auto ep = getEpilogue();
  bool needsAux = ep == EpilogueMode::resadd || ep == EpilogueMode::mul;
  if (needsAux && !getAux())
    return emitOpError("epilogue resadd/mul requires an aux operand");
  if (!needsAux && getAux())
    return emitOpError("aux operand is only valid with epilogue resadd/mul");
  if (ep == EpilogueMode::silu && !(getSiluMi() && getSiluSi() && getSiluShOut()))
    return emitOpError("epilogue silu requires silu_mi/silu_si/silu_sh_out");
  if (getDest() && getDest().getType() != getOutput().getType())
    return emitOpError("dest type must equal the result type");
  if (getNOffset().has_value() != getNSize().has_value())
    return emitOpError("n_offset and n_size must be set together");
  return success();
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===//
LogicalResult GemmOp::verify() {
  if (getM() < 1 || getM() > int64_t(::llaccel::kGemmTM))
    return emitOpError("M must be in [1, 16]");
  if (getN() % 16 || getK() % 16)
    return emitOpError("N and K must be multiples of 16");
  if (getOutI8() && getEpilogue() != EpilogueMode::none)
    return emitOpError("i8 output is only valid with epilogue none");
  bool needsAux = getEpilogue() == EpilogueMode::resadd || getEpilogue() == EpilogueMode::mul;
  if (needsAux != bool(getAux()))
    return emitOpError("aux must be present exactly for epilogue resadd/mul");
  return success();
}

//===----------------------------------------------------------------------===//
// IsaOp interface implementations
//===----------------------------------------------------------------------===//
namespace {
BufAccess sram(Value buf, int64_t off, int64_t size, bool write) {
  BufAccess a;
  a.buf = buf;
  a.offset = off;
  a.size = size;
  a.write = write;
  return a;
}
BufAccess dram(FlatSymbolRefAttr sym, int64_t off, int64_t size, bool write) {
  BufAccess a;
  a.dram = sym;
  a.offset = off;
  a.size = size;
  a.write = write;
  return a;
}
int64_t span(int64_t rows, int64_t rowBytes, int64_t stride) {
  return rows <= 0 ? 0 : (rows - 1) * stride + rowBytes;
}
} // namespace

IsaEngine DmaLoadOp::getEngine() { return IsaEngine::DMA; }
void DmaLoadOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  out.push_back(sram(getDst(), getDstOff(), span(getRows(), getRowBytes(), getDstStride()), true));
  out.push_back(dram(getSrcAttr(), getSrcOff(), span(getRows(), getRowBytes(), getSrcStride()), false));
}

IsaEngine DmaStoreOp::getEngine() { return IsaEngine::DMA; }
void DmaStoreOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  out.push_back(sram(getSrc(), getSrcOff(), span(getRows(), getRowBytes(), getSrcStride()), false));
  out.push_back(dram(getDstAttr(), getDstOff(), span(getRows(), getRowBytes(), getDstStride()), true));
}

IsaEngine GemmOp::getEngine() { return IsaEngine::GEMM; }
void GemmOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  int64_t M = getM(), N = getN(), K = getK();
  int64_t outElem = getOutI8() ? 1 : 2;
  out.push_back(sram(getA(), getAOff(), M * K, false));
  out.push_back(sram(getW(), getWOff(), N * K, false));
  out.push_back(sram(getOut(), getOutOff(), M * N * outElem, true));
  out.push_back(sram(getRq(), getRqOff(), N * 8, false));
  if (getBias())
    out.push_back(sram(getBias(), getBiasOffAttr() ? getBiasOffAttr().getInt() : 0, N * 4, false));
  if (getAux())
    out.push_back(sram(getAux(), getAuxOffAttr() ? getAuxOffAttr().getInt() : 0, M * N * 2, false));
}

IsaEngine IsaRmsNormOp::getEngine() { return IsaEngine::VEC; }
void IsaRmsNormOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  out.push_back(sram(getSrc(), getSrcOff(), getM() * getK() * 2, false));
  out.push_back(sram(getGamma(), getGammaOff(), getK() * 2, false));
  out.push_back(sram(getDst(), getDstOff(), getM() * getK() * 2, true));
}

IsaEngine IsaRopeOp::getEngine() { return IsaEngine::VEC; }
void IsaRopeOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  int64_t bytes = getM() * getH() * getD() * 2;
  out.push_back(sram(getSrc(), getSrcOff(), bytes, false));
  out.push_back(sram(getDst(), getDstOff(), bytes, true));
  out.push_back(sram(getTable(), getTableOff(), getTableRows() * getTableStride(), false));
}

IsaEngine IsaSiluOp::getEngine() { return IsaEngine::VEC; }
void IsaSiluOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  out.push_back(sram(getSrc(), getSrcOff(), getCount() * 2, false));
  out.push_back(sram(getDst(), getDstOff(), getCount() * 2, true));
}

IsaEngine IsaMulOp::getEngine() { return IsaEngine::VEC; }
void IsaMulOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  out.push_back(sram(getA(), getAOff(), getCount() * 2, false));
  out.push_back(sram(getB(), getBOff(), getCount() * 2, false));
  out.push_back(sram(getDst(), getDstOff(), getCount() * 2, true));
}

IsaEngine IsaAddOp::getEngine() { return IsaEngine::VEC; }
void IsaAddOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  out.push_back(sram(getA(), getAOff(), getCount() * 2, false));
  out.push_back(sram(getB(), getBOff(), getCount() * 2, false));
  out.push_back(sram(getDst(), getDstOff(), getCount() * 2, true));
}

IsaEngine IsaQuantOp::getEngine() { return IsaEngine::VEC; }
void IsaQuantOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  out.push_back(sram(getSrc(), getSrcOff(), getCount() * 2, false));
  out.push_back(sram(getDst(), getDstOff(), getCount(), true));
}

IsaEngine IsaAttnOp::getEngine() { return IsaEngine::ATTN; }
void IsaAttnOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  int64_t qBytes = getM() * getH() * getD();
  int64_t region = getHkv() * getKvStride();
  out.push_back(sram(getQ(), getQOff(), qBytes, false));
  out.push_back(sram(getOut(), getOutOff(), qBytes, true));
  out.push_back(sram(getKbase(), getKbaseOff(), region, false));
  out.push_back(sram(getVbase(), getVbaseOff(), region, false));
}

IsaEngine IsaKvWriteOp::getEngine() { return IsaEngine::ATTN; }
void IsaKvWriteOp::getAccesses(SmallVectorImpl<BufAccess> &out) {
  out.push_back(sram(getSrc(), getSrcOff(), getM() * getHkv() * getD(), false));
  out.push_back(sram(getBase(), getBaseOff(), getHkv() * getKvStride(), true));
}

IsaEngine IsaNopOp::getEngine() { return IsaEngine::CP; }
void IsaNopOp::getAccesses(SmallVectorImpl<BufAccess> &) {}

IsaEngine IsaHaltOp::getEngine() { return IsaEngine::CP; }
void IsaHaltOp::getAccesses(SmallVectorImpl<BufAccess> &) {}
