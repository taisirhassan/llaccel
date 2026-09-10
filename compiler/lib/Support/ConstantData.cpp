//===- ConstantData.cpp ---------------------------------------------------===//
#include "llaccel/Support/ConstantData.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::llaccel;

namespace {
Attribute makeResource(RankedTensorType type, StringRef name, ArrayRef<char> data) {
  AsmResourceBlob blob = HeapAsmResourceBlob::allocateAndCopyWithAlign(data, 16);
  unsigned bits = type.getElementTypeBitWidth();
  switch (bits) {
  case 8: return DenseI8ResourceElementsAttr::get(type, name, std::move(blob));
  case 16: return DenseI16ResourceElementsAttr::get(type, name, std::move(blob));
  case 32: return DenseI32ResourceElementsAttr::get(type, name, std::move(blob));
  default: llvm_unreachable("unsupported element width");
  }
}
} // namespace

WeightOp mlir::llaccel::createWeight(ModuleOp module, StringRef name, ArrayRef<int64_t> shape,
                                     unsigned elemBits, ArrayRef<char> data, StringRef kind,
                                     Operation *insertBefore) {
  OpBuilder b(module.getContext());
  auto type = RankedTensorType::get(shape, IntegerType::get(module.getContext(), elemBits));
  assert(int64_t(data.size()) == type.getNumElements() * int64_t(elemBits / 8) && "data size mismatch");
  if (auto existing = dyn_cast_or_null<WeightOp>(SymbolTable::lookupSymbolIn(module, name))) {
    setWeightData(existing, shape, elemBits, data);
    existing.setKindAttr(b.getStringAttr(kind));
    return existing;
  }
  if (insertBefore)
    b.setInsertionPoint(insertBefore);
  else {
    // After the last weight (keeps all constants before the functions).
    Operation *last = nullptr;
    for (Operation &op : module.getBody()->getOperations())
      if (isa<WeightOp>(op)) last = &op;
    if (last) b.setInsertionPointAfter(last);
    else b.setInsertionPointToStart(module.getBody());
  }
  auto w = b.create<WeightOp>(module.getLoc(), b.getStringAttr(name), TypeAttr::get(type),
                              /*data=*/nullptr, b.getStringAttr(kind), /*exp=*/nullptr,
                              /*orig_n=*/nullptr, /*dram_bytes=*/nullptr);
  w.setDataAttr(cast<ElementsAttr>(makeResource(type, name, data)));
  return w;
}

void mlir::llaccel::setWeightData(WeightOp w, ArrayRef<int64_t> shape, unsigned elemBits,
                                  ArrayRef<char> data) {
  auto type = RankedTensorType::get(shape, IntegerType::get(w.getContext(), elemBits));
  assert(int64_t(data.size()) == type.getNumElements() * int64_t(elemBits / 8) && "data size mismatch");
  w.setTypeAttr(TypeAttr::get(type));
  w.setDataAttr(cast<ElementsAttr>(makeResource(type, w.getSymName(), data)));
}

const F32Weights::Entry *F32Weights::find(StringRef name) const {
  for (auto &e : entries)
    if (e.name == name) return &e;
  return nullptr;
}

const float *F32Weights::get(StringRef name, int64_t expectedElems) const {
  const Entry *e = find(name);
  if (!e) return nullptr;
  int64_t n = 1;
  for (auto d : e->shape) n *= d;
  if (n != expectedElems) return nullptr;
  if (e->offset % 4 || e->offset / 4 + n > int64_t(data.size())) return nullptr;
  return data.data() + e->offset / 4;
}

std::optional<std::string> mlir::llaccel::readFile(StringRef path, Location loc) {
  auto buf = llvm::MemoryBuffer::getFile(path, /*IsText=*/false);
  if (!buf) {
    emitError(loc) << "cannot read `" << path << "`: " << buf.getError().message();
    return std::nullopt;
  }
  return std::string((*buf)->getBuffer());
}

LogicalResult mlir::llaccel::writeFile(StringRef path, ArrayRef<char> data, Location loc) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  if (ec)
    return emitError(loc) << "cannot write `" << path << "`: " << ec.message();
  os.write(data.data(), data.size());
  os.close();
  if (os.has_error())
    return emitError(loc) << "error writing `" << path << "`";
  return success();
}

FailureOr<F32Weights> mlir::llaccel::loadF32Weights(StringRef binPath, StringRef jsonPath,
                                                    Location loc) {
  auto bin = readFile(binPath, loc);
  if (!bin) return failure();
  auto js = readFile(jsonPath, loc);
  if (!js) return failure();
  F32Weights w;
  if (bin->size() % 4)
    return emitError(loc) << "weights.bin size is not a multiple of 4";
  w.data.resize(bin->size() / 4);
  std::memcpy(w.data.data(), bin->data(), bin->size());
  auto parsed = llvm::json::parse(*js);
  if (!parsed)
    return emitError(loc) << "weights.json: " << llvm::toString(parsed.takeError());
  auto *arr = parsed->getAsArray();
  if (!arr)
    return emitError(loc) << "weights.json: expected a top-level array";
  for (auto &item : *arr) {
    auto *o = item.getAsObject();
    if (!o)
      return emitError(loc) << "weights.json: entries must be objects";
    F32Weights::Entry e;
    auto name = o->getString("name");
    auto off = o->getInteger("offset");
    auto *shape = o->getArray("shape");
    if (!name || !off || !shape)
      return emitError(loc) << "weights.json: entry needs name/shape/offset";
    e.name = name->str();
    e.offset = *off;
    for (auto &d : *shape) {
      auto v = d.getAsInteger();
      if (!v) return emitError(loc) << "weights.json: shape must be integers";
      e.shape.push_back(*v);
    }
    w.entries.push_back(std::move(e));
  }
  return w;
}

FailureOr<llvm::StringMap<double>> mlir::llaccel::loadCalib(StringRef path, Location loc) {
  auto txt = readFile(path, loc);
  if (!txt) return failure();
  auto parsed = llvm::json::parse(*txt);
  if (!parsed)
    return emitError(loc) << "calib.json: " << llvm::toString(parsed.takeError());
  auto *obj = parsed->getAsObject();
  if (!obj)
    return emitError(loc) << "calib.json: expected a top-level object";
  llvm::StringMap<double> m;
  for (auto &kv : *obj) {
    auto v = kv.second.getAsNumber();
    if (!v)
      return emitError(loc) << "calib.json: value of `" << kv.first.str() << "` is not a number";
    m[kv.first.str()] = *v;
  }
  return m;
}

std::string mlir::llaccel::jsonToString(const llvm::json::Value &v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << llvm::formatv("{0:2}", v);
  return s;
}
