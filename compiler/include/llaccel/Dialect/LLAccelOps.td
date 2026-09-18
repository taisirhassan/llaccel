//===- LLAccelOps.td - llaccel operations ---------------------*- tablegen -*-===//
//
// Section A: high-level ops exactly as in docs/DIALECT.md section 1 (plus the
//            post-quantization forms: integer tensors, integer parameter attrs,
//            `llaccel.quant`, the linear `epilogue`/`aux` fused form and the
//            `n_offset/n_size/into` chunk form produced by llaccel-tile).
// Section B: ISA-level ops `llaccel.isa.*` mapping 1:1 onto docs/ISA.md.
//
//===----------------------------------------------------------------------===//
#ifndef LLACCEL_OPS_TD
#define LLACCEL_OPS_TD

include "llaccel/Dialect/LLAccelDialect.td"
include "llaccel/Dialect/LLAccelInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class LLAccel_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLAccel_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// A. High-level ops
//===----------------------------------------------------------------------===//

def LLAccel_WeightOp : LLAccel_Op<"weight", [Symbol]> {
  let summary = "a named constant tensor (model weight or compiler-generated table)";
  let description = [{
    Before quantization: `llaccel.weight @l0_wq : tensor<128x128xf32>` with the
    data in weights.bin. After `llaccel-quantize` the type is integer and the
    quantized data is attached as a `dense_resource` blob (`data`), so the
    module is self-contained. `kind` classifies the constant for the DRAM image
    (`w` tiled int8 weight, `rq` per-channel {M,S} pairs, `bias`, `gamma`,
    `embed`, `rope_cos`, `rope_sin`, `rope_table`, `consts` (packed resident
    constants), `dram_input`, `dram_logits`, `dram_scratch`). `exp` is the
    i16 exponent of the tensor (gamma/embed). `orig_n` is the pre-padding N of
    a weight padded by llaccel-tile.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<AnyRankedTensor>:$type,
                       OptionalAttr<ElementsAttr>:$data,
                       OptionalAttr<StrAttr>:$kind,
                       OptionalAttr<I64Attr>:$exp,
                       OptionalAttr<I64Attr>:$orig_n,
                       OptionalAttr<I64Attr>:$dram_bytes);
  let assemblyFormat = "$sym_name attr-dict `:` custom<WeightType>($type)";
  let extraClassDeclaration = [{
    ::mlir::RankedTensorType getTensorType() { return ::llvm::cast<::mlir::RankedTensorType>(getType()); }
    /// Raw little-endian bytes of the attached data (empty if none).
    ::llvm::ArrayRef<char> getRawData();
    int64_t getByteSize();
  }];
}

def LLAccel_RmsNormOp : LLAccel_Op<"rmsnorm", [Pure, AllTypesMatch<["input", "output"]>]> {
  let summary = "RMSNorm with a gamma weight symbol";
  let arguments = (ins AnyRankedTensor:$input, FlatSymbolRefAttr:$gamma, F64Attr:$eps,
                       DefaultValuedAttr<I64Attr, "1">:$heads,
                       OptionalAttr<I64Attr>:$eps_t, OptionalAttr<I64Attr>:$C,
                       OptionalAttr<I64Attr>:$sh_post);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input `,` $gamma attr-dict `:` type($input)";
}

def LLAccel_LinearOp : LLAccel_Op<"linear", [Pure, AttrSizedOperandSegments]> {
  let summary = "y = x W^T (+ b) with an optional fused epilogue";
  let description = [{
    `weight` is `[N][K]` like `nn.Linear.weight`; `bias` optional. After
    quantization `rq` names the per-channel requant table. `epilogue` (target
    v2 only): `resadd` adds `aux`, `silu` applies SiLU with `silu_*`, `mul`
    multiplies by `aux` with `aux_shift`. After llaccel-tile a linear may be a
    chunk: `n_offset/n_size` select the output columns it produces and `dest`
    (destination-passing style, `into(...)`) chains chunks that fill the same
    output tensor.
  }];
  let arguments = (ins AnyRankedTensor:$input,
                       Optional<AnyRankedTensor>:$aux,
                       Optional<AnyRankedTensor>:$dest,
                       FlatSymbolRefAttr:$weight,
                       OptionalAttr<FlatSymbolRefAttr>:$bias,
                       OptionalAttr<FlatSymbolRefAttr>:$rq,
                       DefaultValuedAttr<LLAccel_EpilogueAttr, "::mlir::llaccel::EpilogueMode::none">:$epilogue,
                       OptionalAttr<I64Attr>:$aux_shift,
                       OptionalAttr<I64Attr>:$silu_mi,
                       OptionalAttr<I64Attr>:$silu_si,
                       OptionalAttr<I64Attr>:$silu_sh_out,
                       OptionalAttr<I64Attr>:$n_offset,
                       OptionalAttr<I64Attr>:$n_size);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)?
    (`aux` `(` $aux^ `:` type($aux) `)`)?
    (`into` `(` $dest^ `:` type($dest) `)`)?
    attr-dict `:` type($input) `->` type($output)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    int64_t getN() { return getOutput().getType().getShape().back(); }
    int64_t getK() { return getInput().getType().getShape().back(); }
  }];
}

def LLAccel_RopeOp : LLAccel_Op<"rope", [Pure, AllTypesMatch<["input", "output"]>]> {
  let summary = "rotary embedding (HF rotate_half) over `heads` heads";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$heads);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def LLAccel_AttentionOp : LLAccel_Op<"attention", [Pure]> {
  let summary = "causal GQA attention over the KV cache of `layer`";
  let arguments = (ins AnyRankedTensor:$q, AnyRankedTensor:$k, AnyRankedTensor:$v,
                       I64Attr:$layer, I64Attr:$heads, I64Attr:$kv_heads, I64Attr:$head_dim,
                       OptionalAttr<I64Attr>:$Ms, OptionalAttr<I64Attr>:$Ss,
                       OptionalAttr<I64Attr>:$Mo, OptionalAttr<I64Attr>:$So);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$q `,` $k `,` $v attr-dict `:` functional-type(operands, results)";
}

def LLAccel_SiluOp : LLAccel_Op<"silu", [Pure, AllTypesMatch<["input", "output"]>]> {
  let summary = "SiLU";
  let arguments = (ins AnyRankedTensor:$input, OptionalAttr<I64Attr>:$Mi,
                       OptionalAttr<I64Attr>:$Si, OptionalAttr<I64Attr>:$sh_out);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def LLAccel_MulOp : LLAccel_Op<"mul", [Pure, AllTypesMatch<["a", "b", "output"]>]> {
  let summary = "elementwise multiply";
  let arguments = (ins AnyRankedTensor:$a, AnyRankedTensor:$b, OptionalAttr<I64Attr>:$sh);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($a)";
}

def LLAccel_AddOp : LLAccel_Op<"add", [Pure, AllTypesMatch<["a", "b", "output"]>]> {
  let summary = "elementwise add (residual)";
  let arguments = (ins AnyRankedTensor:$a, AnyRankedTensor:$b, OptionalAttr<I64Attr>:$sh_b);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$a `,` $b attr-dict `:` type($a)";
}

def LLAccel_QuantOp : LLAccel_Op<"quant", [Pure]> {
  let summary = "i16 -> i8 requantization: y = sat8(mulshift(x, M, S))";
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$M, I64Attr:$S);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

//===----------------------------------------------------------------------===//
// B. ISA-level ops
//===----------------------------------------------------------------------===//

// Scheduler annotations carried by every instruction op.
def LLAccel_IsaCommonArgs {
  dag args = (ins OptionalAttr<I64Attr>:$wait_sem, OptionalAttr<I64Attr>:$wait_val,
                  OptionalAttr<I64Attr>:$signal_sem);
}

class LLAccel_IsaOp<string mnemonic, dag ownArgs, list<Trait> traits = []>
    : LLAccel_Op<"isa." # mnemonic,
                 !listconcat(traits, [DeclareOpInterfaceMethods<LLAccel_IsaOpInterface>])> {
  let arguments = !con(ownArgs, LLAccel_IsaCommonArgs.args);
}

def LLAccel_AllocOp : LLAccel_Op<"isa.alloc", []> {
  let summary = "an SRAM buffer; `addr` is assigned by llaccel-alloc-sram";
  let description = [{
    `region`: `act` (activation, bank-rotated placement), `weight` (256-B
    aligned rotating weight buffer), `const` (resident constants), `kv`
    (KV cache, placed identically in every program), `tmp` (chunk staging).
  }];
  let arguments = (ins I64Attr:$size, I64Attr:$align, StrAttr:$name, StrAttr:$region,
                       OptionalAttr<I64Attr>:$addr);
  let results = (outs LLAccel_BufType:$buf);
  let assemblyFormat = "attr-dict `:` type($buf)";
}

def LLAccel_DmaLoadOp : LLAccel_IsaOp<"dma_load",
    (ins LLAccel_BufType:$dst, I64Attr:$dst_off, FlatSymbolRefAttr:$src, I64Attr:$src_off,
         I64Attr:$rows, I64Attr:$row_bytes, I64Attr:$src_stride, I64Attr:$dst_stride)> {
  let summary = "DMA_LOAD: DRAM symbol (+offset) -> SRAM buffer (+offset)";
  let assemblyFormat = "$dst custom<Off>($dst_off) `,` $src custom<Off>($src_off) attr-dict";
}

def LLAccel_DmaStoreOp : LLAccel_IsaOp<"dma_store",
    (ins LLAccel_BufType:$src, I64Attr:$src_off, FlatSymbolRefAttr:$dst, I64Attr:$dst_off,
         I64Attr:$rows, I64Attr:$row_bytes, I64Attr:$src_stride, I64Attr:$dst_stride)> {
  let summary = "DMA_STORE: SRAM buffer (+offset) -> DRAM symbol (+offset)";
  let assemblyFormat = "$src custom<Off>($src_off) `,` $dst custom<Off>($dst_off) attr-dict";
}

def LLAccel_GemmOp : LLAccel_IsaOp<"gemm",
    (ins LLAccel_BufType:$a, I64Attr:$a_off, LLAccel_BufType:$w, I64Attr:$w_off,
         LLAccel_BufType:$out, I64Attr:$out_off, LLAccel_BufType:$rq, I64Attr:$rq_off,
         Optional<LLAccel_BufType>:$bias, OptionalAttr<I64Attr>:$bias_off,
         Optional<LLAccel_BufType>:$aux, OptionalAttr<I64Attr>:$aux_off,
         I64Attr:$M, I64Attr:$N, I64Attr:$K,
         DefaultValuedAttr<LLAccel_EpilogueAttr, "::mlir::llaccel::EpilogueMode::none">:$epilogue,
         DefaultValuedAttr<BoolAttr, "false">:$out_i8,
         DefaultValuedAttr<I64Attr, "0">:$aux_shift,
         DefaultValuedAttr<I64Attr, "0">:$silu_mi,
         DefaultValuedAttr<I64Attr, "0">:$silu_si,
         DefaultValuedAttr<I64Attr, "0">:$silu_sh_out), [AttrSizedOperandSegments]> {
  let summary = "GEMM (ISA.md): out = epilogue(requant(A W^T + bias))";
  let assemblyFormat = [{
    $a custom<Off>($a_off) `,` $w custom<Off>($w_off) `,` $out custom<Off>($out_off) `,`
    $rq custom<Off>($rq_off)
    (`bias` `(` $bias^ custom<Off>($bias_off) `)`)?
    (`aux` `(` $aux^ custom<Off>($aux_off) `)`)?
    attr-dict
  }];
  let hasVerifier = 1;
}

def LLAccel_IsaRmsNormOp : LLAccel_IsaOp<"rmsnorm",
    (ins LLAccel_BufType:$src, I64Attr:$src_off, LLAccel_BufType:$gamma, I64Attr:$gamma_off,
         LLAccel_BufType:$dst, I64Attr:$dst_off, I64Attr:$M, I64Attr:$K,
         I64Attr:$eps_t, I64Attr:$C, I64Attr:$sh_post)> {
  let summary = "VEC_RMSNORM";
  let assemblyFormat = "$src custom<Off>($src_off) `,` $gamma custom<Off>($gamma_off) `,` $dst custom<Off>($dst_off) attr-dict";
}

def LLAccel_IsaRopeOp : LLAccel_IsaOp<"rope",
    (ins LLAccel_BufType:$src, I64Attr:$src_off, LLAccel_BufType:$dst, I64Attr:$dst_off,
         LLAccel_BufType:$table, I64Attr:$table_off, I64Attr:$M, I64Attr:$H, I64Attr:$D,
         I64Attr:$table_stride, I64Attr:$table_rows)> {
  let summary = "VEC_ROPE (table row POS+m)";
  let assemblyFormat = "$src custom<Off>($src_off) `,` $dst custom<Off>($dst_off) `,` $table custom<Off>($table_off) attr-dict";
}

def LLAccel_IsaSiluOp : LLAccel_IsaOp<"silu",
    (ins LLAccel_BufType:$src, I64Attr:$src_off, LLAccel_BufType:$dst, I64Attr:$dst_off,
         I64Attr:$count, I64Attr:$Mi, I64Attr:$Si, I64Attr:$sh_out)> {
  let summary = "VEC_SILU";
  let assemblyFormat = "$src custom<Off>($src_off) `,` $dst custom<Off>($dst_off) attr-dict";
}

def LLAccel_IsaMulOp : LLAccel_IsaOp<"mul",
    (ins LLAccel_BufType:$a, I64Attr:$a_off, LLAccel_BufType:$b, I64Attr:$b_off,
         LLAccel_BufType:$dst, I64Attr:$dst_off, I64Attr:$count, I64Attr:$sh)> {
  let summary = "VEC_MUL";
  let assemblyFormat = "$a custom<Off>($a_off) `,` $b custom<Off>($b_off) `,` $dst custom<Off>($dst_off) attr-dict";
}

def LLAccel_IsaAddOp : LLAccel_IsaOp<"add",
    (ins LLAccel_BufType:$a, I64Attr:$a_off, LLAccel_BufType:$b, I64Attr:$b_off,
         LLAccel_BufType:$dst, I64Attr:$dst_off, I64Attr:$count, I64Attr:$sh_b)> {
  let summary = "VEC_ADD";
  let assemblyFormat = "$a custom<Off>($a_off) `,` $b custom<Off>($b_off) `,` $dst custom<Off>($dst_off) attr-dict";
}

def LLAccel_IsaQuantOp : LLAccel_IsaOp<"quant",
    (ins LLAccel_BufType:$src, I64Attr:$src_off, LLAccel_BufType:$dst, I64Attr:$dst_off,
         I64Attr:$count, I64Attr:$M, I64Attr:$S)> {
  let summary = "VEC_QUANT (i16 -> i8)";
  let assemblyFormat = "$src custom<Off>($src_off) `,` $dst custom<Off>($dst_off) attr-dict";
}

def LLAccel_IsaAttnOp : LLAccel_IsaOp<"attn",
    (ins LLAccel_BufType:$q, I64Attr:$q_off, LLAccel_BufType:$out, I64Attr:$out_off,
         FlatSymbolRefAttr:$kbase, I64Attr:$kbase_off, FlatSymbolRefAttr:$vbase, I64Attr:$vbase_off,
         I64Attr:$M, I64Attr:$H, I64Attr:$Hkv, I64Attr:$D, I64Attr:$kv_stride,
         I64Attr:$Ms, I64Attr:$Ss, I64Attr:$Mo, I64Attr:$So)> {
  let summary = "ATTN";
  let assemblyFormat = [{
    $q custom<Off>($q_off) `,` $out custom<Off>($out_off) `,`
    $kbase custom<Off>($kbase_off) `,` $vbase custom<Off>($vbase_off) attr-dict
  }];
}

def LLAccel_IsaKvWriteOp : LLAccel_IsaOp<"kv_write",
    (ins LLAccel_BufType:$src, I64Attr:$src_off, FlatSymbolRefAttr:$base, I64Attr:$base_off,
         I64Attr:$M, I64Attr:$Hkv, I64Attr:$D, I64Attr:$kv_stride)> {
  let summary = "KV_WRITE";
  let assemblyFormat = "$src custom<Off>($src_off) `,` $base custom<Off>($base_off) attr-dict";
}

def LLAccel_IsaNopOp : LLAccel_IsaOp<"nop", (ins)> {
  let summary = "NOP (carries an extra wait when an instruction needs several)";
  let assemblyFormat = "attr-dict";
}

def LLAccel_IsaHaltOp : LLAccel_IsaOp<"halt", (ins)> {
  let summary = "HALT";
  let assemblyFormat = "attr-dict";
}

#endif // LLACCEL_OPS_TD
