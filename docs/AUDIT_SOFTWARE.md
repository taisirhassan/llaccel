# Compiler and Python software audit

This records a source review and targeted regression evidence, not a proof of exhaustive correctness. Runtime C++/RTL/numerics and physical-flow audits are coordinated separately.

## Scope and evidence

| Files or area | Review focus and result |
|---|---|
| `compiler/CMakeLists.txt`, `compiler/lib/*/CMakeLists.txt`, `compiler/tools/CMakeLists.txt` | Full standalone and integrated builds. Fixed missing tool/test sources, LLVM source-list handling and mixed static/shared MLIR linkage causing duplicate type registrations and a parser crash. |
| `compiler/tools/llaccel-{compile,opt,disasm}.cpp` | Added complete compiler pipeline driver and checked disassembler; checked target/schedule constraints and truncated containers. Quantized graph exports now include the sibling tokenizer. |
| `compiler/include/llaccel/Dialect/*`, `compiler/lib/Dialect/*` | Read op/type definitions, value annotations and ISA access/range interfaces. Fixed exported bare tensor syntax parsing for WeightOp under MLIR 23. |
| `compiler/include/llaccel/Support/*`, `compiler/lib/Support/*` | Read quantization arithmetic, model metadata and binary readers. Fixed negative/misaligned/out-of-file weight offsets and overflow-prone shape products; reject duplicate symbols, nonfinite weights and invalid calibration ranges. Validate positive dimensions, head grouping and finite numerical constants. |
| `compiler/lib/Transforms/Quantize.cpp` | Read input/output formats, exponent selection, requant, bias, RoPE and attention metadata. Fixed compile errors, external-function crash and projection headroom across RoPE. Exponents must cover both calibrated pre- and post-rotation maxima. |
| `compiler/lib/Transforms/Fuse.cpp` | Read single-use/dominance/type/exponent conditions and GEMM epilogues. Tested fused/unfused generated programs; no additional defect confirmed. |
| `compiler/lib/Transforms/Tile.cpp`, `LowerToISA.cpp` | Read N padding, chunk chains, dense-block output/aux scratch relayout, KV allocation and POS semantics. Fixed compile errors and tested forced chunks, odd vocabulary, multi-chunk prefill and context boundaries. |
| `compiler/lib/Transforms/AllocSram.cpp`, `Schedule.cpp` | Read liveness allocation, shared KV placement, absolute-range hazards, semaphore dependencies and DMA hoisting. Fixed signed conversion compilation error. Tested all variants with randomized functional engine interleaving; this does not exhaust all legal input graphs. |
| `compiler/lib/Transforms/Emit.cpp`, `QGraphDump.cpp` | Read ISA/container encoding, layout and quantized graph contract. Fixed namespace/JSON compilation errors. Root's separate numerical audit extends attention probabilities via explicit metadata/ISA flag. |
| `python/llaccel/export.py` | Reviewed pattern matcher, layout transforms, embedding/linear/RMSNorm/RoPE/SDPA construction, Llama topology assignment, config/table validation and serialization. Importer intentionally accepts a constrained exported topology, not arbitrary PyTorch or Hugging Face models. Existing negative importer tests and generated full-model exports exercised. |
| `python/llaccel/calibrate.py` | Reviewed exported-graph interpreter and abs-max collection. Added nonfinite-activation rejection, argument validation and explicit corpus token-range diagnostics. Calibration requires a corpus/tokenizer compatible with the model. |
| `python/llaccel/refquant.py` | Reviewed independent graph reconstruction, quantization formulas and constant packing against compiler contracts. Fixed unbounded float-to-int64 conversion before saturation; reject nonfinite data/scales. Carry explicit wide-probability metadata for the new attention mode. |
| `python/llaccel/data.py` | Reviewed tokenizer, corpus loading and sampling. Fixed off-by-one exclusion of the last valid training window and exact context+1 corpus failure. Validate tokenizer entries/uniqueness. |
| `python/llaccel/model.py` | Reviewed configuration, RMSNorm/GQA/RoPE/SwiGLU, model generation and checkpoint loading. Added positive finite dimension/numerical validation. Full-context rolling fp32 generation is intentionally separate from bounded device generation. |
| `python/llaccel/train.py` | Read optimizer grouping, LR schedule, checkpoint/logging and evaluation loops. Existing training path was not retrained during this audit; original trained checkpoint and independently generated models provide execution coverage. Training remains dataset/toolchain dependent. |
| `python/llaccel/verify.py` | Reviewed teacher-forced fp32 comparison and JSON conformance checks. Fixed false MATCH when logits were omitted/truncated; all launch/logit/token arrays and exact integer row values are now required. Accuracy report remains descriptive, without an arbitrary pass threshold. |
| `python/llaccel/golden.py`, `luts.py` | Primitive and E2E tests inspected; numerical width/rounding/attention fixes are owned by the parallel numerical audit. Golden consumes compiler qgraphs in backend tests; independently generated refquant graphs and separate primitive properties are also necessary to expose shared compiler mistakes. |
| `tests/python/*`, `compiler/test/*`, `scripts/regress_models.py` | Audited test dataflow and added rejection/roundtrip/chunked compilation/headroom tests. New model regression generates its own corpus and seeded weights, and requires explicit positive runtime verification per case. |

## Confirmed regression results

Before the later wide-attention/root numerical extension: 62 Python tests passed; standalone compiler regression suite passed seven tests; four normal exported-model variants matched nine launches/eight generated tokens both normally and with seed-7 engine interleaving. Four additional variants forced 8 KiB weight chunks and also matched.

The deterministic model harness completed 168 exact backend checks: D=16 GQA with QKV bias/17-token vocabulary, and D=64 MHA; all four target/scheduling variants; normal functional, randomized functional and RTL; prompt lengths 1, 15, 16, 17, 31 and 32 plus decode through the 32-token context boundary. These generated random models validate dataflow/numerical equivalence, not language quality. The checked-in `scripts/regress_models.py` reproduces this matrix and records detailed JSON/log artifacts.

The root audit reruns the final compiler and these tests after applying the subsequent software/numerical changes; consult the final results report for the final run, rather than treating earlier passes as evidence for later edits.

## Remaining limits

- No claim of arbitrary model support: input topology, integer widths, fixed head dimensions and SRAM/context limits remain explicit design constraints.
- Verification shares the documented numerical contract; independent primitive/property tests and independent quantization are necessary alongside backend/golden agreement.
- Dataset download currently follows a mutable upstream URL; the generated regression corpus avoids it, but a fresh training run is not content-pinned to a dataset digest.
- Calibration is abs-max based and does not prove absence of saturation on unseen prompts. The fp32 quality comparison reports measured degradation rather than enforcing an unsubstantiated threshold.
- The tested toolchain is version checked on macOS ARM64; it is not a hermetic binary image. The unused GitHub workflow was removed; tests run locally.
- No new full training run, broad pretrained-model matrix, FPGA-board validation or physical implementation result follows from this software audit.
