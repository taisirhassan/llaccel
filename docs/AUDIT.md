# Cross-layer audit and validation — 2026-09-10

The project now has a working C++/MLIR-to-RTL transformer path, including stateful KV-cache decode. This audit covered compiler lowering, scheduling/allocation, Python export/calibration/reference code, runtime binary loading and simulation, all compute/DMA/interconnect RTL, testbenches, and synthesis scripts. This is a broad engineering review and regression campaign, not formal proof of all inputs.

## DRAM-KV follow-up

The current image contract is version 2, intentionally incompatible with version-1 images. Persistent KV moves to DRAM; attention supports 4096 positions with a fixed 256-entry score tile and global integer normalization over three K passes. The shared DRAM arbiter now tracks three request owners and holds a selected request through backpressure.

An independent command-processor audit found a concrete HALT race: a prefetched HALT could withdraw a previously stalled fetch after the arbiter had retained its owner. The fix keeps fetch valid until acceptance, accounts for late accepted reads, and drains/discards their responses before completion. The 12-case directed bench fails on the old RTL and passes on the repaired version.

Current follow-up evidence includes 273 Python tests (`work/final-all-python.log`), 714 attention cases plus seven context-traffic profiles, nine seeded arbiter runs across response FIFO depths 1/4/64, 60 system checks per fusion variant, eight CTest suites, and four ASan/UBSan suites. The refreshed architecture sweep has 64 exact RTL matches. These supersede the older performance conclusions below; see [RESULTS.md](RESULTS.md).

Pretrained validation separates full prompt/generation RTL from preinitialized-state boundary launches. Boundary fixtures use independently computed reference-token KV history and compare every vocabulary logit and every KV byte, including untouched tails. Deliberately corrupting a final-vocabulary logit or untouched KV byte is rejected. A Python FP64-BLAS reference path is permitted only after proving an integer sum bound within exact FP64 representation; the ordinary integer path remains available and cross-checked.

The first expanded Smol calibration recipe regressed on the independent 4096-target test despite improving calibration loss. This failed experiment is retained. Independent TRAIN/VALIDATION selection froze replacement recipes for both models. Fresh TEST confirmation reduced the perplexity penalty from 41.16% to 11.77% for Smol and from 76.60% to 17.47% for Qwen, with identical baseline/selected target IDs within each model. Calibration improvement alone is not a model-quality claim.

## Correctness repairs

- Fixed crossbar write classification, unaligned GEMM requant fetching, fused auxiliary-request backpressure, small-divider parameter handling, and DRAM request sampling in the RTL driver.
- Replaced silent engine test stubs/skips with real-engine integration checks and enabled assertions.
- Hardened binary bounds, program metadata and image consistency, input/model validation, illegal shifts, unsigned CLI parsing, and output failure handling.
- Fixed missing/truncated golden logits and malformed fractional/overflowing JSON values being accepted (28 additional malformed-input regressions), sampler endpoint selection, compiler weight/range validation, and absent compiler test registration.
- Reworked signed rounding to avoid 64-bit overflow; independently checked against a 128-bit oracle.
- The earlier implementation added Q15 attention probabilities. The old Q8 mode returned 85 for 171 equal keys with constant value 127. The new mode stays within one LSB across lengths 1–256 and signed constant extrema; Current version-2 images require recompilation; the compatibility behavior described here belongs to the historical version-1 implementation.
- Preserved pre/post-RoPE calibration headroom, saturated quantization before integer conversion, and constrained RMS epsilon to its actual 32-bit ISA operand.
- Made prefill shape metadata consistent across compiler, golden model, and host, including M=1/4 and context divisibility checks.

## Trained-model campaign measurements (before HF compiler changes)

- Python: 76 tests passed (`work/finish/final-python.log`).
- CTest: numerics, runtime, binary loader, and compiler regressions passed; compiler suite includes ten tests (`work/finish/final-ctest.log`).
- AddressSanitizer/UndefinedBehaviorSanitizer: three C++ suites passed (`work/finish/final-sanitize.log`).
- RTL: 500 GEMM cases per fusion variant, 120 DMA cases, 10,011 crossbar checks, 60 integration checks per variant, 1,201 vector cases, and 631 attention cases passed (`work/finish/final-tb.log`). The separate math bench also covers full-width rounding and arithmetic properties.
- Verilator, Slang, and Yosys static gates passed (`work/finish/final-lint.log`).
- 168 generated-model runs passed across D16 GQA with bias and D64 MHA, all compiler variants, functional/randomized-interleave/RTL backends, and prompt/context boundaries (`build/final-model-regressions/results.json`).
- Another 144 configurable-prefill runs passed for M=1/4 (`build/prefill-regressions/results.json`).
- Trained model: all four fusion/schedule variants passed 64-token functional/interleaved and RTL generation against complete integer logits and tokens (`build/final-trained/`).
- 64 architecture measurements passed exact output verification (`build/final-architecture-study/results.json`).

## Historical architecture and numerical conclusions

Increasing the configurable DMA window from 16 to 32 gives about 46.7% fewer decode cycles in the best latency-100 sweep cases. Keep this a selectable configuration until its physical area/timing tradeoff is measured. Fusion lowers SRAM traffic and instruction count but barely changes decode cycles in this workload. The measured bottleneck remains memory delivery; another compute feature is not the first optimization to prioritize.

Q15 attention addresses an independently demonstrated numerical defect; agreement among three implementations alone would not have exposed it. The trained-model smoke check has 95.38% teacher-forced top-1 agreement and 0.998944 mean logit cosine against fp32, on one 64-token prompt. Broader quality claims need a held-out multi-prompt/perplexity evaluation.

Physical work is unfinished. Docker ORFS was exercised with a pinned image and isolated baseline inputs, but emulation/daemon trouble interrupted the earlier baseline mapping. Docker was recovered and the final audited DMA32 physical flow restarted successfully. No finished area, timing closure, power, or achievable-frequency claim is justified yet. External SRAM is excluded from the core boundary. See AUDIT_PHYSICAL.md for assumptions and retained evidence.

For a resume, the strongest remaining deliverables are reproducible physical results and a concise explanation of the measured memory/compute and numerical tradeoffs. The project does not need unrelated features to be substantial. Formal verification, a real SRAM implementation, FPGA/silicon deployment, workload-annotated power, and general arbitrary-model support remain outside the demonstrated result.

See AUDIT_SOFTWARE.md and REPRODUCIBILITY.md for software coverage and exact local toolchain/CI boundaries. The unused GitHub workflow was removed; tests run locally.

## HF extension validation

The later HF work adds streaming constants and output logits for large vocabularies, per-residual exponents, calibrated RMS reciprocal range, exact HF token-ID input, and offline norm/value/FFN channel balancing. These compiler changes mean earlier trained-model performance numbers are historical measurements, not fresh measurements of the updated compiler.

At that earlier HF checkpoint: 211 Python tests; six CTest suites including streamed large-vocabulary functional oracles; 48 HF fixture/backend checks; 168 generated-model/backend checks after residual-scale changes; three rebuilt ASan/UBSan suites. Attention offset-invariance coverage now totals 679 RTL cases; no attention production change was needed for the offset hypothesis. Full pretrained measurements and exact supported model scope are documented in HF_SUPPORT.md.
