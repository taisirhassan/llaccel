# Documentation

Start with the [project overview](../README.md). This index separates current
contracts from measurements and historical review notes; a passing result applies
to its recorded source and binary hashes, not automatically to later edits.

## Design and usage

| Guide | Contents |
| --- | --- |
| [Architecture](ARCH.md) | Engines, memory organization and execution model |
| [ISA](ISA.md) | Instructions, operand layout and synchronization |
| [Numerics](NUMERICS.md) | Integer formats, rounding and attention semantics |
| [MLIR dialect](DIALECT.md) | Model operations and compiler representation |
| [Dense model support](HF_SUPPORT.md) | Accepted Llama/Qwen variants, importer usage and capacity limits |
| [Software regression](DENSE_SOFTWARE_REGRESSION.md) | Seeded compatibility matrix and context-end checks |
| [Reproducibility](REPRODUCIBILITY.md) | Recorded toolchain, provenance and CI limitations |

Diagrams: [accelerator architecture](assets/architecture.svg) ·
[compiler pipeline](assets/compiler-flow.svg).

## Results and validation boundaries

- [Results index](../results/README.md): published evidence bundles and their scope.
- [Dense software snapshot](../results/dense-llama-qwen-2026-09-15/manifest.json):
  88 comparisons, four boundary checks and recorded software test logs; predates
  the subsequent wide-head RTL/backend changes.
- [Hardware tests](../results/verilator-local-2026-09-15/README.md): local Verilator
  testbenches and 44 compiled model comparisons.
- [Head widths and memory limits](DENSE_HARDWARE_GAPS.md).
- [Architecture measurements](RESULTS.md) and [KV bandwidth](KV_BANDWIDTH.md):
  earlier narrow-head RTL cycles, traffic, utilization and stalls.
- [Quantization tuning](QUANTIZATION_TUNING.md): calibration/validation/test
  separation, pretrained quality measurements and remaining error.
- [Quantization profiling](QUANTIZATION_PROFILING.md): per-operation diagnostics
  and interpretation limits.
- [Synthesis guide](../synth/README.md): physical-flow configuration and area scope.

## Historical reviews and project planning

These documents retain intermediate findings and failures for traceability.
They are not substitutes for the source-specific evidence above.

- [System audit](AUDIT.md)
- [Compiler and Python audit](AUDIT_SOFTWARE.md)
- [Physical implementation audit](AUDIT_PHYSICAL.md)
- [Original project plan](PLAN.md)
