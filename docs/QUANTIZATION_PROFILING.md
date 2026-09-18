# Per-operation quantization diagnostics

`scripts/profile_quantization.py` compares the quantized graph with the floating
export on one text prefix, at position zero. It accepts 1–16 tokens. This is a
local diagnostic for locating numerical error; it is not a held-out model
quality benchmark, a calibration procedure, or RTL validation.

```sh
OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
uv run python scripts/profile_quantization.py \
  --export build/dram-kv-hf-regressions/llama-tied/export \
  --qgraph build/dram-kv-hf-regressions/llama-tied/q-v2-overlap \
  --text tests/data/hf-calibration.txt --tokens 8 --threads 2 \
  --out build/profile-quantization-tiny.json
```

The text file is tokenized with the export's saved HF tokenizer, including its
normal special-token behavior. Only the first requested tokens are used; the
report records the actual token count and IDs if the text is shorter. `--threads`
sets Torch threads; the environment variables above also bound BLAS/OpenMP
libraries used by NumPy. The float `weights.bin` and its manifest must exist in
the export. If these regenerable intermediates were removed, regenerate them
from the recorded checkpoint/export recipe before profiling.

For each operation with an output, the report includes:

- **Local error:** dequantized integer output versus the floating operation on
  dequantized integer inputs and floating exported weights. This includes
  weight quantization, fixed-point approximations and output rounding; it is
  not an isolated measurement of output quantization alone.
- **Global error:** dequantized integer output versus the fully floating exported
  graph on the same prefix. This includes accumulated upstream error. It does
  not compare to a separate Transformers execution.
- Cosine similarity, relative L2 error normalized by the reference norm, and
  absolute maxima for both sides. Two zero vectors have cosine 1; only one zero
  vector gives cosine 0. A zero reference norm uses a 1e-15 denominator floor.
- **Integer endpoint occupancy:** fraction equal to the representation's lowest
  or highest code. An exactly representable endpoint counts here even without
  clipping; this metric must not be labeled a clipping or saturation rate.
- **Local float outside endpoint range:** fraction outside the scaled integer
  endpoint range. It describes the local floating reference, not observed RTL
  clipping. Rounding boundaries and intermediate clamps can differ.

The floating path supports attention, GQA, RoPE, normalization, quantization
boundaries, arithmetic, biased linears and fused RESADD/SiLU/MUL epilogues. KV
writes update the corresponding local and global reference state; they have no
standalone output metric. Compiler-padded output channels remain in the
per-operation arrays, including the zero-padded vocabulary tail.

Input file hashes cover the text, export (including tokenizer and float weights)
and quantized graph/weights, and are checked again before writing the report.
Different exported transformations or tokenizer inputs require a new profile.
Avoid generalizing a single prefix's worst operation to overall model quality;
use the independent calibration/validation/test protocol for selection.

## Retained pretrained exploratory profiles

`work/numerics/hf-qwen-baseline.json` contains 507 operation-output diagnostics;
`work/numerics/hf-llama.json` contains 633. Both use the original fixed-alpha 0.5
recipe calibrated on 64 authored sequences of length 32, and positions 0–7 of
the first calibration line, “The history of science includes many careful
experiments”. These are historical exploratory profiles, not fresh profiles of
the final TRAIN/VALIDATION-selected artifacts. Their JSON lacks embedded source
hashes; the maintained tool above adds provenance for new runs.

Local references use the same dequantized integer inputs for each operation;
global references follow the exported fp32 graph. Qwen local NRMSE was 25.27%
for `l2.f.q` and 22.75% for `l16.o`; Smol was 49.50% for `l20.o`, 38.11% for
`l18.o`, and 37.67% for `l14.o`. These outputs had zero endpoint occupancy,
showing why endpoint counts alone cannot establish numerical accuracy. Mean
local RMSNorm NRMSE was 0.179% for Qwen and 0.146% for Smol on this prefix.
The independent held-out comparisons in [QUANTIZATION_TUNING.md](QUANTIZATION_TUNING.md)
establish the selected recipes' measured quality separately.
