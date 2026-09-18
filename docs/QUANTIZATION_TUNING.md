# Calibration-selected quantization

This experiment compares actual integer cross-entropy on a calibration selection cohort before evaluating held-out quality. It does not use the held-out text to choose parameters. The original recipe uses fixed SmoothQuant alpha 0.5, 64 calibration sequences from `tests/data/hf-calibration.txt`, and context/calibration length 32. The expanded corpus is `tests/data/hf-calibration-expanded.txt`, SHA-256 `ef13443b7b4f4a32634e0450975b97737b82c293d24b9878c67a8f3973e6a126`. Calibration sequence construction uses deterministic seed 0; it takes line prefixes first, then contiguous token windows.

The selection cohort consists of the first 32 non-overlapping reference-text windows from the expanded calibration corpus, with 16 next-token predictions per window: **512 scored calibration targets**. Every window resets KV state and follows the actual reference tokens. Cross-entropy delta means integer CE minus original HF fp32 CE, measured in nats per token. These measurements are calibration evidence, not held-out perplexity.

| Model and candidate | Calibration CE delta | Decision |
| --- | ---: | --- |
| Qwen2.5-0.5B, original recipe | +0.28777 | Baseline |
| Qwen2.5-0.5B, expanded corpus, fixed alpha 0.5, calibration length 32 | +0.36706 | Rejected |
| Qwen2.5-0.5B, expanded corpus, per-group alpha, no clipping, calibration length 32 | +0.05526 | Advance to larger-context candidate |
| Qwen2.5-0.5B, expanded corpus, per-group alpha, no clipping, context 1024/calibration length 128 | +0.01309 | Freeze for independent verification |
| SmolLM2-135M, original recipe | +0.41027 | Baseline |
| SmolLM2-135M, expanded corpus, per-group alpha, no clipping, context 1024/calibration length 128 | +0.07867 | Freeze for independent verification |

The 64-target exploratory comparison was too small to order the candidates reliably. The final selection uses the larger 512-target cohort. Qwen's full clipping candidate reduced a local reconstruction-error objective but worsened its end-to-end calibration logits. Even int8-only clipping raised 64-target CE delta from -0.00847 to +0.16468. **Clipping is not enabled in the selected recipe.** This is why minimizing a local proxy is followed by actual integer-model evaluation.

## Independent quality warning and follow-up

The authored-corpus selection above did **not** establish a general quality gain.
The subsequent 4096-target WikiText test diagnostic found Smol CE delta increased
from +0.31540 for the original recipe to +0.42233 for the selected candidate.
Those artifacts remain valid integer-conformance targets, but that result is a
language-quality regression. The original reports are
`work/numerics/smol-baseline-wikitext4096.json` and
`work/numerics/smol-final-wikitext4096.json`.

The follow-up experiment separates three data uses: 128 length-128 windows
uniformly dispersed across pinned WikiText TRAIN for calibration; 1024 targets
uniformly dispersed across VALIDATION for fixed-alpha versus per-group-alpha
selection; then a previously unexamined TEST segment for confirmation. The
export's `--calibration-sampling uniform-windows` records every window start and
the exact sampled-token hash. These candidates use separate scratch artifacts
and did not overwrite the earlier RTL campaign. Completed validation selections
and fresh confirmations are reported below.

The pinned source is `Salesforce/wikitext` revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, configuration `wikitext-2-raw-v1`.
TRAIN text SHA-256 is `c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8`;
VALIDATION text SHA-256 is `714a24a5de7459876f7cfdc64837e551dd7cde2a51f95b763ee083fff273e307`.
The Smol tokenizer produces 2,554,860 TRAIN tokens; the 128 dispersed windows
cover 16,384 calibration tokens, with sampled-token SHA-256
`3e3d05407056c6ae42ce6968af89c8c7f733337b28b2a1845cd406e49562647f`.

## Algorithm and controls

The exact float reparameterizations remain the RMSNorm/projection, V/output, and FFN-product/down transformations described in [HF_SUPPORT.md](HF_SUPPORT.md). `smooth_model(..., auto_alpha=True)` samples a bounded number of activation rows, distributed deterministically across calibration sequences. For each group it considers alpha values 0, 0.25, 0.5, 0.75, 0.9, and 1, plus the caller's alpha. It measures an activation-energy/weight-energy-weighted local quantization-error proxy and records every candidate score and selected alpha. No random search, training, or held-out text enters this choice.

The optional `calibrate_tokens(..., optimize_quantization=True)` uses bounded weighted activation samples to select separate int8 and int16 clipping bounds. It retains raw maxima, selected bounds, and sampled MSE values. The compiler and independent reference quantizer consume bounds named `*.i8_absmax` and `*.i16_absmax`; missing bounds preserve the ordinary max-based behavior. RMS reciprocal/range planning still reads raw maxima. Invalid or inconsistent bounds are rejected. This remains an experimental facility, disabled by default, because lower local MSE did not improve the tested end-to-end candidate.

Both CLI switches are explicitly experimental. `--smoothquant-auto-alpha` must accompany `--smoothquant-alpha`; `--optimize-quantization` should not be added to a reproduction command merely because it is available. Default library/CLI behavior remains fixed alpha and no clipping.

## Remaining precision bottleneck

A calibration-only per-operation profile compared integer results with the exported fp32 graph, using identical dequantized inputs to isolate local errors. The original Qwen recipe showed FFN-product int8 normalized RMS error around 22–25% in several groups. Original Smol output projections reached 35–49% local normalized RMS error in later layers. Those projections showed zero integer endpoint occupancy (not a direct clipping measurement); coarse residual headroom produced output quantization steps as large as one real unit. This points to residual precision as a remaining architectural tradeoff. It is not evidence that all remaining error can be repaired by clipping or more calibration.

The profiles and selection logs from this working run are retained under `work/numerics/`, including the original-recipe Qwen calibration JSON reproduced exactly. Final pretrained conformance and held-out results must be reported separately after the selected artifacts are frozen.

## Reproduce the earlier authored-corpus candidate

After installing the locked HF extras and building the current compiler, the candidate export uses explicit settings:

```sh
uv run --frozen --extra hf python -m llaccel.hf export \
  checkpoints/hf/SmolLM2-135M --out build/hf-llama/export \
  --context 1024 --calibration tests/data/hf-calibration-expanded.txt \
  --calibration-sequences 64 --calibration-length 128 \
  --smoothquant-alpha 0.5 --smoothquant-auto-alpha
```

For Qwen, substitute `checkpoints/hf/Qwen2.5-0.5B` and `build/hf-qwen/export`. Both earlier authored-corpus candidates used the same settings and corpus. Their 512-target selection reports are `work/numerics/qwen-final-calibration-512.json` and `work/numerics/llama-final-calibration-512.json`.

Compile with `--prefill-m 16`, overlap scheduling, and 131072-byte weight chunks. The export manifest records the corpus hash, sampled sequence count, calibration length, all per-group alpha choices, independent float-import validation, and hashes of the generated artifacts. Integer CE is measured with `scripts/evaluate_hf_quality.py --text tests/data/hf-calibration-expanded.txt --windows 32 --window-length 16`. This text is calibration data: do not label that report held-out evaluation.

## Independent test exposed calibration overfitting

The first frozen authored-corpus recipes were evaluated on 4096 targets from pinned WikiText-2 TEST, in 128 non-overlapping windows of 32 predictions. Each window resets positions and KV; each checkpoint uses its own tokenizer. These are bounded diagnostics, not standard full-context WikiText benchmark scores.

| Model | Original recipe excess CE | Authored expanded recipe excess CE |
|---|---:|---:|
| Qwen2.5-0.5B | +0.470532 | +0.453012 |
| SmolLM2-135M | +0.315400 | +0.422332 |

Qwen improves only modestly and Smol regresses. The much larger calibration gains above did not generalize. Reports remain in `work/numerics/*-wikitext4096.json`. No settings were selected by sweeping this test cohort.

The follow-up protocol calibrates on 128 uniformly dispersed 128-token windows from WikiText TRAIN and selects recipes on 1024 uniformly dispersed VALIDATION targets. A subsequent confirmation cohort starts at TEST token offset 8192, beyond the first experiment's 4224 token IDs, with its exact token offsets retained. The original recipe is evaluated on the identical confirmation cohort for a fair comparison. Validation selection and fresh test confirmation are complete for both models below.


## Independent TRAIN/VALIDATION selection

All three Smol recipes were scored on exactly the same 1024 VALIDATION target
IDs: 32 uniformly dispersed windows of 32 next-token predictions, with positions
and KV reset per window. The reconstructed original recipe's calibration JSON
matches the retained original calibration exactly.

| SmolLM2-135M recipe | VALIDATION excess CE (nats/token) |
|---|---:|
| Original authored-corpus recipe | +0.335886 |
| WikiText TRAIN, fixed alpha 0.5 | +0.456919 |
| WikiText TRAIN, per-group alpha | **+0.087958** |

The per-group-alpha candidate was frozen before fresh TEST confirmation; clipping
remains disabled. This is a validation-set selection result, not a test result.
Reports are `work/numerics/smol-independent-*-validation1024.json`; selected
artifact hashes are in `work/numerics/smol-independent/selection.json`. The
selected image and graph are `auto.llbin` and `auto-qgraph` in that directory;
`original-qgraph` was used for the matched fresh-test baseline; its JSON and hashes remain after the completed baseline weights were archived. Discarded
intermediate files and their hashes are recorded in `discarded-intermediates.json`.
An export whose float weights were discarded cannot be reused without rebuilding.

Reproduce the independently calibrated Smol export in a fresh output directory:

```sh
uv run --frozen --extra hf python -m llaccel.hf export \
  checkpoints/hf/SmolLM2-135M --out work/numerics/smol-reproduction/export \
  --context 1024 --calibration work/evaluation-data/wikitext-2-train.txt \
  --calibration-sequences 128 --calibration-length 128 \
  --calibration-sampling uniform-windows \
  --smoothquant-alpha 0.5 --smoothquant-auto-alpha
```

Compile with the same prefill16/overlap/131072-byte-chunk settings above. Validate
with `--text work/evaluation-data/wikitext-2-validation.txt
--min-scored-tokens 1024 --window-length 32 --window-sampling uniform`.


## Fresh Smol TEST confirmation

After freezing the validation-selected recipe, the original and selected models
were compared on the same previously unexamined TEST cohort: 4096 scored targets,
uniformly dispersed after token offset 8192, with 32 predictions per window.
No further Smol tuning used this result.

| Smol metric | Original recipe | TRAIN/VALIDATION-selected recipe |
|---|---:|---:|
| Excess CE, nats/token | +0.344709 | **+0.111252** |
| Integer perplexity | 122.4031 | **96.9177** |
| HF fp32 perplexity | 86.7136 | 86.7136 |
| Top-1 agreement with HF | 65.67% | **79.57%** |
| Mean logits cosine | 0.93670 | **0.97938** |

This confirms improvement on that fresh bounded diagnostic, with remaining
quantization loss. It is not a full-context benchmark or a guarantee across
models and domains. Evidence: `work/numerics/smol-confirmation-original.json`
and `work/numerics/smol-confirmation-selected.json`, which retain exact windows,
artifact/source hashes, per-token losses and tokenizer identity. RTL conformance
of the selected image is verified separately.


## Independent Qwen selection

The same predeclared TRAIN/VALIDATION protocol was applied to Qwen2.5-0.5B,
with its own tokenizer and exact reference IDs. Its TRAIN stream contains
2,517,233 tokens; the 128 dispersed windows total 16,384 tokens, SHA-256
`935c9285c5fd8ef4d6268c5b0eb0b4948b391e7564a97a5131ee03a3e12d644a`.
The original calibration JSON was reproduced exactly before comparison.

| Qwen2.5-0.5B recipe | VALIDATION excess CE (nats/token) |
|---|---:|
| Original authored-corpus recipe | +0.506581 |
| WikiText TRAIN, fixed alpha 0.5 | +0.667974 |
| WikiText TRAIN, per-group alpha | **+0.145267** |

The per-group-alpha candidate was frozen, with clipping disabled, before the
fresh TEST confirmation below. All three reports use the same 1024 VALIDATION targets,
verified by exact token-ID equality. Reports, model/source hashes and selection
are under `work/numerics/qwen-independent/`; selected `auto.llbin` and
`auto-qgraph` remain immutable. The completed `original-qgraph` baseline retains its JSON and hashes; its weights were archived after confirmation. Rejected fixed
artifacts and regenerable fp32 intermediates were removed only after recording
hashes and metrics. Reproduce with the independently calibrated Smol command
above, substituting the Qwen checkpoint and a fresh Qwen output directory.


## Fresh Qwen TEST confirmation

After freezing the validation-selected recipe, both Qwen recipes were evaluated
on the same previously unexamined TEST cohort: 4096 scored targets uniformly
dispersed after token offset 8192, with 32 predictions per window. No further
numerical tuning used these results.

| Qwen metric | Original recipe | TRAIN/VALIDATION-selected recipe |
|---|---:|---:|
| Excess CE, nats/token | +0.568696 | **+0.161037** |
| Integer perplexity | 117.5377 | **78.1868** |
| HF fp32 perplexity | 66.5573 | 66.5573 |
| Top-1 agreement with HF | 60.11% | **76.27%** |
| Mean logits cosine | 0.91439 | **0.97533** |

Excess CE decreased 71.68%; the relative perplexity penalty decreased from 76.60%
to 17.47%. This confirms improvement on this bounded diagnostic, with remaining
quantization loss. It does not establish full-context benchmark performance or
a guarantee across models and domains.

`work/numerics/qwen-confirmation-comparison.json` verifies exact window starts
and token IDs, matching corpus/checkpoint/tokenizer hashes, identical fp32 CE,
and both original and selected graph hashes against the frozen selection record.
It retains hashes of `qwen-confirmation-original.json` and
`qwen-confirmation-selected.json`. Its PASS status means cohort/provenance
checks passed; `improvement_observed` separately records the quality outcome.
The selected run groups 16 independent reference windows per disk-backed batch;
the original used one. This changes retention and execution scheduling, not the
per-window inputs or forward computation. The comparison records both settings
and the selected run's two-thread environment. Selected-image RTL conformance
is reported separately.

## Retained real-model operation profiles

`work/numerics/hf-qwen-baseline.json` contains 507 operation-output measurements
for Qwen2.5-0.5B; `work/numerics/hf-llama.json` contains 633 for SmolLM2-135M.
Their logs are `qwen-profile.log` and `llama-profile.log` in the same directory.
These exploratory profiles used the original fixed-alpha-0.5 recipe calibrated
on 64 authored sequences of length 32, not the final TRAIN-selected recipe.
Both evaluate positions 0–7 of the calibration prefix “The history of science
includes many careful experiments”. Local error compares an operation against
its fp32 counterpart using identical dequantized integer inputs; global error
compares against the complete exported fp32 graph.

Qwen's `l2.f.q` local normalized RMS error was 25.27% and `l16.o` 22.75%; Smol's
`l20.o` was 49.50% and `l18.o` 38.11%. These outputs had zero integer endpoint
occupancy, not a direct clipping measurement. Mean local RMSNorm normalized
RMS error was 0.179% for Qwen and 0.146% for Smol. The historical JSON calls endpoint
occupancy `saturation`; that label must not be interpreted as observed clipping.
These files lack embedded source-artifact hashes and are retained as scoped
exploratory evidence. They do not establish final-recipe error at every layer
or quality over long contexts.


After both fresh confirmations completed, the original baseline `qweights.bin`
files were removed to recover disk space. Their hashes, byte sizes and reasons
are recorded in `work/numerics/completed-baseline-weights-removed.json`; baseline
graph JSON, calibration, checkpoint files and comparison provenance remain.
The original baseline directories therefore require weight reconstruction before
rerunning evaluation. Selected `auto-qgraph` weights and `auto.llbin` images
were preserved for the complete RTL campaigns.
