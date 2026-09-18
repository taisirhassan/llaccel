# Hugging Face checkpoint support

The HF path imports dense floating-point `llama`, `qwen2` and `qwen3` checkpoints into the decoder graph, calibrates with their own tokenizers, and compiles through the C++/MLIR pipeline. The supported configurations are checked in the functional simulator and local Verilator simulation, including head widths through 256. Backends reject incompatible images before upload. See the [hardware test results](../results/verilator-local-2026-09-15/README.md). It retains every layer, weight, and vocabulary entry. The default compiled context is **32 tokens**; this is a deliberate context reduction, not the checkpoint's native context capability.

The initial context-32 pinned pretrained campaign completed with aggregate PASS in `build/pretrained-results.json`. Each model used the prompt `Hello`, generated eight tokens, and compared all logits across nine launches. Quality below is teacher-forced along that generated history; cycle counts are simulated cycles, not measured silicon performance.

| Pinned checkpoint | Complete func/RTL integer comparison | Original fp32 quality comparison | RTL cycles |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B` | MATCH, 9 launches / 8 generated tokens | cosine 0.968879; top-1 77.78% | 251,777,087 total |
| `HuggingFaceTB/SmolLM2-135M` (`model_type=llama`) | MATCH, 9 launches / 8 generated tokens | cosine 0.928518; top-1 66.67% | 67,721,434 total |

SmolLM2 establishes a checkpoint using the implemented Llama architecture contract. It is not a claim that Meta Llama checkpoints, arbitrary Llama derivatives, or every Qwen family are supported.

## DRAM-KV expanded verification

Both complete checkpoints now have context-1024, prefill-M16 images. Eight preinitialized-state boundaries (255, 256, 257, 511, 512, 513, 1023, 1024) passed on both RTL and functional simulation: 32 checks total. Every vocabulary logit and every KV byte, including untouched tails, matched independent golden state. The prefix state was prepared independently; these checks do not claim full RTL execution of the 1024-token pretrained prefix. Methodology, tool/image hashes and counters are in `build/pretrained-boundary-summary.md` and [KV_BANDWIDTH.md](KV_BANDWIDTH.md).

Full prompt-to-generation conformance uses four natural prompts with 16 generated tokens each and a separate 64-token decode. The final selected Smol image completed all ten backend cases in `build/pretrained-expanded/smol-selected/`, with launch-time image, graph, simulator and tokenizer hashes rechecked after completion. The selected Qwen campaign also completed all ten backend cases in `build/pretrained-expanded/qwen-selected/`, with completed-input audits retained for each shard. It is separate from teacher-forced quality evaluation: exact device agreement does not imply acceptable pretrained accuracy.

The first broader 4096-target held-out check exposed calibration overfitting. Qwen's excess CE changed from +0.47053 to +0.45301 nats; Smol worsened from +0.31540 to +0.42233. Independent TRAIN calibration and VALIDATION selection have now frozen replacement recipes for both models. On fresh 4096-target TEST cohorts, Smol's perplexity penalty decreased from 41.16% to 11.77% and Qwen's from 76.60% to 17.47%. The selected Smol image also passed all eight boundaries on both backends; selected Qwen also passed all eight boundaries on both backends. See [QUANTIZATION_TUNING.md](QUANTIZATION_TUNING.md); do not describe the calibration-only gains as held-out improvement.

Regenerable fp32 export `weights.bin` intermediates were deliberately removed after compilation and boundary hash verification to recover disk space. Each export records their original checksum and the removal reason. Checkpoints, integer graphs and device images remain available. A fresh export recreates those intermediates; `--reuse-export` correctly rejects missing artifacts.

## Reproduce the independently selected recipe from a fresh export

Run from the repository root on the native toolchain described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The commands below create a separate native build and fresh export/result directory. The Python lockfile is used unchanged; native tools must already be installed.

```sh
uv sync --frozen --extra hf
uv run --frozen --extra hf python scripts/check_toolchain.py
cmake -S . -B build/hf-native -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DLLACCEL_COMPILER=ON -DLLACCEL_RTL=ON -DLLACCEL_DMA_OUTSTANDING=32
cmake --build build/hf-native -j 3
uv run --frozen --extra hf python scripts/regress_pretrained.py \
  --build build/hf-native --out build/hf-pretrained-fresh \
  --checkpoint-root checkpoints/hf-pinned \
  --models qwen llama --context 1024 --prefill-m 16 --tokens 16 \
  --calibration work/evaluation-data/wikitext-2-train.txt \
  --calibration-sequences 128 --calibration-length 128 \
  --calibration-sampling uniform-windows --smoothquant-auto-alpha --prompt Hello
```

The runner downloads only configuration, tokenizer, and safetensors files at these immutable revisions:

| Repository | Revision |
| --- | --- |
| `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |

Fetch the pinned TRAIN/VALIDATION/TEST splits first using the command in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The checkpoint files are hashed, and their Hugging Face download metadata must establish the requested revision. The command processes models sequentially and uses `llaccel-v1`, overlap scheduling, prefill M=16, 131072-byte weight chunks, and per-group SmoothQuant selection with base alpha0.5. A fresh export is the default. `--reuse-export` and `--skip-download` are explicit reuse modes with provenance checks, not part of the fresh command above.

Inspect `build/hf-pretrained-fresh/pretrained-results.json`. Only aggregate `status: "PASS"` establishes that both requested models finished every required stage. Caught failures and interruptions record `FAILED`; abrupt process termination may leave `RUNNING`. Neither establishes an aggregate pass. Each model directory retains download/export/compile/run logs, source hashes, MLIR and weights, calibration, compiled image, qgraph, golden outputs, simulator outputs, and the independent quality comparison. A full pretrained RTL simulation can take substantial time; model import or compilation finishing alone does not establish end-to-end execution.

## Supported contract and capacity

The importer accepts local single-file or indexed/sharded safetensors with floating-point weights. It validates exact names, shapes, finite values, required tensors, shard ownership, and local path containment. Tied output embeddings, Qwen2 Q/K/V biases, Llama attention/MLP biases and Qwen3 per-head Q/K normalization are handled explicitly. Checkpoint Python code and pickle weights are not executed.

The supported graph uses full causal attention, RMSNorm, SwiGLU, and full-width rotate-half RoPE. Dense `qwen3` adds per-head Q/K RMSNorm, lowered to existing device normalization instructions. The query projection width can differ from the residual width. Multi-head and grouped-query attention require query heads divisible by KV heads.

Static default, linear, Llama3 and YaRN RoPE tables are supported. YaRN amplitude must fit signed Q1.14; custom mscale fields, dynamic NTK/LongRoPE, partial rotary embeddings, active sliding windows, prequantized checkpoints and other model types remain rejected. Qwen MoE, multimodal Qwen, legacy `qwen` and `llama4` are not aliases for these supported graphs. Checkpoint names alone do not establish compatibility.

The software compiler supports head dimensions 16, 32, 64, 128 and 256 and projection widths compatible with 16-element tiles. Local Verilator tests cover all five head widths. The host checks model metadata and actual instruction widths against the selected backend before uploading an image. The model comparisons check correctness; performance depends on the workload. See [DENSE_HARDWARE_GAPS.md](DENSE_HARDWARE_GAPS.md). Context is bounded to 4096 positions and the selected context must fit SRAM. Resident RoPE consumes `context * head_dim * 2` bytes: D=128 at 4096 already consumes all 1 MiB before activations, so it cannot compile with that context. DRAM images remain limited to the 32-bit address space, excluding arbitrary checkpoint sizes.

Inspect a local checkpoint configuration without allocating weights:

```sh
.venv/bin/python3 -m llaccel.hf inspect checkpoints/hf/Qwen2.5-0.5B --context 1024
```

This reports configuration compatibility, compiler prerequisites and current RTL head geometry separately. It does not certify weights, tokenizer, successful allocation or execution. Import, compilation and software comparison remain required. `llaccel.hf run` defaults to functional simulation; RTL requires explicit `--backends rtl`.

See [DENSE_SOFTWARE_REGRESSION.md](DENSE_SOFTWARE_REGRESSION.md) for the seeded software-only compatibility matrix. It is separate from the older pretrained accuracy and RTL evidence above.

The accelerator has 1 MiB of SRAM. Weights are streamed from simulated DRAM; the vocabulary output is tiled and written directly to DRAM instead of allocating the entire logits tensor in SRAM. This permits large checkpoint vocabularies without replacing the language-model head with a small surrogate. Persistent KV resides in DRAM with a fixed 256-entry attention score tile; activations and resident RoPE tables still impose SRAM capacity limits. No native long-context or unlimited model-size claim follows from the tiled head.

## Host/device boundary and verification

The HF tokenizer encodes the prompt and calibration corpus using the checkpoint's real token IDs and special-token rules. The saved tokenizer is reused to decode generated IDs. The host supplies embeddings, launches programs, and chooses the next token by argmax. Transformer layers, normalization, attention, KV updates, MLPs, and the full output projection execute through the compiled ISA; host fp32 inference is used only as an independent validation reference.

The export checks the imported, optionally smoothed fp32 model against the original Transformers model before integer quantization. The current CLI checks two calibration probes of up to eight tokens with explicit tolerances. This is a directed equivalence check, not exhaustive proof across all prompts or positions. `--skip-float-check` explicitly omits that check and is not used by the pinned runner.

The integer golden model and both simulator backends must match complete prompt tokens, generated IDs, step argmax values, and recorded logit rows. Functional simulation also uses randomized legal engine interleaving with seed 7. Exact integer agreement demonstrates compiler/runtime/RTL consistency for those runs. It does **not** establish that quantized text matches the original floating-point model.

The runner separately records teacher-forced logit cosine and top-1 agreement against the original checkpoint, evaluated along the integer-generated token history. These are prompt-specific diagnostics, not held-out perplexity, a broad benchmark, or a claim of negligible language-quality loss. The initial campaign used 64 sequences from the short corpus. The final validation-selected recipe uses 128 uniformly dispersed sequences of length 128 from the pinned WikiText TRAIN split, with group-specific balancing; see [QUANTIZATION_TUNING.md](QUANTIZATION_TUNING.md).

## Offline balancing

[SmoothQuant](https://github.com/mit-han-lab/smoothquant) moves activation range into weights through equivalent parameter rescaling. This implementation adds no operators or hardware instructions. It collects channel maxima using the real calibration token sequences and applies three transformations:

1. **RMSNorm to projections:** divide norm gamma by a channel scale and multiply matching Q/K/V or gate/up weight columns by that scale. The final norm/head group follows the same rule; a tied output head is cloned first so input embeddings remain unchanged.
2. **Value to output projection:** divide V projection rows and biases by a scale and multiply output-projection columns by the same scale, repeated over the query heads sharing each KV head. Q/K and attention probabilities are unchanged.
3. **FFN product to down projection:** divide up-projection rows by a scale and multiply down-projection columns by that scale. The gate projection and SiLU remain unchanged.

Scales use activation and weight-column maxima. The initial pinned campaign used fixed alpha 0.5; the independently selected recipe explicitly enables per-group alpha search. Large weight reductions operate in row chunks; finite scales and composed parameter ranges are checked before mutation. Tests cover nonzero biases, tied embeddings, multiple KV groups, measured outlier reduction, fp32 logit equivalence, and unchanged exported operation graphs. Equivalent float behavior does not by itself guarantee a particular integer-accuracy improvement; the final quality report must measure that separately.

## Offline compatibility regression

The smaller regression needs no checkpoint download:

```sh
uv run --frozen --extra hf python scripts/regress_hf.py \
  --build build/hf-native --out build/hf-offline-regressions
```

It creates seeded Transformers Llama/Qwen2 models with tied and untied heads and a real byte-level BPE tokenizer. It exercises all v1/v2 and inorder/overlap combinations, prefill M=4, forced weight chunking, ordered and randomized functional execution, and RTL against the golden model. The matrix contains 48 backend cases. These random-model fixtures establish compatibility and dataflow coverage; they are not pretrained language-quality evidence.

## Bounded reference-text quality evaluation

```sh
uv run --frozen --extra hf python scripts/evaluate_hf_quality.py \
  --checkpoint checkpoints/hf/Qwen2.5-0.5B --qgraph build/hf-qwen/qgraph \
  --text tests/data/hf-evaluation.txt --out build/hf-qwen/held-out-quality.json
```

Use the SmolLM2 checkpoint and `build/hf-llama/qgraph` for the Llama-architecture run. Pass `--min-scored-tokens 4096` to score thousands of reference targets; the default remains a 64-target smoke check. Explicit window length controls independent segments, using actual text history and resetting KV between windows. It records integer/original-model cross-entropy and perplexity, top-1 agreement, cosine, and input hashes. The checked-in evaluation text is separate from the calibration file. This is a small local diagnostic, not benchmark perplexity. These additional sequences run through the integer golden model; the pretrained RTL conformance campaign remains the separately recorded prompt above.

Historical reference-text results from the initial recipe (64 scored tokens per model; different tokenizers produce different token sequences):

| Model | Mean cosine | Top-1 agreement | FP32 perplexity | Integer perplexity | Extra cross-entropy (nats) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 0.953517 | 67.19% | 62.027 | 96.564 | 0.442632 |
| SmolLM2-135M | 0.956901 | 65.62% | 59.397 | 90.749 | 0.423855 |

The roughly 0.42–0.44 nat increase is material quantization loss. Broader calibration, quantization-aware training, or more flexible activation scales are possible next numerical improvements; these measurements do not establish near-lossless model quality.

## Experimental calibration candidates

`--smoothquant-auto-alpha` selects a separate balancing alpha per group using bounded calibration activation samples and a weighted linear quantization-error proxy. `--optimize-quantization` selects explicit `*.i8_absmax` / `*.i16_absmax` bounds from sampled reconstruction MSE. Both are **experimental and disabled by default**. A lower local error can still worsen end-to-end logits or cross-entropy; these switches do not promise better language quality. Raw maxima remain available separately for arithmetic range planning.

`--calibration-length` decouples the sampled sequence length from a larger compiled context. `tests/data/hf-calibration-expanded.txt` provides a separate, authored multi-domain calibration corpus. Compare candidate recipes on a separate VALIDATION split, freeze the selected settings and TRAIN corpus hash, and only then measure untouched TEST quality. Never tune settings against the held-out report. Final reproduction commands must name the selected flags explicitly rather than assuming that enabling every optional optimization improves the result.

An export publishes `hf-import.json` atomically only after its graph, weights, calibration, and tokenizer are complete. The manifest records their hashes. A failed overwrite removes the previous completion manifest, preventing a partial new export from inheriting an earlier success record.

Per-operation numerical profiling is available through [QUANTIZATION_PROFILING.md](QUANTIZATION_PROFILING.md). It requires a complete fp32 export and reports a bounded calibration-prefix diagnostic, separate from held-out quality and RTL conformance.
