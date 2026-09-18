# Dense Llama/Qwen software regression

`scripts/regress_dense_hf.py` exercises local, seeded Transformers Llama,
Qwen2 and Qwen3 checkpoints through the importer, C++ MLIR compiler, independent
integer reference and functional simulator. It never requests an RTL backend,
Verilator or physical-design tools.

Run from the repository using an isolated software-only build:

```sh
.venv/bin/python3 scripts/regress_dense_hf.py \
  --build build/llama-qwen-software --out work/dense-support
```

The build must have `LLACCEL_RTL=OFF`. The runner retains checkpoint/tokenizer,
export, compiled images, quantized graphs, golden logits, generated IDs and
subprocess logs. `results.json` starts incomplete and is atomically updated;
only the successful end of all requested cases marks it complete. Launch and
completion compiler/simulator hashes must match. `--cases NAME ...` selects
named cases and the report records that narrower scope.

| Case | Behavior exercised |
| --- | --- |
| llama-untied | Dense GQA baseline with separate output weights |
| llama-tied-bias | Shared embeddings, nonzero attention and MLP biases |
| llama-linear | Static linear RoPE scaling |
| llama3-rope | Llama 3 frequency-dependent RoPE scaling |
| llama-head128 | 128-element heads |
| llama-head256 | 256-element heads |
| qwen2-untied | Nonzero Q/K/V biases |
| qwen2-tied-linear | Tied embeddings and static linear RoPE |
| qwen2-yarn | YaRN frequencies and attention-amplitude scaling |
| qwen3-qknorm | Nonuniform Q/K normalization weights |
| qwen3-independent-width | Query projection width different from residual width |

Each case uses both v1/v2 compilation targets and inorder/overlap schedules.
Weight chunks are sized to fit at least one 16-column tile for each model,
while preserving streamed execution for larger matrices. Each compiled image runs with ordered and randomized functional execution:
88 exact integer comparisons for the full matrix. Import validation compares
against Transformers float outputs; generated trajectories also report
teacher-forced float agreement. Exact integer agreement does not imply exact
float agreement. An active sliding window smaller than the compiled context
must be rejected explicitly.

These are small random models intended to expose computational and scheduling
mistakes, not pretrained language-quality benchmarks. Passing establishes only
the tested software paths. It does not establish RTL execution, unrestricted
model sizes/context lengths, every Llama/Qwen release, MoE or multimodal support.
Refer to `HF_SUPPORT.md` for the accepted architecture contract and remaining
limitations. Previously measured pretrained and RTL campaigns remain separate.

The focused context-end companion uses the compiled Qwen3 independent-width
v2/overlap fixture:

```sh
.venv/bin/python3 scripts/regress_dense_boundary.py
```

It compares full integer reference logits and generated IDs for 31-token
prefill plus one decode and 32-token prefill plus zero decode, each with ordered
and randomized functional execution. Its separate report is
`work/dense-boundary/results.json`; input image, graph, weights and simulator
hashes are checked at launch and completion. These four checks supplement the
88-run matrix without changing its scope.

Local run on 2026-09-15: all 88 matrix comparisons and all four context-end
checks passed. The largest independent Transformers float-import absolute
error across matrix probes was 6.56e-7. These observations concern the hashes
recorded in the reports, not future revisions.

The subsequent larger-head RTL source extension is documented separately in [DENSE_HARDWARE_GAPS.md](DENSE_HARDWARE_GAPS.md). The software results above do not validate that unbuilt, untested RTL.
