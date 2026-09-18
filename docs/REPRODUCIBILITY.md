# Reproducing the tested toolchain

`toolchain.json` records the exact macOS ARM64 versions observed in this review. The C++ compiler and MLIR/LLVM must come from the same LLVM installation. The compiler links the MLIR shared library consistently; mixing its shared library and component static libraries caused duplicate MLIR type registrations and a parser crash.

Python is selected by `.python-version`; `uv sync --frozen --extra hf` installs the checked-in `uv.lock` resolution without changing it. Native tools are preinstalled. Run `uv run --frozen python scripts/check_toolchain.py` to fail on a tool, Python, library, or package version mismatch. This is a version lock and preflight, not a hermetic image or a guarantee of identical native binary provenance.

The tested native versions are LLVM/MLIR 23.1.1, Verilator 5.052, CMake 4.4.3, Ninja 1.13.2, uv 0.12.12, nlohmann-json 3.12.0 and LZ4 1.10.0. Python is 3.14.7. The local build expects those tools to be installed, with LLVM at `/opt/homebrew/opt/llvm`.

Run `bash scripts/ci.sh` for the full local test suite. It builds the compiler, functional runtime and both RTL variants, runs CTest, Python tests and every engine testbench, then generates two seeded model shapes and checks 168 combinations of model/variant/prompt/backend. It also checks 48 HF fixture/backend cases with independent Transformers float comparisons, tied and untied weights, QKV bias, and channel balancing. Calibration uses a generated matching corpus, so this regression requires no trained checkpoint or dataset download. Reports and artifacts are under `build/ci/` (override with `LLACCEL_CI_BUILD`).

Tests run locally; no GitHub Actions runner is required. Linux, other compiler releases, FPGA execution and physical ASIC implementation are outside this tested toolchain.

The optional full pretrained campaign is `uv run --frozen --extra hf python scripts/regress_pretrained.py` after configuring/building `build/dma32` with `-DLLACCEL_DMA_OUTSTANDING=32`. It downloads pinned safetensors checkpoints and tokenizer data, then runs both complete models sequentially through fresh export, compilation, functional simulation and RTL. Exact integer verification and original-model quality are separate report fields. See `HF_SUPPORT.md` for capacity limits and checkpoint revisions. This larger campaign is not part of the default CI fixture suite.

## Independent numerical selection and confirmation

Fetch the pinned WikiText splits with checksum verification:

```sh
uv run --frozen --extra hf --with pyarrow python scripts/fetch_evaluation_data.py \
  --splits train validation test
```

The dataset is `Salesforce/wikitext`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, raw WikiText-2. Text and source-Parquet hashes are recorded under `work/evaluation-data/`; see the [source dataset card](https://huggingface.co/datasets/Salesforce/wikitext) for CC-BY-SA/GFDL attribution. Generated text is kept outside version control.

The follow-up export uses `--context 1024 --calibration-sequences 128 --calibration-length 128 --calibration-sampling uniform-windows --smoothquant-alpha 0.5 --smoothquant-auto-alpha` with `--calibration work/evaluation-data/wikitext-2-train.txt`. Compile for `llaccel-v1`, overlap, prefill16 and131072-byte weight chunks. Validation uses `--text work/evaluation-data/wikitext-2-validation.txt --min-scored-tokens 1024 --window-length 32 --window-sampling uniform`. Freeze the selected artifacts before evaluating TEST.

Final confirmation uses `--text work/evaluation-data/wikitext-2-test.txt --min-scored-tokens 4096 --window-length 32 --window-sampling uniform --token-offset 8192`. The original recipe must use the identical token IDs/window starts. The offset separates confirmation from the first failed authored-corpus experiment's4224token IDs. Each window resets KV and positions; report these bounded diagnostics with their protocol rather than calling them standard full-context benchmark perplexity. Checkpoint tokenizers differ, so cross-model absolute perplexities are not directly comparable.

Successful export manifests hash their complete inputs. Large regenerable fp32 intermediates may be deliberately removed after compilation, with their checksum/reason recorded in `intermediates-removed.json`. Such directories cannot be reused as complete exports; rerun export to restore them. Integer graphs, device images, pinned source checkpoints and campaign provenance are retained separately.

### Completed measurement binaries and storage cleanup

The completed selected pretrained campaigns and final full4096 RTL prefill used the simulator archived in `build/validated-pre-memory-release/`. A later CLI-only change releases its original DRAM image buffer after synchronous upload; the shared Host API is unchanged. `results/dram-kv-2026-09-10/host-upload-memory.json` records old/new binary hashes and functional, boundary, RTL and tokenizer validation. No peak-memory measurement is claimed.

After completed input audits, disk recovery removed baseline reference weight blobs and the selected Smol `auto-qgraph/qweights.bin`. The selected Smol reference was subsequently regenerated with the frozen TRAIN recipe and restored. Its qweights SHA-256 exactly matches `d26ca32bbf3d8622ed1466120292966b6d326bea34f745f248595d44bd9df10c`; regenerated graph and compiled image also match the original bytes. `results/dram-kv-2026-09-10/smol-reference-restored.json` records the proof. Both selected reference graphs are complete.
