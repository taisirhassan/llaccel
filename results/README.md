# Evidence bundles

Each bundle applies to its recorded source and binary hashes. Later documentation
and comment edits do not rewrite those historical manifests.

| Bundle | Scope |
| --- | --- |
| [Local Verilator tests](verilator-local-2026-09-15/README.md) | All testbenches passed; 44 compiled model runs matched the reference. |
| [Dense software](dense-llama-qwen-2026-09-15/manifest.json) | 88 seeded comparisons, four context-end checks, 309 Python and nine CTest passes; predates later RTL/backend edits. |
| [Wide-head source](rtl-wide-head-2026-09-15/manifest.json) | Historical implementation record from before testing; see the local Verilator results above. |
| [Earlier measurements](dram-kv-2026-09-10/) | Architecture, pretrained quality and incomplete physical implementation. |

## Earlier measurement records


`dram-kv-2026-09-10/` contains compact copies and summaries of the measured
architecture, numerical, synthesis and software-test evidence. Its manifest is
currently **PARTIAL_PHYSICAL_BLOCKED**: selected-model RTL campaigns are complete, while routed physical results
are blocked by Docker OOM at detailed-route track assignment. V1 global routing and both synthesis runs completed; see `physical-status.json`. A successful numerical evaluation means that the comparison
completed, not that its remaining accuracy loss meets a quality threshold.

The manifest hashes the original local reports. Full checkpoints, generated
images, corpora, full-logit traces and physical databases stay outside Git.
The reproduction commands and dataset/checkpoint pins are in
[REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md),
[HF_SUPPORT.md](../docs/HF_SUPPORT.md), and
[RESULTS.md](../docs/RESULTS.md). Synthesis measurements exclude external SRAM
and DRAM infrastructure and must not be interpreted as routed chip results.
