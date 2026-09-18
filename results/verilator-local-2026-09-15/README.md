# Local Verilator tests

Verilator 5.052, macOS ARM64. All commands completed successfully.

| Test | Result |
| --- | --- |
| Full-design lint | Passed with `-Wall` |
| Matrix engine | 500 cases each, fusion off/on |
| DMA | 120 cases |
| SRAM crossbar | 10,011 cases |
| Full system | 60 cases each, fusion off/on |
| Integer math | 12,000 operand sets |
| Vector engine | 1,203 cases |
| Attention / KV writes | 724 cases |
| DRAM arbiter | Depths 1, 4 and 64 |
| HALT / command handling | 12 scenarios |
| Compiled models | 44 runs matched the integer reference |

Attention includes D128/256, 16 query rows, contexts 257 and 4096, grouped-query
attention and randomized backpressure. RoPE includes D128/256 with 16 rows and
three heads. The testbench suite used seed 1.

The model checks used 11 existing seeded Llama/Qwen fixtures, two compiler targets
and two schedules. Each run prefills its prompt and generates two tokens. These
are correctness checks on small models, not pretrained quality measurements.

```sh
make -C tb all JOBS=3
verilator --lint-only -Wall -f rtl/filelist.f --top-module llaccel_top
```

The runtime was built in `build/verilator-local` with `LLACCEL_RTL=ON`. Reusing
`build/review-rtl` first produced a generated-code linker error; a fresh build
resolved it. No hardware logic changes were needed to pass the tests.

- [Testbench log](verilator-tests.log)
- [Lint log](verilator-lint.log)
- [Model runs](model-tests.log) and [case list](models.json)

Physical implementation and timing closure are separate from these simulations.
