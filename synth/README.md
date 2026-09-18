# Synthesis checks and physical design

From the repository root:

```sh
synth/run_yosys.sh elab v1
synth/run_yosys.sh elab v2
synth/run_yosys.sh stat v1
synth/run_yosys.sh stat v2
synth/run_orfs.sh v1 synth
synth/run_orfs.sh v2 finish
```

`elab` renders the Yosys command template with fusion disabled/enabled, checks
hierarchy and connections, and exports a Verilog netlist under `build/synth`.
`stat` performs generic synthesis and connection checks. Both save complete
logs there. The `.ys` sources are templates: invoke the wrapper instead of
passing them directly to Yosys, whose command scripts do not expand shell
variables.

ORFS runs with three worker cores and propagates failures from both setup and
the make pipeline. Full console logs are saved under `build/orfs`. Set
`ORFS_IMAGE` to an image digest for a reproducible toolchain; the default is
the installed digest `openroad/orfs@sha256:d4598d07ce4dbdbed3c1baaa98860f983c305d1141ce3fdc4a31da788c32db61`. Supported stages are `synth`, `floorplan`, `place`,
`cts`, `route`, and `finish`.

Generic cell counts and successful elaboration are not physical PPA results.
Area and timing claims require the corresponding ORFS reports and identified
toolchain. The synthesized boundary is `llaccel_core`; the external 1 MiB SRAM
banks are excluded from its area. The SDC requests a 3 ns clock; this is a
constraint, not a measured achievable clock. Power additionally needs a stated
switching-activity/workload methodology.

For an immutable comparison, put the two exported netlists and `constraint.sdc`
under `build/synth/<snapshot>/`, then set `ORFS_SNAPSHOT=<snapshot>`. This skips
elaboration and writes results under `build/orfs/<snapshot>/`; for example,
`ORFS_SNAPSHOT=baseline synth/run_orfs.sh v1 finish`. Snapshot names contain
letters, digits, underscores, or hyphens only.

On Apple Silicon, `run_native_orfs.py` can use host Yosys/ABC for mapping and
then hand the mapped netlist to the pinned Docker ORFS physical flow:

```sh
uv run --extra hf python synth/run_native_orfs.py --snapshot final-dram-kv \
  --variant v1 --skip-adder-extraction --single-pass-abc --physical
# Repeat with --variant v2. Reuse a verified mapped netlist for a physical retry:
uv run --extra hf python synth/run_native_orfs.py --snapshot final-dram-kv \
  --variant v1 --skip-adder-extraction --single-pass-abc --reuse-mapped --physical
```

The initial asset capture needs a running synthesis container from the pinned
image (`--container NAME`). Later runs use the captured platform/scripts and
environment. Manifests under `build/native-orfs/<snapshot>/` record immutable
input, library, script, executable, environment and output hashes. Physical
results go under `build/orfs/<snapshot>-native-no-fa-single-pass/` when using
both options above. This is a mixed native synthesis / Docker physical
toolchain, recorded explicitly; it is not an all-Docker run.

`--skip-adder-extraction` disables the optional recursive full-adder recognition
pass; normal gate mapping remains enabled. `--single-pass-abc` copies the ORFS
synthesis driver and changes only its ABC script selection, while also saving
a pre-ABC RTLIL checkpoint. It omits SAT choice sweeping and five repeated LUT
rewrite/remap sequences, retaining standard-cell mapping, buffering, sizing,
and the surrounding Yosys synthesis and connection checks. Use identical
options for comparisons. These checks are not formal equivalence verification.
The default wrapper uses unchanged ORFS synthesis scripts. Native synthesis
completion alone is not physical completion: `--physical` requires final
routed artifacts and writes `ppa.json` before reporting `PHYSICAL_FINISHED`.

For a declared diagnostic optimization budget, add
`--repair-max-iterations 500`. Supported ORFS pre-stage hooks cap each timing
repair invocation at 500 iterations while preserving the original clock and
all timing checks. The profile and hook hash are recorded in the physical run
and PPA JSON. This can bound runtime when a structural timing violation has
plateaued; it does not establish timing closure or represent the default
unbounded repair flow. Use the same cap for both variants in a comparison.
