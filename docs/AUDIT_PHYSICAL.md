# Physical/tooling audit

Audit scope: `synth/**`, `scripts/collect_ppa.py`, exported core-netlist handoff,
full RTL static elaboration, and the physical evidence produced by ORFS.

## Confirmed defects fixed

- Yosys `.ys` scripts do not expand `$FUSION` or `$OUT`: added a validated
  template runner, used by the physical flow for both fusion variants.
- The prior Verilog export requested internal-cell output with `-noexpr`.
  Removed that option and explicitly lower `$bwmux`, which otherwise remains
  unresolved in the downstream Verilog frontend. Export now undergoes a
  fresh Verilog parse, hierarchy check, process lowering, and `check -assert`.
- The Docker `make | tail` pipeline could report success after failed synthesis.
  The inner shell now uses `pipefail`, propagates environment-setup failures,
  records the complete console log, and restricts valid stages.
- Physical runs now use a pinned installed ORFS image digest by default, three
  worker cores, and optionally isolated immutable snapshot inputs/results.

## Completed static checks

- Slang full `llaccel_top`: zero errors and warnings.
- Verilator full `llaccel_top`, `--lint-only -Wall`: passes with repository waivers.
- Yosys `llaccel_core` elaboration, connection check and exported-Verilog
  readback pass for fusion 0 and 1.
- Word-level cell counts after optimization: v1 6,755; v2 7,435. These counts
  contain arithmetic and memory cells and are **not mapped gate counts or area**.
- Docker test against the installed image confirmed setup succeeds and a failed
  command piped to `tail` returns status 1 under the repaired shell settings.
- Optional local gate mapping was stopped during SAT sharing; no mapped gate
  count is claimed from that incomplete run.

## Physical run boundary and assumptions

Baseline netlists, the 3 ns SDC, elaboration/readback logs and SHA256 manifest
are frozen under `build/synth/baseline`. Baseline physical reports are isolated
under `build/orfs/baseline`. A later RTL revision requires its own snapshot and
completed reports before it can replace this baseline's results.

The boundary is `llaccel_core`: external 1 MiB SRAM banks are excluded. Nangate45
uses its typical standard-cell library. The 3 ns clock and 0.6 ns I/O delays
are constraints, not an achieved frequency claim. Physical metrics require
completed route/extraction reports; timing violations must be retained in the
report rather than treated as successful timing closure.

Without a VCD/SAIF annotation, OpenSTA power is vectorless. Installed OpenSTA
help documents default input activity 0.1 transitions per minimum clock period
and duty 0.5. Such power is an estimate under those assumptions, not measured
LLM workload or silicon power. External SRAM power is excluded too.

## In progress

Final DMA32 v1 ORFS through `finish` is blocked by memory exhaustion at detailed-route track assignment. Updated-design physical evaluation remains pending; no completed PPA values are claimed until its reports exist.

## Additional flow findings

ORFS SAT sharing was prohibitively slow for the scheduled datapaths; configurations now set `SYNTH_ARGS=-noshare`. The FIFO register mapping limit was too small for the existing 16x512-bit queue, and is explicitly 16384 bits to permit the tested 32-entry alternative. This does not add an external SRAM macro.

Final DMA32 netlist snapshots are distinct from the old baseline. The old baseline mapping run encountered Docker emulation/daemon issues, and its incomplete logs are retained. Native width inspection found that redundant multiplier sign extensions are reduced correctly; there is no evidence justifying arbitrary narrowing of arithmetic intermediates. Completed physical finish artifacts remain required before quoting PPA.

### Physical recovery and current limits

Both final variants passed mapped-netlist connection checks. Mapped cell area is 1,824,750.158 µm² for v1 and 2,047,287.886 µm² for v2. Routing and final STA remain incomplete. The v1 placement checkpoint is preserved. Its initial CTS attempt failed in the optional Kepler equivalence child with an illegal instruction; resumption explicitly sets `LEC_CHECK=0`. Formal equivalence is not established. Setup repair is capped at 500 iterations; hold repair retains the tool’s native bounds. The 3 ns constraint is unchanged and is not an achieved clock claim.

Completed ODB checkpoints were transparently compressed on APFS to recover disk space. SHA-256 checks prove logical content unchanged, and original mtimes are retained for Make dependencies. Compression records are in `work/physical-transition/apfs-compression.json`.

The CTS vectorless `report_power` calculation later exceeded ten minutes without output. A separate debugger could not inspect Rosetta registers. The preserved run is in `work/physical-transition/cts-power-timeout/`. The explicit `--skip-vectorless-power` reporting profile keeps timing, electrical checks and area, omits power estimates, and saves CTS ODB/SDC before metrics. The original and transformed ORFS metrics script hashes are recorded. This changes reporting only; power is unmeasured for this profile. Apply the same profile to both variants.

After the numerical campaigns completed, routing was restarted from the saved v1 CTS checkpoint with `--physical-cores 8` to use available Docker CPU capacity. V1 placement/CTS were created with three threads; current routing uses eight. The profile explicitly scopes the thread count to the current invocation, and per-stage logs retain earlier settings. Thread count does not change the frozen RTL or SDC.

The eight-thread v1 global route completed 861,530 pin-access groups and routed 1,503,072 nets with zero congestion overflow, then was killed for memory use in post-route `repair_design`. Docker was increased from 12 GiB RAM/2 GiB swap to 14 GiB RAM/4 GiB swap. `--checkpoint-global-route` now saves the database, SDC and raw route segments before repair, with a completion marker written only after all saves and a source/profile key recorded in the manifest. This checkpoint is not a final routed result.

The post-route optimization attempt later slowed to roughly 1,000 entries over several minutes with about 184,000 entries left. The final diagnostic profile explicitly sets `SKIP_INCREMENTAL_REPAIR=1 RECOVER_POWER=0` (`--skip-post-route-repair`); placement and CTS repairs remain, while final timing, electrical, area and detailed-route checks are retained. This is not a timing-closed or fully optimized PPA claim. The route-resume path verifies checkpoint and upstream hashes before loading raw route segments. Earlier partial optimization edits are discarded by resuming the saved pre-repair checkpoint.

The resumed v1 global route completed and saved `5_1_grt.odb`: 2,083,809 µm² core cell area, −4.13 ns setup slack at the unchanged 3 ns constraint, and a 7.13 ns estimated minimum period. These are global-route estimates. Detailed routing then exhausted memory at track assignment with eight threads. The retry uses `--detail-route-cores 2 --reuse-pin-access`, loading the completed global-route database and its persisted pin-access data. The original detailed-route connectivity, DRC and antenna checks remain enabled. The two-thread retry also failed with a Docker-confirmed OOM kill at track assignment (2026-09-11 00:48 UTC). Final detailed-route and extracted timing results are blocked on resource capacity. Failure evidence and the exact resume command are in `results/dram-kv-2026-09-10/physical-status.json`.
