# Oracle Recoverability Census startup — 2026-09-20

Started 2026-09-20T22:45:05 Asia/Shanghai. Source revision: `2da2125`.
The immutable plan contains 205 historical failure-predecessor states and 561
bounded branches: A=67/182, B=48/48, C=90/331 (states/branches). Sixteen
single-thread CPU workers own disjoint round-robin subsets. No NPU/GPU use.

This is a privileged diagnostic census, not a new full policy, MPC, training,
benchmark retuning, or 98% kill test. See ../../ORACLE_RECOVERABILITY_CENSUS.md.

Nine focused tests passed. The one real engineering smoke reconstructed historical
state r00_s920260927_e00009 at simulation time 132.3 after 526 replay policy steps,
with all nine prefix events and the complete decision observation exactly matching.
Two bounded branch steps verified full-state clone isolation and midflight disk
resume equality. This fixture contributes no census outcome.

All 16 first resumable checkpoints passed checksum/deserialization, exact historical
prefix comparison, worker contract/plan identity, finite observation and actual
single-thread checks. Some checkpoints remain in replay; startup health does not
claim that all 205 roots or all 561 branch outcomes have already been validated.
Every root must pass exact complete-prefix/observation comparison before any branch.
A replay mismatch stops that worker rather than becoming an unsafe label.

Runtime source/model/dependency provenance and native binary hashes match B5.
Only the separate diagnostic script/tests/docs were added. Frozen estimator,
physics, action legality, original horizon and navigation timeout are unchanged.
No physical reset is inserted in a branch. Direct return stops at charger arrival,
not after charging; horizon censoring is unknown, navigation failure distinct from
energy depletion. No successful tested continuation is not a maximal-viability or
regime-infeasibility proof.

Output: `/mnt/workspace/zjl-exp/repo/persistent_uav_throughput_v1/artifacts/oracle_recoverability_census_20260920`.

`launch.json` stores PIDs and exact commands. Each worker saves atomic checkpoints
every 100 replay/branch policy steps or 60 seconds, plus branch-root snapshots and
completed branch traces. To pause, first match the live PID command to launch.json,
then SIGTERM; resume only verified stopped workers with its corresponding line in
resume_commands.txt. Never start a second writer in an active worker directory.
The error record preserves the last valid checkpoint rather than accepting a
mismatching reconstructed state.

Manual collection, only after all workers complete:

```bash
/mnt/workspace/zjl-exp/.venv/bin/python /mnt/workspace/zjl-exp/repo/persistent_uav_throughput_v1/scripts/oracle_recoverability_census.py collect --output /mnt/workspace/zjl-exp/repo/persistent_uav_throughput_v1/artifacts/oracle_recoverability_census_20260920
```

Collection verifies all state IDs, branch specifications, exact reconstructed roots,
checkpoint/results correspondence and branch-trace hashes. It reports A/C safe-task
existence and alternatives, legal direct-return outcomes, and the B immediate-return
test, retaining timeout/depletion/censoring distinctions. No next policy starts.

This directory preserves immutable plan/manifests, startup health, tests and the
engineering smoke checkpoint. It does not copy changing production checkpoints or
claim completion. Handed back running after first-checkpoint health; no automatic
collector, completion monitoring, or promotion to full OracleSafe-SJF.
