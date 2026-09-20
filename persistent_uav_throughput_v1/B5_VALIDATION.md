# B5 reserve-SJF validation

Completed 2026-09-20 at 20:59 Asia/Shanghai: 270/270 unique jobs; all 27 regimes
fail empirical eligibility. [Final results and raw evidence](evidence/b5_validation_complete_20260920/README.md)
include the initial descriptive failure audit. No subsequent stage was launched.

User-authorized scope after review of c158cc07: 27 frozen regimes × 10 original
validation seeds = 270 runs, only `reserve_sjf`. No B0–B3 sweep, evaluation,
neural training, MPC/Oracle, or 98% kill test. B4's negative finding applies only
to its four declared candidates, not all possible SOC thresholds.

## Unchanged policy and experiment

The scheduler selects the predicted shortest task among candidates satisfying
`predicted_task_energy + predicted_return_energy < current_battery`. No margin
is added and no calibration is rerun. If none is feasible it recharges when legal;
at a full station it retains the existing minimum-predicted-total-energy fallback.
The existing empty-queue 0.25 SOC rule is unchanged. The job field `threshold=0.25`
records that existing idle rule, not a newly tuned reserve margin.

The frozen SAC, collision protocol, LiDAR observations, finite battery dynamics,
navigation timeout, workload generation, horizon and all 27 regimes are unchanged.
Regime feasibility is unresolved. A failed B5 cannot by itself establish a need
for RL/planning; an apparently successful B5 is not a statistical safety certificate.

## Added passive records

Every B5 decision event carries `reserve_prediction`: starting battery, all
candidate task energies/times, hypothetical task-endpoint return energies, strict
feasibility flags and signed reserve margins. The selected candidate is explicit;
recharge also records the predicted direct return from the actual current position.
For recharge/idle there is no selected task. Predictions use only the public
observation and unchanged frozen estimator.

Failure events and run summaries label `task`, `return`, `waiting` or `charging`.
Every completed `run_*.json` additionally has `decision_diagnostics`, linking each
decision to its leg outcome, measured leg energy when available, and explicitly
separate final-run depletion/navigation/recharge flags. A later failure is not
assigned to an earlier completed leg. A predicted task-endpoint return is
hypothetical unless recharge is actually chosen next; it is not an oracle label.
Cutoff-censored legs do not receive an invented completed energy measurement.

Telemetry enriches existing events after the original environment step; no
scheduler input or action is changed. Tests compare physical summaries, original
events and midflight recovery with telemetry enabled. Archived B4 results and
their original source contracts remain untouched.

## Execution and handoff

`scripts/b5_validation.py` supports explicitly invoked `launch`, `check`, and
`collect`. Launch uses disjoint regime-modulo CPU shards and exact 270-job coverage;
each worker uses the existing locked, atomic checkpoint runner. The bounded
`check` validates one first checkpoint per worker and returns. There is no
automatic collector, evaluation or training stage.

Host records live under `/mnt/workspace/zjl-exp/migration_20260920/b5_validation`.
The B5 compatibility record must verify unchanged calibration/model/native runtime
and allow only the added diagnostics module and evaluator telemetry changes.
Prior B4 compatibility records must continue to reject the new source revision.

Manual collection requires every worker complete, exact job coverage, unchanged
manifests, matching checkpoint/results, and decision telemetry. It emits all four
required outcome metrics via the standard per-regime summary: completed tasks,
depletion, navigation failure and recharge failure. Empirical eligibility uses
the unchanged 5% budget (thus 0/10 observed depletions in each regime).

Directory consolidation is complete: `zjl-exp/repo` is now a real directory and
the old `persistent-uav` root is absent. Historical path references in the B4
evidence describe its execution history, not current launch commands.
