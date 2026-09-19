# P2-A2 safe-set regenerative optimal-stopping kill test — startup

**P2-A1 passed user review. P2-A2 is implemented and running; no kill result is
available in this startup handoff.** At the startup-health observation, 13 nodes
were checkpointed and 12 remained on the frontier. No further progress monitoring
is performed in this turn, following the repository's fixed startup workflow.

## Authorized question and frozen decision

Compare one baseline, viability-boundary (VB), against the optimal renewal ratio
on the same robust-safe exact task-sequence tree. Kill the current main story if
`rho_VB / rho_star >= 0.98`. A lower ratio is not automatic permission to train:
it still needs multiple interpretable safe-but-return-optimal states and evidence
that the gap is not driven by one pathological prefix. Task distribution, map,
capacity, collision mechanism and frozen SAC remain unchanged.

This is an exact solution of the **enumerated frozen deterministic task-sequence
model under the declared immediate-return robust safe set**, not an unrestricted
continuous-control oracle or a result about all persistent-control problems.

## Safe set and exact tree

- One fixed map, layout seed `319190002`, capacity 60, task value 1,
  task types 100/300/600 with probabilities exactly 1/3.
- Begin at an actual paid recharge completion H. A recorded setup task100→R
  supplies canonical H; this setup prefix is excluded from cycle accounting.
  No manual reset is used to fabricate a recharged state.
- H forces its first IID task; all three outcomes are explicitly enumerated.
  Every D node is a full simulator snapshot following a specific task prefix.
- R is admissible only if it succeeds and reaches the same canonical physical H.
  Failure of the forced-H task/return prerequisite stops with an error, not an
  invented recovery transition.
- C is admissible only if all three original-energy task→immediate-R sequences
  succeed. Task delivery success alone is insufficient. Timeout or depletion
  makes the corresponding C illegal.
- Each successful task consumes strictly positive battery in the recorded
  transition. Expand every successor of legal C; illegal C needs no descendant
  value because it is unavailable to **both** policies.
- No state bins, nearest-neighbor mapping or physical-key merging. Different
  prefixes retain separate snapshots even if their measured physics agrees.
- Task RNG is replaced with the P2-A1 guard that raises on access. Branch records
  contain no RNG state. Full map is audit data only; frozen actor input is unchanged.
- A 10,000-node work limit is an operational protection, not a terminal state.
  An unfinished frontier yields INCOMPLETE and no ratio. No artificial depth
  cutoff converts unexpanded safe states into R leaves.

Safe-set admissibility always uses original battery/capacity. Failed missions
with observed depletion receive a separately labelled energy/capacity10000
counterfactual, solely to distinguish resource limitation from navigation timeout.
`energy` means depletion with successful high-energy mission; `navigation` means
observed timeout without depletion; `both` means observed depletion plus a
high-energy navigation timeout. These are operational diagnostic categories,
not claims of independent causal effects. Diagnostic states/costs never enter DP.
Unknown or unfinished diagnostics fail explicitly. SAC is not modified.

## Renewal ratio solution

For a candidate rate rho, at D:

```
F_R(s) = -rho * time_R(s)
F_C(s) = mean_d[1 - rho * time_task_d(s) + F(successor_d)]
F(s)   = max over admissible R/C
```

The three forced H tasks are included in reward and time. Dinkelbach iterations
solve the root excess equation. Each iteration carries expected task count and
expected duration separately: the objective is their ratio, never the mean of
branch reward/time ratios. A root bracket and excess residual accompany the
answer. If the numerical ratio interval straddles .98, the decision is unresolved.
R/C ties do not count as strict safe-but-return-optimal findings (strict Delta
threshold −1e-9). VB always takes C when legal and otherwise R; both policies
have zero failed branches in their represented admissible cycle trees.

The solver refuses missing legal successors. It runs only after the whole
canonical-H tree and the nominated-state diagnostic tree are closed.
No after-1 rate or P2-A1 feasibility-derived labels are used as the oracle.

## Nominated early-return state

Original ID:
`e5352ba4f4a9a467a1ba50cbf4b747ba522fdf1cacad4be12d036f02d62d300a`.
Original battery 46.7180678, max margin 4.3572116, local contrast −0.00191359.
Its original hash-checked snapshot starts a separate exact subtree. It has **zero
weight** in canonical-H renewal optimization; its C/R advantage is evaluated at
the resulting rho-star. This prevents optimizing the experiment around the anomaly.
No candidate duplicate is counted as extra population sign-reversal evidence.

## Outputs when the run finishes

`result.json` includes rho-star, rho-VB, ratio/bracket, expected tasks/time per
cycle, every node's optimal action and advantage, VB/optimal visitation masses,
candidate decision, and infeasibility cause counts. No further experiment launches.

The read-only report command below then computes:

- strict safe-but-return-optimal cycle-prefix count and distinct physical count;
- probability of early return under the optimum;
- each prefix's contribution to VB excess loss and the largest single-prefix share;
- an independent check that the summed VB-weighted advantage loss equals
  `rho_star * E[T_VB] - E[N_VB]`.

This separates a substantial repeated phenomenon from one special state. Physical
key deduplication is only for reporting distinct-state counts, never for solving.
A ratio below .98 is reported as a gap requiring review, not a Spotlight claim.

## Startup verification

Seven focused tests pass: exhaustive-policy agreement on a finite toy tree;
VB-optimal kill case; tie handling; missing-successor rejection; complete robust
safe-set check; checkpoint reuse/corruption rejection; and independently checked
loss decomposition with candidate duplicates excluded.

A one-node real frozen-SAC execution smoke wrote an atomic checkpoint. The formal
run reused it; a newly completed node and its successor snapshot hashes were
checked. Source/checkpoint hashes, uniform task weights, canonical H, task-free
decisions and positive task energy consumption passed startup checks.

Frozen checkpoint SHA256:
`fb79482ae7d832b40137ff1a080ee84912eabb95867e118aca80c846f6ae62be`.
No optimizer, no new training, no random census, no environment modification.

The first background launcher mistakenly resolved the virtualenv interpreter
symlink to the system interpreter and failed importing torch before any simulation.
It was corrected without changing the experiment or its checkpoint; the failed
launch receipt and original log entry are retained.

## Run and handoff

Runtime directory: `artifacts/regenerative_p2a2_20260919`.
Formal process PID at launch: **370798**, four CPU workers. This is a startup
snapshot, not a claim that the PID will remain alive after completion.

- `startup_health.json`: immutable startup observation (13 nodes, frontier12).
- `progress.json`: atomic live progress; `status.json`: RUNNING/INCOMPLETE/ERROR/COMPLETE.
- `nodes/`: hashed node receipts; `snapshots/`: full simulator state checkpoints.
- `root.json`: canonical-H setup, forced branches and nominated-state provenance.
- `run.log`: execution output including the retained initial launcher failure.
- `result.json`: only written after complete exact-tree solution.

Gracefully stop after the current worker batch:

```bash
touch artifacts/regenerative_p2a2_20260919/STOP
```

Resume after the process has exited, from this unchanged source checkout:

```bash
rm -f artifacts/regenerative_p2a2_20260919/STOP
PYTHONPATH=. .venv/bin/python -m research.regenerative_control.stopping_p2a2 \
  --output artifacts/regenerative_p2a2_20260919 --workers 4
```

An exclusive lock rejects concurrent writers. Source or checkpoint mismatch
refuses resume; existing node/snapshot hashes are validated before reuse.
After completion, summarize without launching or waiting for any simulation:

```bash
PYTHONPATH=. .venv/bin/python -m research.regenerative_control.report_p2a2 \
  --input artifacts/regenerative_p2a2_20260919
```

Per [AGENTS.md](../AGENTS.md): “After startup health is confirmed, hand the
resumable run back to the user; do not monitor it to completion or auto-promote it.”
Accordingly this turn ends at startup handoff. The run itself may finish the
explicitly authorized P2-A2 tree/ratio calculation; it cannot launch learning,
new task processes, uncertainty, or another research phase.

Portable startup evidence:
`research/regenerative_control/evidence/p2a2_startup_20260919`.
These are startup receipts, **not** completed kill-test results.
