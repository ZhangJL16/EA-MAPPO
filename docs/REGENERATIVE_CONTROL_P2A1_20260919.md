# P2-A1: reachable D-state census and exact task branching


**Result: no systematic comparable-state sign reversal demonstrated. Stop here;
no full oracle or learning promotion.** The bounded census yielded 885 natural
D observations but only 65 distinct physical keys, below the 200-state target.
All 260 core branches completed. Exact D-successor closure is only 62/128
(48.44%); no tested abstraction has sufficient support. Navigation confounding
also remains. This is an incomplete hypothesis test, not evidence that the
simple reserve rule is optimal.

Scope: user-approved P2-A1 only. P1/P1.1 passed review. No SMDP solver,
policy learner, new critic, task process, map sweep or dynamics modification.
This document supersedes earlier P2-hold wording only for this bounded census.

## Frozen construction

- One UAV, fixed map seed `319190002`, 24 obstacles, capacity 60; unchanged
  P1.1 post-delivery decisions and paid docking/recharge mechanics.
- Frozen SAC SHA256:
  `fb79482ae7d832b40137ff1a080ee84912eabb95867e118aca80c846f6ae62be`.
  Actor weights only; no optimizer loaded, no training updates.
- Six occupancy collectors: after-1, after-2, SOC thresholds 20/30/40,
  aggressive always-C. These are state collectors, not policy-value benchmarks.
- Predeclared budget: 20 rounds × six trajectories, up to 12 D observations
  or 1,500 simulator seconds per trajectory, stopping also on actual failure.
  Stream seed is `819190001 + 6*round + policy_index`; no favorable-seed search.
- Select first 200 distinct physical keys in fixed round/policy/observation
  order, or stop at budget exhaustion. Retain every source failure and repeated
  occupancy occurrence. No battery grid or interventions in occupancy.
- Key fields: exact position, velocity, energy, previous/last position,
  both contact-memory flags, current goal, phase and map hash. A key is not a
  proof of complete Markov sufficiency. Full resumable simulator snapshots,
  including replay RNG, remain in the local artifact tree with SHA256 receipts.
  Logging-only trajectory arrays are removed from snapshot copies.
- For each selected state, independently restore the same original snapshot
  for C100, C300, C600 and R. C probabilities are exactly 1/3. The task RNG is
  replaced with an object that raises on access; branch observations/outcomes
  exclude RNG state. Full map is audit information, never an actor input.
- Evaluation cutoff is extended to allow each branch to finish; physical
  clocks/energy/pose are not reset. The frozen 4,000-policy-step navigation
  limit remains in force. Failure is absorbing, with no rescue.

## Complete mission requirements and local comparison

After each successful C branch, execute R from its *actual* successor.
This measures joint task→return consumption directly, not summed quantiles.
If this sequence fails at the actual battery, retain that failure and run a
separately labelled diagnostic with energy **and capacity** set to 10,000.
This diagnostic estimates otherwise censored flight/service energy; it is
not natural occupancy and cannot turn an actual failure into a success.
A diagnostic navigation timeout leaves the full mission requirement undefined.
Diagnostic charging times are not used in the local comparison.

The initial diagnostic implementation raised energy without capacity, which
could yield negative service time. This was caught before the full branch run.
V1 source/manifest and its one successful, non-intervened smoke state are retained.
V2 changes only this diagnostic's capacity; natural collector receipts are reused
unchanged with explicit cross-version provenance. A regression test checks paid,
positive diagnostic service time. Frozen environment/actor files were not edited.

Let `rho_1` be the explicitly enumerated canonical-H after-1 renewal rate:

```
rho_1 = 1 / mean_d(time[H → task_d → H])
Delta_local(s) = 1 - rho_1 * (mean_d(time[s → task_d → H]) - time[s → H])
```

This compares two bounded continuations ending at the same H, under a **fixed
reference opportunity cost**. It is neither an optimal bias estimate nor Q*.
Delta is defined only when direct R and all three task→R sequences succeed at
original energy. Undefined contrasts remain in raw data and are counted in all
figures; no invented failure penalty or counterfactual completion is substituted.
The reference rate is not oracle throughput.

For finite complete energy requirements, save their mean and maximum and margins
`e - mean(E_mission)` and `e - max(E_mission)`. No q95 is estimated.
The auxiliary viability-first local label chooses R if R succeeds and any mission
fails; otherwise it uses the local contrast sign. Threshold fit to that label
partly restates feasibility, and must not be interpreted as a scientific kill.
All fits are descriptive in-sample fits, with constant-label accuracy reported.
No long-run policy comparison, holdout accuracy or independent-sample CI is claimed.

## Abstraction and closure protocol

Predeclared bins: energy 2, home distance 10, speed 1, position coordinates 10,
velocity coordinates 1. Test `e`, `(e,d_H)`, `(e,d_H,|v|)`, `(e,x,v)`.
For each bin containing at least three distinct physical keys, report branch
energy/time ranges and variances, mixed successes, and successor coordinate /
velocity / energy spread. Cost stability requires energy range ≤1 and time range
≤5 with no success mixture. At least 50% state coverage and 20 tested bins are
required before declaring adequate support. Passing costs alone would still not
prove lumpability of next-state transitions. Sparse bins cannot establish sufficiency.

Exact D-successor closure uses physical-key equality, never nearest neighbors.
Successful R branches enter canonical H; H forced-task branches are separately
enumerated. H support reported by pose/velocity/energy is explicitly a partial
check, not exact full-state closure. Failure branches are separate absorbing outcomes.

Comparable pairs are unordered, from distinct trajectories, with `|Δe|<2`,
`|Δmargin|<1`, `|Δd_H|<10`, excluding local ties ≤1e-6. A secondary count also
requires vector-velocity distance <0.5. Pairs share states and are not IID.
The mean- and max-margin matching rules are frozen separately.

## Reproducibility

Runtime: `artifacts/regenerative_p2a1_20260919_v2`; original collector and V1
receipts: `artifacts/regenerative_p2a1_20260919`. V2 `sources` points to the
unchanged original natural-trajectory store. Completed branch rows are atomic,
hash-checked and skipped on resume; source trajectories checkpoint independently.
Never mix a changed runtime source with an existing contract.

```
PYTHONPATH=. .venv/bin/python research/regenerative_control/census_p2a1.py \
  --output artifacts/regenerative_p2a1_20260919_v2 --phase branch
PYTHONPATH=. .venv/bin/python research/regenerative_control/audit_p2a1.py \
  --input artifacts/regenerative_p2a1_20260919_v2
PYTHONPATH=. .venv/bin/python research/regenerative_control/analyze_p2a1.py \
  --input artifacts/regenerative_p2a1_20260919_v2 \
  --output artifacts/regenerative_p2a1_20260919_v2/analysis
```

The old collector contract pins its archived V1 source; running the current V2
collector against that old directory intentionally refuses a contract mismatch.
A new independent collection would require a new output directory. No extra
collection is authorized or automatically started by the analysis commands.

Portable evidence is in `research/regenerative_control/evidence/p2a1_20260919`:
raw branch envelopes, raw source-trajectory receipts, census and checkpoint/source
hashes, fixed after-1 reference, audit, all state values, abstraction bins, and
three editable SVG/PDF figures with PNG previews. Local pickle snapshots are not
committed as portable data; their hashes and full state outcome records are.

## A. Actual reachable-state census

- 120 trajectories, 885 natural D observations, **65 distinct physical keys**.
  Target 200 not reached at the frozen 20-round budget; no synthetic fill-in.
- Battery range: **1.7028–56.0206** (capacity 60).
- Position coordinate ranges: x 2276.672–2777.908, y 2095.808–2596.868,
  z 199.053–202.292. Speed: **9.359–14.394**.
- Velocity component ranges: vx 5.087–10.700, vy 5.822–10.422,
  vz −5.000 to −1.334. Residual motion is substantial and remains in state.
- All six source policies represented. First-occurrence dedup provenance counts:
  after-1 3, after-2 5, SOC20 18, SOC30 14, SOC40 7, aggressive 18.
  These counts depend on deterministic selection order, not policy occupancy value.
- Source failures: after-1 0/20, after-2 10/20, SOC20 20/20,
  SOC30 13/20, SOC40 7/20, aggressive 20/20. These deliberate boundary-collection
  runs are not fair policy-performance or reliability comparisons.

## B. Exact four-branch completeness

| Branch | Attempted | Success | Failure | Censored | Contact |
|---|---:|---:|---:|---:|---:|
| C100 | 65 | 54 | 11 | 0 | 0 |
| C300 | 65 | 46 | 19 | 0 | 0 |
| C600 | 65 | 28 | 37 | 0 | 0 |
| R | 65 | 54 | 11 | 0 | 0 |

C success here means task delivery, **not** guaranteed subsequent return.
All 78 original-energy core failures end in depletion. That terminal reason
alone does not establish a pure resource limitation: of 98 separately labelled
failed-mission diagnostics, 85 complete with increased energy/capacity and
**13 still reach navigation timeout**. Those 13 are not converted into finite
mission costs. They cover 13 of the 195 task-conditioned mission branches;
no claim of independent failure probability is made.

Raw invariant audit passes for 65 snapshots and 260 branches: initial energy/pose/
velocity match source snapshots, no decision task, exact draw counts, canonical
successful R, source/model hashes unchanged, no branch RNG feature/access,
no actual-energy interventions. Frozen depletion logging can overshoot the
clipped battery by one integration step: maximum recorded discrepancy 0.013651
energy units. This is disclosed, not repaired by changing frozen physics.

## C. Closure and state abstraction

Successful C branches yield 128 D successors; **62 are in exact selected support**
(48.4375%). Successful R branches yield 54 canonical H states. All three forced
H task outcomes match selected pose/velocity/energy, which is a partial check.
No nearest-neighbor mapping or closed SMDP is constructed.

| Abstraction | Tested bins | States covered | Stable states in tested bins |
|---|---:|---:|---:|
| e | 12 | 47/65 | 0/47 |
| e + home distance | 4 | 13/65 | 3/13 |
| e + home distance + speed | 3 | 9/65 | 6/9 |
| e + full position + velocity | 3 | 9/65 | 6/9 |

Battery-only bins show substantial outcome mixture (8 of 12 tested bins have
mixed successes). Adding geometry/velocity reduces observed spread in some bins,
but finer bins become sparse. **No level is established as transition-sufficient.**
Do not treat the last row as a Markov-state PASS or within-bin spread as conditional
physical noise. Per-bin successor coordinate/velocity/energy spreads are exported.

## D. Local contrast and three figures

Fixed after-1 reference: `rho_1 = 0.00881209518` task value per simulator second.
Only **15/65 states** have all four cycle alternatives viable and hence finite
local Delta. The other 50 remain visible as explicit exclusions from these plots.
There are 14 positive contrasts and one negative contrast; range
**−0.00191359 to +0.47037923**. The single slightly negative state is not itself
systematic sign-reversal evidence and is not an optimal-action label.

- [Battery versus local Delta](../research/regenerative_control/evidence/p2a1_20260919/energy_vs_local_delta.svg)
- [Mean reserve versus local Delta](../research/regenerative_control/evidence/p2a1_20260919/m_mean_vs_local_delta.svg)
- [Max reserve versus local Delta](../research/regenerative_control/evidence/p2a1_20260919/m_max_vs_local_delta.svg)

On 43 states with finite mission requirements and a defined viability-first local
label, the best in-sample battery/mean/max thresholds fit 41/43, 41/43, **42/43
(97.67%)**, respectively. The max-reserve zero threshold also fits 42/43. This is
mostly a feasibility-label comparison, **not** a throughput ratio. On the 15
common-viable states, all three best threshold fits are 14/15, identical to
always-C. No `rho_simple/rho_star` was computed; the ≥0.98 safety-matched
long-run kill criterion cannot be evaluated here.

## E. Comparable sign reversals and decision

| Matching rule | Comparable pairs | Opposite-sign pairs | With velocity matched | Flips after velocity match |
|---|---:|---:|---:|---:|
| battery + home distance + mean margin | 4 | 0 | 3 | 0 |
| battery + home distance + max margin | 5 | 0 | 3 | 0 |

**在控制 battery 和简单 mission reserve 后，本轮未发现系统性 C/R sign reversal；
但只有 4–5 对可比状态，不能据此证明简单 reserve 足够。**

No mechanism claim is warranted. The missing independent state coverage,
nonclosed support, sparse abstraction checks, and residual navigation timeouts
block escalation. Stop at this report, preserve the fixed task distribution and
98% criterion, and leave any further diagnostic/sampling decision to user review.
No full P2 solver, neural network training, task-family expansion or hidden rescue.

## Verification and figure QA

Five focused tests pass: explicit RNG-free enumeration, RNG-change/log-compaction
branch parity, isolated intervention, physical snapshot roundtrip, and positive
paid diagnostic recharge. Raw provenance/invariant audit passes. Original frozen
navigation, P1 and P1.1 source files were not modified.

Python/matplotlib figures use all 15 defined contrasts, with counts of 50 undefined
states stated on-panel; no jitter, selective point removal, CI or significance
claim. Shape and color both encode preceding task type. Editable SVG/PDF and
300-dpi PNG previews are included. Source preflight: no failures; TIFF/600-dpi
warnings are accepted because these are vector diagnostic plots with preview PNGs,
not a journal raster submission. Rendered figures are visually inspected for
labels, overlap and zero-line visibility. Source values are in `state_values.json`.
