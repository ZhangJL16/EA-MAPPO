# Measurement-bundling calibration at T=4096

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent; nature-figure
- Origin Mode: run + descriptive analysis
- Origin Date: 2026-09-16
- Verification Status: ANALYZED — one paired seed, not statistically verified
- Version: bundling-calibration-result-4096-v1
- Authority: user-approved continuation to first scientific horizon only

## Outcome

**The requested learner-direction check did not succeed at this budget.**
The resource-path learner has P_high,on(4096)=4.30 versus P_low,on(4096)=4.10.
The capacity-only negative control is exact: P_high,off=P_low,off=4.10, with
identical observable continuations except battery. No routing library, additional
seeds, horizon changes, hypothesis changes or task-throughput analysis followed.

This is a single realized learner trajectory's pseudo-regret, not an estimate
of expected regret across seeds and not a counterexample to the asymptotic theorem.
There is currently **no finite-budget support** for the primary learning benefit
in this calibration. Nothing here justifies a general capacity-benefit claim.

## Execution and integrity

Executed in `/home/zjl/mappo`:

```bash
.venv/bin/python scripts/run_bundling_calibration.py resume --output artifacts/bundling_calibration_startup_v1_20260916
```

Exit code 0; internal grid execution about 0.368 seconds. All 20 checkpoints
contain exactly 4096 primitive steps; the original continuing seed is 20260916.
Source/prediction hashes still match the pre-sampling manifest. The analysis
replayed all 81,920 primitive actions, checking position/state/time/reload,
recomputed pseudo-regret from conditional means, checked every completed path
cost and the incomplete-path remainder, and revalidated the stored 128-step
checkpoint values against the same trace prefix. No automatic experiment retry.

Analysis and plotting are separate from the hash-pinned learner/runner:

```bash
.venv/bin/python scripts/analyze_bundling_calibration.py --source artifacts/bundling_calibration_startup_v1_20260916
```

No source in the frozen scientific closure was edited. Analysis exports were
subsequently rerendered from the SAME saved curves to make vector exports visible
to source preflight and PNG previews 600 dpi. This did not rerun a learner,
change numerical data, change cell selection or fit any coefficients.

## Primary results

All values below use primary truth theta_A, unchanged optimal gain .1 and span .2.

| Cell | Sealed C | P(4096) | P/log(4096) | P − C log(4096) |
|---|---:|---:|---:|---:|
| low/on | .0943398 | 4.10 | .492921 | 3.315304 |
| high/on | .0188680 | 4.30 | .516966 | 4.143061 |
| low/off | .0943398 | 4.10 | .492921 | 3.315304 |
| high/off | .0943398 | 4.10 | .492921 | 3.315304 |

Observed primary difference-in-differences, defined as low-minus-high benefit,
is **−.20**. The sealed leading coefficient is +.0754718382; multiplying it by
log(4096) gives +.627757. That number is the asymptotic leading term, NOT a
promised finite-budget contrast. No intercept subtraction, rescaling, fitted
slope, CI or p-value was used to conceal this discrepancy.

### Qualitative checks requested by the author

1. Mechanism direction: **not supported at T=4096** for resource-path learning
   or the strongest cost-aware reference. High/on is slightly worse, not better.
2. Capacity-only negative control: **consistent**. High/off and low/off are
   identical throughout, apart from the intended battery field, for all methods.
3. Prediction residual: **material finite-budget remainder remains**. Residuals
   are roughly 3.32 split and 4.14 bundle at the endpoint. They do not establish
   approach to the predicted coefficient. A single finite path cannot determine
   systematic bias in an expectation or refute a T-to-infinity claim.

The curves after the early exploration phase mainly reflect accumulated overhead
divided by log T, with cycle-phase oscillations. A falling P/log T by itself is
not evidence of the predicted limiting value.

## Mechanistic accounting — no learner retuning

For resource-path learning:

| Completed path | Split count | High/on count | Calendar cost per completed path |
|---|---:|---:|---:|
| dock | 8 | 8 | .05 |
| empty | 8 | 8 | .20 |
| A | 1349 | 1338 | 0 |
| B-only | 8 | 8 | .25 |
| AB | unavailable | 8 | .05 |

Costs on complete sorties sum to **4.00** split and **4.40** high/on.
Incomplete-path remainders are **+.10** split and **−.10** high/on, giving the
logged totals **4.10** and **4.30**. Recounting exactly matches evaluator P.

The high/on B-only count has NOT fallen. Initialization/forced coverage adds
AB rather than replacing the expensive B-only queries at this budget. Seven
of the eight AB sorties are forced; none is a correct-model information-LP
quota play in this observed trace. Complete forced-sampling cost is 3.25 split
and 3.85 high/on; initialization costs are .50 and .55 respectively. A single
split B information-quota sortie contributes the remaining .25. A sorties
have zero calendar cost at truth even when selected under a different MLE.

**Acquisition cost must not be confused with observation count.** Potential
opportunity cost per B measurement is .25 on B-only versus .05 on AB. The
actually executed mixture, however, has eight B measurements at cost 2.00 in
split versus sixteen at cost 2.40 in high/on: average B-containing-sortie cost
per B measurement is .25 versus .15, not the LP-optimal .05. Dock/empty overhead
and final cycle phase remain separate. This explains why merely receiving more
measurements did not produce the predicted regret ordering in this run.

## All predeclared methods retained

| Method | P_low,on | P_high,on | P_low,off | P_high,off |
|---|---:|---:|---:|---:|
| resource-path | 4.10 | 4.30 | 4.10 | 4.10 |
| cost-aware reference | 4.10 | 4.30 | 4.10 | 4.10 |
| cost-blind | 4.10 | 4.30 | 4.10 | 4.10 |
| independent duration-aware UCB | 154.80 | 125.90 | 154.80 | 154.80 |
| oracle allocation (privileged) | 1.10 | .20 | 1.10 | 1.10 |

The equal cost-aware reference is expected and cannot be turned into a claimed
algorithmic win. The privileged schedule's ordering shows the intended allocation
can express the cost difference; it does NOT certify unknown-model finite-budget
attainability. The weak independent-UCB ordering cannot replace the failed primary
check or establish the predicted coefficient. No known-model execution/tasks-hour
comparison was analyzed.

## Figure and data QA

Outputs under the run's `analysis_4096` directory include clean curves.csv,
curves/summary/contrasts JSON, original analysis provenance receipt, render-only
receipt, and both primary and all-method SVG/PDF/PNG panels.

- Python-only rendering; nominal 183 mm width; editable SVG/PDF text; 600 dpi PNG.
- All twenty trajectories retained; 81,920 source curve rows; no outcome exclusions.
- P/log T is undefined at T=1: that ONE point per normalized curve is omitted,
  explicitly represented as null in source data. Every valid T>=2 point is used;
  the residual uses every T>=1 point.
- No error bars, smoothing or uncertainty intervals. Dense apparent bands are
  rapidly repeated cycle-phase line oscillations, not shaded confidence regions.
- Static source preflight: 11 PASS, 3 WARN, 0 FAIL. Warnings are documented:
  PNG is a preview rather than TIFF submission raster; width parsing misreads
  the `183/25.4` inch conversion (actual nominal width 183 mm); all logarithm
  inputs are positive integer time or guarded T>1.
- Primary and all-method PNGs visually inspected; axes, legend and labels readable.
- Not a submission-readiness or real-robot deployment certification.

## Statistical/methodological fallacy scan — 11/11

| Check | Disposition |
|---|---|
| Simpson reversal | Methods shown separately; no pooling to hide primary direction |
| Ecological inference | No broader embodied-agent claim from this one fixture |
| Berkson selection | Fixed calibration/seed; no outcome-selected instances |
| Collider adjustment | No fitted post-intervention controls or adjustment |
| Base-rate neglect | No classifier/sensitivity metric used |
| Regression to mean | No seed chosen for extreme pre/post performance |
| Survivorship | All trajectories and incomplete paths retained |
| Look-elsewhere | All cells/methods shown; favorable oracle/UCB not substituted |
| Forking paths | Horizon, seed, hypotheses and learner unchanged after outcomes |
| Correlation/causation | Controlled model intervention only; no deployment generalization |
| Reverse causality | Resource/protocol fixed before observations |

No inferential hypothesis tests were run; checkpoints/time steps are NOT independent
replicates. Confidence in any broad claim remains CAUTION. An independent proof
audit and seed-distribution evidence are not supplied by this descriptive run.

## Stop boundary

Execution is stopped at the approved first scientific horizon. Do not enter a
descriptor library on this result. Do not extend the horizon, alter the confidence
rule, eliminate forced paths, change seeds or change means to obtain a positive
ordering. Any further research action needs author direction; the valid negative
calibration remains the result of the frozen experiment.
