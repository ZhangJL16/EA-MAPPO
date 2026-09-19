# FPL second batch: task families, independent planning and scaling

Base: `b71002f78ee659c8eca4e9493507bd8dfd5a416c` (`feedback_protocol`).
The tracked working tree was clean at entry. This batch is local and uncommitted.

## Delivered scope

1. Observation/utility separation through `UtilitySpec`; old Bernoulli-bit reward
   remains a backward-compatible fixture adapter. Task utility is evaluator-only
   unless already encoded in the declared observations.
2. Complete/chain/star/ring generators with channel/hypothesis counts, shared
   parameters, cost ratios, capacities and budgets; family and structural-clone
   split enforcement. Twenty engineering instances were generated and registered.
3. `ExactMinimax`: tiny finite policy-tree regret game, rational mixture and
   least-favorable-prior certificate, explicit computation limits.
4. `BeamBayes`: bounded prefix generation and finite-depth Bayesian task-value
   planning; `PosteriorSampling`: independent seeded posterior sampling with
   cost-aware selection. Neither requires the complete protocol catalogue.
5. Configured, sealed, resumable scaling profiler with actual CPU measurements.

See [current contract](../research/feedback_protocol/CONTRACT_V02.md) and
[commands](../research/feedback_protocol/README.md). Dependencies remain stdlib
only. Runtime feedback is still finite-hypothesis Bernoulli, not general discrete
or continuous likelihoods.

## Validation

Twenty-one focused tests pass. The 13 original tests, including the fixed committed
solver regression, still pass. Added tests establish utility can differ without
changing policy observations; information-only sensing can improve future task
value; exact minimax randomization and primal/dual equality; a contingent sensing
tree with zero minimax regret in a tiny game; explicit minimax-limit failures;
cross-family renamed-clone and within-template split rejection; config round-trip;
public parameter sharing; deterministic generation; bounded planners functioning
when full enumeration is disabled; and catalogue count identities.

Two new debug episodes (beam and posterior sampling), each T=6, finish safely at
reset. Outputs: `/home/zjl/fpl_debug_beam_20260919_v2.json` and
`/home/zjl/fpl_debug_ps_20260919_v2.json`. These are smoke checks, not expected-regret
estimates. Generated engineering dataset: `/home/zjl/fpl_family_debug_20260919_v2/`.
Its 20-instance manifest validates train/dev/test/ood grouping but is not a final
scientific test-set freeze.

Frozen original source allowlist hashes remain unchanged. No old CONFIRM data,
training job, original calibration or theorem certificate was accessed/recomputed.

## Measured scaling

Final source-sealed raw run: `/home/zjl/fpl_scaling_20260919_v2_final.jsonl`.
[Versioned measurement summary with source hashes](../research/feedback_protocol/provenance/second_batch_profile.json).
Explicit completed-run resume left exactly 12 unique rows and did not duplicate
measurements. An earlier engineering run is retained separately; the final run
uses the completed source, with identical cases and limits, not favorable reseeding.

Fixed profile: three hypotheses, public seed 17, capacity and horizon m+2,
unit duration/energy, m distinct measurement channels. Exhaustive enumeration
is capped at 20,000 expanded prefix nodes; exact Bayes at 200 states and a
cooperative one-second budget. Beam width 4, depth 2, 500 expansions per candidate
generation and one-second cooperative deadline. These are engineering caps,
not demonstrated machine capacity or equal total computation budgets.

| Topology | Channels | Exact protocol count | Enumeration seconds | Enumeration Python peak MiB | Bayes states | Bayes outcome branches | Bayes status | Beam seconds |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| complete | 2 | 8 | 0.0001 | 0.006 | 17 | 43 | solved | 0.0079 |
| complete | 3 | 19 | 0.0002 | 0.003 | 56 | 211 | solved | 0.0258 |
| complete | 4 | 68 | 0.0010 | 0.012 | 200 | 1121 | state cap | 0.0621 |
| complete | 5 | 329 | 0.0063 | 0.064 | 200 | 884 | state cap | 0.1300 |
| complete | 6 | 1960 | 0.0429 | 0.504 | 200 | 354 | state cap | 0.2212 |
| complete | 8 | unresolved under cap | 0.6586 | 5.777 | 1 | 0 | enumeration cap | 0.7620 |
| star | 2 | 6 | 0.0001 | 0.001 | 13 | 35 | solved | 0.0064 |
| star | 3 | 7 | 0.0001 | 0.001 | 24 | 79 | solved | 0.0132 |
| star | 4 | 8 | 0.0001 | 0.001 | 75 | 207 | solved | 0.0192 |
| star | 5 | 9 | 0.0001 | 0.001 | 168 | 592 | solved | 0.0261 |
| star | 6 | 10 | 0.0001 | 0.001 | 200 | 644 | state cap | 0.0290 |
| star | 8 | 12 | 0.0001 | 0.002 | 200 | 442 | state cap | 0.0332 |

Times include tracemalloc overhead and have no repeated-run error bars. Peak is
Python traced allocation, not whole-process resident memory. Beam returned a legal
candidate in every row and reported pruning in every row. It has no exact-value
claim. The detailed file also records Bayes and beam Python peaks and expansion
counts. Capped state/branch counts are partial work, not full tree sizes.

For this complete template, ordered measurement routes have analytical count
`4 + sum(m!/(m-k)!, k=1..m)`; star count is `m+4`. The tests check this at m=2..4.
The complete m=8 analytical count is 109,604, but the run deliberately does not
label an interrupted enumerator as having measured that count.

## Research implications supported by these measurements

There are two distinguishable computation bottlenecks. Dense complete routes
create ordered-path combinatorics; star routes do not. Yet even the star Bayes
reference can consume many belief/time states. Therefore attributing all planning
cost to path enumeration would be incorrect.

At m=6 the complete catalogue still took about 0.043 seconds; the measured cap at
m=8 is not proof that a real system cannot enumerate it. The deliberately low
200-state Bayes cap also cannot support a claim that exact planning is infeasible
at m=4. Raising caps is a compute-allocation decision, not evidence already collected.

This batch does not establish a neural-method motivation by itself. Beam finishes
within these profiling budgets, but its reward/regret gap to Bayes on representative
tasks has not been measured. A matched-compute, multi-instance quality comparison
is still required before concluding either that simple planning suffices or that
a learned protocol generator is warranted. No neural training was started.

## Remaining scope

The requested five implementation targets are present and tested. Full A/B/C is
not complete: scientific distributions/splits, mechanism-subset sampling,
systematic expected-risk evaluator, multi-instance learning checkpoints, broad
baseline comparison, teacher datasets, B2/B3 neural methods, and an external-task
adapter remain future work. The tiny minimax solver returns actual mixture trees
and an exact certificate; it does not establish a new general minimax theorem.
