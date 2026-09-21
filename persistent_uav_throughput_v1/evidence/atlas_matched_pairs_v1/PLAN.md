# Archived Atlas matched-pair analysis v1

Retrospective descriptive analysis requested after Atlas outcomes were reviewed; not a preregistered experiment. No simulator imports or new rollouts. Original Atlas immutable.

Population: the 128 multi-safe roots with all safe-first continuations resolved. Enumerate every unordered safe-action pair, including equal-throughput pairs. Keep 17 unresolved roots out of complete-case comparisons and disclose them. Pair identity ordered by task ID, never outcome. Report pair counts and roots separately, plus source and battery strata. No IID p-values, regression claims or causal mediation fractions.

User-confirmed primary time match: abs(T_i-T_j)/mean(T_i,T_j)<0.10; battery match: abs(e_i-e_j)/capacity<0.05; joint is intersection. User explicitly selected the mean first-task duration denominator before matched results were computed. Also retain both relative-time and W-normalized differences in pair data. No outcome-dependent threshold changes.

Features: measured task duration/energy, completion battery/position, Euclidean 3D charger distance, root queue positions and pairwise distances, first-task arrivals/accepted/rejected counts, post queue IDs/occupancy and Euclidean distances, immediate post-task oracle-safe task count (unknown if absent/incomplete). Read only actual continuation top-level events: nested counterfactual arrivals are not actual arrivals. Post-state must match first-task completion time. Future overflow/forced returns/time-by-mode are outcomes, never post-task predictors.

For each matching category report all pairs, divergent pairs, represented roots, roots with a divergent pair, mean absolute N difference, root-balanced mean absolute difference. Describe signed high-N vs low-N feature differences for divergent joint pairs, with null/tie counts; do not interpret direction counts as causal explanation. Same arrival counts are weaker than same IDs, so preserve both. Residual time/battery differences remain: caliper matching cannot exclude nonlinear local indices or prove a nonlocal mechanism.

Integrity: validate input summary hashes and every used raw B file hash; verify A/B first completion alignment, post-task oracle timing and candidate coverage. Expose missing features instead of reconstruction by new simulation.
