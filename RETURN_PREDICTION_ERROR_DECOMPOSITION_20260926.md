# Moving-state return prediction-error decomposition

Date: 2026-09-26. This is a descriptive analysis plan written while the
frozen v4 return matrix is still running. It does not change its maps,
controller, SOC grid, estimator, or outcome labels. Maps 112–114 were used to
calibrate the backup controller; only maps 115–127 are independent confirmation
for that controller version. The endpoint is the 2 m flight handoff point,
not landing or charging.

After the complete return matrix and the frozen multi-map SOC analysis finish,
report per-map results rather than treating consecutive flight states or the
57 payload/profile labels as independent flights. Keep `arrived`, contact,
controller infeasibility, model capability violations, and energy exhaustion
separate. A return that arrives with an infeasible controller step is not a
valid return under the existing strict rule.

For each state and payload, let `p` be the frozen route-summary prediction,
`b` the executed return energy under the base energy profile, and `y` the
executed return energy under a tested profile. Then

    y - p = (b - p) + (y - b).

The first term describes the old rest-start route estimator's error on a
moving-state return under its own base model. The second is the effect of a
changed energy assumption on the same physical trajectory. For each SOC and
profile, assess false-safe events using that profile's observed remaining
energy after the outbound prefix. Among predicted-safe, energy-failing states,
record whether `b` alone already exceeds that remaining energy (base-model
error sufficient) or whether `b` fits but `y` does not (profile shift needed).
The capacity and outbound-prefix changes of a profile remain in the observed
remaining-energy calculation; these event labels are an accounting
decomposition, not causal attribution.

Report the base-model residual distribution by map and payload, with speed
bands fixed now at `<2`, `2–5`, and `>=5` m/s only as exploratory diagnostics.
Report map-level false-safe counts by SOC and profile type. Do not fit a new
predictor, tune a safety margin, select SOC thresholds, or claim online safety
from these retrospective labels. Source and input hashes must accompany the
output. The already frozen `return_multimap_soc` result remains the primary
confirmation readout; this decomposition is a secondary explanation audit.
