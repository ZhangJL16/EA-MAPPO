# External hostile review request

Status: NOT DISPATCHED; no independent review received. Prepared by the manuscript-producing agent. This request is not a correctness or novelty certificate.

Target: ICML theory submission. Please evaluate the manuscript, not the authors' positioning documents or previous mock reviews. Do not assume novelty from the physical interpretation. Do not assign Spotlight probabilities.

## Round 1: manuscript-only novelty attack

Read `ICML_RESOURCE_SEPARATION_PAPER_CORE_20260916.md` without the positioning audits or prior review scores.

Try to reject the novelty claim. In particular, either derive Theorem 1 and the two-certificate diagnostic as a direct corollary of an existing controlled-sensing, structured-bandit, side-observation, or sequential-experiment theorem, or identify precisely which conclusion the closest theorem does not supply.

For a proposed reduction provide: verified source and theorem number; a mapping of hypotheses, actions, feedback, durations, policy class, terminal convention and comparator; verification of every required assumption; and the exact conclusion inherited. Distinguish a direct corollary from a representation inside a parent framework and from a new construction using standard tools. If the result is not a direct corollary, still judge whether the residual construction is significant rather than merely technically absent from a theorem statement.

Challenge the choice of the maximum fixed-instance coefficient. Does preserving that scalar, while an individual coefficient improves, substantively support the claimed comparison failure? Which wording should be weakened? A failure to find a reduction is not proof of priority.

Return evidence-anchored objections and one of: direct corollary established; substantial overlap but no direct corollary established; residual separation potentially significant; or unresolved. Record search/read scope and uncertainty. Commit this report before seeing the authors' novelty narrative.

## Round 2: separate correctness audit

Prefer a different reviewer. Reconstruct the proof from definitions and source derivations before running the supplied checkers. Inspect the anchor bounds, legal arbitrary-terminal low-capacity policy coverage, feasibility of the randomized high-capacity witness, average versus finite-terminal comparators, individual and maximum coefficient formulas and attainability assumptions, perturbation bound over the structural open box, and all-budget bundling-off coupling. Check quantifiers in the diagnostic, especially every old least-favorable prior and dual survival at an old maximizing hypothesis.

Requested source files after the first report is committed:

- `STRICT_FINITE_BUDGET_CAPACITY_BENEFIT_20260916.md`
- `ROBUST_RESOURCE_BUNDLING_SEPARATION_20260916.md`
- `RESOURCE_SEPARATION_DIAGNOSTIC_AND_TRANSFER_20260916.md`
- `RESOURCE_PATH_CATALOGUE_ATTAINABILITY_20260916.md`

All are in this directory. Numerical replay is supplementary, not a substitute for policy-class and proof validation. Report each conclusion as checked, contradicted, or unresolved, with exact locators and any countercalculation. Do not infer practical learner superiority from the optimal-risk theorem.

## Provenance and dispatch boundary

Record reviewer identity or stable pseudonym, expertise, prior involvement in this work, materials received, access to author narrative, date, and whether this is a human or model-assisted report. An external model review is not a human domain-expert review; a different implementation is not a blind external proof audit. No unpublished materials have been uploaded or sent by this task. Neither Claude nor Gemini CLI is present on the current PATH; authentication and API tests were skipped after that detection failure. A named recipient/contact channel or explicit provider-and-content consent is required before dispatch.
