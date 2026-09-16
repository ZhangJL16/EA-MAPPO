# Paper-core assembly audit — 2026-09-16

## Completed deliverables

- Standalone English Introduction, comparison setting, Theorem 1, explanatory bottleneck paragraph, related-work table, and Discussion: `ICML_RESOURCE_SEPARATION_PAPER_CORE_20260916.md`.
- Figure 1: editable SVG/vector PDF and 300-dpi PNG preview under `figures/resource_separation/`.
- Reproducible renderer: `scripts/render_resource_separation_figure.py` (repository-relative location).
- Figure source CSV and provenance JSON accompany the figure.

Writing axes: manuscript / research / Introduction + theorem + Discussion + related work / Chinese-to-English / generic ICML. The user supplied the argument, evidence, section hierarchy, and boundaries; no further proposal confirmation was needed. Skills used: nature-writing for evidence-first assembly, nature-figure for certificate-based figure construction and QA. Existing Python preference was honored.

## Argument and paragraph jobs

Single argument: two specific summaries can agree under resource-realizable experiment nesting while optimal finite-budget minimax regret strictly differs.

Introduction paragraph jobs, in order: physical learning decision; precise summaries and credited prior art; parent framework and claim boundary; certified separation and robustness; causal controls; distinct bottlenecks; evaluation implication and negative calibration.

Theorem hierarchy: separation first; standard duality as explanation; resource realization and controls as evidence. Discussion distinguishes evaluation insufficiency, physical realizability, scalar-maximum limitation, adaptive continuation, negative calibration, and scope. No new foundational theorem is advertised.

## Claim–evidence map

| Claim | Existing evidence | Scope enforced |
| --- | --- | --- |
| Equal hypothesis-wise average gains | `ROBUST_RESOURCE_BUNDLING_SEPARATION_20260916.md` | Same optimum across capacities at each hypothesis, not one optimum across hypotheses |
| Equal finite execution values | Same source and strict certificate | Charger-terminal; T=12,24 only |
| Equal maximal fixed-instance coefficients | Existing coefficient formulas and q dominance | A coefficient changes; no full geometry equality or limit/sup interchange |
| Strict finite-budget minimax gap | `STRICT_FINITE_BUDGET_CAPACITY_BENEFIT_20260916.md` | Low lower bound covers primitive-feedback/arbitrary legal terminal; high upper uses committed terminal witness |
| Open-region gap >0.038509475 | Existing policy Lipschitz bound, anchor gap | Open four-dimensional structural family, not all independent reward perturbations |
| Off-control equality | Existing virtual-counter/route correspondence | Old feedback and durations preserved |
| Feedback-free control cannot explain witness | Existing exact 32/115 terminal control | Does not prove within-sortie necessity |
| Operational learner benefit unsupported | Frozen T=4096 calibration and diagnostic | No empirical positive claim or population inference |
| Analytical tools are established | Four primary sources in main table; existing locator ledger | No formulation/tool novelty or global priority clearance |

## Terminology ledger

| Use | Do not substitute |
| --- | --- |
| Two specific summaries | All statistical summaries / statistical equivalence |
| Maximal fixed-instance logarithmic regret coefficient | Asymptotic minimax constant / full asymptotic information geometry |
| Certified lower and attainable upper bounds | Exact minimax values / sampled learning curves / confidence intervals |
| Between-sortie adaptive continuation | Necessary within-sortie adaptivity / timing-only causal explanation |
| Resource-realizable robust separation regime | New sensing formulation / new foundational duality / new exploration principle |

## Figure contract and checks

Conclusion: the two specified equal summaries do not imply equal finite-budget risk.
Evidence: existing T=12 anchor certificate, not additional horizon evaluation or trajectories.
Layout: schematic-led two-panel evidence composite; final size 183 mm × 88 mm; minimum base font 7.2 pt. Panel a identifies the unchanged quantities and route feasibility. Panel b displays one-sided certificate arrows, not point estimates.

The renderer checked exact rational gap subtraction, certificate flags, and canvas text clipping. Python figure validator: 10 PASS, 4 WARN, 0 FAIL. Actual PNG was visually inspected: table, labels, arrows, gap bracket, and scope footnotes are legible with no visible overlap or clipping.

Warning dispositions: SVG/PDF are primary vector outputs; PNG is a 300-dpi review preview, so no TIFF/600-dpi raster submission claim is made. The validator's apparent 4648.2-mm width is a static-expression parsing artifact; the renderer and provenance record the actual 183-mm width after millimeter-to-inch conversion. The logarithm uses fixed positive interior Bernoulli anchor means, not arbitrary unchecked data. These dispositions do not certify venue-specific submission compliance.

Source certificate: `artifacts/bundling_finite_budget_theory_20260916/strict_benefit_certificate.json`, SHA-256 `d05eac70ec41d7247aef4fd18f65ad494352ea5705e13b4ead12275e950324af`. Figure metadata records zero sampled trajectories, five source rows, and no new horizons evaluated.

## Integrity boundaries

No new theory, numerical policy search, sampled calibration, seeds, routing library, learner changes, or runtime changes. Existing negative experiment retained. This completes the requested paper-core writing and figure task; independent proof/priority review and full submission preparation are not claimed completed.

Final mechanical checks passed: all paper-core local links resolve; exact anchor gap and neighborhood margin agree arithmetically; `git diff --check` reports no whitespace errors. Frozen calibration core and runner SHA-256 hashes remain unchanged from the startup registry.
