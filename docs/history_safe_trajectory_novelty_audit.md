# Novelty Audit: History-Conditioned Safe Trajectory Control

## Verdict first

**CLAIM_LEVEL = ENGINEERING_ONLY.**

Current conference readiness for a theory/main-method paper is low. Development potential as a robotics/autonomous-systems engineering study is medium. The method is scientifically sound because learning never certifies safety, but the central mechanism is heavily covered and the new experiments do not show a history-conditioned proposal advantage over random safe MPPI.

## Search basis

Seventy primary papers were screened. The decisive closest works are SC-MPPI, GS-MPPI, DualGuard MPPI, BR-MPPI, Safe Beyond the Horizon, ARMTD, reachability-based trajectory design with neural implicit constraints, Stein MPC, STORM, and history-conditioned generative trajectory methods.

## Normalized idea

- Problem: single-step HOCBF can be myopic, infeasible, or freeze.
- Mechanism: history/motion features concentrate proposals; cheap pruning reduces compute; exact physics and verification preserve safety; CLF prevents freeze; telemetry energy ranks survivors.
- Evidence: high safe-state recall and sub-50 ms state-level latency.
- Unresolved: closed-loop B5 safety/progress/energy, dynamic obstacle recall, and a nonstandard theorem.

## Closest-prior attack

| Closest work | What it already covers | Residual difference | Risk |
|---|---|---|---|
| SC-MPPI | Barrier-state control embedded in MPPI | Separate exact quartic verifier | Fatal to generic safe-sampling novelty |
| GS-MPPI | Composite CBF and safe sampled trajectories | UAV clipping/energy implementation | Fatal to CBF-safe MPPI novelty |
| DualGuard MPPI | Reachability-safe sampling MPC | Different guard and energy stack | Fatal to reachability-guided novelty |
| Safe Beyond the Horizon | Terminal safety for sampling MPC | Static-sphere interval verifier | Fatal to terminal-safety novelty |
| BR-MPPI | Barrier-rate-guided sampling | History/motion proposal inputs | High |
| ARMTD / neural implicit reachability | Certified trajectory design | Different dynamics and proposal | High |
| Diffusion Policy / AgentFormer / MTR | History-conditioned generation | Independent safety certificate | High |

## Novelty delta after subtraction

The residual is an integration:

1. causal LiDAR motion estimation with ego compensation;
2. clipped UAV physics with realized acceleration and telemetry energy;
3. exact quartic inter-sample sphere verification;
4. lexicographic safety, progress, and energy selection.

No screened paper has the identical stack, but stack identity is not a theorem-level insight. The coarse-rejection and epsilon-cover results are standard bounded-error and Lipschitz-cover arguments.

## Empirical novelty attack

- B3 random safe-MPPI-style proposals achieved the same 100% state-level recall as B5 on both exact subsets.
- B5's compute advantage is against B4 exact-all, not an optimized closest safe-MPPI implementation.
- History was not used by the static B5 benchmark.
- Motion history helps constant-velocity estimation, but L16 fails under abrupt motion.
- No closed-loop B5 path, freeze, collision, or energy comparison exists.

## Independent reviewer panel

### Reviewer A — control theory

- Score tendency: 5/10.
- Confidence: high.
- Positive: exact finite-horizon certificate and role separation are correct.
- Reject concern: recursive feasibility is conditional on an uninstantiated backup set; the adaptive tube is missing.
- Score-change condition: prove a non-vacuous causal error set under feedback, or narrow to engineering.

### Reviewer B — novelty

- Score tendency: 3/10.
- Confidence: high.
- Positive: exact UAV implementation differs in details.
- Reject concern: safe MPPI, reachability-guarded sampling, terminal safety, and history-conditioned proposal generation already exist.
- Score-change condition: identify a theorem-level object unavailable to closest work.

### Reviewer C — systems and experiments

- Score tendency: 5/10.
- Confidence: medium-high.
- Positive: 100M coarse sequences and P99 latency expose a real compute tradeoff.
- Reject concern: only 55 exact-reference states and no B5 closed-loop evaluation; B3 equals B5 recall.
- Score-change condition: same-stream dynamic-obstacle evaluation showing lower freeze/energy or higher recall than B3/B4.

### Area-chair synthesis

The implementation is credible but the claimed research mechanism is not differentiated. The strongest result is an engineering reduction from 1.44 s exact-all to 31.20 ms coarse-to-fine on ordinary states while preserving recall in a small exact subset. That is insufficient for a theory-centered main contribution and under-evidenced for a systems paper.

## Dimension scores

| Dimension | Weight | Score | Confidence | Deduction / evidence basis | Repair condition |
|---|---:|---:|---:|---|---|
| Problem importance | 12 | 4 | 4 | Myopic filters and real-time lookahead matter | Keep |
| Novelty | 14 | 2 | 5 | SC/GS/DualGuard/BR-MPPI cover mechanism | New certified history-adaptive object |
| Conceptual innovation | 12 | 3 | 4 | Mostly coherent module composition | Mechanism unavailable to random MPPI |
| Method soundness | 14 | 4 | 4 | Proposal/physics/certificate separated | Instantiate terminal and perception assumptions |
| Elegance | 8 | 3 | 4 | Many components; history unnecessary in static run | Remove noncausal-value components |
| Feasibility | 8 | 4 | 5 | P99 below 50 ms after pruning | Validate full stack |
| Experimental convincibility | 10 | 2 | 5 | Small exact subset, no B5 closed loop | Same-stream multi-seed closed loop |
| Venue fit | 8 | 3 | 3 | Better robotics fit than ML theory | Reposition and strengthen systems evidence |
| Timeliness | 6 | 4 | 4 | Safe sampling is active | Keep |
| Risk-adjusted acceptance | 8 | 2 | 4 | Fatal overlap and no history advantage | Resolve one decisive axis |

**Weighted score: 3.08/5 = 6.16/10.**

The fatal novelty gate caps the recommendation at pivot-with-rescue-route.

## Keep / reject decisions

- KEEP: exact physics rollout, continuous-time verifier, 20-survivor coarse-to-fine path, causal ego compensation, lexicographic role separation.
- PARTIAL: history; useful under smooth motion, harmful when stale.
- REJECT as novelty: generic history safe MPPI, terminal backup theorem, epsilon-cover bound, candidate-set optimality.
- REJECT for now: learned proposal network; random B3 already matches recall.
- REJECT as paper core: current complete framework.

## Strongest rescue route

The only theory rescue is a tight causal set-membership trajectory tube whose width adapts to observable residuals and remains valid under closed-loop selection. The only empirical rescue is a robotics paper showing that an adaptive history window plus physical motion compensation improves dynamic-obstacle safe recall and reduces closed-loop freeze/energy at P99 below 50 ms against optimized safe-MPPI and sampled-data HOCBF.

Neither rescue is established in the current bounded study.

## Final recommendation

**PIVOT-WITH-RESCUE-ROUTE.** Preserve the implementation as a research platform. Do not present it as a new general safety theory or paper-core method. If continued, target a robotics/autonomous-systems phenomenon study centered on abrupt-motion robustness and causal history-window selection, not another generic safe-RL rebranding.
