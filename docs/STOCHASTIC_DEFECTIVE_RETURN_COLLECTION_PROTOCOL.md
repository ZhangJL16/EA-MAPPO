# Stochastic defective return-law collection protocol

Frozen 2026-09-06 before collection. Development feasibility data; no confirmation or automatic model promotion.

## Research object

For anchor state \(x\), executed interface \(I\in\{\pi,\,\pi\circ\mathrm{HOCBF}\}\), and future execution disturbance \(\omega\), collect

\[
Y_I(x,\omega)=
\begin{cases}
\sum_{t<\tau_C}e_t,&\tau_C\le4000\text{ and }N_{\rm contact}=0,\\
+\infty,&\text{otherwise}.
\end{cases}
\]

The experiment asks whether repeated outcomes at fixed \((x,I)\) have enough structural and finite-energy variation to compare mean-only prediction with a factorized defective distribution in the next phase.

## Fixed policy, anchors and interfaces

- Scratch SAC seed 0, exactly 524,288 training transitions; deterministic proposal.
- All 310 task-leg anchors from the preserved 150-scene development inventory.
- Raw SAC and the identical proposal passed through the existing default HOCBF.
- Eight physical disturbance replicates per anchor/interface: 4,960 total return trajectories.
- Each raw/HOCBF pair uses the same anchor and disturbance seed. Seeds are keyed by anchor index and replicate, independent of scheduling or trajectory length.

## Physically specified disturbance law

The disturbance is post-controller acceleration-execution error, not policy sampling. In normalized action coordinates, once per 0.2 s policy step,

\[
\eta_{t+1}=0.95\eta_t+0.04\sqrt{1-0.95^2}\,\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,I_3),
\]

with each component clipped to \([-0.12,0.12]\). The same \(\eta_t\) is applied across the four 0.05 s physics substeps after the raw policy or HOCBF output and before plant integration. Actions remain clipped to their physical normalized range. Under the current acceleration limits, one stationary standard deviation corresponds to 0.20 m/s² horizontally and 0.12 m/s² vertically; the componentwise bound is 0.60/0.36 m/s². The nominal correlation time is \(-0.2/\log(0.95)\approx3.90\) s.

This is a controlled simulator stress law, not a calibrated real-aircraft uncertainty claim. Because actual post-clipping acceleration commands drive both motion and the existing energy calculation, the perturbation affects realized dynamics and energy consistently inside the current plant.

## Collision and outcome contract

Contact never terminates a trajectory. Repair position, zero every velocity component at repair, and continue toward the same charger. Reward remains the locked −1.2/−0.42 rule. Save only one unified contact count per policy step; never save separate boundary/obstacle counts. Safe return means charger arrival within 4,000 steps and total unified contact count zero.

## Data and next decision

Every anchor/interface/replicate trajectory is saved atomically with its disturbance seed, arrival, safe arrival, timeout, unified contact count, energy, path ratio, intervention count, and realized execution-error RMS/max. Scene, anchor and replicate provenance are retained.

This collection has no performance gate. After the user reports completion:

- quantify within-anchor structural and energy variation;
- if variation is nondegenerate, freeze scene-disjoint train/calibration/test splits and compare finite mean/capped regression, structural-only prediction, and factorized defective-law prediction;
- if almost all repeated outcomes are identical, do not fit a distributional neural model—revise the physical disturbance/domain first.

Collection is resumable at each trajectory file. Only a focused test, a two-anchor/two-replicate smoke, and a first formal atomic-batch health check precede handoff. No predictor fitting starts automatically.
