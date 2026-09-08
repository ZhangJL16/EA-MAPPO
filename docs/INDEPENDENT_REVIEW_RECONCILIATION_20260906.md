# Independent review reconciliation after scratch SAC/PPO

Date: 2026-09-06. Sources: all top-level records in `independent-review-20260906` plus the completed immutable 500-task scratch comparison. This note distinguishes reviewer judgment from evidence and updates only decisions affected by new data.

## Retain

1. **Navigation is an enabling platform, not the paper's novelty.** Safe-RL literature normally reports success and budgeted cost separately; the project's old conjunction of 98% arrival, zero contacts, and path ratio≤1.10 is a defensible deployment target but not a field-standard scientific prerequisite.
2. **Do not continue action-level option-critic work on the old data.** The completed 3,410 rollouts showed only 21/310 anchors with structurally mixed action outcomes, Cochran Q p=.776, and energy Kendall W=.017. This rejects that candidate set and one-step executed interface as a rich direct `Q_op(x,a)` label source. It does not prove that actions can never affect returnability.
3. **The core object should be state-level resource-to-recharge first passage under the executed interface.** Structural failure must be an atom at infinity, not folded into mean energy. A successor model may later evaluate actions through `V_R(F(x,a))` if action intervention becomes necessary.
4. **Expected-energy MAE is insufficient.** The new SAC result makes this concrete: five tasks create 90% of its contacts despite a zero median and near-perfect arrival. The paper should target dangerous underestimation, failure probability/calibration, and the stranding–throughput frontier.
5. **Oral-level novelty is not yet established.** Identifiability, margin stability, conformal abstention, killed Feynman–Kac operators, and return controllers each have substantial prior art. The plausible new joint object is the killed resource first-passage law under policy∘filter composition shift plus an irreversible stopping decision. It needs a separation construction, a non-tautological bound/lower bound, and a prediction that experiments can falsify.

## Revise with new evidence

1. **“The current recovery reward cannot learn avoidance” is too strong.** Scratch SAC, without HOCBF and under the fixed nonterminal repair/reward contract, reaches 496/500 goals and 423/500 with zero contact. The reward is learnable by at least one configuration. The remaining defect is rare catastrophic contact-stall behavior and declining long-distance zero-contact rate, not complete absence of avoidance.
2. **R3 need not remain the nominal platform.** It was a useful historical reference, but the scratch SAC has 99.2% arrival and path ratio 1.063 on the immutable test set. It is now the stronger learned nominal candidate. Its return reliability from preserved off-trajectory anchors is still unknown.
3. **A fixed 97% return gate is not adopted.** The old recommendation is useful as an engineering aspiration, not a literature-derived theorem threshold. The next anchor experiment reports rates, intervals, failure types, and interface effects without turning another arbitrary number into a pre-research blocker.
4. **The paper should not assume HOCBF as the only platform.** New SAC supplies an unshielded learned-policy arm. A shielded arm remains valuable because the theory's executed interface explicitly includes policy∘filter composition, but shield-on results cannot be presented as autonomous policy safety.

## Reject or defer

1. **Do not run another PPO repair now.** Target-KL early stopping is not dominant (median 10/10 epochs; 92/128 full blocks), the critic fits its on-policy returns well, and the failure grows with distance. A lambda, entropy, or target-KL ablation would improve PPO diagnosis but would not unlock the resource-return research question.
2. **Do not make a 2×2 policy/filter study the immediate next experiment.** It is a later mechanism experiment after a viable return policy and stochastic resource variation exist. Running it now would multiply an unresolved return baseline.
3. **Do not claim oral potential from theorem count.** The previous 33-result package is theorem sprawl. The target is one central identifiability/risk-transfer theorem, one matching impossibility or lower-bound result, and direct experimental consequences.

## Selected next experiment

Re-evaluate immediate return from the same 310 preserved task-leg anchors using the new scratch SAC. Use two execution interfaces—raw learned action and the repository's standard HOCBF—inside one matched diagnostic. In both arms, a contact is repaired, all velocity is zeroed at repair, and the return episode continues; record one unified contact count at most once per policy step. This deliberately corrects the legacy diagnostic's first-contact termination semantics.

Primary quantities are charger arrival, zero-contact charger arrival, timeout, unified contacts, energy-to-recharge, and distance-bucket breakdown. The comparison identifies whether the new policy repairs the old 88.71% return bottleneck and whether the filter changes the resource first-passage law enough that `policy∘filter` must be explicit in the theory. It is a mechanism diagnostic, not a navigation gate or confirmation result.

If return arrival is high but a rare contact/timeout tail remains, proceed to fresh stochastic wind/execution-noise data and state-level distributional `V_R`. If both interfaces fail broadly, train the same SAC navigation objective on return-target states before energy modeling. If HOCBF substantially improves zero-contact return but reduces arrival or increases energy, that trade-off becomes the first concrete composition-shift phenomenon for the paper.
