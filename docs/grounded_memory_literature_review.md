# Control-Grounded Structured Memory: Literature Review

## Scope and evidence protocol

- Review date: 2026-08-20.
- Primary papers screened: **71**; see `docs/grounded_memory_literature_matrix.csv`.
- Deep full-text reads: ExpoComm, MASIA, RIMs, Slot Attention, SlotFormer, FOCUS, MERLIN, approximate information states, nonlinear output-feedback information states, DBC, task-driven information bottlenecks, KalmanNet, Lyapunov neural observers, measurement-robust CBFs, belief CBFs, risk-aware belief CBFs, modular safety filters, and data-driven safe-set expansion.
- Remaining papers were read at abstract-plus-method/theorem level from their primary arXiv or proceedings source. Surveys were used only to discover primary papers and are not counted in the 71.
- The review distinguishes four claims: **what a representation stores**, **how it updates**, **what objective grounds it**, and **what theorem (if any) connects it to control**.

## Required paper: ExpoComm

Primary source: [ExpoComm, ICLR 2025](https://arxiv.org/abs/2502.19717).

### What its memory/message stores

ExpoComm has two recurrent quantities at each agent:

1. a **local history** `h_i^t`, updated from the current local observation and the previous local history;
2. a **message memory** `m_i^t`, updated from the previous local message and the messages received at the previous step.

The message is therefore an accumulated communication state, not merely the current sender observation. Its intended semantics are global information propagated over the exponential graph, but its coordinates are not assigned physical meanings.

### Temporal update and distinction between history and message

Algorithm 1 updates local history first, then updates the message from prior message traffic, sends the current message, and chooses an action from current history plus current message. For the one-peer graph the message aggregator is recurrent; for the static graph it is attention-based. The paper's key temporal idea is that a small graph diameter is useful only if message processors preserve information across the corresponding number of timesteps.

### Grounding objective

With a training-time global state, a discarded MLP decoder reconstructs the current global state from each local message. Without a compact global state, InfoNCE treats same-time messages from different agents as positives and temporally distant messages as negatives. The total objective is the MARL TD loss plus a scalar-weighted auxiliary loss.

### Downstream use

The action network reads both the local history and the current accumulated message. There is no typed routing to Safety, Route, or Energy modules and no separation between learned and certified channels.

### Ablations

The reported ablations remove either message memory or the auxiliary grounding loss. Removing memory is most harmful in temporally correlated scenarios; removing auxiliary grounding can make communication unhelpful or detrimental. These are task-return ablations, not physical reconstruction, uncertainty calibration, or safety-filter ablations.

### What the theory actually proves

The sole formal theorem proves a Boolean matrix-product property of the one-peer exponential topology: after `ceil(log2(N-1))` steps every node is graph-reachable from every other node. The accompanying remark additionally assumes that information remains valid and aggregation causes no information loss. The paper does **not** prove:

- that the recurrent message preserves all received information;
- that state reconstruction makes a message control sufficient;
- that the auxiliary loss improves policy value;
- that grounded messages remain valid under distribution shift;
- any physical safety property.

### Transferable and non-transferable ideas

Transferable:

- separate local history from accumulated inter-module message;
- make update topology explicit;
- ground messages with auxiliary physical/control targets;
- ablate memory and grounding independently;
- treat communication bandwidth and update latency as first-class metrics.

Not directly transferable:

- exponential graph diameter is relevant to homogeneous many-agent dissemination, not to a five-module single-UAV control stack;
- reconstructing a global MARL state is not the same as retaining barrier-, route-, or energy-sufficient information;
- temporal contrast among agents does not certify obstacle state or actuator bounds;
- ExpoComm's action network consumes messages end-to-end, whereas hard safety here must not accept unverified latent relaxations.

## Closest-work clusters

### 1. Grounded communication and sparse routing

MASIA reconstructs global state and predicts future aggregate representations, then applies agent-specific masks. Autoencoder-grounded communication and contrastive grounded communication predate ExpoComm. IMAC applies an information bottleneck; ToM2C uses target-aware routing; learned structured communication and DCMAC provide sparse or receiver-conditioned routing. Thus the claims “ground messages”, “compress them”, and “route different messages to different receivers” are all established separately.

**Implication:** a fixed Safety/Route/Energy routing graph is an architectural design choice, not by itself a research contribution.

### 2. Modular recurrent memory

RIMs already use independently parameterized recurrent modules, sparse input activation, persistence of inactive modules, and sparse inter-module attention. Fast/slow RIMs add timescale separation. Structured state-space models provide stable compact alternatives to GRUs. Recent AURA and HERA explicitly gate memory writes or route historical evidence.

**Implication:** replacing a monolithic GRU with several recurrent modules, sparse attention, or an SSM is not sufficient novelty.

### 3. Object-centric memory

Slot Attention provides exchangeable object slots with competitive binding and slot-wise GRU updates. SlotFormer adds temporal object dynamics. FOCUS and related object-centric world models use object reconstruction and world-model losses for downstream control. Recent Slot-MPC directly couples object slots to planning, while 2025–2026 diagnostic work shows that better slots can help but may saturate and that policy learning can still fail despite good object prediction.

**Implication:** per-obstacle slots are well motivated, but object decomposition alone does not establish control value or safety.

### 4. Predictive/world-model memory

MERLIN, PlaNet, and Dreamer show that memory can be grounded by predictive modeling and then consumed by a controller. These systems already use multiple prediction heads and recurrent latent states. Prediction is therefore neither a novel grounding mechanism nor automatically control sufficient. The 2026 controlled study on predictive objectives explicitly warns that predictive losses can discard exogenous control-relevant features.

**Implication:** motion prediction, future-clearance prediction, or energy prediction must be tied to a declared downstream quantity; prediction accuracy alone is insufficient.

### 5. Control-sufficient representations

Exact information-state theory defines a recursive statistic sufficient for cost evaluation and self-prediction. Approximate information-state theory replaces equality with terminal-cost and transition-set errors, producing a horizon-accumulated value bound weighted by Lipschitz constants. Agent-state POMDP work extends this idea to finite learned memory. Bisimulation and task-driven information bottleneck work connect compressed representations to value/control robustness.

**Implication:** a theorem of the form `representation error -> value error` is already known. A new result must exploit the concrete modular contract or a genuinely different information structure, not simply sum three Lipschitz errors.

### 6. Learned observers and uncertainty

KalmanNet and its variants combine known model structure with recurrent learned gains. Lyapunov-constrained neural observers provide estimation-error stability under assumptions. These methods are stronger closest works than a generic physics-plus-residual GRU because they make the estimator update and guarantees explicit.

**Implication:** “explicit physical state plus a small learned residual” is an existing hybrid observer pattern.

### 7. Output-feedback and belief-space safety

Measurement-robust CBFs consume a point estimate plus a valid state-dependent error bound. Belief CBFs consume belief mean/covariance and prove conditional invariance/probabilistic reset results. Recent output-feedback backup CBFs explicitly handle estimator error and input bounds. Conformal perception-based safe control constructs calibrated uncertainty sets for robust constraints.

**Implication:** a certified uncertainty memory routed into a robust safety filter is established. Learned latent information may improve proposals or nominal estimates, but safety remains conditional on the certified set.

### 8. Modular safety filters and permissiveness

The safety-filter literature already separates arbitrary nominal controllers from independently defined admissible sets. Data-driven safe-set expansion explicitly targets lower conservatism, and robust-optimization set inclusion immediately gives a larger feasible action set when uncertainty contracts soundly.

**Implication:** “learned memory cannot break hard safety because a filter projects its proposal” is correct but is a specialization of the established modular safety-filter principle.

## Candidate contribution audit

| Candidate statement | Prior-art status | Assessment |
|---|---|---|
| Object slots plus recurrent updates | Slot Attention, SlotFormer, object-centric world models | Existing |
| Sparse modular recurrent memory | RIMs, SSMs | Existing |
| Auxiliary physical/control grounding | MASIA, ExpoComm, MERLIN, world models, control-centric representations | Existing |
| Receiver-specific message routing | ToM2C, DCMAC, structured communication | Existing |
| Information-state/value error bound | approximate information states, bisimulation, information bottleneck control | Existing |
| Learned/certified channel separation | robust output-feedback CBFs and modular safety filters | Existing |
| Smaller certified set gives larger admissible action set | robust optimization and safe-set expansion | Existing |
| One typed graph jointly instantiates all of the above for persistent UAV control | no exact match found | Systems integration gap, not yet a theorem-level gap |

## The strongest defensible gap

No reviewed work gives one end-to-end theorem connecting all four of the following in a typed UAV control graph:

1. module-specific approximate information states;
2. a certified uncertainty channel that cannot be relaxed by learned messages;
3. sound information contraction to robust HOCBF action-set expansion;
4. downstream intervention/path/energy consequences.

However, each edge of this chain is already covered by mature theory. The missing end-to-end composition is not automatically novel: under straightforward assumptions it follows by composing approximate-information-state error bounds, robust-set monotonicity, and safety-filter separation. A contribution would require a nontrivial coupling result, a strictly weaker assumption set, or a new control-relevant memory construction with a proved contraction property.

## Literature-driven architecture constraints

The review supports the following **engineering** choices:

- explicit per-object physical state and confidence/age rather than a single LiDAR-history vector;
- learned residuals may propose centers or rank routes but may not shrink certified uncertainty without calibration/evidence;
- fixed typed routing is preferred over learned all-to-all routing unless learned routing demonstrates a measurable advantage;
- Safety consumes certified geometry and uncertainty plus explicit ego/actuator state;
- Route consumes object predictions, avoidance-side history, progress, and prior safe trajectory;
- Energy consumes executed-motion and intervention/detour context;
- grounding heads are separate and normalized, and their success is evaluated both by probes and closed-loop metrics;
- an adversarial learned-message test is mandatory.

## Literature gate conclusion

The literature supports a strong **control architecture study**, but it does not yet support a novelty score high enough for a theory-first long experiment. The exact theorem candidates must be attacked against approximate information states, task-driven information bottlenecks, output-feedback CBFs, belief CBFs, and modular safety filters before any long run.

## Final targeted attack: verified contraction and proposal checking

The last search closes the remaining gap rather than reopening a broad survey.

- [Li et al. (ICML 2024)](https://proceedings.mlr.press/v235/li24ci.html) already provides non-asymptotic convergence rates for set-membership uncertainty-set learning.
- [Tang et al. (L4DC 2024)](https://proceedings.mlr.press/v242/tang24a.html) explicitly uses constraint pruning to make certified set-membership over-approximations tractable.
- [Misra et al.](https://arxiv.org/abs/1802.09639) learns relevant active constraint sets for repeated constrained optimization.
- [Certified Control](https://arxiv.org/abs/2104.06178) formalizes the asymmetric architecture where a complex component finds a certificate and a smaller trusted monitor checks it.
- [Adaptive Learning-based MPC with Set Membership Identification](https://arxiv.org/abs/2404.16514) directly couples learned uncertainty-set contraction to robust control adaptation.

Consequently, `network proposes history constraints; verifier accepts; robust controller uses accepted intersection` is not an uncovered theorem-level object. A defensible contribution would need either a materially stronger contraction guarantee or a downstream effect unavailable to deterministic set-membership and simple route hysteresis. The real diagnostic provides neither.
