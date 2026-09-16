# Research Question Card: C/R latent factor and legal observability

Question: What low-dimensional latent causal factors determine the sign and
magnitude of the paired C/R advantage, and which of those factors are observable
from deployment-legal sensor history?

Type: exploratory, followed by a separately frozen confirmatory study only if
the exploratory factorization is coherent.

Hypothesis: The C/R advantage is governed primarily by two latent margins:

1. a **recoverability margin**—remaining usable energy and dynamic reachability
   to the charger under the executed low-level interface; and
2. a **task-opportunity margin**—the additional near-term task utility available
   before return becomes operationally unsafe.

The full 2,912-dimensional history failed because its generic summaries mix
irrelevant LiDAR variation with these margins, not necessarily because the
margins are fundamentally unobservable.

Why it matters: Gate H established real decision heterogeneity, while Gate I
showed that the tested full-history representation and fixed ridge/RFF models do
not identify it. Understanding the missing state is necessary before any new
actor, critic, recurrent model, or larger history encoder is scientifically
justified.

Current evidence:

- ER-20260915-pai-gate-h-01: 96-world paired-CRN DEV experiment; Gate H passed
  with selector gain 0.15959, 95% CI [0.07650, 0.25668]. Source type:
  experiment artifact. Claim strength: supported for this DEV environment.
- ER-20260915-pai-gate-i-01: Gate I failed; full-history MSE 1.31760 exceeded
  latest-frame MSE 0.80292 and constant MSE 0.68407. Source type: experiment
  artifact. Claim strength: supported negative result for the tested features
  and model families.
- ER-20260915-pai-raw-01: all 96 worlds, 24,576 branch outcomes, hashes, paired
  seeds, utility and collision semantics passed raw audit. Source type:
  experiment artifact. Claim strength: strong integrity support.

Missing evidence:

- A decomposition of paired advantage into task-increment, operational-failure,
  and unified-contact contributions.
- Identification of which pre-decision physical margins explain the component
  differences, using simulator state only as an explicitly privileged diagnostic.
- A measurement audit showing whether each candidate margin is identifiable
  from legal LiDAR/action/velocity/battery/charger/task history rather than exact
  map geometry or future outcomes.
- Fresh-world confirmation of any selected low-dimensional factorization.

What would support it:

- One or two interpretable pre-decision margins explain most stable sign changes
  in paired advantage across held-out worlds.
- Legal, low-dimensional summaries estimate those margins out of world and add
  selected utility beyond the latest-frame/constant baselines without increasing
  history-model capacity.
- A privileged oracle improves factor prediction only modestly, or legal-vs-
  privileged gaps can be traced to a specific sensor ambiguity.

What would falsify it:

- Advantage sign is dominated by downstream paired disturbance noise rather than
  stable pre-decision margins.
- No compact privileged factorization predicts held-out advantage better than a
  constant, meaning the proposed margins are the wrong causal abstraction.
- Privileged factors predict advantage but no legal statistic predicts those
  factors, establishing practical partial observability under the current sensor.
- Legal low-dimensional factors predict latent margins but still do not improve
  held-out decisions, showing that the margins are not decision-sufficient.

Minimal next action: On the completed DEV data, perform a no-new-training
advantage decomposition and a measurement/observability audit using only
pre-decision variables. Treat exact simulator geometry as a labeled privileged
diagnostic. Do not access CONFIRM, fit a larger history model, or start actor
training.

Decision: explore.

## Claim Candidate

Claim: Stable C/R heterogeneity exists, but the missing decision state has not
yet been identified or shown legally observable.

Source evidence: ER-20260915-pai-gate-h-01 and
ER-20260915-pai-gate-i-01.

Allowed wording: “The paired experiment reveals stable C/R advantage
heterogeneity, while the tested generic full-history representation fails to
predict or exploit it out of world.”

Forbidden stronger wording: “History is useless,” “the true latent state is
recoverability margin,” or “the decision problem is unobservable.”

Uncertainty: The failure may arise from irrelevant high-dimensional summaries,
measurement insufficiency, the wrong factorization, or residual outcome noise.

Next check: component decomposition followed by legal-versus-privileged factor
observability diagnostics.

Decision: keep as a bounded research hypothesis.
