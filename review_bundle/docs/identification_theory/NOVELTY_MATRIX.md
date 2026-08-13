# Novelty Matrix

## Gate

The direction requires `Theory >= 7/10`, `Novelty >= 6/10`, `Methodology >= 6/10`, `AC >= 6/10`, and zero unresolved `FATAL` or `CRITICAL` concerns. Descriptive application differences do not count.

| Candidate claim | Nearest established object | Candidate difference | Is the difference theorem-bearing? | Verdict |
|---|---|---|---|---|
| Rejected actions create missing collision/energy labels | Selective labels; logged bandit feedback | Physical UAV action and multiple outcomes | No | Covered |
| Rejected-action boundary is non-identified | No-overlap OPE and partial identification | Filter intentionally creates zero propensity | No; cause of zero support does not change observational equivalence | Covered |
| Charger commitment censors task continuation | Censored sequential outcomes; zero history-action occupancy | Absorbing one-way stop rather than administrative censoring | No; augmented state/action reduction is exact | Covered |
| Two censoring levels form a new information structure | Partial monitoring over an augmented MDP | Diagnostic separation of action and suffix mechanisms | No new identified set or minimax rate shown | Covered as representation |
| Conservative filtering causes permanent under-expansion | SafeOpt/SafeMDP safe-set expansion; pessimistic safe exploration | Learned filter creates its own support | No; Proposition 4 follows from no-update-without-sample and SafeMDP already requires structure | Covered |
| Uniform safety conflicts with identifying an unknown action | Safe exploration impossibility; risk-sensitive abstention | Explicit Le Cam bound $(1-\beta)/2$ for a catastrophic action pair | Correct but generic; embeds a two-action bandit | Incremental, not ICLR-level |
| Boundary-targeted safe probes restore learning | SafeOpt, Active Learning with Safety Constraints, ActSafe | Probe followed by charger commitment | No; return/escape action is an application constraint | Covered |
| Probe labels consume battery, collision risk, and horizon | BMDP, CMDP, BwK | Multiple physical resources coupled to data acquisition | No; state augmentation and knapsack constraints apply | Covered |
| Current safety policy determines future identifiability | Performative prediction/RL; adaptive design; safe-set expansion | Occupancy support rather than changed physical law | Potentially useful language, but no irreducible theorem found | Framing only |
| Repeated charger-origin sorties self-supervise the boundary | Episodic safe exploration/reset-free RL | Natural repeated physical reset at charger | No unique statistical assumption or rate derived | Application opportunity |

## Reviewer B Required Answer

**Why is this not ordinary selective labels, no-overlap OPE, safe active learning, or a budgeted MDP?**

It currently cannot be distinguished at theorem level:

- the static target is ordinary no-overlap;
- safe boundary expansion is ordinary safe active learning/safe exploration;
- the dynamic lower bound embeds a two-action cautious bandit;
- resource depletion is a budgeted state variable;
- commitment is an absorbing augmented action.

Therefore the required answer is not available.

## Reformulation Ledger

| Round | Core object | Outcome |
|---|---|---|
| 1 | Two-level censoring plus static observational equivalence | Novelty collapses to support mismatch |
| 2 | Uniform safety–identification lower bound plus dynamic lock-in | Sounder theorem, but reducible to safe-exploration/abstention impossibility |

No second algorithmic reformulation is justified because the closest work directly occupies the proposed repair mechanism.
