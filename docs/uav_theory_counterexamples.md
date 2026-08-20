# Counterexample Ledger

| Candidate claim | Counterexample | Result |
|---|---|---|
| Current \(\rho\ge0\) implies next-sample feasibility | \(x^+=x+0.2u\), \(u\in[-1,1]\), row \(u\ge x\); \(x=0.9,u=1\) gives \(\rho:0.1\to-0.1\) | Claim false |
| Adding \(\dot\rho+\alpha(\rho)\ge0\) cannot hurt feasibility | Same system at \(x=0.75\): original row \(u\ge0.75\), margin-CBF row \(u\le0.25\) | Circular construction; claim false |
| Per-obstacle stopping-distance barriers imply a joint braking action | Two nearby obstacles can have opposing outward normals; each scalar stopping condition can hold while their required braking half-spaces have empty intersection under bounded acceleration | Single-obstacle braking test is not a multi-obstacle recursive-feasibility certificate |
| Safe sample endpoints imply safe ZOH interval | \(r_0=(-2,0,0)\), \(v=(8,0,0)\), \(u=0\), \(T=0.5\), radius \(0.5\); both endpoints have barrier 3.75 but \(h(0.25)=-0.25\) | Endpoint-only theorem false |
| Lower instantaneous acceleration energy implies lower mission energy | Stage cost \(0.6+u^2\); one \(u=1\) step costs 1.6, two \(u=0.5\) steps cost 1.7 | Trajectory dominance false |
| Deployed MC network admits a finite global gradient-Lipschitz constant | Across a ReLU kink, required descent-lemma constant scales as \(1/(2\epsilon)\) and diverges | Global deterministic Taylor certificate unavailable |
| Conformal upper coverage is a deterministic Energy-to-Go certificate | Exchangeable marginal coverage permits violations on individual states and does not bound Bellman residuals | Deterministic claim invalid |
| Max-margin feasibility diagnostic is a new recursive-feasibility theorem | Xiao et al. derive feasibility CBF constraints; Breeden--Panagou construct viability domains; backup-CBF work proves uniform feasibility | Generic claim covered by prior art |
| Lexicographic pointwise energy minimization preserves useful navigation | On 334 adversarial scenarios it succeeds on only 69, has path ratio 3.456, and uses 15.956 energy versus 4.847 for standard sampled-data HOCBF | Practical candidate falsified |
| Reducing infeasible-step count solves recursive feasibility | Lexicographic filter reduces 305 baseline infeasible steps to 15 but does not reach zero and still uses uncertified fallback | Required theorem conclusion not achieved |

Executable evidence:

- `artifacts/uav_theory_candidate_validation_20260819/report_100k_all_final.json`;
- `artifacts/uav_hard_feasibility_set_20260819_v3/summary.json`;
- `artifacts/uav_hard_feasibility_set_20260819_v3/hard_states.jsonl`;
- `artifacts/uav_theory_adversarial_1002_20260819/summary.json`;
- `artifacts/uav_theory_adversarial_1002_20260819/rollouts.csv`.
