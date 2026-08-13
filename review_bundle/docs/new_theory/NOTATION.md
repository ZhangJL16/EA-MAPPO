# Notation

| Symbol | Meaning |
| --- | --- |
| `x_t` | Markov state or sufficient observable history/belief used by the theory |
| `s_t`, `h_t` | physical state and observable history; `x_t=s_t` only under state sufficiency |
| `g_t`, `g_c` | current task goal and always-observable charger goal |
| `a_t` | continuous navigation action |
| `pi_theta(a|x,g)` | shared goal-conditioned base navigation policy |
| `U_C^1(x,a)` | calibrated upper bound on one-transition swept-collision hazard |
| `H_C` | optional predictive screening horizon; exact additive recursion uses `H_C=1` or disjoint macro-blocks |
| `B_{t+1}` | collision on the swept transition from decision `t` to `t+1` |
| `N_{t+1}` | near-collision event; auxiliary unless assigned its own declared budget |
| `Delta_C` | initial sortie collision-risk budget |
| `b_t` | remaining collision-risk budget before decision `t` |
| `d_t` | predictable risk spend attached to action `a_t` |
| `kappa_{C,eta}(a,d|x,b)` | charger-goal deployment kernel induced from `pi_theta(.|x,g_c)` by collision filtering and allocation |
| `eta` | immutable tuple of policy, collision model, calibration, selector, and allocation versions |
| `c_t` | measured nonnegative one-step energy expenditure |
| `T_c`, `T_B` | first charger hitting time and first collision time |
| `T_empty` | first time the filtered charger kernel has empty support |
| `Z_tilde_E^eta(x,a,d,b)` | extended cumulative energy: finite on charger arrival before failure, `+infinity` otherwise |
| `F_E^eta(z|x,a,d,b)` | defective finite-energy CDF `P(Z_tilde_E^eta<=z)` |
| `p_fail^eta` | missing finite mass `1-lim_{z->infinity} F_E^eta(z)` |
| `Z_E^eta` | finite energy return conditional on successful collision-free charger arrival, used only with explicit success mass |
| `Q_E^eta` | mean finite return when the relevant expectation exists |
| `q_tau^eta` | conditional quantile of a declared finite or extended return law |
| `U_E^eta(x,a,d,b)` | calibrated energy upper prediction bound for the declared selected-action population |
| `e_t`, `e_t^-` | true usable energy and conservative lower measurement |
| `m` | explicit reserve for excluded consumption |
| `A_DB(x,e,b)` | dual-budget admissible action-spend set |
| `z_t` | mode in `{TASK, CHARGER_COMMITTED}` |
| `tau_commit` | irreversible charger-commitment stopping time |
| `alpha_E` | allowed energy-tail violation probability on the stated calibration population |
| `beta_C`, `beta_E`, `beta_e` | validity-failure probabilities for collision, energy calibration, and battery lower bound |
| `P_eta^y` | augmented charger path law from initial state-budget `y=(x,b)` under version `eta` |
| `epsilon(y)` | local total-variation drift between induced charger kernels at `y` |
| `B_gamma` | set of augmented states with an action within `gamma` of collision admission |
| `rho_eta->eta'` | path/return-law transport debit between filter versions |
| `W` | versioned action opportunity `(x,a,d,b,eta)` |
| `Z_tilde^c(W)` | potential extended return under immediate charger commitment after `W` |
| `M` | indicator that a randomized/observed commitment probe reveals the outcome |
| `p(W)` | logged commitment-probe propensity |
| `F_c(z|W)` | target defective CDF of the commitment potential outcome |

`b_t` is a probability-accounting resource, not a physical state variable. `e_t` is a physical resource. Neither can compensate for the other.
