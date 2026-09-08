# Idea-Grounding Packet

## Scope And Evidence Boundary

- Topic: collision safety as fixed infrastructure for an energy-safe return-to-charge paper.
- Search date: 2026-09-01.
- Source-supported: HOCBF addresses high-relative-degree constraints; sampled-data variants address inter-sample safety; Composite CBF provides recent dense-LiDAR quadrotor evidence; predictive and learned filters address different uncertainty/model regimes.
- Repository-supported: ordinary frozen R3 records collisions; a separate controlled prototype reports zero collisions for sampled-data HOCBF under exact conditions but failures under dropout/combined stress.
- Optimizer inference: sampled-data robust HOCBF is the best fit because it aligns with the current known model and already implemented timing/input contract while minimizing scope drift.
- Unknown: whether a faithfully integrated sampled-data layer passes the full frozen navigation and energy protocol.

## Evidence Cards

| Source | Supported observation | Reported limitation / condition | Mechanism primitive | Protocol anchor | Transfer condition | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| Xiao & Belta 2022 | HOCBF handles high-relative-degree safety constraints | requires stated dynamics and feasibility assumptions | recursive barrier derivatives | double-integrator obstacle avoidance | correct model and feasible input set | direct |
| Breeden et al. 2022 | sampled-data conditions can preserve forward invariance | requires fixed sample/hold model and computable bounds | inter-sample compensation | piecewise-constant QP controller | valid residual bound at project timestep | direct |
| Oruganti et al. 2024 | bounded disturbance and measurement error can enter robust sampled-data barriers | known uncertainty bounds required | doubly robust high-order barrier | noisy sampled obstacle avoidance | calibrated bounds and no unmodeled dropout | direct |
| Harms et al. 2025 | Composite CBF scales dense LiDAR constraints on quadrotors | faithful full-order formulation and feasibility conditions required | soft composition plus QP | indoor/outdoor onboard LiDAR | reimplementation matches published dynamics/controller | direct |
| Wabersich & Zeilinger 2021 | predictive filters use backup trajectories and terminal safety | online MPC/model/terminal-set burden | receding-horizon safety verification | nonlinear constrained learning control | reliable model and runtime budget | direct |
| Lavanakul et al. 2024 | learned input half-spaces can modularize black-box safety | learned/verified invariant data or RL training required | discriminating hyperplane | black-box dynamics | genuine model uncertainty justifies learning | direct |
| R3 platform contract | selected platform does not pass collision/navigation gate | conditional energy study authorization is not safety evidence | ordinary HOCBF execution | 500 fixed tasks | cannot support hard-safety wording | direct |
| Repository safety benchmark | sampled-data HOCBF removes observed collisions in exact controlled prototype | nonformal; unsafe fallbacks and sensor-dropout failure remain | inter-sample robust HOCBF | 125-rollout controlled benchmark and 30-rollout stress diagnosis | formal multi-seed integration audit | direct |

## Cross-Source Relations

| Source pair / cluster | Relation | Open gap or conflict | Why it matters | Evidence needed next |
| --- | --- | --- | --- | --- |
| ordinary HOCBF vs sampled-data CBF | leaves-open | continuous-time constraint does not cover sample-and-hold execution | explains observed gap between source claim and deployed behavior | exact inter-sample/fallback audit |
| sampled-data HOCBF vs Composite CBF | conflicts-with on priority, not correctness | timing robustness versus dense-constraint scalability | current scene needs timing correctness more than 10k-point scale | matched implementation benchmark |
| CBF family vs predictive filter | depends-on | local invariance versus backup-trajectory feasibility | predictive method adds scope and compute | runtime/fairness contract if included |
| known-model filters vs learned filters | depends-on | learned filter useful only under genuine model/barrier uncertainty | prevents unnecessary collision-method novelty | declare environment uncertainty model |
| collision filter vs Resource-to-Go | supports | filter interventions alter executed occupancy and energy requirement | collision layer must be frozen before energy study | recalibration plus Oracle Gate |

## Idea Constraints

- Already covered central claims: collision avoidance with HOCBF/CBF, sampled-data robustness, dense-LiDAR composition, and modular safety filtering.
- Transferable mechanism primitive: sampled-data residual compensation plus bounded-input QP and exact swept-clearance validation.
- Protocol suitable for direct comparison: no filter; ordinary HOCBF; sampled-data robust HOCBF; faithful Composite CBF only if implemented.
- Stale/overcrowded route: claiming another HOCBF variant as the ICLR paper's novelty.
- Minimum viable research question: after freezing a collision layer that meets its declared safety contract, can executed-interface Resource-to-Go reduce energy stranding at matched throughput under composition shift?

