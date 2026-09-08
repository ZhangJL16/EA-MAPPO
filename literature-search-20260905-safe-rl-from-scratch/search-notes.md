# Search and provenance notes — 2026-09-05

## Question

Which established safe/CMDP algorithm is an appropriate minimal from-scratch baseline for continuous-action, partial-LiDAR UAV navigation after reach-avoid fine-tuning destroyed goal reachability?

## Search coverage

Public searches covered CPO, FOCOPS, PID Lagrangian, CRPO, CUP, CVPO, Sauté RL, Safety-Gymnasium, CAL/off-policy primal-dual safety, FCSRL, safety curricula, C-TRPO, ActSafe and SafeMPO. Queries combined algorithm names with official proceedings and author implementations. No local research data, private code or checkpoints were uploaded. Read-only access to public GitHub raw files was used to inspect implementation details.

Seventeen candidates screened: fourteen retained in papers.md; three additional discovery records:

- PCPO / Projection-Based Constrained Policy Optimization: older projection family already covered by CPO/CUP; not prioritized for deeper reading in this pass. [Author project](https://sites.google.com/view/iclr2020-pcpo). Do not describe this as a full-paper review.
- Safe, Trust Region Policy Optimization for Constrained Reinforcement Learning (sTRPO): a NeurIPS-related event record was found, but main-track versus workshop status was not resolved. Excluded from verified top-conference claims. [Event record](https://neurips.cc/virtual/2025/loc/san-diego/136134).
- FuzRL: official NeurIPS 2025 poster discovery; not deeply inspected because robustness-specific components were not the first diagnostic priority. [Poster](https://neurips.cc/virtual/2025/poster/117895). No technical efficacy claim made here.

## Screening criteria

Continuous actions; compatible with learning a deployable neural policy; understandable physical cost semantics; behavior under initially unsafe policies; handling of value-estimation error; reproducible source implementation; reasonable simulation/update compute; theory assumptions that can be stated without claiming unproved UAV invariance.

MDPI material was not used. No citation-count or venue-rank proxy was used as a claim of project suitability. The shortlist scores are provisional and limited by reading depth. The review does not establish which algorithm is empirically best in this environment.

## Source/access limitations

The raw PMLR C-TRPO PDF returned unsupported application/octet-stream in the browser. Read the author-institution repository mirror instead: https://pure.mpg.de/rest/items/item_3680903_2/component/file_3681565/content . The official proceedings page independently verified the venue. This mirror is a later arXiv version and is identified as such; exact proof numbering can differ from the conference PDF.

SafeMPO's ICLR 2026 proceedings entry was verified. Its author implementation was not located in this search; absence from this search is not evidence that none exists. Other source repositories are moving branches; implementation work must pin a revision or content hash before claiming reproducibility.

## Local diagnostic provenance

Read existing RESULT.json and ten saved model checkpoints in artifacts/r3_reach_avoid_pair_50k_each_20260905_v1. No training or environment rollout was performed during the value audit. Load hard/checkpoints/00010000/replay.pkl; use NumPy default_rng(930001), draw 1,024 replay row indices uniformly from buffer.size(), then 1,024 environment indices uniformly from buffer.n_envs. Use the exact same observations, stored potentials and replay actions for every checkpoint.

For each arm at 10k,20k,30k,40k,50k compute min(Q1,Q2)(obs, deterministic_actor(obs)) + stored_phi. Also inspect values for replay actions. This tests a necessary value-range property on a fixed old-state sample; it is NOT success-probability calibration, a causal intervention, or an independent-seed statistical test. Values and interpretation are recorded in docs/SAFE_NAVIGATION_FROM_SCRATCH_REDESIGN.md.

## Handoff

Next deliverable is the isolated implementation and contract tests described in the design document, followed by a measured pilot. Do not auto-resume the failed pair or auto-scale from finite losses alone. No experiment was launched as part of this literature review.
