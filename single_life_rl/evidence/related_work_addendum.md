# Additional primary-source audit during fixed execution

This read-only literature check did not change experiment parameters, code, seeds,
selection or stopping. No intermediate performance was consulted.

- Yang, Zheng & Li (2026), [On the Equilibrium between Feasible Zone and Uncertain
  Model in Safe Exploration](https://arxiv.org/html/2602.00636v1), especially
  Sections III–IV. In a deterministic dynamics setting, their framework alternates
  uncertainty reduction and feasible-zone expansion and analyzes the limiting
  equilibrium. Therefore the qualitative coupling between knowledge and safe
  actions, or iterative safe expansion alone, is not a new contribution here.
  A noisy, decision-relevant finite-model characterization and lifetime-risk
  information complexity would require a precise comparison rather than renaming
  this coupling a refinement tree. This audit does not establish equivalence or
  non-equivalence of the full theorem statements.

- Biyik et al. (ACC 2019), [Efficient and Safe Exploration in Deterministic Markov
  Decision Processes with Unknown Transition Models](https://iliad.stanford.edu/pdfs/publications/biyik2019efficient.pdf).
  Definitions 3–4 and Remark 2 relate known transitions to recovery and expansion
  of the safe set under Lipschitz assumptions. Distinguish their deterministic
  information assumptions from the full-support noisy observation obstruction.

- Moldovan & Abbeel (ICML 2012), [Safe Exploration in Markov Decision Processes](https://icml.cc/2012/papers/838.pdf).
  Recovery/ergodicity-based safe exploration predates this project. Do not claim
  the regenerative motivation or returnability concept is new.

Current assessment: prior work presents substantial novelty threats. The narrow
proof draft is useful for checking the protocol's correctness, but is not yet a
verified original theorem contribution or a publication-level novelty clearance.
