# Primary-source audit before protocol freeze

Checked 2026-09-20. This is a focused threat audit, not a novelty clearance.

1. Nguyen & Cheng (ICML 2023), [Provable Reset-free Reinforcement Learning by
   No-Regret Reduction](https://proceedings.mlr.press/v202/nguyen23b.html).
   Their abstract states a reduction to a two-player game and sublinear performance
   regret plus sublinear reset count. This is not a zero-catastrophe guarantee.

2. Prajapat et al., [arXiv:2509.16650v2](https://arxiv.org/html/2509.16650v2), revised
   2026-02-21, currently titled *Safe and Near-Optimal Control with Online Dynamics
   Learning*. The supplied older title should not be used without a version.
   Theorem 2 gives safety and reward guarantees with probability at least 1-delta,
   under stated assumptions. Theorem 3 / Appendix D supplies a dynamics-learning
   sample-complexity lower bound using GP optimization. Thus “we have a lower
   bound, they only have an algorithm” is not a defensible distinction. A finite
   model, decision-relevant adaptive safety-unlocking characterization would need
   a precise theorem comparison; none is certified novel here.

3. Begzadic et al. (L4DC 2025), [Back to Base](https://proceedings.mlr.press/v283/begzadic25a.html).
   Reach-avoid safety filtering enables return to a desired region and autonomous
   resets. Compare operational recovery assumptions rather than a broad claim of
   first safe reset-free learning.

4. Degenne & Koolen (NeurIPS 2019), [Pure Exploration with Multiple Correct
   Answers](https://proceedings.neurips.cc/paper/2019/hash/60cb558c40e4f18479664069d9642d5a-Abstract.html).
   Relevant to answer-dependent alternatives, stopping criteria and matching
   identification complexity. Policy identification rather than complete dynamics
   identification is not by itself a novelty claim.

5. [Safe Exploration in Markov Decision Processes](https://arxiv.org/abs/1205.4810)
   and [Information-Theoretic Safe Exploration with Gaussian Processes](https://arxiv.org/abs/2212.04914)
   are additional primary-source leads. Full theorem comparisons remain outstanding;
   do not claim this search excluded prior characterization results.

Mathematical counterexamples in PRE_FREEZE_AUDIT.md are direct derivations from
the proposed protocol, not assertions attributed to these sources.
