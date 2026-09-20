# Safe Identifiability: When Is Reset-Free Reinforcement Learning Possible?

Separately scoped research project. The user has selected strict zero-catastrophe
impossibility plus lifetime delta-safe learning (primary delta=.05), with no
statistical budget reset at physical regeneration. Deterministic conditional
safety; observation noise only. Existing persistent-UAV experiments are unchanged.

- [Frozen protocol specification](SINGLE_LIFE_PROTOCOL_V1.md)
- [Manuscript skeleton, finite-menu characterization and staged-family proofs](paper/THEORY.md)
- [Initial mathematical audit](theory/PRE_FREEZE_AUDIT.md): retained historical objections,
  resolved by the revised scope rather than suppressed.
- [Related work and remaining novelty questions](theory/RELATED_WORK.md)
- [Reducibility audit: restricted separation, controlled-sensing encoding, unresolved novelty](theory/REDUCIBILITY_AUDIT_20260921.md)
- `configs/protocol_v1.json`: exact parameters and seeds.
- `scripts/run_all.py`: freeze, resumable 16-worker execution, complete-only collect/analyze.

The iff characterization is restricted to a finite reusable experiment abstraction.
Matching bounds are for a staged Bernoulli testing benchmark, not arbitrary MDPs.
UAV uses an explicit semi-empirical mission library with observable clock; its
multiplier may be identified in one cycle. No Level-5 or novelty claim is made.
