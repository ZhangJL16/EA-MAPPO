# Recurrence-tail Lean checks

This minimal project machine-checks only the deterministic tail-sequence core
used by `docs/RECURRENCE_TAIL_DERIVATION_PACKAGE_20260916.md`.

It does **not** yet formalize the probability tail-sum identity, defective
return-time laws, controlled policies, Markov recurrence, Wasserstein
identities, or renewal-reward conclusions.

Build with:

```bash
lake update
lake exe cache get
lake build
```

The toolchain and mathlib revision are pinned in `lean-toolchain` and
`lake-manifest.json`.
