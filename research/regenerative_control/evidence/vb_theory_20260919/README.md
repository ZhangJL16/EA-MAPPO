# Completed P2-A2 and theory-only certificate

The original high-level learning story is KILLED at the fixed >=.98 threshold.
See [closure report](../../../../docs/VIABILITY_BOUNDARY_THEORY_20260919.md) and
[derivation package](../../../../DERIVATION_PACKAGE.md).

- `result.json`, `structural_summary.json`, `status.json`: completed original P2-A2.
- `completed_nodes.jsonl.gz`: all46 original hashed node envelopes, including
  the10 candidate-subtree nodes. Restore individual files with JSON indent2 and
  a final newline to preserve original serialized receipt bytes.
- `completed_root.json.gz`: exact original root receipt bytes.
- `contract.json`: original experiment contract, NOT a new experiment.
- `certificate.json`: new deterministic algebra on only the36 cycle nodes;
  no oracle result is an input, no simulator/actor imports or sampling.
  It rechecks recorded admissibility, transition times, probabilities and hashes.
- `checks.json`: existing-result consistency and exact rational analytic example.
- `source_receipt_sha256.json`: fingerprints of the original completed result files.

VB-only certificate: ratio to optimum >=.9861853328; exact known ratio=.9913422775.
This supports the confirmed KILL, not renewed high-level learning.
The conditional theory uses standard Wald/renewal/stopping tools. Novelty is
unestablished; the derivation package separates proofs, assumptions and nonclaims.
No new UAV run, neural training, distribution change or multi-agent experiment.
