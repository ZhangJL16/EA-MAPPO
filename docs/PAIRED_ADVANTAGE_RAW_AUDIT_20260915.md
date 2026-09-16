# Paired-advantage DEV v3 raw-evidence audit

Date: 2026-09-15. Scope: the completed 96-world DEV split in
`artifacts/paired_advantage_identification_20260914_v3/pai_dev_raw_v1`.

## Verdict

The raw audit passed. It checked the externally anchored contract and receipt,
the complete ordered checkpoint, every staged ticket, every bound JSON/NPZ
hash, every paired disturbance seed, every frozen utility calculation, the
unified collision accounting and PAI-v3 collision-free derivation, and every
numeric array used to construct the legal latest/full features. CONFIRM access
remains zero.

## Counts

- Worlds/tickets/DEV access events: 96/96/96.
- Included anchors: 192, with 96 at step 256 and 96 at step 768.
- Paired branch records: 24,576 (64 paired CRN replicates for each C/R action
  and anchor).
- Total branch policy steps: 30,154,281.
- Legal features: 192 latest-frame rows of dimension 416 and 192 full-history
  rows of dimension 2,912; every value is finite.
- Unified collisions: 2 branches with any contact, 2,719 total policy-step
  contacts. Collision-free is derived only from total collision count equal to
  zero, as required by the locked protocol.
- Operational failures/returns: 1,938/22,638 branch outcomes.

These counts are integrity summaries, not Gate H/I conclusions.

## Audit implementation note

The first audit invocation reused a legacy CMI helper whose collision-free
condition included successful return. That invocation stopped without writing
an artifact. The successful audit separately enforced the frozen PAI-v3 rule
`collision_free_arrival == (collision_count == 0)` and used a private copy only
to reuse the legacy helper's remaining exact-schema, utility, numeric-bound and
disturbance-bound checks. No experimental record was changed.

The audit also exposed an analysis-execution compatibility bug: the frozen PAI
analyzer imports a loader that hard-codes the predecessor CMI-v3 contract ID.
Raw evidence is unaffected, but Gate analysis must not run until a transparent,
hash-linked loader-only repair is frozen and checked.
