# PSPS-v1 execution-only 3×8 amendment

Date: 2026-09-15

The user authorized reducing PSPS-v1 DEV execution parallelism from four world
processes × eight branch workers to three × eight after the 4×8 service reached
a 19.89 GiB peak under its 20 GiB cgroup limit. The earlier service interruption
was a Python 3.12 SIGSEGV, not an OOM kill, but the observed headroom made 4×8 an
unnecessary continuation risk.

## Preserved boundary

The 4×8 collector exited normally as `PAUSED_RESUMABLE` at 15:48:55 CST. At the
amendment boundary it had an ordered generation-20 checkpoint, 22 complete
staged records, 6,377 partial replicate JSON files, and 8,285,985 committed
branch policy steps. A snapshot records the SHA-256 of all 6,444 pre-amendment
immutable evidence files. Its inventory hash is
`sha256:b5ddf5ede6c52e5fa1e2de523e1517b1bba0ec52b7c18d5a9fe2a65a2b0be609`.

The amendment record is
`sha256:222c85c0358fd974cae76aefb1f90473b907ce2f763859116f626ff0e43ed191`.
It is locally hash-bound to the original preaccess record, freeze receipt,
scientific manifest, access-chain head, evidence snapshot, original collector,
3×8 wrapper, and amendment validator. It was not separately externally
anchored.

## Scope

Only `world_processes` changes from 4 to 3. `workers_per_world=8` and
`max_worlds_per_child=1` remain fixed. Physical worlds, seeds, split membership,
anchor steps, legal feature state, paired 64-replicate C/R CRN grid, branch
guard, collision semantics, utility, Gate O/D estimators and thresholds, and all
stopping rules remain unchanged. The original manifest and all pre-amendment
evidence remain byte-identical.

The resume remains DEV-only. Automatic analysis, CONFIRM access, and training
remain disabled; METHOD-TRAIN remains unauthorized.

## Startup health

The amended service `psps-v1-dev-resume-3x8.service` started at 15:52:41 CST.
It committed generation 21 at 15:52:54 from the preserved partial state. The
new checkpoint contains 9,829,783 cumulative branch policy steps and valid
2-anchor × 64-replicate arrays. Post-resume validation confirmed all 6,444
pre-amendment evidence files remain byte-identical. The service reported three
active world processes, eight branch workers per world, approximately 4.6 GiB
current/peak memory, zero swap, and no new kernel OOM or SIGSEGV event at the
health handoff. Monitoring stopped after this first checkpoint.
