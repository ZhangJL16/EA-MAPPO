# Feedback Protocol Lab: first-batch implementation

Date: 2026-09-19. Status: A0/A1 and executable independent comparison skeleton complete. No method-performance finding is claimed.

## Repository and evidence

Local HEAD equals the audited `fff2b026ca3e4ea6b68cacc3964971b0749bfeb3`. Initially the tracked tree was clean; `expert_advice/` was untracked. No pull/reset was performed. The earlier user request to keep master only is retained: isolation is by `research/feedback_protocol/`, not an extra Git branch. This batch is uncommitted and has not been pushed.

The supplied read-only inventory script ran with system Python and hardware metadata enabled. Original report: `/home/zjl/ea_mappo_local_handoff_20260919.json`. Versioned metadata extracts:

- [Repository baseline](../research/feedback_protocol/provenance/repository_baseline.json).
- [Evidence existence inventory](../research/feedback_protocol/provenance/evidence_inventory.json).

All 26 allowlisted evidence paths exist: six manifest/prediction/health/analysis/certificate/audit paths and 20 checkpoints. Only metadata was read; no artifact-content/hash-integrity or scientific certificate revalidation is implied. The original script excludes untracked paths, so the initially untracked advice directory is separately recorded here. No CONFIRM directory was accessed.

Hardware observed: 24 logical CPUs; approximately 23 GiB RAM visible to this environment; NVIDIA RTX 5060 Laptop GPU, 8151 MiB reported VRAM. Package metadata in the inventory refers to the interpreter used, not automatically the old virtual environment. No GPU was used and no package was installed. Human-time/GPU-hour budgets remain unspecified and are unnecessary for this bounded implementation batch.

## Implemented delivery

The [contract](../research/feedback_protocol/CONTRACT.md) fixes reset-terminal finite-budget cumulative reward, complete committed protocols, independent time/energy accounting and batch-end feedback. [README](../research/feedback_protocol/README.md) supplies executable commands.

| File/module | Delivered behavior |
|---|---|
| `problem.py`, `configs/fixture.json` | Immutable public graph/model/prior contract; rational probabilities; parameterized fixture and capacity/bundling overrides |
| `protocols.py` | Budget-bounded complete protocol enumeration and pre-reload resource legality |
| `belief.py` | Released history, public planner state, exact finite-family posterior and joint outcome law |
| `environment.py` | Evaluator-owned truth and RNG; operation/resource event log; feedback released on reset return |
| `teachers/exact_bayes.py` | Rational finite-budget Bayes recursion; explicit computation-limit failures; separate known-model oracle |
| `policies/channel_cover.py` | Informative-channel coverage followed by posterior reward/time; independent of Bayes recursion and frozen allocation code |
| `cli.py` | One bounded debug episode with source/config hashes, public decision states and evaluator output |
| `tests/test_contract.py` | 13 focused tests, including old committed-solver numerical regression |

There are no runtime third-party dependencies. Production package imports only stdlib and itself; the test suite reads the existing stdlib-only `audit_resource_separation_bridge.py` as a reference. No frozen implementation was edited. Source allowlist hashes other than explicitly updated navigation files were verified unchanged after implementation.

## Validation actually executed

`python3 -m unittest discover -s research/feedback_protocol/tests -v`: **13/13 passed**.

Coverage: four fixture cells and off-control, resource debit before reload, duration different from energy, reset terminal and protocol closure, batch feedback, exact Bayes posterior, truth absent from policy state, shared-channel coverage without all-path initialization, known-model/Bayes distinction, arbitrary hypothesis/channel counts, zero-energy cycles bounded by time, invalid input rejection, explicit solver-limit exceptions, budget-conditioned choice, and exact agreement with the frozen committed-reference Bayes value at T=6 for both capacities.

Two new debug-only smoke episodes ran at budget 6, evaluator truth 0, noise seed 0:

- Exact Bayes: two complete decisions, consumed time 6, reset terminal. `/home/zjl/fpl_debug_bayes_20260919.json`.
- Channel coverage: three complete decisions, consumed time 6, reset terminal. `/home/zjl/fpl_debug_cover_20260919.json`.

These check the new execution entry points. No reward ordering, superiority, confidence interval or minimax claim is inferred. `git diff --check` passed.

## Information boundary and remaining scope

Policy `select` takes only PublicProblem at construction and PlannerState at invocation. The CLI owns PrivateTruth; oracle evaluation happens after execution. The same public history gives the same policy input regardless of evaluator label. This is API discipline rather than a security sandbox.

The new primary comparator is the known-model finite-budget value with matching committed/reset-terminal constraints. The old arbitrary-terminal primitive lower certificate and T*rho metric are not relabeled as new results. The catalogue retains BA, while the old calibration collapsed that feedback-equivalent committed route type. This distinction is documented and checked against the old committed reference, not the primitive lower-bound solver.

This is a small-instance enumeration-based comparison foundation. It does not complete general task generation, dataset splitting, bounded approximate search, minimax solving, teacher datasets, neural decoding, resumable pending-protocol checkpoints, DAD/Step-DAD adapters, or layered statistical evaluation. Those remain later A/B/C work. The concrete profiling plan is in the contract; no long experiment or training was started.
