# Frozen inference-only return-action diagnostic

The paired 2D CPU training and all validation jobs finished before this
diagnostic was specified. The diagnostic does not select a new policy or change
the original validation results. It uses final control and feedback checkpoints
from the vectorized seed 101 panel and the independent seeds 102–106 panel,
all on the already-used validation maps 32–39. The untouched holdout maps
56–71 remain unused.

For each of the 48 control-policy/map combinations, replay the exact archived
shielded control trajectory. Require every chosen action, completed-target
count, forced-return count, charging count and terminal time to match the
archive. At each flight decision, query both frozen final policies on that
identical observation. Record their probability of voluntary return (action
9), the feedback policy's deterministic action, energy fraction and whether
the control action immediately caused a shield return takeover. The feedback
policy does not drive the replayed plant. Run 16 CPU processes with one math
thread each. No simulation parameters, reward, model weights or environment
files change.

Report the two panels separately. Compare return probabilities at all control
flight states and just before forced takeover, and count how often the feedback
policy would choose return in those same states. These are diagnostic
counterfactual logits, not realized feedback-policy outcomes or a causal
estimate of improved safety.

Run from the repository root with CPU-thread limits:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../.venv-2d-cpu/bin/python -m analysis2d.audit_return_logits_cpu
```
