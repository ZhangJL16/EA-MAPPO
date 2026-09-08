# Fixed user research constraints

The user explicitly froze the collision mechanism on 2026-09-05. Do not change
the following in new research, ablations, acceleration, or reward tuning unless
the user explicitly revises this instruction:

- Contact alone must NOT end a navigation episode. Repair position and continue
  the same task; zero all velocity components at the collision repair instant.
- Raw boundary penalty and raw obstacle penalty each equal -1.2 for a contact
  following a clean policy step (also the first contact after reset).
- If the preceding policy step had ANY contact, each applicable penalty is
  -1.2 * 0.35 = -0.42. This is a fixed discount, not geometric decay.
- One clean policy step resets the next contact penalty to -1.2. Boundary and
  obstacle penalties add if both happen in the same policy step.
- Safety cost is 1 if a policy step has any contact, otherwise 0; never discount
  this cost with the reward. Record ONE unified collision count; do not expose
  separate boundary/obstacle collision counts in new experiment statistics.
  Count at most once per policy step, including simultaneous or substep contacts.
  Derive collision-free arrival from total collision count == 0.
- Do not silently replace or overwrite historical first-contact experiments.
  Those are legacy evidence, NOT the authorized environment for new research.
- Keep experiments resumable. Check startup health then wait for the user;
  do not monitor to completion or automatically promote to long training.

Contract and current research plan: docs/LOCKED_COLLISION_RECOVERY_PROTOCOL.md.

# User-specified deployment observations

The user clarified the observation assumptions on 2026-09-06:

- The charging-station location is known and remaining battery is available
  online. These are legitimate inputs for energy-aware decisions.
- Local obstacle layout is available only through LiDAR. Do not supply actors
  or deployable energy predictors with simulator obstacle centers/radii,
  full obstacle layouts, or exact map-derived clearance unavailable from the
  sensor. Geometry features must be derived from the available LiDAR data.
- Simulator geometry may still be used for physics, ground-truth evaluation,
  and explicitly labeled privileged-information diagnostic controls.
- The user questioned the necessity of the 1,024 LiDAR hit flags; this does not
  itself authorize changing an ongoing experiment or removing distance rays.

# Fixed experiment-startup constraint

The user froze the experiment-startup workflow on 2026-09-06.  Do not build a
long sequence of preliminary gates before an experiment unless the user
explicitly requests it:

- Run only the checks needed to prevent an invalid or unrecoverable run:
  focused tests/static checks for changed code, one tiny execution smoke when
  the execution path is new, and one first-checkpoint startup-health check.
- Test pause/resume only when checkpoint or resume code changed.  Reuse a
  previously passing resume result when that path is unchanged.
- Do not require tens or hundreds of exploratory/evaluation gates before
  starting the requested experiment.  Questions about performance belong in
  the requested experiment or its post-run analysis, not in a growing pre-gate
  chain.
- Any additional pre-experiment gate must address a concrete identified risk,
  have a stated stopping rule, and require the user's explicit approval.
- After startup health is confirmed, hand the resumable run back to the user;
  do not monitor it to completion or auto-promote it.
