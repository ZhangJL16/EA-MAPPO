# PSPS-v1 best observed DEV world return GIF

Date: 2026-09-15

## Purpose

This post-hoc qualitative visualization answers whether the frozen SAC can
execute a real mid-task return in the PSPS environment. It is not Gate O/D
evidence and does not claim cross-world generalization.

The task schedule is an unbounded keyed stream: completing a task immediately
creates the next task. Therefore the environment has no natural “all tasks are
finished” state. The truthful visualized unit is one registered battery-sortie
branch from reset through a mid-task R commitment and then to a real return,
energy exhaustion, or registered deadline.

## Selection and replay

Only complete, already accessed PSPS DEV records available at selection time
were considered. Worlds were ranked by mean frozen C/R utility, then R return
rate, C return rate, and mean R duration. Seed `1137000014`, identity
`sha256:a51b837e7ee2cf234f3c499eefd284cd4aa225bf315d70660d2f3644acd6407c`,
was selected. Its recorded C and R return rates were both 100%; among tied
worlds it had the shortest mean R duration.

At the registered step-768 anchor, successful R replicate 27 was selected by
median—not minimum—return duration. The replay uses the same frozen SAC, world,
anchor, action, and disturbance seed `9100246209313219006`. Its outcome exactly
reproduced the saved formal branch.

## Observed trajectory

- Reset at the charger with energy 378.7263 synthetic units.
- Completed task 1 at policy step 360; the keyed stream immediately issued task 2.
- At step 768, while still in TASK mode and 632.94 distance units from task 2,
  R committed to the charger without teleporting or resetting velocity.
- Returned in 210 further policy steps, for 978 visualized steps total.
- Final pre-service charger distance was 4.46, inside the real arrival tolerance.
- Arrival energy before recharge was 337.6823; service then restored 378.7263.
- Unified collision count was zero.

The GIF shows the top-down path and obstacles, task/return path colors, active
goal, UAV, battery trace, return-commit marker, and distance-to-charger trace.
It contains 181 frames at 1100×616 pixels. Its hash is
`sha256:6993add2e1e596031b11aa1b0e198f042bf90d870748244d89f6b160479357a3`.

## Limits

The R commitment is the registered experimental intervention at step 768, not
an output of a PSPS selector. Gate O/D has not run, so no PSPS-driven deployment
policy exists yet. The world was selected post hoc for visualization and must
not be used as an unbiased performance estimate. CONFIRM was not accessed.
