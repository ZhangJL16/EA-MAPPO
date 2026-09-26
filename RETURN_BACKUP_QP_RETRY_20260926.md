# Backup QP numerical retry, 2026-09-26

The calibrated k1 = k2 = 1.5 return controller removed the map112 ground
route's infeasible actions. In the roof route, 22 of 1282 moving states still
had one infeasible action each. Instrumentation found OSQP's 2000-iteration
limit, not a proven empty feasible region. On a representative state, solving
the same QP with 20000 iterations and polishing converged after about 2500
iterations and satisfied the original constraints.

Version 4 retries only when the first solve reports maximum iterations.
It keeps the same dynamics, obstacles, bounds, arrival rule, and strict
zero-infeasible-step criterion. Every retry solution is checked against the
original linear constraints; a failed retry remains infeasible. The number
of retries is recorded per return state.

This change was selected using map112. Maps112-114 remain calibration data;
maps115-127 remain independent confirmation data.
