# Backup CBF calibration, 2026-09-26

The first multi-map diagnostic run exposed QP infeasibility during some moving
state returns, usually near the ground while descending. Those trajectories
arrived without contact, but the zero-infeasible-step evidence rule rejected
them. The user chose to retain this strict rule and recalibrate the backup
controller, then rerun.

Only backup return simulations use k1 = k2 = 1.5; the qualified outbound
navigation controller remains unchanged. This value was selected after probes
on maps 112-114. Those maps are calibration data. Maps 115-127 are the
independent confirmation cohort for this setting. All 16 maps will be rerun
for complete descriptive coverage. No failed return or infeasible action is
relabelled as valid.
