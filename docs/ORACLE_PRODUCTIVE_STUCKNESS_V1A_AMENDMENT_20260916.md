# Oracle Productive-Stuckness v1a — Prospective Analysis/Execution Amendment

## Status and timing

This amendment was created after the v1 integrity-only startup audit and before
any tasks/hour value, method comparison, Gate O/M/N result, or CONFIRM outcome
was inspected. The remaining 23 DEV worlds were not accessed. It is therefore
a prospective correction of the measurement and inference contract, not an
outcome-dependent change.

The v1 contract, sources, smoke output, and first-DEV startup output are retained
unchanged. All v1 outcome files are designated **integrity/startup only** and
are excluded from every v1a estimate. The first DEV world is rerun under v1a and
contributes exactly once if the complete v1a grid is later authorized.

## Unchanged scientific intervention contract

M0--M4 decision logic, physical worlds, seeds, pairing, SOC40, 4000-step timeout
and oracle horizon, 256-step no-progress window, 7200-second horizon, frozen SAC
navigator, collision semantics, recharge mechanics, and plant physics are
unchanged. M3 remains an experiment-only action outside Continue/Return. M4
remains a completion-within-4000 privileged oracle, not a perfect oracle.

## Amendment A — completion timestamps and common-alive accounting

The runner records the simulated timestamp of every completed task. Counts at
an analysis horizon use only timestamps not exceeding that horizon. For world
\(w\), define

\[
T_w^{\mathrm{common}}=\min(T^{\mathrm{alive}}_{w,0},T^{\mathrm{alive}}_{w,4}),
\]

\[
\Delta N_w^{\mathrm{common}}
=N_{w,4}(T_w^{\mathrm{common}})-N_{w,0}(T_w^{\mathrm{common}}),
\quad
\Delta N_w^{\mathrm{total}}=N_{w,4}(7200)-N_{w,0}(7200).
\]

The frozen survival-explanation fraction is

\[
F_{\mathrm{survival}}=
\frac{\sum_w[\Delta N_w^{\mathrm{total}}-
\Delta N_w^{\mathrm{common}}]_+}
{\sum_w[\Delta N_w^{\mathrm{total}}]_+}.
\]

Gate M requires \(F_{\mathrm{survival}}\le 0.5\), in addition to the existing
20% unproductive-dwell reduction and integrity/collision checks. A zero positive
total-gain denominator fails Gate M rather than being assigned a favorable value.

## Amendment B — simultaneous Gate N contrasts

The primary simultaneous max-\(t\) family contains exactly four paired,
whole-world contrasts:

\[
M4-M0,\quad M4-M1,\quad M4-M2,\quad M4-M3.
\]

Gate O uses the first contrast plus its previously frozen magnitude, sign, and
intervention-frequency requirements. Conditional on O and M, Gate N requires
strictly positive simultaneous 95% lower bounds for all three M4-versus-simple
contrasts. Define

\[
G_{\mathrm{simple}}=\max_{j\in\{1,2,3\}}(\bar Y_j-\bar Y_0),
\qquad
G_{\mathrm{oracle}}=\bar Y_4-\bar Y_0.
\]

Gate N additionally requires \(G_{\mathrm{simple}}/G_{\mathrm{oracle}}<0.8\).
LOO cross-fitted selection is retained only as descriptive secondary output and
does not enter a gate or confidence interval.

## Amendment C — execution path

The v1a runner accepts a resumable `--stop-after-worlds` value rather than
hard-coding one DEV world. Creation of that path does not authorize its use.
Current authorization remains one excluded 120-second smoke world followed by
one complete first-DEV five-method startup grid. Continuing the other 23 worlds
requires a new explicit user authorization.

