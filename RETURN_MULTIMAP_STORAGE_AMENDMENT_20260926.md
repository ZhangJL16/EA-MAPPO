# Return matrix storage amendment, 2026-09-26

The initial matrix manifest capped output at 4 GiB. During the frozen v4
run, complete compressed return traces from long routes indicated that this
cap could stop the 16-map matrix before completion. The user chose to retain
every return trace and raise the budget. The revised cap is 8 GiB for this
matrix. This is a resource ceiling, not a change to maps, states, controls,
energy profiles, SOC grid, or outcome rules.

The matrix manifest records both the initial and revised budget and the
SHA-256 of this amendment. The run is resumed through its existing result
and trace hash checks. No previously completed result is discarded.
