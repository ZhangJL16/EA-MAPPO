# Return matrix execution amendment, 2026-09-26

After the 32 navigation flights were qualified, the user selected full execution
of all 16 held-out maps, with at most 16 worker processes. This changes only
the computational schedule in the frozen multi-map protocol. The maps, routes,
energy profiles, SOC grid, and outcome definitions remain as specified there.

The return accounting engine is version 2. It uses an array implementation of
the same per-step energy equations and is cross-checked against version 1
before the matrix starts. Each worker is capped at 2 GiB address space, the
number of concurrent workers at 16, and the matrix output at 4 GiB. Results
are checkpointed after every route and can be resumed without changing inputs.
