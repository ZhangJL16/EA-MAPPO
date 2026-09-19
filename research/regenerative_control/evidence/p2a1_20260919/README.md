# P2-A1 portable evidence

See [report](../../../../docs/REGENERATIVE_CONTROL_P2A1_20260919.md).
65 distinct physical keys from 885 natural D observations; 260 core branches.
No oracle, no training, no established sign reversal or reserve sufficiency.

`raw_branches.jsonl.gz`: all 65 hashed envelopes, four original-energy branches,
three actual task→return sequences, and separately marked higher-energy/capacity
diagnostics where required. Branch state has no task RNG. `raw_sources.jsonl.gz`:
all 120 natural trajectory receipts, repeated occupancy states and failures;
replay RNG in source audit snapshots is permitted and never an oracle feature.
Local resumable pickle snapshots are retained under the artifact paths recorded
in `census.json`, with hashes; they are not included in this portable bundle.
Full map in audit outcomes is privileged ground truth and is never actor input.

`collector_source_v1.txt` matches `collector_contract_v1.json`; the V2 branch
contract fixes diagnostic capacity accounting only, with cross-version provenance.
No diagnostic failure from V1 is reused. Original physics is unchanged.

All three figures use the same 15 common-viable states. The other 50 contrasts
are undefined, not negative, zero or silently discarded from source data.
No IID/CI/significance claim. These are bounded local contrasts, not Q*.
Python SVG/PDF editable figures are the vector outputs; PNG files are 300-dpi
previews. Figure preflight has no failures; TIFF/600-dpi warnings do not apply
to this diagnostic vector delivery. Final PNGs were visually inspected.

Recompute analysis using the scripts linked by the report and local artifacts.
For a portable-only reconstruction, decompress each branch envelope to
`branches/<row.state.state_id>.json`; census and after1_reference are already here.
The plotting/analysis script needs these three data inputs only.
