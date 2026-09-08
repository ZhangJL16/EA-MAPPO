# Search Notes

## Safe Queries Used

- sampled-data control barrier function safety filter collision avoidance quadrotor
- composite control barrier function LiDAR quadrotor ICRA 2025
- robust control barrier function measurement disturbance sampled-data
- predictive safety filter nonlinear constrained learning control
- LiDAR online control barrier function unknown environment
- safety filter input limits quadrotor

No private manuscript sentence, unpublished theorem text, result value, or local path was sent in a web query.

## Sources Checked

- ICLR 2027 official author/reviewer/AI-policy pages
- IEEE DOI and publisher records
- PMLR proceedings
- ICRA 2025 proceedings/DOI and the authors' public implementation page
- arXiv records for current preprints
- DBLP and proceedings tables only for discovery/status confirmation

Nineteen distinct candidates were screened; twelve decision-relevant sources were retained. Discovery duplicates, non-primary summaries, unrelated multi-agent barrier papers, and methods requiring a substantially different task model were excluded from the final table.

## Excluded Sources

- MDPI and policy-excluded venues: excluded.
- Search snippets without a stable primary record: excluded from claims.
- ICLR papers whose collision filter was only a small application detail: excluded.
- The 2025 MPC-SdHOCBF work is retained only as a labeled arXiv preprint, not treated as peer-reviewed SOTA.

## Unknowns

- No single method is universally state of the art across known/static, unknown/dynamic, black-box, and hardware settings.
- The exact publication metadata for some recent sampled-data robustness variants should be rechecked before adding them to `references.bib`.
- A faithful Harms et al. Composite CBF implementation has not been verified in this repository.
- The current repository aggregate-HOCBF prototype must not be treated as equivalent to the ICRA 2025 method.

## Handoff Notes

- For review: the current paper's collision-authority wording conflicts with the frozen R3 evidence.
- For idea optimization: freeze sampled-data robust HOCBF as inherited infrastructure; make energy/resource stopping the only headline safety contribution.
- For experiment design: changing the safety layer changes the executed occupancy and energy law, so battery calibration and all downstream Oracle/Pareto artifacts must be regenerated.
- For writing: cite the selected filter and list its assumptions; do not spend contribution bullets on collision avoidance.

