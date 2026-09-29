# Information before exploration design — 2026-09-23

User asks what information avoids dedicated scouting and whether AI conference methods exist. No environment change or run authorized here.

UnderMind discovery and PDF retrieval: Cha20b, Tor19. PDFs downloaded to /tmp/energy_literature. New PDFs imported into Zotero collection 7BA7MPRB with attachment MD5 verified. Import record below.

## Reading coverage and transferable mechanisms

Learning to Explore using Active Neural SLAM (ICLR 2020): inspected full-text methods and experimental setup PDF pp.4–6, results and ablation table pp.7–8. RGB + odometry; persistent obstacle/explored map; learned global waypoint selection; analytical Fast Marching planner; local imitation policy. Unknown cells treated as free for planning (p.5). Exploration coverage reward, not delivery throughput or energy safety. Ablation includes deterministic local planner. This is partial full-text reading, not exhaustive appendix audit.

FASTER (IROS 2019): inspected PDF pp.1–4 and start of experiments p.5. JPS global planning, MIQP local planning; optimistic whole trajectory through known/unknown space plus safe terminal-stop trajectory entirely in known free space. Execute only committed trajectory in known free space; retain previous committed trajectory on solve failure or missed timing. Online occupancy map from depth. Terminal stop is not charger return, and does not solve hovering energy depletion. No independently checked or Lean-verified proof is claimed.

## Proposed clarification, not a frozen decision

Distinguish (1) navigation collision safety, (2) maintained energy-feasible return option, and (3) advance guarantee that a whole unseen delivery route can complete. Order coordinates plus current LiDAR do not themselves reveal hidden route connectivity or energy. A priori traversable corridor graph plus valid route energy bounds could support no dedicated scouting without requiring full obstacle geometry, but supplying such prior data requires an explicit observation-contract decision. Online mapping during ordinary delivery can reduce the need for dedicated scouting. No guarantee is asserted for arbitrary unseen layouts.
