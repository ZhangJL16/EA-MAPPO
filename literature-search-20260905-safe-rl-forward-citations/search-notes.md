# Forward-citation research provenance

Date: 2026-09-05. Public queries only; no unpublished project data uploaded.

## Method

Search method names/titles alongside navigation, benchmark, fails, implementation,
and later conference years. Negative-query bias was countered with neutral
navigation/benchmark searches and positive examples (CRAX, FOCOPS in CarGoal,
Point Robot Navigation). A citation edge is recorded only after reading the
downstream body/reference list, not from citation counts or snippets.

Key queries included FOCOPS navigation benchmark; FOCOPS fails safe RL;
SafeMPO; Embedding Safety into RL 2026; A Multiplicative Value Function for Safe
and Efficient Reinforcement Learning; Langevin Policy safe RL; SafeOR-Gym;
ProSh; FCSRL feasibility-consistent later works; robust safe transfer.

Search was manual forward-citation discovery, not an exhaustive citation-index
export. No verified independent follow-up numerical study was found for
SafeMPO itself. C-TRPO follow-up search had inaccessible/insufficient and
future-issue candidates; no strong independent validation is asserted.

## Fifteen screened candidate records

Ten retained core records are in papers.csv. Five additional records:

1. ReDMan (Machine Learning 2025), https://link.springer.com/article/10.1007/s10994-025-06825-x . Venue and FOCOPS/CPO references verified. Publisher access exposed abstract/references but not the whole results section consistently. Search snippets about multiplier oscillations were NOT used as decisive empirical evidence.
2. Robust Transfer of Safety-Constrained Reinforcement Learning Agents (ICLR 2025), https://proceedings.iclr.cc/paper_files/paper/2025/hash/e59bf9a9077c361ed57180fc79c5b8d8-Abstract-Conference.html . Source-task robustness/transfer is a later issue; not deep-read as a direct FOCOPS implementation validation.
3. Safety-Prioritizing Curricula (ICLR 2025), https://proceedings.iclr.cc/paper_files/paper/2025/hash/120ed726cf129dbeb8375b6f8a0686f8-Abstract-Conference.html . Existing candidate; do not conflate a generic curriculum reference in CRAX with direct adoption of this algorithm.
4. Proactive Constrained Policy Optimization with Preemptive Penalty (AAAI 2026), https://ojs.aaai.org/index.php/AAAI/article/view/39978 . Official abstract inspected; barrier plus intrinsic reward adds two mechanisms. Deferred. Its PCPO acronym is NOT the same as Projection-Based CPO.
5. Safe policy optimization with stretchable penalties, https://www.sciencedirect.com/science/article/pii/S0925231226023660 . Discovery record has a December 2026 volume date, after this review date, and unclear online-first chronology. Not used as established current C-TRPO follow-up evidence.

SafePO repository was inspected as the implementation associated with the
Safety-Gymnasium paper, not counted as another independent paper. FSRL is
explicitly classified as a system/tool, not a novel method paper.

## Access/version issues

- OpenReview returned challenge pages or 429; no attempts to bypass access controls.
- Some browser PDF reads returned internal/unsupported-content errors. Public
  proceedings/arXiv PDFs were read in memory using `uv run --with pypdf` and
  urllib. This uses an ephemeral tool dependency, not a project dependency change.
- SafeMPO pages2–3,7–10; Safety-Gymnasium pages5–10; multiplicative-value
  pages5–7; Langevin pages6–8,15 were read for the claims reported here.
- CUP supplement includes main paper: printed Table1 is PDF page10.
- C-TRPO inspection uses arXiv v4, not silently the earliest conference version.
- CRAX uses v1; main text500M versus appendix100M discrepancy recorded, not
  resolved by picking the more convenient number.
- SafeOR HTML version v4 was guessed incorrectly and returned404; canonical
  record verified v2 (2026-07-16), which is the version used in the review.
- ProSh long paper v2 was read. Author homepage verifies AAMAS2026 extended
  abstract and NeurIPS2026 submission, contradicting the impression from a
  university ICLR news grouping. Do not label it ICLR accepted on that basis.
- SafeOR-Gym TMLR2026 publication is supported by indexed official TMLR PDF
  metadata and DBLP; official forum fetch was rate limited. Numerical claims
  here point to the accessible author arXiv v2, not an uninspected revision.

## Evidence safeguards

MDPI excluded. No citation counts used. Every numeric example includes the
source table, metric and budget in papers.md. Means with no uncertainty shown
are not significance tests. No paper's safety term is silently upgraded to
zero-collision UAV safety. The review checks the critic's own method too, not
only its negative baseline descriptions.

## Handoff implications

FOCOPS and standard PPO-Lagrangian should be compared under the same physical
event cost contract; plain PPO is a learning sanity reference. Do not treat
FOCOPS-versus-unconstrained-PPO as isolation of one safety mechanism. SafeMPO
is downgraded to theoretical inspiration due to its own reported cost overruns.
SAC is not ruled out by its data regime. No training or source code was modified.
