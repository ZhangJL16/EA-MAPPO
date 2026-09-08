# ICLR 2027 Return-to-Charge Paper

Working title: **When Should an Agent Return? Reliable Resource-to-Go under Closed-Loop Composition Shift**

This is an anonymous, evidence-bounded pre-results manuscript. Red `RESULT PENDING` markers are deliberate and must not be removed until the corresponding gate in `notes/PENDING_RESULTS.md` passes with audited artifacts.

## Build

From this directory in WSL:

```bash
/mnt/d/Program_Files/texlive/2025/bin/windows/latexmk.exe \
  -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=build main.tex
```

Clean auxiliary outputs without removing the PDF:

```bash
/mnt/d/Program_Files/texlive/2025/bin/windows/latexmk.exe \
  -c -outdir=build main.tex
```

The project uses the unmodified official ICLR 2027 style files stored in this directory. Keep `\iclrfinalcopy` commented for double-blind review.

## Evidence rules

- The current simulator supports deterministic point Resource-to-Go plus epistemic reliability, not physical q95/q99 tails.
- HOCBF remains the hard collision-safety layer; the learned resource predictor is not a certificate.
- Historical pilots, smoke tests, failed gates, and live training values do not fill formal result slots.
- Every final number needs code revision, configuration, checkpoint, seeds, raw artifact, and evaluation-script provenance.

## Key ledgers

- `notes/CLAIM_EVIDENCE_MATRIX.md`
- `notes/CITATION_AUDIT.md`
- `notes/PENDING_RESULTS.md`
- `notes/PAGE_BUDGET.md`
- `notes/CYNICAL_REVIEW.md` (created after the first compiled audit)
- `notes/AUTHOR_REWRITE_GATE.md` (must pass before the narrow AI-use disclosure is finalized)
