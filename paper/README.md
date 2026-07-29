# Manuscript files

## Final

**`contamsens_manuscript_Q1_rewrite.docx`** (and the matching `.pdf`) is the
submission version:

> Contamination Robustness Values: Sensitivity Analysis for Benchmark Claims
> Under Unverifiable Data Leakage

20 pages, IEEE numeric citation style, 84 bracketed citations. Where this
file and anything below disagree, this file wins.

## Working draft, kept for traceability

The Q1 version grew out of a generated pipeline that is still in the
repository, because every number in it is produced by a script rather than
typed by hand:

| file | role |
|---|---|
| `manuscript.md` | body text, hand-edited |
| `appendix_proofs.md` | Propositions 1–3, Theorems 4–5, Lemmas A.0–A.3 |
| `references.bib`, `references.md` | 28 arXiv entries fetched live from the arXiv API; 6 classical statistics entries checked against publisher records |
| `PAPER_FINAL.md` | body + appendix + references, assembled by `scripts/assemble_paper.py` |
| `contamsens_manuscript.docx` / `.pdf` | rendered from `PAPER_FINAL.md` by `scripts/make_docx.js` |

Regenerating the draft:

```
python scripts/assemble_paper.py
node scripts/make_docx.js
```

These commands rewrite `PAPER_FINAL.md` and `contamsens_manuscript.docx`.
They do **not** touch the Q1 file, which was prepared outside this pipeline
and is edited directly.

## Titles differ

The working draft is still titled *"How Contaminated Would It Have To Be?
Partial Identification and Sensitivity Analysis for Benchmark Claims Under
Unmeasured Data Contamination"*. The Q1 rewrite retitled the paper and
switched from author–year to IEEE numeric citations. The scientific content,
every reported number, and the pre-registered protocol are unchanged between
them; `PROVENANCE.md` maps each number to its generating script either way.
