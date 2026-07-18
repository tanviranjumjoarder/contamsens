# Filing the pre-registration on OSF — 10-minute guide

The protocol is frozen (PREREGISTRATION.md v1.0, git tag `prereg-freeze-v1.0`).
Filing it on OSF gives it an independent, third-party timestamp — the thing a
reviewer trusts. Do this before publicizing any confirmatory number.

1. Go to https://osf.io and sign in (or register — free).
2. **Create a new project**: "contamsens — Contamination Sensitivity Analysis
   for Benchmark Claims". Keep it private for now if you prefer (the
   registration itself gets its own DOI and timestamp either way).
3. Upload `PREREGISTRATION.md` (and optionally `results/lambda_priors.csv`,
   which §4 freezes) to the project's files.
4. Click **Registrations → New registration**.
   - Template: **"OSF Preregistration"** (or "Open-Ended Registration" —
     simplest; paste the full text of PREREGISTRATION.md into the summary box).
   - In the description, note: *"Analysis code frozen at git tag
     prereg-freeze-v1.0; commit hash <paste hash from `git rev-parse
     prereg-freeze-v1.0`>."*
5. Choose **"Make registration public immediately"** (recommended — an
   embargoed registration still timestamps, but public is cleaner for the
   paper) and submit.
6. Copy the registration DOI/URL into:
   - `PREREGISTRATION.md` header (add: "OSF registration: <URL>, filed <date>"),
   - the manuscript's §9 and reproducibility statement.

That's the whole procedure. The audit results in this repository are labeled
"protocol v1.0 (frozen 19 Jul 2026); OSF timestamp pending" until step 6 —
after filing, update the label to the OSF URL and the chain of custody is
complete: priors frozen → protocol frozen → code tagged → OSF timestamp →
results.
