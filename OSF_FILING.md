# Filing the pre-registration on OSF — detailed walkthrough

The protocol is frozen (`PREREGISTRATION.md` v1.0, git tag `prereg-freeze-v1.0`).
Filing on OSF adds an independent third-party timestamp and a DOI — the thing a
reviewer actually trusts, because it cannot be back-dated by you.

**Do this before publicising any confirmatory number.** Total time ~15 minutes.

---

## Step 0 — Pre-flight (have these ready)

Four files to upload, all in this repository:

1. `PREREGISTRATION.md` — the frozen protocol
2. `OSF_ADDENDUM_chain_of_custody.md` — exactly what was frozen vs. written after
3. `results/lambda_priors.csv` — **required, not optional.** §4 of the protocol
   freezes this table; if it is not filed, "Λ_ref = 0.355 was frozen" is an
   unverifiable assertion.
4. `THEORY.md` — defines the estimators §3 refers to

The commit hash you will paste (already looked up for you):

```
ca7773b57b385e5f3c605fba0f66fd4ef773787d
```

> **Why the addendum matters.** `scripts/run_confirmatory_audit.py` is named in
> §2 but was committed *after* the tag. A reviewer who checks the tag will not
> find it. Disclosed up front this is a non-issue; discovered by a reviewer it
> looks like concealment. The addendum states it plainly.

---

## Step 1 — Account

Go to <https://osf.io> and sign in, or click **Sign Up** (free).

- You can register with email, ORCID, or an institutional login. **ORCID is
  recommended** — it ties the registration to your researcher identity, which is
  what you want on a paper.
- If you sign up by email you must **click the confirmation link** before you can
  create a registration. Check spam.

---

## Step 2 — Create the project

Dashboard → **Create new project**.

| Field | What to enter |
|---|---|
| **Title** | `contamsens — Contamination Sensitivity Analysis for Benchmark Claims` |
| **Storage location** | Pick the region nearest you. **This cannot be changed later.** |
| **Description** | paste the block below |

```
Partial identification and sensitivity analysis for machine-learning benchmark
leaderboard claims under unmeasured training-data contamination. Rather than
attempting to detect contamination, this project asks how strong contamination
would have to be to overturn a given leaderboard claim, and calibrates that
threshold against measured contamination strengths.

This project hosts the frozen pre-registration for a confirmatory re-audit of
Open LLM Leaderboard v1 claims.
```

Leave the project **private** if you prefer — the registration gets its own DOI
and timestamp either way, and can be public even while the project is private.

---

## Step 3 — Upload the files (⚠️ must happen BEFORE Step 4)

Open the project → **Files** → **OSF Storage** → drag the four files from Step 0 in.

> **This ordering is not optional.** OSF snapshots project files *at the moment
> the registration draft is created* (up to 5 GB). Files added afterwards are
> **not** in the registration. Upload first, register second.

Confirm all four appear under OSF Storage before continuing.

---

## Step 4 — Start the registration

From the project, open the **Registrations** tab → **Add a Registration**.
(Equivalently: **My OSF → My Registrations → Add a Registration**.)

When asked whether you have existing project content, choose **Yes** and select
this project from the dropdown — that is what attaches the Step 3 files.

---

## Step 5 — Choose the template

Pick **Open-Ended Registration**, then **Create Draft**.

- **Open-Ended Registration** — recommended. One free-text summary box. Your
  protocol is already written and only ~720 words, so pasting it verbatim is
  both faster and *more faithful* than re-typing it into someone else's field
  structure.
- **Secondary Data Preregistration** — a defensible alternative, arguably the
  most technically apt (you are analysing pre-existing public data), but it asks
  many structured questions you would answer by pointing at the same document.
- **OSF Preregistration** — the standard template. Designed around
  hypothesis-testing studies with data collection; several required fields do
  not map onto a computational re-analysis.

Any of the three produces the same timestamp and DOI. Choose Open-Ended unless
you have a reason not to.

---

## Step 6 — Fill in the fields

**Title:**

```
Confirmatory Contamination Re-Audit of Open LLM Leaderboard Claims (contamsens Phase 4)
```

**Summary / description box** — paste, in this order:

1. the header block below,
2. then the **entire verbatim text** of `PREREGISTRATION.md`.

```
This is a frozen pre-registration, filed before any confirmatory result was
publicised. The full protocol text follows verbatim.

Analysis code frozen at git tag `prereg-freeze-v1.0`, commit
ca7773b57b385e5f3c605fba0f66fd4ef773787d (committed 2026-07-19 00:17:46 +0600).
The frozen calibration table (results/lambda_priors.csv, git blob
590d29523466e11525d72ef6992039dcd5e0e75b) is attached to this registration, as
is a chain-of-custody addendum stating exactly which files were frozen at the
tag and which were written afterwards.

Primary hypothesis (H4): at the meta-analytically calibrated contamination
strength, at least 25% of adjacent-pair leaderboard claims in the confirmatory
corpus are not contamination-robust. Falsification is informative in either
direction; if fewer than 25% are non-robust, the result is published as a
defence of current leaderboard practice under this same protocol.

--- PREREGISTRATION.md v1.0-FREEZE follows verbatim ---
```

Do **not** paraphrase or tidy the protocol while pasting. Verbatim is the point.

The draft **autosaves** as you move between sections.

---

## Step 7 — Review and register

Go to the **Review** page, read it once, then click **Register**.

You will be asked to choose:

- **Make registration public immediately** — recommended. Cleanest for the
  paper; the DOI resolves right away.
- **Enter an embargo** (up to 4 years) — the timestamp still exists and is still
  valid; the content just isn't visible until the embargo lifts. Choose this only
  if you have a reason to withhold the protocol.

---

## Step 8 — The 48-hour approval window

After you click Register, the submission enters a **pending** state and every
admin contributor is emailed to approve or reject it.

- You are the sole admin, so **approve your own submission from that email** to
  finalise immediately.
- If nobody acts, it **auto-approves after 48 hours**.
- This window is your only chance to cancel. **Once approved, the registration
  and its attached files are permanently immutable** — you cannot edit them.
  A registration can later be *withdrawn*, but withdrawal leaves a public
  tombstone with the metadata; it does not erase it.

So: read the Review page carefully before clicking Register.

---

## Step 9 — Collect the DOI

Once public, OSF mints a DOI automatically — this is your registration number.
A short form shows on the overview page; the full DOI is under **Metadata**.

Record both:

- Registration URL: `https://osf.io/xxxxx/`
- DOI: `10.17605/OSF.IO/XXXXX`

---

## Step 10 — Propagate it back into the repository

Send me the URL and DOI and I will update all of these in one pass:

- `PREREGISTRATION.md` header — add `OSF registration: <URL>, filed <date>`
- `paper/manuscript.md` §9.1 and the reproducibility statement
- `scripts/run_confirmatory_audit.py` — the summary label, replacing
  "OSF timestamp pending"
- `results/confirmatory_summary.txt` — regenerated with the real label
- `PROVENANCE.md` — the freeze row
- `README.md`

At that point the chain of custody is complete and independently checkable:

**priors frozen → protocol frozen → code tagged → OSF timestamp → results.**

---

## Gotchas

- **Files before draft.** Registering first and uploading after silently
  produces a registration with no files attached. (Step 3.)
- **Storage region is permanent.** Chosen at project creation.
- **Immutable after approval.** No edits, ever. Only withdrawal.
- **The commit hash proves nothing while the repo is private.** The OSF
  timestamp is independently valid regardless, but a reviewer can only verify
  `ca7773b…` once the repository is published. This is an argument for pushing
  the repo publicly — the hash then becomes a genuine cryptographic anchor.
- **Don't edit `PREREGISTRATION.md` before filing.** It is currently
  byte-identical to the tag; any edit breaks that and weakens the freeze. All
  clarifications belong in the addendum instead.
