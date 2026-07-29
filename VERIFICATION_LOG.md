# Verification log

Every recorded attempt to re-run this artifact, in chronological order, with
who ran it and what they actually checked. Entries are appended, never
rewritten. A failed or partial run is as welcome here as a clean one: the
point of the log is that it shows its work rather than asserting a verdict.

**Independence is stated explicitly in each entry, because it is what the
entry is worth.** Three levels are used:

| level | meaning |
|---|---|
| `author` | run by an author of the paper, or by the agent that wrote the code |
| `coauthor` | run by a listed author other than the one who wrote the code |
| `independent` | run by someone who is not an author and has no stake in the result |

Only an `independent` entry can retire limitation §11(6) of the paper. A
`coauthor` entry establishes something smaller and real: that the artifact
runs for a second pair of hands, working from the public repository rather
than the development copy. Whether it also establishes a second *machine*
depends on the environment recorded in the entry, so every entry pastes its
`run_manifest.json` and says plainly what its environment does not prove.

## How to add an entry

Run the fifteen-minute path in [REPLICATION.md](REPLICATION.md), then either
open a GitHub issue titled "Replication report" (we will transcribe it here)
or append your own entry below and open a pull request. Include the fields in
the template at the end of this file. Paste `results/run_manifest.json`
verbatim rather than retyping it.

---

## Entry 1 — 2026-07-29 — protocol dry run

- **Who:** Claude (Anthropic agent), the same agent that wrote the code and
  the manuscript, acting on the author's instruction.
- **Independence:** `author`. This entry establishes **nothing** about
  independent verification and is logged so that the absence of an
  independent run is visible rather than implied. It is the author checking
  their own homework, and is recorded as such.
- **Purpose:** to confirm that the instructions in `REPLICATION.md` actually
  work from a clean clone, so that the first real replicator does not waste
  their time on a broken guide.
- **What was run:** fresh `git clone` of the public repository into an empty
  directory (not the working copy), `pip install -e .[experiments,test]`,
  then the freeze check, the ruler check, and `python reproduce.py`.
- **Repository state:** clone at `6260eda`; tag `prereg-freeze-v1.0` resolves
  to `ca7773b57b385e5f3c605fba0f66fd4ef773787d`.

**Results.**

1. *Freeze verification* — PASS. `git diff prereg-freeze-v1.0` for
   `PREREGISTRATION.md` and `results/lambda_priors.csv` is empty; the tag
   hash matches the one filed in the OSF registration
   (DOI `10.17605/OSF.IO/2436Z`).
2. *Ruler check* — PASS. Recomputing the frozen rule `margin < 0.1 × Λ_ref`
   directly from `results/confirmatory_audit.csv`, without calling any
   project code, reproduces all 58 `non_robust_primary` labels and the
   49/58 headline.
3. *Fifteen-minute path* — **FAILED on first attempt**, then passed after a
   fix. The `adjacent-pair selection check` step reads the `data/oll_audit/`
   download cache, which is gitignored and therefore absent from every fresh
   clone. Any replicator would have hit this. Fixed in commit `6260eda`
   (the step is now conditional, with a notice naming the command that
   rebuilds the cache). Re-run from the clone: 8 steps, all passed, 72.5 s.

```json
{
  "timestamp_utc": "2026-07-29T05:51:58Z",
  "python": "3.13.1",
  "platform": "Windows-10-10.0.19045-SP0",
  "machine": "AMD64",
  "packages": {
    "numpy": "2.2.2", "scipy": "1.15.1", "pandas": "3.0.3",
    "matplotlib": "3.11.0", "scikit-learn": "1.6.1", "pytest": "9.1.1"
  },
  "seed": 42,
  "wall_seconds": 72.5
}
```

- **Not checked:** the three-hour path (re-downloading the leaderboard data
  and regenerating `confirmatory_audit.csv` from nothing) was not run in this
  entry. The proofs in Appendix A were not independently verified by a human
  mathematician.
- **Outcome:** the replication guide is now known to be executable end to
  end. Limitation §11(6) stands unchanged.

---

## Entry 2 — 2026-07-29 — coauthor clean-clone run

- **Who:** Md. Sadman Anjum Joarder, Rajshahi University of Engineering &
  Technology.
- **Independence:** `coauthor`. A listed author has a stake in the result,
  so this entry does **not** retire limitation §11(6). What it does
  establish is narrower and still worth recording: a second person, working
  from the public repository rather than the development copy, re-derived
  every committed number.
- **What was run:** `git clone` of the public repository into a new
  directory, `pip install -e .[experiments,test]`, then
  `python reproduce.py`.
- **Repository state:** clone at `3e8303f`, the current HEAD at the time of
  the run.

**Results.** All 8 steps passed in 53.3 s.

Every file under `results/` regenerated **byte-identically** to the
committed versions. `git diff` across the directory reports one changed
file, `run_manifest.json`, which necessarily differs because it records the
run's own timestamp and step list. The headline numbers re-derived from
this clone's own regenerated outputs: 58 claims, 49 non-robust, 8 BH
certificates, 6 BY.

```json
{
  "timestamp_utc": "2026-07-29T08:49:47Z",
  "python": "3.13.1",
  "platform": "Windows-10-10.0.19045-SP0",
  "machine": "AMD64",
  "packages": {
    "numpy": "2.2.2", "scipy": "1.15.1", "pandas": "3.0.3",
    "matplotlib": "3.11.0", "scikit-learn": "1.6.1", "pytest": "9.1.1"
  },
  "seed": 42,
  "wall_seconds": 53.3
}
```

- **Environment caveat, stated because it limits what this entry proves:**
  the platform string, Python version, and every package version are
  identical to Entry 1. The run happened on the same machine and the same
  interpreter. It therefore says nothing about cross-platform or
  cross-version reproduction. A replicator on Linux, or on different
  package versions, would test something this entry cannot.
- **Not checked:** the three-hour path (re-downloading the leaderboard data
  and rebuilding `confirmatory_audit.csv` from nothing) was not run. The
  proofs in Appendix A were not verified by hand.
- **Outcome:** the committed results are reproducible from the public
  repository by someone other than the code's author. Limitation §11(6)
  stands: no `independent` entry exists yet.

---

## Entry template

```
## Entry N — YYYY-MM-DD — <short label>

- **Who:** name, affiliation, contact or GitHub handle
- **Independence:** author | coauthor | independent
- **What was run:** which path, which commands
- **Repository state:** clone commit hash
- **Results:** what passed, what failed, exact numbers where they differ
- **Environment:** paste results/run_manifest.json
- **Not checked:** anything you did not verify
```

Discrepancies of any size are wanted, including ones you suspect are your own
setup. They will be acknowledged in the paper. Anyone completing the
three-hour path is offered an acknowledgment by name.
