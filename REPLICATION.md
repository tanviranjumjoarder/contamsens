# Invitation to replicate

This repository claims that 84.5% of adjacent Open LLM Leaderboard claims
are not contamination-robust under a pre-registered protocol
(OSF: https://osf.io/2436z). The paper's own limitation SS11(6) says the
artifact was audited only by its author. If you are willing to spend
roughly fifteen minutes checking it independently, this file is for you.

## The fifteen-minute replication

    git clone https://github.com/tanviranjumjoarder/contamsens
    cd contamsens
    pip install -e .[experiments,test]
    python reproduce.py

Expected: 10 steps pass in ~2 minutes on any CPU; every number printed
matches the paper via PROVENANCE.md. The confirmatory audit re-derivation
needs no network: results/confirmatory_audit.csv is committed, and the
frozen rule is checkable with a ruler (margin < 0.0355; SS9.2 of the paper).

## The three-hour replication

Re-download the leaderboard data yourself and re-run the audit from
nothing (network required; ~64k items):

    rm -rf data/oll_audit results/confirmatory_audit.csv
    python scripts/run_confirmatory_audit.py

Expected: byte-identical CSV (seed 42, frozen protocol).

## Verifying the freeze independently

    git rev-parse prereg-freeze-v1.0        # ca7773b57b3...
    git diff prereg-freeze-v1.0 -- PREREGISTRATION.md results/lambda_priors.csv

Both diffs must be empty; the hash must match the one in the OSF
registration (DOI 10.17605/OSF.IO/2436Z, filed 2026-07-20).

## The verification log

Every recorded run lives in [VERIFICATION_LOG.md](VERIFICATION_LOG.md), each
entry labelled `author`, `coauthor`, or `independent`. As of this writing it
holds exactly one entry, at `author` level: a dry run of these instructions
that found and fixed a step which broke on fresh clones. No independent run
has happened yet. That is stated in the paper (SS11) rather than glossed, and
your entry would be the first of its kind.

## Reporting

Open a GitHub issue titled "Replication report" with your platform,
package versions (results/run_manifest.json), and any discrepancy, however
small. Discrepancies will be acknowledged in the paper. Replicators who
complete the three-hour path will be offered an acknowledgment by name.
