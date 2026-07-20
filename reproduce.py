"""One-command reproduction of every committed number.

    python reproduce.py            # tests + fast experiments (~5 min)
    python reproduce.py --full     # also the Phase-2 twin-training grid (~30 min)

Order respects dependencies; each step's wall time is printed. Every output
lands in results/ and maps to a claim via PROVENANCE.md. Seed 42 throughout.
The Phase-4 pilot requires data/atlas_pilot/*.csv (from the author's private
TSFM-atlas repository) and is skipped with a notice if absent.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FAST_STEPS = [
    ("unit tests", [sys.executable, "-m", "pytest", "tests", "-q"]),
    ("P0 validation + gate", [sys.executable, "experiments/run_validation.py"]),
    ("P1 estimation theory", [sys.executable, "experiments/run_p1_theory.py"]),
    ("P5 design sensitivity", [sys.executable, "experiments/run_p5_design.py"]),
    # Reads the committed confirmatory_audit.csv; no network, no GPU.
    ("Lambda_ref sensitivity (f18)",
     [sys.executable, "scripts/run_lambda_sensitivity.py"]),
]
PILOT_STEP = ("P4 pilot (atlas)", [sys.executable, "experiments/run_p4_pilot_atlas.py"])
LORA_STEP = ("P2c LoRA analysis", [sys.executable, "experiments/run_p2c_lora_analysis.py"])
FULL_STEPS = [
    ("P2 CONTAM-CTRL twins", [sys.executable, "experiments/run_p2_contamctrl.py"]),
    ("P2b spillover fix", [sys.executable, "experiments/run_p2b_spillover_fix.py"]),
    ("f12 two-panel figure", [sys.executable, "experiments/make_f12_two_panel.py"]),
]


def run(label: str, cmd: list[str]) -> None:
    t0 = time.perf_counter()
    print(f"\n=== {label} ===")
    result = subprocess.run(cmd, cwd=ROOT)
    dt = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"FAILED after {dt:.0f}s: {label}")
        sys.exit(result.returncode)
    print(f"ok ({dt:.0f}s)")


def write_manifest(steps: list, seconds: float) -> None:
    """Run manifest (platform, versions, steps) -> results/run_manifest.json."""
    import importlib.metadata
    import json
    import platform

    pkgs = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn",
                 "pytest"):
        try:
            pkgs[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pkgs[name] = "absent"
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": pkgs,
        "seed": 42,
        "steps": [label for label, _ in steps],
        "wall_seconds": round(seconds, 1),
    }
    (ROOT / "results" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf8"
    )


def main() -> None:
    full = "--full" in sys.argv
    steps = list(FAST_STEPS)
    if (ROOT / "data" / "atlas_pilot" / "atlas_delta.csv").exists():
        steps.append(PILOT_STEP)
    else:
        print("note: data/atlas_pilot missing -> P4 pilot skipped "
              "(see experiments/run_p4_pilot_atlas.py header for provenance)")
    if (ROOT / "results" / "contamctrl_lora_peritem.csv").exists():
        steps.append(LORA_STEP)
    else:
        print("note: results/contamctrl_lora_peritem.csv missing -> P2c "
              "skipped (run the Kaggle notebook to produce it)")
    if full:
        steps += FULL_STEPS
    else:
        print("note: pass --full for the Phase-2 twin-training grid (~30 min)")
    t0 = time.perf_counter()
    for label, cmd in steps:
        run(label, cmd)
    dt = time.perf_counter() - t0
    write_manifest(steps, dt)
    print(f"\nALL STEPS PASSED in {dt:.0f}s ({len(steps)} steps). "
          f"Manifest -> results/run_manifest.json; claims map via PROVENANCE.md.")


if __name__ == "__main__":
    main()
