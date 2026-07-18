"""Fetch per-item Open LLM Leaderboard (v1 archive) details and audit them.

The v1 archive (`open-llm-leaderboard-old/details_<org>__<model>`) is fully
open: per-task parquet files with one row per item, including binary
`acc`/`acc_norm` and the full `example` text. Items are paired across models
by a content hash of `example` (row order is not trusted).

Usage (demo):
    python scripts/fetch_oll.py            # 4 models on ARC-Challenge -> audit

*** EXPLORATORY: the confirmatory corpus is governed by PREREGISTRATION.md.
Single-draw binary scores -> the audit auto-routes to the simple bound. ***
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contamsens import audit  # noqa: E402

HUB = "https://huggingface.co"
ARCHIVE = "open-llm-leaderboard-old"
RESULTS = ROOT / "results"

DEMO_MODELS = [
    "meta-llama__Llama-2-7b-hf",
    "meta-llama__Llama-2-13b-hf",
    "mistralai__Mistral-7B-v0.1",
    "meta-llama__Meta-Llama-3-8B",
]
DEMO_TASK = "arc:challenge|25"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "contamsens/0.2"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def task_files(model: str, task: str) -> list[str]:
    """All snapshot files for this task, oldest -> newest. Later snapshots can
    be partial or tiny debug re-runs, so callers walk newest -> oldest until
    an acceptable item count is found."""
    tree = json.loads(
        _get(f"{HUB}/api/datasets/{ARCHIVE}/details_{model}/tree/main"
             f"?recursive=true")
    )
    marker = f"details_harness|{task}_"
    hits = sorted(t["path"] for t in tree
                  if t["type"] == "file" and marker in t["path"]
                  and t["path"].endswith(".parquet"))
    if not hits:
        raise RuntimeError(f"no parquet for task {task!r} in details_{model}")
    return hits


def _extract_scores(df: pd.DataFrame, model: str) -> pd.Series:
    """Per-item score across the archive's format generations."""
    for col in ("acc_norm", "acc"):
        if col in df.columns:
            return df[col].astype(float)
    if "metrics" in df.columns:  # newer lighteval format: dict per row
        vals = [
            float(m.get("acc_norm", m.get("acc")))
            for m in df["metrics"]
        ]
        return pd.Series(vals, index=df.index)
    raise RuntimeError(
        f"{model}: no accuracy column found; columns = {sorted(df.columns)}"
    )


def _item_key_column(df: pd.DataFrame) -> pd.Series:
    """The stable cross-model pairing field, across format generations."""
    for col in ("example", "full_prompt", "query"):
        if col in df.columns:
            return df[col]
    raise RuntimeError(f"no pairing column; columns = {sorted(df.columns)}")


def fetch_task_items(model: str, task: str, min_items: int = 100) -> pd.DataFrame:
    """Per-item rows for one model on one harness task.

    Walks snapshots newest -> oldest and keeps the first with >= min_items
    rows (tiny debug re-runs exist in the archive); falls back to the largest.
    """
    df = best = None
    for fname in reversed(task_files(model, task)):
        url = (f"{HUB}/datasets/{ARCHIVE}/details_{model}/resolve/main/"
               f"{urllib.parse.quote(fname)}")
        cand = pd.read_parquet(io.BytesIO(_get(url)))
        if best is None or len(cand) > len(best):
            best = cand
        if len(cand) >= min_items:
            df = cand
            break
    if df is None:
        df = best
    item_ids = [
        hashlib.sha1(str(ex).encode("utf8")).hexdigest()[:16]
        for ex in _item_key_column(df)
    ]
    out = pd.DataFrame({
        "model": model,
        "item": item_ids,
        "score": _extract_scores(df, model),
    })
    if out["item"].duplicated().any():
        out = out.groupby(["model", "item"], as_index=False)["score"].mean()
    return out


def build_corpus(models: list[str], task: str, min_items: int = 100) -> pd.DataFrame:
    frames = []
    for m in models:
        rows = fetch_task_items(m, task, min_items=min_items)
        print(f"  {m}: {len(rows)} items, mean acc_norm "
              f"{rows.score.mean():.3f}")
        frames.append(rows)
    df = pd.concat(frames, ignore_index=True)
    # keep only items every model answered (paired design)
    counts = df.groupby("item")["model"].nunique()
    shared = counts[counts == len(models)].index
    return df[df.item.isin(shared)].reset_index(drop=True)


def main() -> None:
    print(f"Fetching {DEMO_TASK} details for {len(DEMO_MODELS)} models "
          f"(v1 archive, EXPLORATORY demo)")
    corpus = build_corpus(DEMO_MODELS, DEMO_TASK, min_items=1000)
    n_items = corpus.groupby('item').ngroups
    print(f"paired items: {n_items}")
    out = audit(corpus, pi=0.1, lam_ref=0.2, n_boot=500)
    out["task"] = DEMO_TASK
    out["status"] = "EXPLORATORY_DEMO"
    out.to_csv(RESULTS / "p6_oll_demo.csv", index=False)
    cols = ["model_a", "model_b", "margin", "gamma_star_display",
            "robust_at_ref", "fragility_p", "certified_robust_fdr", "regime"]
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
