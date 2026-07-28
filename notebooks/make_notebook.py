"""Generate contamctrl_lora_kaggle.ipynb -- with per-cell compile validation.

Cell sources are raw strings (backslashes survive verbatim), and every code
cell is syntax-checked with compile() before the notebook is written, so an
escaping bug can never ship again. Regenerate with:

    python notebooks/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "contamctrl_lora_kaggle.ipynb"

MD_HEADER = r"""# CONTAM-CTRL — LoRA-injected ground-truth contamination (full scale)

**Protocol (THEORY.md §8, red-team fix #5):** fine-tune small open LLMs on controlled
fractions of a benchmark's test items at controlled dosage, then measure the *emergent*
per-item lift against the clean twin. Tests externally whether real LLM memorization
respects the CSM A1 envelope `lift <= Lambda * (1 - y*)`, and estimates Lambda, the
violation rate (rho evidence), and the spillover eps at real scale.

- **Grid:** fractions pi in {0.05, 0.10, 0.25, 0.50} x dose in {1, 4, 16} epochs-over-leaked-items
- **Models:** Qwen2.5-1.5B, TinyLlama-1.1B
- **Benchmark:** MMLU (4 subjects, ~1,100 items) — per-item score = softmax prob of the
  correct choice letter (continuous, regime R1 -> sharp knapsack applies)
- **Runtime:** ~2-4 h full grid on a Kaggle T4; QUICK=True gives a ~20-min smoke run

Kaggle setup: Accelerator -> GPU (T4), Internet -> On, then Run All."""

CELL_SETUP = r"""# torchao 0.10 ships in the Kaggle image and makes recent peft RAISE during
# LoRA injection (it wants >= 0.16). We don't use torchao -- remove it.
%pip -q uninstall -y torchao
%pip -q install transformers peft datasets accelerate
# RUN_MODE:
#   "grid"       -- the main pi x dose grid (COMPLETE, 24/24, 18 Jul 2026)
#   "corpus_mix" -- the dilution experiment: leaked items interleaved with
#                   neutral wikitext at MIX chunks per leaked copy, so the
#                   same dose arrives diluted through a mixed corpus --
#                   interpolating the concentrated-exposure ceiling toward
#                   the pretraining regime (~1-2 h on a T4)
RUN_MODE = "corpus_mix_rep"
SEED = 42  # base; corpus_mix_rep sweeps SEEDS
# Same-family pair isolates CAPACITY in the dose-response comparison
# (TinyLlama dropped: fp16 llama-family training NaN'd on T4; Qwen2.5-0.5B
# is the controlled small-capacity replacement and ships safetensors).
if RUN_MODE == "grid":
    MODELS = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B"]
    CONFIGS = [(pi, dose, 0) for pi in [0.05, 0.1, 0.25, 0.5]
               for dose in [1, 4, 16]]
elif RUN_MODE == "corpus_mix":
    MODELS = ["Qwen/Qwen2.5-1.5B"]
    CONFIGS = [(0.1, dose, mix) for dose in [1, 4] for mix in [0, 4, 20]]
else:  # corpus_mix_rep: seeds x models replication of the dilution sweep.
    # 1.5B seeds 43,44 (seed 42 = the original run) + 0.5B seed 42.
    # ~5-6 h total on a T4; checkpointed per config, safe to split sessions.
    MODELS = ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B"]
    CONFIGS = [(0.1, dose, mix) for dose in [1, 4] for mix in [0, 4, 20]]
    SEEDS = {"Qwen/Qwen2.5-1.5B": [43, 44], "Qwen/Qwen2.5-0.5B": [42]}
SUBJECTS = ["college_biology", "high_school_statistics", "philosophy", "marketing"]"""

CELL_DATA = r"""import numpy as np, pandas as pd, torch, random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
LETTERS = ["A", "B", "C", "D"]

def load_items():
    rows = []
    for s in SUBJECTS:
        ds = load_dataset("cais/mmlu", s, split="test")
        for i, ex in enumerate(ds):
            rows.append({"item_id": f"{s}/{i}", "question": ex["question"],
                         "choices": ex["choices"], "answer": int(ex["answer"])})
    return pd.DataFrame(rows)

def prompt_of(r):
    ch = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, r["choices"]))
    return f"Question: {r['question']}\n{ch}\nAnswer:"

items = load_items()
print(len(items), "items across", len(SUBJECTS), "subjects")"""

CELL_EVAL = r'''@torch.no_grad()
def per_item_scores(model, tok, items, batch=16):
    """Per-item score = softmax prob over the 4 letter tokens on the correct letter."""
    model.eval(); scores = []
    device = next(model.parameters()).device
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS]
    for start in range(0, len(items), batch):
        chunk = items.iloc[start:start + batch]
        enc = tok([prompt_of(r) for _, r in chunk.iterrows()],
                  return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        logits = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        lg = logits[torch.arange(len(chunk)), last][:, letter_ids]
        p = torch.softmax(lg.float(), dim=-1)
        for j, (_, r) in enumerate(chunk.iterrows()):
            scores.append(float(p[j, r["answer"]]))
    return np.array(scores)'''

CELL_TRAIN = r'''import time
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, TensorDataset

_NEUTRAL = None
def neutral_chunks(tok, n, max_length=512):
    """Wikitext-103 chunks as the neutral corpus for the dilution experiment."""
    global _NEUTRAL
    if _NEUTRAL is None:
        wiki = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                            split="train[:20000]")
        _NEUTRAL = [t for t in wiki["text"] if len(t) > 200]
    rng = np.random.default_rng(SEED)
    return [_NEUTRAL[i] for i in rng.integers(0, len(_NEUTRAL), n)]

def lora_contaminate(base, tok, items, contaminated_idx, dose, mix=0, tag=""):
    """Fine-tune a LoRA adapter on the leaked items (prompt + correct letter).

    mix > 0 dilutes the exposure: each epoch's corpus holds `mix` neutral
    wikitext chunks per leaked item, shuffled together, so the same dose
    arrives embedded in a mixed corpus (the pretraining-like regime).
    mix = 0 recovers the concentrated-exposure grid runs exactly.

    Takes an ALREADY-LOADED base model (loaded once per model). Caller must
    unload() the returned adapter to restore the clean base. Prints step
    progress so long configs never look hung. Base is float32 (fp16 NaN'd
    on llama-family; T4 lacks bfloat16; LoRA keeps trainables tiny).
    """
    peft = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                           target_modules="all-linear"))
    device = next(peft.parameters()).device
    leaked = [prompt_of(items.iloc[i]) + " " + LETTERS[items.iloc[i]["answer"]]
              for i in contaminated_idx]
    texts = leaked + (neutral_chunks(tok, mix * len(leaked)) if mix > 0 else [])
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"])
    opt = torch.optim.AdamW((p for p in peft.parameters() if p.requires_grad), lr=1e-4)
    peft.train()
    n_steps = dose * ((len(texts) + 3) // 4)
    step, t0 = 0, time.time()
    for epoch in range(dose):
        for ids, mask in DataLoader(ds, batch_size=4, shuffle=True):
            ids, mask = ids.to(device), mask.to(device)
            loss = peft(input_ids=ids, attention_mask=mask, labels=ids).loss
            loss.backward(); opt.step(); opt.zero_grad()
            step += 1
            if step % 50 == 0 or step == n_steps:
                print(f"    [{tag}] step {step}/{n_steps} "
                      f"loss {float(loss):.3f} ({time.time() - t0:.0f}s)",
                      flush=True)
    return peft'''

CELL_MAIN = r"""# Checkpointed grid: results are appended to the CSV after EVERY config,
# and already-completed configs are skipped -- an interrupted session
# resumes by simply re-running this cell (keep the CSV in /kaggle/working).
import os

OUT_CSV = {"grid": "contamctrl_lora_peritem.csv",
           "corpus_mix": "contamctrl_corpusmix_peritem.csv",
           "corpus_mix_rep": "contamctrl_corpusmix_rep_peritem.csv"}[RUN_MODE]
if not os.path.exists(OUT_CSV):
    # recover a prior checkpoint attached as a Kaggle input dataset
    import glob, shutil
    prior = sorted(glob.glob(f"/kaggle/input/**/{OUT_CSV[:-4]}*.csv",
                             recursive=True))
    if prior:
        shutil.copy(prior[-1], OUT_CSV)
        print("recovered prior checkpoint from:", prior[-1])
results, done = [], set()
if os.path.exists(OUT_CSV):
    prev = pd.read_csv(OUT_CSV)
    if "mix" not in prev.columns:
        prev["mix"] = 0
    results.append(prev)
    if "seed" not in prev.columns:
        prev["seed"] = SEED
    done = set(zip(prev["model"], prev["pi"], prev["dose"], prev["mix"],
                   prev["seed"]))
    print(f"resuming: {len(done)} configs already checkpointed")

for model_id in MODELS:
    seeds = (SEEDS.get(model_id, [SEED]) if RUN_MODE == "corpus_mix_rep"
             else [SEED])
    todo = [(pi, dose, mix, sd) for pi, dose, mix in CONFIGS for sd in seeds
            if (model_id, pi, dose, mix, sd) not in done]
    if not todo:
        print(f"{model_id}: all configs already done"); continue
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # load the base ONCE per model; adapters are attached and unloaded per
    # config; float32 for numerical stability (see lora_contaminate docstring)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="auto")
    y_clean = per_item_scores(base, tok, items)
    probe = y_clean[:16].copy()
    for pi, dose, mix, sd in todo:
        if (model_id, pi, dose, mix, sd) in done:
            continue
        rng = np.random.default_rng(sd)
        torch.manual_seed(sd)
        c_idx = rng.choice(len(items), int(pi * len(items)), replace=False)
        tag = (f"{model_id.split('/')[-1]} pi={pi} dose={dose} "
               f"mix={mix} seed={sd}")
        print(f"config {tag}: {len(c_idx)} leaked items", flush=True)
        t0 = time.time()
        m = lora_contaminate(base, tok, items, c_idx, dose, mix=mix, tag=tag)
        y_obs = per_item_scores(m, tok, items)
        base = m.unload()  # strip the adapter, restore the clean base
        del m; torch.cuda.empty_cache()
        if np.isnan(y_obs).any():
            raise RuntimeError(
                f"{tag}: NaN scores after training (numerical overflow) "
                f"-- config NOT checkpointed; do not trust this session's "
                f"remaining output")
        # twin-integrity check: the restored base must score like the clean run
        drift = float(np.abs(per_item_scores(base, tok, items.iloc[:16])
                             - probe).max())
        assert drift < 2e-3, f"adapter unload left residue (drift {drift})"
        df = items[["item_id"]].copy()
        df["y_clean"], df["y_obs"] = y_clean, y_obs
        df["contaminated"] = np.isin(np.arange(len(items)), c_idx)
        df["model"], df["pi"], df["dose"], df["mix"] = model_id, pi, dose, mix
        df["seed"] = sd
        results.append(df)
        pd.concat(results).to_csv(OUT_CSV, index=False)  # checkpoint
        print(f"  done in {time.time() - t0:.0f}s | mean lift on leaked: "
              f"{float((y_obs - y_clean)[df.contaminated].mean()):.4f} "
              f"| checkpointed", flush=True)
    del base; torch.cuda.empty_cache()
print(f"{RUN_MODE} complete ->", OUT_CSV)"""

CELL_ANALYSIS = r"""# A1 channel-shape analysis (mirrors experiments/run_p2_contamctrl.py)
raw = pd.concat(results)
for (mid, pi, dose), g in raw.groupby(["model", "pi", "dose"]):
    c = g[g.contaminated]
    lift = c.y_obs - c.y_clean
    hr = 1 - c.y_clean
    ok = hr > 0.02
    lam_q95 = float(np.quantile((lift[ok] / hr[ok]), 0.95))
    viol = float((lift < -0.01).mean())
    spill = float((g[~g.contaminated].y_obs - g[~g.contaminated].y_clean).abs().mean())
    print(f"{mid} pi={pi} dose={dose}: lambda_q95={lam_q95:.3f} "
          f"violation={viol:.3f} spillover={spill:.4f}")
# -> transfer lambda_q95 / violation / spillover rows into results/lambda_priors.csv
#    (quality: 'measured') and THEORY.md SS11 at LLM scale"""

CELL_DOWNLOAD = r"""# Package EVERYTHING this run produced into one zip and show a download link.
# (On Kaggle the file also appears under the Output tab after "Save Version".)
import zipfile
from pathlib import Path
from IPython.display import FileLink, display

zip_path = Path("contamctrl_output.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for f in Path(".").iterdir():
        if f.is_file() and f.suffix in (".csv", ".json", ".png") \
                and f.name != zip_path.name:
            z.write(f)
            print("packed:", f.name, f"({f.stat().st_size/1e6:.2f} MB)")
print("\nzip ready:", zip_path, f"({zip_path.stat().st_size/1e6:.2f} MB)")

# AUTO-DOWNLOAD: in an interactive session the browser download starts by
# itself (base64 data-URI + scripted click). In a committed "Save & Run All"
# run there is no browser attached -- grab the zip from the Output tab.
import base64
from IPython.display import HTML

b64 = base64.b64encode(zip_path.read_bytes()).decode()
display(HTML(
    f"<a id='dl' href='data:application/zip;base64,{b64}' "
    f"download='{zip_path.name}'>backup link: {zip_path.name}</a>"
    "<script>document.getElementById('dl').click();</script>"))
display(FileLink(str(zip_path)))  # manual fallback"""

CODE_CELLS = [CELL_SETUP, CELL_DATA, CELL_EVAL, CELL_TRAIN, CELL_MAIN,
              CELL_ANALYSIS, CELL_DOWNLOAD]


def validate(src: str, name: str) -> None:
    """Syntax-check a cell, ignoring notebook magics."""
    stripped = "\n".join(
        line for line in src.split("\n")
        if not line.lstrip().startswith(("%", "!"))
    )
    compile(stripped, f"<{name}>", "exec")


def main() -> None:
    for i, src in enumerate(CODE_CELLS, 1):
        validate(src, f"cell{i}")
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": MD_HEADER},
            *(
                {"cell_type": "code", "metadata": {}, "source": src,
                 "outputs": [], "execution_count": None}
                for src in CODE_CELLS
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf8")
    print(f"all {len(CODE_CELLS)} code cells compile -> {OUT.name} written")


if __name__ == "__main__":
    main()
