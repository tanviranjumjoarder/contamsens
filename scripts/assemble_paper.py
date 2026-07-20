"""Assemble paper/PAPER_FINAL.md from the manuscript, proofs appendix and
verified bibliography.

Order: body -> Appendix A (proofs) -> References. Regenerate after editing
any component:  python scripts/assemble_paper.py
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def main() -> None:
    man = (PAPER / "manuscript.md").read_text(encoding="utf8")
    app = (PAPER / "appendix_proofs.md").read_text(encoding="utf8")

    i = man.index("## References")
    body, refs = man[:i], man[i:]
    app = re.sub(r"^#\s+.*\n", "", app, count=1)  # drop the appendix's own H1

    out = "\n".join([
        body.rstrip(),
        "\n---\n",
        "# Appendix A — Proofs\n",
        app.strip(),
        "\n---\n",
        refs.strip(),
        "",
    ])
    dest = PAPER / "PAPER_FINAL.md"
    dest.write_text(out, encoding="utf8")

    print(f"wrote {dest.relative_to(ROOT)}")
    print(f"  words    : {len(out.split())}")
    print(f"  sections : {len(re.findall(r'^## ', out, re.M))}")
    pend = out.count("[PENDING")
    print(f"  PENDING  : {pend}" + ("  <- corpus-mix run outstanding"
                                    if pend else ""))


if __name__ == "__main__":
    main()
