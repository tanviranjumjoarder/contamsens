"""Generate paper/references.bib from VERIFIED arXiv metadata.

Every arXiv key is fetched live from the arXiv API, so titles, author lists
and years are authoritative rather than recalled. Non-arXiv classics are
listed explicitly below and are flagged in the output for a final human
check against the publisher record.

Run:  python scripts/make_bibliography.py
"""

from __future__ import annotations

import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

NS = {"a": "http://www.w3.org/2005/Atom"}

# Short cite keys for readability in the manuscript.
KEYS = {
    "2005.14165": "brown2020gpt3",
    "2112.13398": "chernozhukov2021ovb",
    "2203.08242": "magar2022contamination",
    "2302.13971": "touvron2023llama",
    "2310.17589": "li2023contamreport",
    "2402.01781": "alzahrani2024benchmarks",
    "2402.14992": "polo2024tinybenchmarks",
    "2403.14152": "heng2024robust",
    "2404.00699": "ravaut2024survey",
    "2405.16281": "dekoninck2024constat",
    "2406.04244": "xu2024survey",
    "2407.21783": "grattafiori2024llama3",
    "2410.18966": "fu2024doescontam",
    "2501.17200": "federiakin2025psychometric",
    "2502.00678": "choi2025kds",
    "2505.18102": "ishida2025capbencher",
    "2510.13654": "meyer2025tsfm",
    "2510.23191": "freiesleben2025epistemology",
    "2511.04703": "bean2025construct",
    "2601.04301": "schaeffer2026quantifying",
    "2601.06103": "kocyigit2026posttraining",
    "2601.19334": "chai2026inferencetime",
    "2602.24261": "sium2026timevarying",
    "2604.03244": "jiang2026itemlevel",
    "2605.15761": "oyarhoseini2026perturbation",
    "2605.23628": "gordienko2026socialchoice",
    "2605.26161": "li2026tsfmaudit",
    "2606.08679": "neuhof2026rankintervals",
}

# Classical statistics references (not on arXiv). VERIFY against publisher
# records before submission -- these are transcribed, not fetched.
CLASSICS = r"""
@article{rosenbaum1987sensitivity,
  author  = {Rosenbaum, Paul R.},
  title   = {Sensitivity Analysis for Certain Permutation Inferences in
             Matched Observational Studies},
  journal = {Biometrika},
  volume  = {74},
  number  = {1},
  pages   = {13--26},
  year    = {1987}
}

@article{tan2006distributional,
  author  = {Tan, Zhiqiang},
  title   = {A Distributional Approach for Causal Inference Using Propensity
             Scores},
  journal = {Journal of the American Statistical Association},
  volume  = {101},
  number  = {476},
  pages   = {1619--1637},
  year    = {2006}
}

@article{vanderweele2017evalue,
  author  = {VanderWeele, Tyler J. and Ding, Peng},
  title   = {Sensitivity Analysis in Observational Research: Introducing the
             {E}-Value},
  journal = {Annals of Internal Medicine},
  volume  = {167},
  number  = {4},
  pages   = {268--274},
  year    = {2017}
}

@article{imbens2004confidence,
  author  = {Imbens, Guido W. and Manski, Charles F.},
  title   = {Confidence Intervals for Partially Identified Parameters},
  journal = {Econometrica},
  volume  = {72},
  number  = {6},
  pages   = {1845--1857},
  year    = {2004}
}

@article{benjamini1995controlling,
  author  = {Benjamini, Yoav and Hochberg, Yosef},
  title   = {Controlling the False Discovery Rate: A Practical and Powerful
             Approach to Multiple Testing},
  journal = {Journal of the Royal Statistical Society, Series B},
  volume  = {57},
  number  = {1},
  pages   = {289--300},
  year    = {1995}
}

@article{benjamini2001control,
  author  = {Benjamini, Yoav and Yekutieli, Daniel},
  title   = {The Control of the False Discovery Rate in Multiple Testing
             under Dependency},
  journal = {The Annals of Statistics},
  volume  = {29},
  number  = {4},
  pages   = {1165--1188},
  year    = {2001}
}
"""


def tex_escape(s: str) -> str:
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def fetch(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 20):
        chunk = ids[i:i + 20]
        url = ("http://export.arxiv.org/api/query?id_list="
               + ",".join(chunk) + "&max_results=100")
        root = ET.fromstring(urllib.request.urlopen(url, timeout=90).read())
        for e in root.findall("a:entry", NS):
            aid = e.find("a:id", NS).text.split("/abs/")[-1].split("v")[0]
            out[aid] = {
                "title": " ".join(e.find("a:title", NS).text.split()),
                "authors": [a.find("a:name", NS).text
                            for a in e.findall("a:author", NS)],
                "year": e.find("a:published", NS).text[:4],
            }
        time.sleep(3)
    return out


def main() -> None:
    ids = sorted(KEYS)
    print(f"fetching {len(ids)} arXiv records ...")
    meta = fetch(ids)
    missing = [i for i in ids if i not in meta]
    if missing:
        raise SystemExit(f"NOT FOUND on arXiv, refusing to emit: {missing}")

    parts = ["% references.bib -- generated by scripts/make_bibliography.py",
             "% arXiv entries fetched live from the arXiv API (authoritative).",
             "% Entries below the CLASSICS banner are transcribed; verify",
             "% volume/number/pages against the publisher before submission.",
             ""]
    for aid in ids:
        m = meta[aid]
        authors = " and ".join(m["authors"])
        parts.append(
            f"@article{{{KEYS[aid]},\n"
            f"  author       = {{{tex_escape(authors)}}},\n"
            f"  title        = {{{tex_escape(m['title'])}}},\n"
            f"  journal      = {{arXiv preprint arXiv:{aid}}},\n"
            f"  eprint       = {{{aid}}},\n"
            f"  archivePrefix= {{arXiv}},\n"
            f"  year         = {{{m['year']}}}\n"
            f"}}\n")
    parts.append("\n% ---------------- CLASSICS (verify manually) ----------\n")
    parts.append(CLASSICS.strip() + "\n")

    (PAPER / "references.bib").write_text("\n".join(parts), encoding="utf8")

    # Human-readable list for the markdown draft, same verified metadata.
    md = ["## References", "",
          "*Generated by `scripts/make_bibliography.py`; every arXiv entry was",
          "fetched live from the arXiv API. BibTeX: `paper/references.bib`.*",
          ""]
    for aid in sorted(ids, key=lambda a: (meta[a]["year"], meta[a]["authors"][0]
                                          if meta[a]["authors"] else "")):
        m = meta[aid]
        au = m["authors"]
        who = (au[0] if len(au) == 1 else
               f"{au[0]} and {au[1]}" if len(au) == 2 else
               f"{au[0]} et al.")
        md.append(f"- {who} ({m['year']}). *{m['title']}.* "
                  f"arXiv:{aid}. `[{KEYS[aid]}]`")
    md += ["",
           "Classical statistics references (transcribed; verify against the",
           "publisher record before submission):", "",
           "- Rosenbaum, P. R. (1987). *Sensitivity analysis for certain "
           "permutation inferences in matched observational studies.* "
           "Biometrika 74(1), 13–26. `[rosenbaum1987sensitivity]`",
           "- Tan, Z. (2006). *A distributional approach for causal inference "
           "using propensity scores.* JASA 101(476), 1619–1637. "
           "`[tan2006distributional]`",
           "- VanderWeele, T. J. and Ding, P. (2017). *Sensitivity analysis in "
           "observational research: introducing the E-value.* Annals of "
           "Internal Medicine 167(4), 268–274. `[vanderweele2017evalue]`",
           "- Imbens, G. W. and Manski, C. F. (2004). *Confidence intervals "
           "for partially identified parameters.* Econometrica 72(6), "
           "1845–1857. `[imbens2004confidence]`",
           "- Benjamini, Y. and Hochberg, Y. (1995). *Controlling the false "
           "discovery rate.* JRSS-B 57(1), 289–300. "
           "`[benjamini1995controlling]`",
           "- Benjamini, Y. and Yekutieli, D. (2001). *The control of the "
           "false discovery rate in multiple testing under dependency.* "
           "Annals of Statistics 29(4), 1165–1188. `[benjamini2001control]`",
           ""]
    (PAPER / "references.md").write_text("\n".join(md), encoding="utf8")
    n_arxiv = len(ids)
    n_class = len(re.findall(r"^@", CLASSICS, re.M))
    print(f"wrote paper/references.bib: {n_arxiv} verified arXiv entries "
          f"+ {n_class} classics")
    for aid in ids:
        m = meta[aid]
        first = m["authors"][0] if m["authors"] else "?"
        tail = " et al." if len(m["authors"]) > 1 else ""
        print(f"  {KEYS[aid]:32s} {first}{tail} ({m['year']})")


if __name__ == "__main__":
    main()
