#!/usr/bin/env python
"""Derive submission_tmlr/main_tmlr.tex from main.tex.

The blinded copy used to be hand-maintained, so every prose fix had to land twice and
the two drifted (383 body lines apart before this script existed). Here the body is
taken verbatim from main.tex and only the deltas that double-blind review actually
requires are applied:

  1. preamble  -- tmlr.sty in place of geometry/titlesec/natbib, empty PDF metadata
  2. title     -- the one-line subtitle tmlr's narrower title block fits
  3. author    -- removed; tmlr.sty prints "Anonymous authors" itself
  4. artefact  -- the Code:/Data: URLs replaced by the withheld-for-review sentence

Run from anywhere; writes submission_tmlr/main_tmlr.tex and reports the delta count.
Anything other than these four deltas showing up in that count is drift, not design.
"""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from deep_research.paths import PROJECT_ROOT

PAPER = f"{PROJECT_ROOT}/papers/paper_a_bounded_returns"
SRC, DST = f"{PAPER}/main.tex", f"{PAPER}/submission_tmlr/main_tmlr.tex"

PREAMBLE = r"""\documentclass{article}

%,, - TMLR style (blinded: no [accepted]/[preprint] => double-blind, Anonymous authors),, -
% tmlr.sty loads natbib itself and sets author-year via \setcitestyle; it also fixes
% its own page geometry and section formatting, so the paper's geometry/titlesec/natbib
% are removed below to avoid clashes.
\usepackage{tmlr}
"""

TITLE_SRC = r"""\author{Peter McCann Strain\\
\normalsize \texttt{peter@petermccannstrain.com}\\
\normalsize \href{https://petermccannstrain.com}{petermccannstrain.com}}
\date{\today}
"""
TITLE_DST = r"""% Author block removed for double-blind submission.
% tmlr.sty prints "Anonymous authors / Paper under double-blind review".
"""

AVAIL_SRC = r"""Code: \url{https://github.com/Peter-McCann-Strain/bounded-returns-deep-research}.
Data: \url{https://huggingface.co/datasets/PeterStrain77/bounded-returns-deep-research},
the"""
AVAIL_DST = r"""Code and data are released publicly; the repository and dataset URLs are withheld here to
preserve double-blind review and will be supplied on acceptance. The release comprises
the"""

# Preamble packages tmlr.sty supplies or overrides. Each is commented out rather than
# deleted so a reader of the blinded source can see what was dropped and why.
DROP = [
    (r"\usepackage[letterpaper,margin=1in]{geometry}",
     "% (geometry removed: tmlr sets \\textwidth/\\textheight/margins)"),
    (r"\usepackage{titlesec}", "% (titlesec removed: tmlr defines its own \\section/\\subsection formatting)"),
    (r"\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{0.6em}{}", None),
    (r"\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}", None),
    (r"\titlespacing*{\section}{0pt}{1.4ex plus .3ex}{0.8ex}", None),
    (r"\titlespacing*{\subsection}{0pt}{1.1ex plus .2ex}{0.5ex}", None),
    (r"\usepackage[numbers,sort&compress]{natbib}",
     "% (natbib removed: tmlr loads natbib in author-year mode via \\setcitestyle)"),
]

s = open(SRC).read()
applied = []

assert s.startswith(r"\documentclass[11pt]{article}"), "main.tex preamble moved"
s = s.replace(r"\documentclass[11pt]{article}", PREAMBLE.rstrip("\n"), 1); applied.append("documentclass+tmlr")

for old, new in DROP:
    if old not in s:
        sys.exit(f"ERROR: preamble line vanished from main.tex, refusing to guess: {old}")
    s = s.replace(old, new, 1) if new else re.sub(re.escape(old) + r"\n", "", s, count=1)
applied.append(f"{len(DROP)} preamble drops")

# Strip the identifying PDF metadata rather than carrying the author's name in the file.
s = re.sub(r"pdftitle=\{[^}]*\},pdfauthor=\{[^}]*\}", "pdftitle={},pdfauthor={}", s, count=1)
applied.append("pdf metadata blanked")

for name, old, new in [("title/author", TITLE_SRC, TITLE_DST),
                       ("availability", AVAIL_SRC, AVAIL_DST)]:
    if s.count(old) != 1:
        sys.exit(f"ERROR: {name} block not found verbatim in main.tex ({s.count(old)} matches)")
    s = s.replace(old, new, 1); applied.append(name)

open(DST, "w").write(s)

ident = [t for t in ("Peter", "McCann", "Strain", "petermccannstrain",
                     "github.com/Peter", "PeterStrain77", "orcid") if t in s]
print("wrote main_tmlr.tex; applied: " + ", ".join(applied))
print(f"identifying tokens remaining in source: {len(ident)}" + (f" -> {ident}" if ident else " (clean)"))
if ident:
    sys.exit(1)
