#!/usr/bin/env python
"""Re-derive every downstream bundle from main.tex, in one command.

Four bundles ship the same paper: the TMLR blinded copy, the arXiv source, the Zenodo
deposit, and the public release. Keeping them in step by hand is what let the TMLR copy
drift 383 body lines from main.tex. Everything here is a copy or a scripted transform;
nothing is authored in a bundle.

  submission_tmlr/   <- make_tmlr_submission.py (preamble + anonymisation transform)
  arxiv_submission/  <- main.tex verbatim + tables/ + figures/ + bibliography
  zenodo_v3_upload/  <- same, plus the built main.pdf
  <release>/paper/   <- built PDF only, when --release-dir is given

Usage:  python sync_artefacts.py [--release-dir DIR]
"""
import argparse, filecmp, os, shutil, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from deep_research.paths import PROJECT_ROOT

PAPER = f"{PROJECT_ROOT}/papers/paper_a_bounded_returns"
BIBS = ["references.bib", "references_new.bib"]

# `main.tex` stays `main.tex` -- every bundle, script and build path depends on it. But no
# reader should receive a file called `main.pdf`: it says nothing, collides with every other
# paper in a downloads folder, and the live Zenodo record already uses the title-based name,
# so matching it keeps the new version consistent with the one it supersedes.
PDF_NAME = "bounded-returns-to-orchestration.pdf"
BLIND_PDF_NAME = "bounded-returns-to-orchestration-anonymous.pdf"


def sync_dir(src, dst):
    os.makedirs(dst, exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete", src + "/", dst + "/"], check=True)


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def drop_stale(d):
    """Remove superseded copies so a bundle can never be uploaded with junk in it."""
    for f in os.listdir(d):
        if ".stale" in f or f.endswith("~"):
            os.remove(os.path.join(d, f)); print(f"  removed stale {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", help="export tree to place the built PDF into")
    a = ap.parse_args()

    if not os.path.exists(f"{PAPER}/main.pdf"):
        sys.exit("ERROR: main.pdf missing -- build main.tex first")

    print("submission_tmlr/")
    subprocess.run([sys.executable, f"{PAPER}/analysis/make_tmlr_submission.py"], check=True)
    for b in BIBS:
        copy(f"{PAPER}/{b}", f"{PAPER}/submission_tmlr/{b}")
    for sub in ("tables", "figures"):
        sync_dir(f"{PAPER}/{sub}", f"{PAPER}/submission_tmlr/{sub}")

    for bundle, want_pdf in (("arxiv_submission", False), ("zenodo_v3_upload/paper_source", True)):
        print(f"{bundle}/")
        d = f"{PAPER}/{bundle}"
        os.makedirs(d, exist_ok=True)
        copy(f"{PAPER}/main.tex", f"{d}/main.tex")
        for b in BIBS:
            copy(f"{PAPER}/{b}", f"{d}/{b}")
        for sub in ("tables", "figures"):
            sync_dir(f"{PAPER}/{sub}", f"{d}/{sub}")
        drop_stale(d)
        if want_pdf:
            copy(f"{PAPER}/main.pdf", f"{PAPER}/zenodo_v3_upload/{PDF_NAME}")
            stale = f"{PAPER}/zenodo_v3_upload/main.pdf"
            if os.path.exists(stale):
                os.remove(stale); print(f"  removed main.pdf (superseded by {PDF_NAME})")

    # Reader-facing copies, named for the paper rather than for the build system.
    copy(f"{PAPER}/main.pdf", f"{PAPER}/{PDF_NAME}")
    if os.path.exists(f"{PAPER}/submission_tmlr/main_tmlr.pdf"):
        copy(f"{PAPER}/submission_tmlr/main_tmlr.pdf",
             f"{PAPER}/submission_tmlr/{BLIND_PDF_NAME}")

    if a.release_dir:
        print(f"{a.release_dir}/paper/")
        copy(f"{PAPER}/main.pdf", f"{a.release_dir}/paper/{PDF_NAME}")

    # A bundle whose main.tex differs from the source is drift by definition; the TMLR
    # copy is excluded because its four deltas are the point of make_tmlr_submission.py.
    bad = [b for b in ("arxiv_submission", "zenodo_v3_upload/paper_source")
           if not filecmp.cmp(f"{PAPER}/main.tex", f"{PAPER}/{b}/main.tex", shallow=False)]
    print("VERIFY: " + ("all bundles match main.tex" if not bad else f"DRIFT in {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
