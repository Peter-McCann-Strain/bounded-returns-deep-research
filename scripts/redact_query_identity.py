#!/usr/bin/env python
"""Redact real-world identities from every released artefact, one query at a time.

Two benchmark queries carry third-party identity. One names a small business and a named
individual; the other quotes a researcher's email address scraped from an academic byline.
Both query *prompts* were scrubbed long ago; nothing scrubbed the citations the agents
retrieved, the judge reasoning quoting them, or the rubric criteria, which were generated
against the unredacted prompt. Redacting an input does not redact what was derived from it.

Three rules make this safe, and all three are load-bearing:

  1. **Scope by query_id, never by token alone.** An earlier pass matched bare tokens across
     the whole corpus and destroyed 5,000+ unrelated strings. Several denylist terms are
     substrings of ordinary English words and of unrelated third-party domains, so a
     corpus-wide match damages far more than it redacts.
  2. **Longest term first.** One denylist term is a prefix of another; replacing the short
     form first leaves a mangled tail of the long one behind.
  3. **Preserve the column's dtype.** Writing a column back as pandas `string` when it was
     `object` turns every missing value from `None` into `<NA>`. No content changes, but it
     is a schema change to a published frame, and it makes a redaction diff unreadable --
     6,685 rows of a 13-cell edit looked "changed".

Terms live in an untracked, unreleased file: shipping the exact strings that were scrubbed,
in the repository that ships the scrubbed data, hands the identity straight back.

Idempotent: running it twice changes nothing the second time, which is what demonstrates
the first run was complete.
"""
import json, re, sys, pathlib
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from deep_research.release_audit import _PII_TERMS_FILE, load_pii_scopes

PLACEHOLDER = "[REDACTED]"

FRAMES = [
    ("data/analysis/df_citations.parquet", "parquet"),
    ("data/analysis/df_verdicts.parquet", "parquet"),
    ("data/analysis/df_citations.csv", "csv"),
]
JSONS = [
    "data/eval_queries_v2.json",
    "papers/paper_a_bounded_returns/zenodo_v3_upload/data/eval_queries_v2.json",
]


def compile_scope(scope):
    terms = sorted(scope.get("terms", []), key=len, reverse=True)
    word = scope.get("terms_word_boundary", [])
    if not terms and not word:
        return None
    return re.compile("|".join([re.escape(t) for t in terms]
                               + [r"\b%s\b" % re.escape(t) for t in word]), re.I)


def redact_frame(path, kind, scopes):
    d = pd.read_csv(path, low_memory=False) if kind == "csv" else pd.read_parquet(path)
    rows_before, cols_before = len(d), list(d.columns)
    dtypes_before = {c: str(d[c].dtype) for c in d.columns}
    if "query_id" not in d.columns:
        return f"{path}: no query_id column, skipped"

    qid = d["query_id"].astype(str)
    total = 0
    for scope in scopes:
        pat = compile_scope(scope)
        mask = qid == scope["query_id"]
        if pat is None or not mask.any():
            continue
        for c in d.columns:
            if str(d[c].dtype) not in ("object", "string", "category"):
                continue
            original_dtype = d[c].dtype
            s = d[c].astype("string")
            target = s.where(mask)                   # only this query's rows are eligible
            hits = int(target.str.contains(pat, na=False, regex=True).sum())
            if not hits:
                continue
            s = s.mask(mask, target.str.replace(pat, PLACEHOLDER, regex=True))
            # Restore the published dtype: `string` would rewrite None as <NA> corpus-wide.
            d[c] = s.astype(object).where(s.notna(), None) if str(original_dtype) == "object" \
                else s.astype(original_dtype)
            total += hits

    assert len(d) == rows_before and list(d.columns) == cols_before, "shape changed"
    after = {c: str(d[c].dtype) for c in d.columns}
    drift = {c: (dtypes_before[c], after[c]) for c in d.columns if dtypes_before[c] != after[c]}
    assert not drift, f"dtype drift: {drift}"

    d.to_csv(path, index=False) if kind == "csv" else d.to_parquet(path, index=False)
    return f"{path}: redacted {total} cell(s); {rows_before:,} rows and all dtypes preserved"


def redact_json(path, scopes):
    """The rubric criteria were written against the unredacted prompt, so they leak too."""
    p = pathlib.Path(path)
    if not p.exists():
        return f"{path}: absent, skipped"
    raw = json.loads(p.read_text())
    by_id = {s["query_id"]: compile_scope(s) for s in scopes}
    n = 0

    def walk(node, pat):
        nonlocal n
        if isinstance(node, dict):
            return {k: walk(v, pat) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, pat) for v in node]
        if isinstance(node, str) and pat.search(node):
            n += 1
            return pat.sub(PLACEHOLDER, node)
        return node

    def scoped(node):
        if isinstance(node, dict):
            key = node.get("query_id") or node.get("id")
            if key in by_id and by_id[key] is not None:
                return walk(node, by_id[key])
            return {k: scoped(v) for k, v in node.items()}
        if isinstance(node, list):
            return [scoped(v) for v in node]
        return node

    p.write_text(json.dumps(scoped(raw), indent=2, ensure_ascii=False) + "\n")
    return f"{path}: redacted {n} string(s) inside the scoped query entries"


if __name__ == "__main__":
    import os
    os.chdir(pathlib.Path(__file__).resolve().parents[1])
    scopes = load_pii_scopes()
    if not scopes:
        sys.exit(f"ERROR: {_PII_TERMS_FILE} not found or empty. It is deliberately untracked; "
                 "obtain it from the maintainer. Refusing to guess what to redact.")
    print(f"{len(scopes)} scope(s) loaded")
    for path, kind in FRAMES:
        print(redact_frame(path, kind, scopes))
    for path in JSONS:
        print(redact_json(path, scopes))
