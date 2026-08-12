"""Release tree audit utilities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_NAME = "PUBLIC_MANIFEST.json"
EXPORT_REPORT_NAME = "PUBLIC_EXPORT_REPORT.json"

SAFE_ENV_TEMPLATES = {".env.example", ".env.template"}

# Defence in depth: paths that must never reach a public tree even if a manifest
# edit would otherwise allow them. This list deliberately does NOT block the
# paper's statistical analysis code (papers/**/analysis/*.py) or the tidy
# analysis frames (data/analysis/**) — the public release ships both so the
# reported numbers can be recomputed. It does block the LaTeX build around them.
FORBIDDEN_PATH_PATTERNS = [
    # Unredacted data backups. data/analysis/** is deliberately released so the
    # reported numbers recompute, and the include glob would otherwise sweep these
    # up -- they hold the personal identifiers the released frames were scrubbed of.
    # A release audit that ships these silently defeats the entire redaction.
    "**/.pre_redaction_backup/**",
    "**/*pre_redaction*",
    # Secrets and local environment
    ".env",
    ".env.*",
    ".cudatk/**",
    "venv/**",
    ".venv/**",
    # Caches and build metadata
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".mypy_cache/**",
    "**/__pycache__/**",
    "*.pyc",
    "*.egg-info/**",
    # Assistant/agent working files and private notes
    ".claude/**",
    "**/CLAUDE.md",
    "CLAUDE.md",
    "**/AGENTS.md",
    "AGENTS.md",
    "memory-bank/**",
    "**/memory-bank/**",
    "scratchpad*/**",
    "**/*.workflow.js",
    "CODEX_REVIEW_PROMPT.md",
    "REBOOT_STATE_*.md",
    "RESUME_AFTER_REBOOT.sh",
    "err.txt",
    "docs/claude_code_evaluation.md",
    "docs/publication/publication_control_room.md",
    # Generated forests, weights, and vendored upstreams
    "artifacts/**",
    "results/**",
    "reports/**",
    "logs/**",
    "models/**",
    "checkpoints/**",
    "external_frameworks/**",
    "archive/**",
    "zenodo_upload/**",
    "analysis/**",
    # Paper: LaTeX build, drafts, submissions, and private material
    "papers/drafts/**",
    "papers/archive/**",
    "papers/**/*.tex",
    "papers/**/*.bib",
    "papers/**/*.txt",
    "papers/**/*.bak*",
    "papers/paper_a_bounded_returns/archive_unused_figures/**",
    "papers/paper_a_bounded_returns/audit_*/**",
    "papers/paper_a_bounded_returns/arxiv_submission*",
    "papers/paper_a_bounded_returns/arxiv_submission*/**",
    "papers/paper_a_bounded_returns/blog/**",
    "papers/paper_a_bounded_returns/build/**",
    "papers/paper_a_bounded_returns/docs/**",
    "papers/paper_a_bounded_returns/reports/**",
    "papers/paper_a_bounded_returns/reviews/**",
    "papers/paper_a_bounded_returns/sections/**",
    "papers/paper_a_bounded_returns/submission_tmlr/**",
    "papers/paper_a_bounded_returns/zenodo_v*/**",
    "papers/paper_a_bounded_returns/personal_website_export/**",
    "papers/paper_a_bounded_returns/public_release/**",
    "papers/paper_a_bounded_returns/*outreach*",
    # LaTeX and binary build products
    "*.aux",
    "*.out",
    "*.log",
    "*.blg",
    "*.zip",
    "*.pt",
    "*.bin",
    "*.safetensors",
    "*.gguf",
]

TEXT_SCAN_EXTENSIONS = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

LOCAL_OR_PRIVATE_MARKERS = (
    "Deep_" + "Research_Projects",
    "file" + "://",
)
# Matches absolute and scheme-relative URLs. Stored citation values are not
# always scheme-complete, so the scheme is optional.
URL_RE = re.compile(r"(?:\bhttps?)?://\S+", re.IGNORECASE)
LOCAL_PATH_RE = re.compile(
    r"(?:/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|"
    r"[A-Za-z]:(?:\\+|/+)(?:Users|Documents and Settings)(?:\\+|/+)"
    r"[A-Za-z0-9._ -]+(?:\\+|/+))",
    re.IGNORECASE,
)

PROVIDER_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:OPENAI|ANTHROPIC|AZURE_OPENAI|TAVILY|SEMANTIC_SCHOLAR|HF|"
    r"HUGGINGFACE|WANDB)_[A-Z0-9_]*(?:API_KEY|TOKEN)\b[ \t]*=[ \t]*"
    r"(?P<value>[^\s#]+)"
)
GENERIC_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:api[_-]?key|password|secret|token)\b[ \t]*[:=][ \t]*"
    r"(?P<value>['\"]?[A-Za-z0-9][A-Za-z0-9_\-./+=]{15,}['\"]?)",
    re.IGNORECASE,
)

PLACEHOLDER_VALUES = {
    "",
    "''",
    '""',
    "none",
    "null",
    "test",
    "key",
    "k",
    "fake",
    "dummy",
    "changeme",
    "your-api-key",
    "your_api_key",
}


@dataclass(frozen=True)
class AuditFinding:
    severity: str
    path: str
    message: str


@dataclass(frozen=True)
class AuditResult:
    root: str
    findings: list[AuditFinding]

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_json(self) -> str:
        return json.dumps(
            {"root": self.root, "ok": self.ok, "findings": [asdict(f) for f in self.findings]},
            indent=2,
        )


def load_public_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and minimally validate the public release manifest."""
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object")
    return manifest


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_release_files(root: Path):
    """Yield files in a release tree, ignoring VCS metadata."""
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file():
            yield path


@lru_cache(maxsize=4096)
def _compile_public_glob(pattern: str) -> re.Pattern[str]:
    pattern = pattern.replace("\\", "/").lstrip("/")
    if pattern.endswith("/"):
        pattern += "**"
    chunks: list[str] = []
    idx = 0
    while idx < len(pattern):
        char = pattern[idx]
        if char == "*":
            if idx + 1 < len(pattern) and pattern[idx + 1] == "*":
                idx += 2
                if idx < len(pattern) and pattern[idx] == "/":
                    idx += 1
                    chunks.append("(?:.*/)?")
                else:
                    chunks.append(".*")
                continue
            chunks.append("[^/]*")
        elif char == "?":
            chunks.append("[^/]")
        else:
            chunks.append(re.escape(char))
        idx += 1
    return re.compile("^" + "".join(chunks) + "$")


def matches_public_glob(rel_path: str, pattern: str) -> bool:
    return bool(_compile_public_glob(pattern).match(rel_path.replace("\\", "/").lstrip("/")))


def matches_any_public_glob(rel_path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(matches_public_glob(rel_path, pattern) for pattern in patterns)


def manifest_patterns(manifest: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Return include, exclude, and required path patterns from a manifest.

    The current manifest uses explicit include/exclude globs. Legacy manifests
    from earlier release prep are also accepted to keep the CLI stable.
    """
    includes = list(manifest.get("include_globs", []))
    excludes = list(manifest.get("exclude_globs", []))
    required = list(manifest.get("required_paths", []))

    if not includes:
        includes.extend(manifest.get("allowed_top_level_files", []))
        includes.extend(manifest.get("allowed_paper_files", []))
        includes.extend(f"{root.rstrip('/')}/**" for root in manifest.get("allowed_roots", []))

    return includes, excludes, required


def is_manifest_allowed(rel_path: str, manifest: dict[str, Any]) -> bool:
    includes, excludes, _ = manifest_patterns(manifest)
    if rel_path in SAFE_ENV_TEMPLATES:
        return matches_any_public_glob(rel_path, includes)
    if matches_any_public_glob(rel_path, excludes):
        return False
    return matches_any_public_glob(rel_path, includes)


def _is_forbidden(rel_path: str) -> str | None:
    if rel_path in SAFE_ENV_TEMPLATES:
        return None
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if matches_public_glob(rel_path, pattern):
            return pattern
    return None


def _normalise_secret_value(value: str) -> str:
    return value.strip().strip("'\"").strip()


def _is_placeholder_secret_value(value: str) -> bool:
    cleaned = _normalise_secret_value(value)
    lowered = cleaned.lower()
    if lowered in PLACEHOLDER_VALUES:
        return True
    if lowered.startswith("<") and lowered.endswith(">"):
        return True
    if lowered.startswith(("your-", "your_", "example-", "example_", "test-", "dummy-")):
        return True
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", cleaned):
        return True
    if re.fullmatch(r"[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN)", cleaned):
        return True
    return cleaned.startswith(("values.get(", "os.getenv(", "getenv("))


def _read_parquet_text(path: Path) -> str | None:
    """Return parquet cell/column text, or None if it cannot be decoded.

    Scanning raw parquet bytes is unreliable: the payload is compressed and
    dictionary-encoded, so a URL can be split such that its scheme and path land
    in different chunks. A cited publisher URL then looks like a local home
    directory path. Reading the actual column values avoids this.
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    chunks: list[str] = [" ".join(str(column) for column in frame.columns)]
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object or str(series.dtype).startswith("string"):
            chunks.append("\n".join(series.dropna().astype(str).unique().tolist()))
    return "\n".join(chunks)


def _read_scannable_text(path: Path, *, text_hint: bool) -> str:
    if text_hint:
        return path.read_text(errors="ignore")
    if path.suffix.lower() == ".parquet":
        decoded = _read_parquet_text(path)
        if decoded is not None:
            return decoded
    data = path.read_bytes()
    # Latin-1 preserves byte values and makes embedded PDF/binary metadata visible
    # without treating arbitrary bytes as a decoding failure.
    return data.decode("latin-1", errors="ignore")


def _find_secret_or_local_marker(text: str) -> str | None:
    lowered = text.lower()
    for marker in LOCAL_OR_PRIVATE_MARKERS:
        if marker.lower() in lowered:
            return f"contains private/local marker `{marker}`"

    # Cited web URLs legitimately contain segments that look like home or user
    # directories (a publisher path such as `<host>/en/home/<section>/`). Those
    # are remote paths, not local ones, so strip URLs before looking for
    # filesystem paths. Local-file URL schemes stay in scope via
    # LOCAL_OR_PRIVATE_MARKERS above.
    match = LOCAL_PATH_RE.search(URL_RE.sub(" ", text))
    if match:
        return "contains absolute local user path"

    for match in PROVIDER_SECRET_ASSIGNMENT_RE.finditer(text):
        if not _is_placeholder_secret_value(match.group("value")):
            return "contains provider API key/token assignment with a non-placeholder value"

    for match in GENERIC_SECRET_ASSIGNMENT_RE.finditer(text):
        if not _is_placeholder_secret_value(match.group("value")):
            return "contains secret-like assignment with a non-placeholder value"

    return None


def audit_release_tree(
    root: Path,
    *,
    max_file_mb: int | None = None,
    manifest_path: Path | None = None,
    enforce_manifest: bool | None = None,
) -> AuditResult:
    """Audit a candidate public release tree."""
    root = root.resolve()
    findings: list[AuditFinding] = []

    manifest: dict[str, Any] | None = None
    candidate_manifest = manifest_path or (root / DEFAULT_MANIFEST_NAME)
    if candidate_manifest.exists():
        manifest = load_public_manifest(candidate_manifest)
    elif manifest_path is not None:
        findings.append(
            AuditFinding("error", DEFAULT_MANIFEST_NAME, f"manifest not found: {manifest_path}")
        )

    if enforce_manifest is None:
        enforce_manifest = manifest is not None
    if max_file_mb is None:
        max_file_mb = int(manifest.get("max_file_mb", 10)) if manifest else 10
    max_bytes = max_file_mb * 1024 * 1024

    if not root.exists():
        return AuditResult(
            root=str(root), findings=[AuditFinding("error", ".", "root does not exist")]
        )

    if manifest and enforce_manifest:
        _, _, required = manifest_patterns(manifest)
        for required_path in required:
            if not (root / required_path).exists():
                findings.append(
                    AuditFinding("error", required_path, "required public file is missing")
                )

    for path in iter_release_files(root):
        rel = _relative(path, root)

        if manifest and enforce_manifest and not is_manifest_allowed(rel, manifest):
            findings.append(AuditFinding("error", rel, "not allowed by PUBLIC_MANIFEST"))

        forbidden_pattern = _is_forbidden(rel)
        if forbidden_pattern:
            findings.append(
                AuditFinding("error", rel, f"forbidden public-release path: {forbidden_pattern}")
            )

        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            findings.append(
                AuditFinding("error", rel, f"file exceeds public size limit of {max_file_mb} MB")
            )

        # The export report is generated by this tool and quotes the findings
        # verbatim, so scanning it would re-flag the very markers it reports.
        # Its own paths are repository-relative by construction.
        if rel == EXPORT_REPORT_NAME:
            continue

        try:
            text_hint = path.suffix.lower() in TEXT_SCAN_EXTENSIONS or path.name.startswith(".env")
            text = _read_scannable_text(path, text_hint=text_hint)
        except OSError:
            continue
        marker = _find_secret_or_local_marker(text)
        if marker:
            findings.append(AuditFinding("error", rel, marker))

    return AuditResult(root=str(root), findings=findings)

# --- PII gate -------------------------------------------------------------
# audit_release_tree previously checked manifest membership, forbidden paths,
# size and secret markers -- but had NO personal-data check. The released
# frames were scrubbed of a named individual, a small-business identifier and
# a private planning-board reference; nothing verified that the scrub held.
# The denylist itself is sensitive: a list of the exact strings scrubbed from the data,
# published in the same repository as that data, hands back what the redaction removed. So
# the terms live in an untracked, unreleased file and this module ships with none of them.
#
# One display form of a name is also not the name -- a single tidy spelling missed every URL
# slug, domain and community name the same identity travelled under. Record the variants.
_PII_TERMS_FILE = "config/pii_terms.local.json"


def load_pii_scopes():
    """Return the private denylist as [{query_id, terms, terms_word_boundary}], or [].

    Absent on a clean clone by design. The generic patterns below still run, so the gate
    degrades to shape-matching rather than silently passing everything.
    """
    import json, pathlib
    for base in (pathlib.Path.cwd(), pathlib.Path(__file__).resolve().parents[1]):
        p = base / _PII_TERMS_FILE
        if p.exists():
            try:
                return json.loads(p.read_text()).get("scopes", [])
            except Exception:
                break
    return []


def _load_pii_terms():
    """Flatten every scope's terms for scanning, where scope does not matter."""
    terms, word = [], []
    for s in load_pii_scopes():
        terms += s.get("terms", [])
        word += s.get("terms_word_boundary", [])
    return terms, word


# A named-term list only catches identities someone already knew to look for. The frames
# also quote scraped web text verbatim, which can carry a third party's contact details
# from a page byline -- so scan for the shape of an address too.
PII_PATTERNS = [r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"]

# The author's own contact details are printed in the paper on purpose. They are the one
# thing this gate must not flag, or every run fails on the byline and the gate gets ignored
# -- which is how a real finding ends up dismissed as noise.
PII_ALLOW = [
    "peter@petermccannstrain.com",
    "noreply@",
    "example.com",
    "user@host",
]


def _allowed(match):
    """True for the author's own published contact details and placeholder addresses."""
    m = str(match).lower()
    return any(a.lower() in m for a in PII_ALLOW)


def _pii_pattern(literal_only=False):
    """Compile the denylist so glued and split forms cannot slip past.

    Personal names need care. A both-sides `\\b` misses "<Name>99"; a left-only `\\b` misses
    "99<Name>" and "Mr<Name><Surname>". But a bare case-insensitive substring match fires on
    every ordinary word that happens to contain the name -- and the real corpus contains
    several, plus a third-party domain.

    The rule that separates them is CASE, not position. A name embedded in an English word
    appears in that word's own lowercase run; a name used as a name keeps its capital. So
    match the capitalised form anywhere at all, and additionally match any casing when the
    term stands alone between word boundaries.
    """
    terms, terms_word = _load_pii_terms()
    parts = [re.escape(t) for t in terms]
    for t in terms_word:
        parts.append("(?-i:%s)" % re.escape(t[:1].upper() + t[1:]))   # glued but capitalised
        parts.append(r"\b%s\b" % re.escape(t))                        # standalone, any case
    if literal_only:
        return re.compile("|".join(parts), re.I) if parts else re.compile(r"(?!x)x")
    return re.compile("|".join(parts + PII_PATTERNS), re.I)


def _normalise_for_scan(text):
    """Yield variants of ``text`` that defeat literal-splitting tricks.

    A name split across a string concatenation, across adjacent literals, or across a
    newline inside a triple-quoted string is invisible to a plain search -- which is exactly
    how one leak survived a grep sweep. Rejoining adjacent quoted fragments, and collapsing
    whitespace, makes all three forms searchable.
    """
    yield text
    yield re.sub(r"['\"]\s*(?:\+\s*)?['\"]", "", text)      # rejoin adjacent literals
    yield re.sub(r"\s+", "", text)                          # defeat newline splitting


def _pii_matches(path, pat):
    """Return every denylist match in a file, whatever its type."""
    import subprocess
    try:
        raw = path.read_bytes()
    except Exception:
        return []
    text = raw.decode("utf-8", errors="ignore")
    # A PDF stores its text in compressed streams, so the raw bytes reveal nothing. This is
    # the one shipped file type where reading the bytes is actively misleading.
    if path.suffix.lower() == ".pdf":
        try:
            text = subprocess.run(["pdftotext", str(path), "-"], capture_output=True,
                                  text=True, timeout=120).stdout
        except Exception:
            return []
    # Literal denylist terms are hunted in every variant; the generic shape patterns run on
    # the raw text alone. Collapsing whitespace welds unrelated tokens into strings that look
    # like addresses ("...asyncio" next to "@pytest.mark..."), and a gate that cries wolf on
    # ordinary test files is a gate that gets switched off.
    # Compressed binaries (images, archives, fonts) are byte soup: the address-shaped
    # pattern hits random spans in a PNG's pixel data. Scan them for
    # literal denylist terms only -- those can still turn up in embedded metadata --
    # and leave the shape patterns to text.
    BINARY = {".png", ".jpg", ".jpeg", ".gif", ".zip", ".gz", ".tar", ".woff", ".woff2",
              ".ttf", ".otf", ".ico", ".pyc", ".so", ".npy", ".npz", ".pkl"}
    literal = _pii_pattern(literal_only=True)
    if path.suffix.lower() in BINARY:
        return literal.findall(text)
    out = list(pat.findall(text))
    for variant in list(_normalise_for_scan(text))[1:]:
        out += literal.findall(variant)
    return out


def scan_release_for_pii(root):
    """Return [(path, term, n_rows)] for any released data file still carrying PII.

    Checked against the released artefacts, not the working tree, so a stale or
    partially-redacted copy cannot slip through.
    """
    import re, pathlib
    try:
        import pandas as pd
    except ImportError:
        return []
    pat = _pii_pattern()
    hits = []
    for f in pathlib.Path(root).rglob("*"):
        if not f.is_file():
            continue
        if f.suffix in (".parquet", ".csv"):
            pass  # handled by the dataframe branch below
        else:
            # Scan EVERY other shipped file, whatever its extension. An allowlist of
            # suffixes silently skipped LICENSE, NOTICE, CITATION.cff, .gitattributes and
            # main.pdf -- seven real shipped files the gate could not read at all.
            n = len([m for m in _pii_matches(f, pat) if not _allowed(m)])
            if n:
                hits.append((str(f), "PII", n))
            continue
        try:
            d = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        # Selecting on `dtype == object` alone silently skipped every pandas
        # `string` and `category` column. df_verdicts.evidence -- the verbatim
        # quoted-source column, the likeliest place in the release for a third
        # party's details -- is `string`, so the gate could not see it and
        # reported clean while an address sat in it.
        text = [c for c in d.columns
                if d[c].dtype == object or str(d[c].dtype) in ("string", "category")]
        if not text:
            continue
        blob = d[text].astype(str).agg(" ".join, axis=1)
        n = int(blob.str.contains(pat, regex=True, na=False).sum())
        if n:
            hits.append((str(f), "PII", n))
    return hits
