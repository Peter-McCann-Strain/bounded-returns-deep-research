#!/usr/bin/env python3
# =============================================================================
# judge_human_gold_panel.py  —  T1_judge_vs_human_panel build engineer artifact
# -----------------------------------------------------------------------------
# Finish the cross-FAMILY human-anchor panel so judge_vs_human is no longer
# single-family (only gpt52 was scored). Adds SONNET (subscription harness) +
# LOCAL-7B (RTX 5080) judges over the on-disk PUBLIC human-label gold sets, then
# `build_judge_vs_human.py --write` picks the verdicts up with ZERO code change.
#
# OPUS IS DROPPED (HARD RULE: no new Opus judging anywhere). This harness can
# only target the SONNET (subscription) and LOCAL-7B endpoints; it refuses Opus.
#
# WHY THIS SHAPE
#   build_judge_vs_human.load_family_verdicts(family, source, gold_rows) already
#   reads, READ-ONLY, from:
#       results/judge_<family>/_human_<source>/*.json
#   where each file is {"item_id": str, "score": float in [0,1], "judge": str}.
#   It joins on item_id to the common-schema gold row from
#   deep_research/benchmarks/gold_loaders.py. So if a NEW judge writes verdicts
#   into that exact store, the canonical builder finalises the cell with no code
#   change. This harness writes EXACTLY that store; it never touches canonical.
#
#   Family -> store dir (read by load_family_verdicts FAMILY_VERDICT_DIRS):
#       claude_sonnet -> results/judge_claude_sonnet48/_human_<source>/
#       local         -> results/judge_local/_human_<source>/
#   (claude_sonnet48 is checked before claude_sonnet by the loader; we use the
#   48 dir so the Sonnet-4.8 subscription judge is unambiguous.)
#
# SOURCES (item-level human grades; HealthBench has its OWN standalone path):
#   expertqa, deepfactbench, draco_full   (each item_id joins straight to gold)
#   HealthBench is finished separately (run_healthbench_judge.py +
#   wire_healthbench_into_judge_vs_human.py) and is NOT re-judged here.
#
# JUDGE INDEPENDENCE (feedback_judge_independence.md):
#   The LOCAL-7B judge must NOT share family/scale with a COMPARED arm. P9's base
#   is Qwen2.5-7B-Instruct and P10 is GAIR/DeepResearcher-7b, so NEITHER may judge.
#   Default local judge = mistralai/Mistral-7B-Instruct-v0.3 (on disk, independent
#   of every compared pattern). Guard REFUSES Qwen2.5-7B / DeepResearcher as judge.
#
# ENDPOINTS
#   --endpoint local        : in-process LocalLLMCaller on the RTX 5080 (no API,
#                             no cost). Writes verdicts directly. Idempotent
#                             (--resume skips items already on disk).
#   --endpoint sonnet-prep  : subscription/file-based harness (feedback_judge_
#                             agents.md: judge agents CANNOT use Bash; pre-generate
#                             criteria, agents use Read+Write only). Writes ONE
#                             task .json per gold item under a tasks/ tree and a
#                             responses/ tree the agent fills in. ZERO API.
#   --endpoint sonnet-parse : read the agent-filled responses/, parse the binary
#                             verdict, and write the {item_id,score,judge} store.
#
# DETERMINISM / SAFETY
#   * Deterministic stratified sample (seeded) per source; balanced across the
#     binary human grade so agreement stats are not degenerate.
#   * Idempotent: existing verdict files are never clobbered; --resume only adds
#     the still-missing items. Writes ONLY under results/judge_<family>/ and
#     (for sonnet) reports/judge_human_panel/. NEVER writes canonical, NEVER
#     touches results/judge_gpt52, results/experiments, data/analysis,
#     reports/eval_v2/verdicts.
#   * No network for local/sonnet-prep/sonnet-parse. (The Sonnet "API" is the
#     subscription harness — a human/agent fills the response files; this script
#     makes ZERO cloud calls in every mode.)
#
# USAGE (ordered; see the run-card the agent emits)
#   # 0. independence + plan, ZERO spend:
#   python scripts/judge_human_gold_panel.py --endpoint local --source deepfactbench --dry-run
#
#   # 1. LOCAL-7B judge (Mistral-7B), all three sources, resumable:
#   python scripts/judge_human_gold_panel.py --endpoint local --source deepfactbench --n 600 --resume
#   python scripts/judge_human_gold_panel.py --endpoint local --source draco_full    --n 800 --resume
#   python scripts/judge_human_gold_panel.py --endpoint local --source expertqa      --n 800 --resume
#
#   # 2. SONNET subscription judge — prep task files (ZERO API), dispatch the
#   #    Claude Code judge agents (Read+Write only), then parse:
#   python scripts/judge_human_gold_panel.py --endpoint sonnet-prep  --source deepfactbench --n 600
#   #    <agents read tasks/*.json, write responses/*.json>
#   python scripts/judge_human_gold_panel.py --endpoint sonnet-parse --source deepfactbench
#
#   # 3. finalise the canonical cells (the existing builder; --write required):
#   python papers/paper_a_bounded_returns/analysis/build_judge_vs_human.py --write
# =============================================================================
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Optional

# repo root on path regardless of cwd
from deep_research.paths import PROJECT_ROOT
REPO_ROOT = PROJECT_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.benchmarks.gold_loaders import LOADERS  # noqa: E402

# ── families / stores (must match build_judge_vs_human.FAMILY_VERDICT_DIRS) ────
FAMILY_STORE_DIR = {
    "claude_sonnet": REPO_ROOT / "results" / "judge_claude_sonnet48",
    "local": REPO_ROOT / "results" / "judge_local",
}
ENDPOINT_TO_FAMILY = {
    "local": "local",
    "sonnet-prep": "claude_sonnet",
    "sonnet-parse": "claude_sonnet",
}
JUDGE_ID = {
    # concrete judge id stamped into each verdict (used by CARE inter-judge corr)
    "local": "mistral7b_local",
    "claude_sonnet": "claude_sonnet_4_8",
}

# Item-level human-label sets this harness covers. HealthBench is finished by the
# dedicated run_healthbench_judge.py + wire_healthbench_into_judge_vs_human.py.
SUPPORTED_SOURCES = ["expertqa", "deepfactbench", "draco_full"]

# Local judge default — INDEPENDENT of every compared pattern (NOT Qwen2.5-7B =
# P9's base, NOT GAIR/DeepResearcher = P10). On disk in the HF cache.
DEFAULT_LOCAL_JUDGE = "mistralai/Mistral-7B-Instruct-v0.3"
FORBIDDEN_LOCAL_JUDGES = {  # share family/scale with a COMPARED arm -> refuse
    "qwen/qwen2.5-7b-instruct",        # P9 base model
    "gair/deepresearcher-7b",          # P10
}

SEED = 20260615  # same seed family as build_judge_vs_human for reproducibility

# Subscription (Sonnet) file-based harness root (NEW; outside the corpus).
SONNET_PANEL_ROOT = REPO_ROOT / "reports" / "judge_human_panel"

# Protected, never-write corpus paths.
PROTECTED = [
    REPO_ROOT / "results" / "judge_gpt52",
    REPO_ROOT / "results" / "experiments",
    REPO_ROOT / "data" / "analysis",
    REPO_ROOT / "reports" / "eval_v2" / "verdicts",
    REPO_ROOT / "papers" / "paper_a_bounded_returns" / "analysis" / "canonical_numbers.json",
]


# ── binary single-criterion judge prompt (mirrors run_healthbench_judge.py) ────
JUDGE_SYSTEM_PROMPT = (
    "You are an expert research-report evaluator using the DRACO evaluation "
    "methodology. You assess whether a single statement/claim/criterion is "
    "SATISFIED by the evidence presented.\n\n"
    "Return a binary verdict:\n"
    "- VERDICT: \"SATISFIED\" or \"NOT_SATISFIED\"\n"
    "- REASONING: one sentence.\n\n"
    "Rules:\n"
    "- Only mark SATISFIED if the criterion is clearly and fully met.\n"
    "- Partial or vague fulfilment counts as NOT_SATISFIED.\n"
    "- Judge ONLY against the stated criterion.\n"
    "Respond with valid JSON only."
)

# Per-source instruction wrapper: each gold row's `text` is the unit to grade and
# `dimension` is the rubric axis. The binary human grade (>=0.5 -> SATISFIED) is
# the target; the judge sees ONLY the claim/criterion text + the axis (it never
# sees the human label), so this is a genuine judge-vs-human agreement test in the
# SAME binary format the GPT-5.2 cell used.
SOURCE_TASK_HINT = {
    "expertqa": ("Axis: {dim}. Decide whether the following expert-domain claim is "
                 "factually correct (for factual_accuracy) or fully attributed (for "
                 "attribution_quality). Claim:\n{text}"),
    "deepfactbench": ("Axis: {dim}. Decide whether the following sentence from a deep-"
                      "research report is FACTUALLY SUPPORTED by reliable evidence.\n"
                      "Sentence:\n{text}"),
    "draco_full": ("Axis: {dim}. The following is an expert evaluation criterion for a "
                   "deep-research report. Decide whether it is a legitimate, well-formed "
                   "MET-target the report SHOULD satisfy (SATISFIED) versus a critical-"
                   "failure the report should NOT trigger (NOT_SATISFIED).\nCriterion:\n{text}"),
}

JUDGE_USER_TEMPLATE = (
    "{task_hint}\n\n"
    "Return JSON in this exact format:\n"
    "{{\n"
    '  "verdict": "SATISFIED" or "NOT_SATISFIED",\n'
    '  "reasoning": "one sentence"\n'
    "}}"
)


# ── safety guards ──────────────────────────────────────────────────────────────
def _is_rel(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_out_safe(*paths: Path) -> None:
    for out in paths:
        out = out.resolve()
        for prot in PROTECTED:
            prot = prot.resolve()
            if out == prot or _is_rel(out, prot) or _is_rel(prot, out):
                raise SystemExit(
                    f"REFUSING: output {out} collides with protected corpus path {prot}."
                )


def assert_independent_local_judge(model_id: str) -> None:
    if model_id.strip().lower() in FORBIDDEN_LOCAL_JUDGES:
        raise SystemExit(
            f"REFUSING: local judge {model_id!r} shares family/scale with a COMPARED "
            f"arm (P9=Qwen2.5-7B / P10=DeepResearcher). Judge independence violated. "
            f"Use an independent model (default {DEFAULT_LOCAL_JUDGE})."
        )


# ── deterministic balanced sample over a source's scoreable gold rows ──────────
def load_scoreable(source: str) -> list[dict]:
    """Gold rows with a usable numeric human grade (None = unscoreable -> dropped)."""
    return [r for r in LOADERS[source]() if r["human_label"] is not None]


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic, balanced 50/50 across the binarised human grade. n<=0 = all.

    Sorted by item_id first so the sample is reproducible regardless of generator
    order; then seeded-shuffled within each class.
    """
    rows = sorted(rows, key=lambda r: r["item_id"])
    if n is None or n <= 0 or n >= len(rows):
        return rows
    pos = [r for r in rows if float(r["human_label"]) >= 0.5]
    neg = [r for r in rows if float(r["human_label"]) < 0.5]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    n_pos = min(half, len(pos))
    n_neg = min(n - n_pos, len(neg))
    if n_pos + n_neg < n:  # top up from the larger class
        n_pos = min(len(pos), n - n_neg)
    chosen = pos[:n_pos] + neg[:n_neg]
    chosen.sort(key=lambda r: r["item_id"])  # stable on-disk order
    return chosen


def build_user_prompt(row: dict) -> str:
    hint = SOURCE_TASK_HINT[row["source"]].format(dim=row["dimension"], text=row["text"])
    return JUDGE_USER_TEMPLATE.format(task_hint=hint)


def parse_verdict(content: str) -> Optional[float]:
    """Parse a judge JSON/string -> 1.0 / 0.0 / None (unparseable)."""
    try:
        obj = json.loads(content)
        verdict = str(obj.get("verdict", "")).strip().upper()
    except Exception:
        verdict = str(content).strip().upper()
    if "NOT_SATISFIED" in verdict or "NOT SATISFIED" in verdict:
        return 0.0
    if "SATISFIED" in verdict:
        return 1.0
    return None


# ── verdict store I/O (the dir load_family_verdicts reads) ─────────────────────
def store_dir(family: str, source: str) -> Path:
    return FAMILY_STORE_DIR[family] / f"_human_{source}"


def existing_item_ids(family: str, source: str) -> set[str]:
    d = store_dir(family, source)
    if not d.is_dir():
        return set()
    out = set()
    for fp in d.glob("*.json"):
        try:
            rec = json.loads(fp.read_text())
            if rec.get("item_id") is not None and rec.get("score") is not None:
                out.add(rec["item_id"])
        except Exception:
            continue
    return out


def write_verdict(family: str, source: str, item_id: str, score: float, judge: str) -> None:
    d = store_dir(family, source)
    d.mkdir(parents=True, exist_ok=True)
    # filename keyed on item_id (sanitised) so it is unique + idempotent.
    safe = item_id.replace("/", "_").replace("|", "_")[:180]
    fp = d / f"{safe}.json"
    if fp.exists():  # never clobber
        return
    tmp = fp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"item_id": item_id, "score": float(score), "judge": judge}))
    tmp.replace(fp)


# ── endpoint: LOCAL-7B (in-process, RTX 5080) ──────────────────────────────────
async def run_local(args) -> None:
    family = "local"
    assert_independent_local_judge(args.local_judge)
    out_d = store_dir(family, args.source)
    assert_out_safe(out_d)

    rows = stratified_sample(load_scoreable(args.source), args.n, args.seed)
    done = existing_item_ids(family, args.source) if args.resume else set()
    todo = [r for r in rows if r["item_id"] not in done]

    print("=" * 78)
    print("LOCAL-7B human-gold judge")
    print("=" * 78)
    print(f"  source        : {args.source}")
    print(f"  local judge   : {args.local_judge}  (independent of P9/P10)")
    print(f"  store (WRITE) : {out_d}")
    print(f"  sampled       : {len(rows)}  already-on-disk: {len(done)}  to judge: {len(todo)}")
    print(f"  cost          : $0.00 (local inference)")
    if args.dry_run:
        if todo:
            ex = todo[0]
            print("-" * 78)
            print("EXACT PROMPT (first item):")
            print("[SYSTEM]\n" + JUDGE_SYSTEM_PROMPT)
            print("[USER]\n" + build_user_prompt(ex))
        print("[DRY RUN] no model loaded, nothing written.")
        return
    if not todo:
        print("  nothing to do (resume).")
        return

    from deep_research.tools.local_llm_caller import LocalLLMCaller
    caller = LocalLLMCaller(model_id=args.local_judge)
    judge = JUDGE_ID[family]
    n_ok = n_unparsed = 0
    for i, row in enumerate(todo, 1):
        try:
            content = await caller.complete(
                prompt=build_user_prompt(row),
                system=JUDGE_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=256,
            )
        except Exception as e:  # surface, skip the item (resume re-tries it)
            print(f"  [{i}/{len(todo)}] ERROR {row['item_id'][:40]}: {type(e).__name__}")
            continue
        score = parse_verdict(content)
        if score is None:
            n_unparsed += 1
            continue
        write_verdict(family, args.source, row["item_id"], score, judge)
        n_ok += 1
        if i % 50 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] written={n_ok} unparsed={n_unparsed}")
    print("-" * 78)
    print(f"  DONE source={args.source} written={n_ok} unparsed={n_unparsed} -> {out_d}")
    print("  NEXT: build_judge_vs_human.py --write to finalise the canonical cell.")


# ── endpoint: SONNET subscription harness (prep / parse; ZERO API) ─────────────
def _sonnet_paths(source: str) -> tuple[Path, Path]:
    base = SONNET_PANEL_ROOT / source
    return base / "tasks", base / "responses"


def run_sonnet_prep(args) -> None:
    family = "claude_sonnet"
    tasks_d, resp_d = _sonnet_paths(args.source)
    out_store = store_dir(family, args.source)
    assert_out_safe(tasks_d, resp_d, out_store)

    rows = stratified_sample(load_scoreable(args.source), args.n, args.seed)
    done = existing_item_ids(family, args.source)
    todo = [r for r in rows if r["item_id"] not in done]

    tasks_d.mkdir(parents=True, exist_ok=True)
    resp_d.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for row in todo:
        safe = row["item_id"].replace("/", "_").replace("|", "_")[:180]
        tfp = tasks_d / f"{safe}.json"
        if tfp.exists():
            continue
        # The agent reads this packet (Read), judges, and writes responses/<safe>.json
        # containing {"item_id","verdict"} — NO Bash, NO API (feedback_judge_agents.md).
        packet = {
            "item_id": row["item_id"],
            "source": args.source,
            "dimension": row["dimension"],
            "system_prompt": JUDGE_SYSTEM_PROMPT,
            "user_prompt": build_user_prompt(row),
            "response_path": str((resp_d / f"{safe}.json").relative_to(REPO_ROOT)),
            "response_schema": {"item_id": row["item_id"],
                                "verdict": "SATISFIED|NOT_SATISFIED"},
        }
        tfp.write_text(json.dumps(packet, indent=2))
        n_written += 1

    print("=" * 78)
    print("SONNET subscription judge — PREP (file-based; ZERO API)")
    print("=" * 78)
    print(f"  source        : {args.source}")
    print(f"  tasks (WRITE) : {tasks_d}   (+{n_written} new packets)")
    print(f"  responses dir : {resp_d}   (agents write {{item_id,verdict}} here, Read+Write only)")
    print(f"  already done  : {len(done)}  sampled: {len(rows)}  pending packets: {len(todo)}")
    print("  cost          : $0.00 (subscription harness; no API key, no Bash for agents)")
    print("  NEXT: dispatch Claude Code Sonnet judge agents over tasks/*.json, then run")
    print(f"        --endpoint sonnet-parse --source {args.source}")


def run_sonnet_parse(args) -> None:
    family = "claude_sonnet"
    tasks_d, resp_d = _sonnet_paths(args.source)
    out_store = store_dir(family, args.source)
    assert_out_safe(out_store)
    judge = JUDGE_ID[family]

    if not resp_d.is_dir():
        raise SystemExit(f"REFUSING: no responses dir {resp_d}; run --endpoint sonnet-prep first.")

    n_ok = n_unparsed = n_missing = 0
    for tfp in sorted(tasks_d.glob("*.json")) if tasks_d.is_dir() else []:
        try:
            packet = json.loads(tfp.read_text())
        except Exception:
            continue
        item_id = packet["item_id"]
        rfp = resp_d / tfp.name
        if not rfp.exists():
            n_missing += 1
            continue
        score = parse_verdict(rfp.read_text())
        if score is None:
            n_unparsed += 1
            continue
        write_verdict(family, args.source, item_id, score, judge)
        n_ok += 1

    print("=" * 78)
    print("SONNET subscription judge — PARSE (responses -> verdict store)")
    print("=" * 78)
    print(f"  source        : {args.source}")
    print(f"  responses     : {resp_d}")
    print(f"  store (WRITE) : {out_store}")
    print(f"  written={n_ok}  unparsed={n_unparsed}  missing-response={n_missing}")
    print("  NEXT: build_judge_vs_human.py --write to finalise the canonical cell.")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", required=True,
                    choices=["local", "sonnet-prep", "sonnet-parse"],
                    help="local = RTX 5080 in-process; sonnet-prep/parse = subscription harness. "
                         "OPUS is intentionally NOT an option (no new Opus judging).")
    ap.add_argument("--source", required=True, choices=SUPPORTED_SOURCES,
                    help="human-label gold set (HealthBench has its own dedicated path).")
    ap.add_argument("--n", type=int, default=0,
                    help="balanced sample size (0 = all scoreable gold rows).")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--local-judge", default=DEFAULT_LOCAL_JUDGE,
                    help=f"local judge model (default {DEFAULT_LOCAL_JUDGE}; must be "
                         "independent of P9/P10).")
    ap.add_argument("--resume", action="store_true",
                    help="skip items already in the verdict store (idempotent).")
    ap.add_argument("--dry-run", action="store_true",
                    help="(local) print plan + exact prompt; load no model, write nothing.")
    args = ap.parse_args()

    if args.endpoint == "local":
        asyncio.run(run_local(args))
    elif args.endpoint == "sonnet-prep":
        run_sonnet_prep(args)
    else:
        run_sonnet_parse(args)


if __name__ == "__main__":
    main()
