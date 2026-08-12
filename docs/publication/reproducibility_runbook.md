# Reproducibility Runbook

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
venv/bin/python -m compileall -q deep_research scripts
```

Required secrets live in `.env` and are never committed. Recommended keys:

- `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT`
- `JUDGE_OPENAI_API_KEY` / `JUDGE_OPENAI_ENDPOINT` if different from search/generation
- `SEMANTIC_SCHOLAR_API_KEY` for healthy S2 retrieval
- Hugging Face token only when gated local models are required

## Layout rules

- Canonical compact inputs: `data/`
- Manuscripts: `papers/`
- Generated outputs: `artifacts/`
- Live programme docs: `docs/publication/`
- Legacy compatibility: ignored symlinks at `results/`, `models`, `logs`, `checkpoints`, and `data/*_cache`

## Safety checks before commit/push

```bash
git status --short --branch
find . -path ./.git -prune -o -type f -size +50M -printf '%s %p\n'
git diff --cached --stat
```

Do not stage raw caches, local model weights, generated report forests, logs, or checkpoints.

## Paper A rebuild gate

From `papers/paper_a_bounded_returns/`:

1. Compile `main.tex`.
2. Regenerate extracted text from the PDF.
3. Compare manuscript numbers with `analysis/canonical_numbers.json`.
4. Verify the retrieval relabel/disclosure text remains present.
5. Record the build timestamp and output checks in the paper lane README.

## Judging rule

GPT-5.2 is authoritative. Claude, GPT-4o, DR-Judge, and local models may be comparison panels, subjects, selectors, or detectors, but not the primary judge unless a future protocol explicitly changes that rule.
