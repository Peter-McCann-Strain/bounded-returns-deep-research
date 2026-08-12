# The Hugging Face dataset repo

**Published:** <https://huggingface.co/datasets/PeterStrain77/bounded-returns-deep-research>

The Hugging Face release is a **dataset** repo holding the tidy evaluation
frames. The code lives on GitHub; the two are complementary, not mirrors.

| Target | Contents | Why |
|---|---|---|
| GitHub | Full codebase, docs, compact inputs, analysis scripts | Code review, issues, CI |
| Hugging Face (dataset) | The `data/analysis/` frames + dataset card | Native parquet viewer, `datasets` loading, LFS for the large verdict frame |

`huggingface/README.md` is the **dataset card** and is the repo-root `README.md`
on the Hub — its YAML frontmatter drives the dataset viewer and the `configs`
split names. It is deliberately a different file from the GitHub `README.md`,
which is a code README with no frontmatter.

## Published layout

Files sit at the **repo root**, not under `data/analysis/`, and the card's
`configs` paths match that:

```
README.md                              <- dataset card (this repo's huggingface/README.md)
.gitattributes                         <- LFS rules for *.parquet
DATA_DICTIONARY.md                     <- column reference; read before using the frames
build_manifest.json
df_queries.parquet                     df_runs.parquet
df_overall_scores.parquet              df_scores.parquet
df_citations.parquet                   df_citations_protocol_a.parquet
df_c0_verdicts.parquet                 df_c0_per_report.parquet
df_e14_oracle_verdicts.parquet         df_e14_oracle_per_report.parquet
df_verdicts-0000{0..5}-of-00006.parquet   <- sharded, see below
```

`df_verdicts` is **sharded across 6 parquet files** rather than shipped as one
28.8 MB file. The card's `verdicts` config globs them with
`data_files: "df_verdicts-*.parquet"`, so `load_dataset` and the viewer treat
them as a single 248,536-row table. Sharding is conventional on the Hub and
keeps individual files small enough for browser upload.

To regenerate the shards from the source frame:

```python
import pandas as pd
df = pd.read_parquet("data/analysis/df_verdicts.parquet")
n, shards = len(df), 6
size = -(-n // shards)
for i in range(shards):
    df.iloc[i*size:(i+1)*size].to_parquet(
        f"df_verdicts-{i:05d}-of-{shards:05d}.parquet", index=False)
```

## Before every push: run the PII gate

Two separate publish attempts were stopped by a leak that a "redaction is done" claim had
already declared clean. Do not skip this because a previous run passed -- the frames change,
and the failures were both in places nobody had thought to look: a `string`-dtype column the
scanner's `dtype == object` filter could not see, and rubric criteria in a JSON manifest that
were generated against a query prompt *before* it was redacted.

```bash
python -c "from deep_research.release_audit import scan_release_for_pii as s; \
           r = s('data/analysis'); print(r or '0 findings')"
```

It must print `0 findings`. If it does not, stop and run
`python scripts/redact_query_identity.py`, then re-run the gate.

Two rules if you extend the redaction:

- **Scope by `query_id`, never by token alone.** A corpus-wide token match once destroyed
  5,000+ unrelated strings. Denylist terms turn up as substrings of ordinary English words
  and of unrelated third-party domains; matched corpus-wide they damage far more than they
  redact. Always confirm afterwards that those unrelated strings are still present.
- **List an identity's variants, not its tidiest spelling.** One tidy display form of a
  business name caught none of the domains, URL slugs, subdomains or community names the
  same identity travelled under.

## Updating the dataset

The token must have **write** permission. A read-scoped token authenticates and
resolves `hf auth whoami` correctly but fails with
`403 Forbidden: You don't have the rights to create a dataset`. Check the role
with `hf auth whoami`; create a write token at
<https://huggingface.co/settings/tokens> if it is `read`.

```bash
pip install -U huggingface_hub      # the `hf` CLI ships with the package;
                                    # the old [cli] extra no longer exists
hf auth login                       # paste a WRITE token

# Stage the payload flat, matching the published layout above.
mkdir -p /tmp/hf-stage
cp huggingface/README.md      /tmp/hf-stage/README.md
cp huggingface/.gitattributes /tmp/hf-stage/.gitattributes
cp data/analysis/*.parquet data/analysis/DATA_DICTIONARY.md \
   data/analysis/build_manifest.json /tmp/hf-stage/
# then shard df_verdicts.parquet as above and remove the unsharded original

hf upload PeterStrain77/bounded-returns-deep-research /tmp/hf-stage . --repo-type dataset
```

The web uploader at `/upload/main` is an alternative that needs no token, but it
flattens directory structure and is impractical for large files.

## Notes

- `df_citations.csv` is intentionally omitted — it duplicates
  `df_citations.parquet` at three times the size. Add it only if a non-parquet
  consumer needs it.
- The dataset viewer takes a few minutes to index after a fresh upload; a
  "should be available soon" placeholder is expected, not an error.
- If you rename the repo, update the card's `Quick start` and `load_dataset`
  examples and the `url` in its citation block together with the commands here.
- Publishing is public and hard to reverse. Re-read the card's "Scope and
  limitations" before pushing new frames.
