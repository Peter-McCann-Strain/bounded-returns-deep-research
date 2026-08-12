# Data And Artifact Inventory

## Versioned data

`data/` now holds compact canonical inputs and checked analysis outputs. See `data/DATA_DICTIONARY.md` and `data/analysis/DATA_DICTIONARY.md`.

## Local artifacts

`artifacts/` holds the physical generated outputs:

| Path | Contents |
|---|---|
| `artifacts/experiments/canonical` | Main generated report corpus formerly at `results/experiments` |
| `artifacts/judges/` | Judge verdict roots formerly under `results/judge_*` |
| `artifacts/experiments/e4_cite` | E4 transformed reports |
| `artifacts/experiments/e5_oracle_dose` | E5 generated reports |
| `artifacts/experiments/e11_vtr` | E11 VTR generated reports |
| `artifacts/experiments/e12_extval` | E12 external validation workspace |
| `artifacts/experiments/drbrace` | DRB-RACE staging/judge outputs |
| `artifacts/caches/` | Search and judge caches formerly under `data/` |
| `artifacts/models/local` | Local model adapters/checkpoints formerly under `models/` |
| `artifacts/checkpoints/local` | Run checkpoints formerly under `checkpoints/` |
| `artifacts/logs` | Logs formerly under `logs/` |

## Manifests

- `artifacts/manifests/pre_migration_inventory.json`: size/count inventory before the migration.
- `artifacts/manifests/migration_ledger.json`: source-to-destination move ledger and compatibility symlinks.

## Compatibility

Legacy paths remain usable locally via symlinks. New code should prefer `deep_research.config` constants such as `ARTIFACTS_DIR`, `EXPERIMENTS_DIR`, `JUDGES_DIR`, `PHASE_REPORTS_DIR`, `PAPERS_DIR`, and `CACHE_DIR`.
