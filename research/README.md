# research/ — exploratory scripts

One-off analyses kept for provenance. They are **not** part of the `hkjc` package and sit
outside the lint / type / test gate that governs `src/` and `tests/` (see `extend-exclude`
in `pyproject.toml`). Findings that survived were folded back into the package — e.g. the
13-factor residual group in `src/hkjc/features/base.py` came out of `reports/w456.py`.

| Directory | What it holds |
|---|---|
| `experiments/` | Ad-hoc backtests (WIN / PLACE / quinella / Harville), calibration checks, LightGBM ablations and regularisation sweeps, data-quality diagnostics, and an LLM-agent prototype (`hkjc_crew_system.py`, needs `OPENAI_API_KEY`). |
| `legacy/` | The first pandas + XGBoost scripts that predate the package (`hkjc_model.py`, `hkjc_quinella.py`, `evaluate_*.py`). They read flat CSVs exported from the data lake. |

These scripts read root-level CSVs / `data/` exports that are gitignored — they are a record of
the exploration, not a supported entry point. Use the `hkjc` CLI documented in the top-level
README for the reproducible pipeline.
