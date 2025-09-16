# Repository Guidelines

## Project Structure & Module Organization
- `common/` holds indicator math (`indicators.py`), signal wiring (`strategy.py`), and environment helpers (`env.py`).
- `scripts/` provides the CLIs for downloading inputs, filtering candidates, and running the backtest (`download_data.py`, `analyze_strategy.py`, `backtest_strategy.py`).
- `data/` stores raw Tushare exports (`stock_basic.csv`, `daily.csv`) and stays git-ignored; generated artifacts land beside it (`output.csv`, `backtest_nav.csv`, `backtest_trades.csv`).
- `requirements.txt` pins pandas/numpy/tushare versions; use a local `.venv/` and `.env` with secrets to keep the root clean.

## Build, Test, and Development Commands
Activate a virtual environment before invoking scripts.
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py --days 60 --out data
python scripts/analyze_strategy.py --data data --out output.csv
python scripts/backtest_strategy.py --data data --hold-days 5 --out-trades backtest_trades.csv --out-daily backtest_nav.csv
```
Use `--start`/`--end` (`YYYYMMDD`) on the backtest to isolate regimes, and keep outputs in CSV for quick diffing.

## Coding Style & Naming Conventions
Write Python 3.10+ with four-space indents, `snake_case` names, and type hints that match the existing modules. Keep strategy thresholds centralized in `common/strategy.py` and add short docstrings (English or bilingual) explaining the business rule. Favor vectorized pandas/numpy operations instead of per-row loops, and follow the existing `if __name__ == "__main__"` + `argparse` entrypoint pattern for new tools.

## Testing Guidelines
There is no formal test suite yet; when adjusting indicators or signals, add lightweight `pytest` cases under a new `tests/` folder that exercise DataFrame edge cases, and mirror them against exported CSVs. At minimum, rerun `python scripts/analyze_strategy.py --data data` and `python scripts/backtest_strategy.py --data data --start 20230101 --end 20231231` to confirm selection counts and backtest stats stay sane. Capture summaries (trade count, win rate, drawdown) in the PR so reviewers can compare before/after.

## Commit & Pull Request Guidelines
Follow the existing Conventional Commit style (`feat:`, `fix:`, `chore:`) and keep messages under 72 characters. Each pull request should describe the change, reference related issues, call out data windows used for validation, and attach a short table or screenshot of the new outputs when visuals shift. Tag any required re-downloads or env variable changes clearly, and request review from strategy owners before merging.

## Environment & Secrets
Store `TUSHARE_TOKEN` (or aliases accepted by `common/env.py`) in `.env`; never commit keys or downloaded CSVs. If you rotate tokens, document the update in the PR and nudge teammates to refresh their `.env`. Treat large historical datasets as disposable—regenerate them with `download_data.py` rather than storing them in git.
