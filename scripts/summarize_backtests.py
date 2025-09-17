import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _parse_meta(name: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    # weight scheme
    if "equal" in name:
        meta["scheme"] = "equal"
    elif "momentum_heavy" in name or "mom" in name:
        meta["scheme"] = "momentum_heavy"
    elif "momentum_tilt" in name or "tilt" in name:
        meta["scheme"] = "momentum_tilt"
    else:
        meta["scheme"] = "unknown"

    # TopN: look for _n{num}_ or equal{num}
    m = re.search(r"_n(\d+)_", name)
    if m:
        meta["topn"] = m.group(1)
    else:
        m2 = re.search(r"equal(\d+)", name)
        if m2:
            meta["topn"] = m2.group(1)

    # turnover: _t{token}, e.g., t5e6, t1e7
    m = re.search(r"_t([0-9a-zA-Z]+)", name)
    if m:
        meta["turnover_tag"] = m.group(1)

    # streak bonus: _sb{num}
    m = re.search(r"_sb(\d+)", name)
    if m:
        try:
            meta["streak_bonus"] = str(float(m.group(1)) / 1000.0)
        except Exception:
            meta["streak_bonus"] = m.group(1)
    return meta


def _max_drawdown(nav_vals: np.ndarray) -> float:
    peak = -np.inf
    max_dd = 0.0
    for v in nav_vals:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _cagr(nav_vals: np.ndarray) -> float:
    num_days = max(1, len(nav_vals))
    years = num_days / 252.0
    if years <= 0:
        return 0.0
    return float(nav_vals[-1] ** (1.0 / years) - 1.0)


def summarize_pair(trades_fp: Path, nav_fp: Path) -> Dict[str, float | int | str]:
    name = trades_fp.name
    trades = pd.read_csv(trades_fp) if trades_fp.exists() else pd.DataFrame()
    nav = pd.read_csv(nav_fp) if nav_fp.exists() else pd.DataFrame()

    stats: Dict[str, float | int | str] = {"file": name}
    stats.update(_parse_meta(name))

    num_trades = int(trades.shape[0]) if not trades.empty else 0
    stats["trades"] = num_trades
    if num_trades > 0:
        stats["win_rate"] = float((trades["net_return"] > 0).mean())
        stats["avg_net_return"] = float(trades["net_return"].mean())
        stats["median_net_return"] = float(trades["net_return"].median())
    else:
        stats["win_rate"] = 0.0
        stats["avg_net_return"] = 0.0
        stats["median_net_return"] = 0.0

    if not nav.empty:
        nav_vals = nav["nav"].astype(float).values
        stats["portfolio_return"] = float(nav_vals[-1] - 1.0)
        stats["max_drawdown"] = _max_drawdown(nav_vals)
        stats["cagr"] = _cagr(nav_vals)
    else:
        stats["portfolio_return"] = 0.0
        stats["max_drawdown"] = 0.0
        stats["cagr"] = 0.0
    return stats


def main():
    parser = argparse.ArgumentParser(description="汇总 backtest_* 输出为单表")
    parser.add_argument("--dir", type=str, default=".", help="扫描目录")
    parser.add_argument("--out", type=str,
                        default="backtest_summary.csv", help="输出CSV")
    args = parser.parse_args()

    base = Path(args.dir)
    rows: List[Dict] = []
    for trades_fp in sorted(base.glob("backtest_trades*.csv")):
        nav_fp = Path(str(trades_fp).replace("trades", "nav"))
        stats = summarize_pair(trades_fp, nav_fp)
        rows.append(stats)

    if not rows:
        print("未发现 backtest_trades*.csv 文件。")
        return

    df = pd.DataFrame(rows)
    # 合理排序列
    prefer_cols = [
        "file", "scheme", "topn", "turnover_tag", "streak_bonus",
        "trades", "win_rate", "avg_net_return", "median_net_return",
        "portfolio_return", "max_drawdown", "cagr",
    ]
    cols = [c for c in prefer_cols if c in df.columns] + \
        [c for c in df.columns if c not in prefer_cols]
    df = df[cols]

    out = Path(args.out)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"已输出汇总: {out} ({df.shape[0]} 行)")


if __name__ == "__main__":
    main()
