import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from common.indicators import compute_all_indicators
    from common.strategy import add_signal_columns, add_score_columns
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.indicators import compute_all_indicators
    from common.strategy import add_signal_columns, add_score_columns


def _parse_date(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # 允许 YYYY-MM-DD 或 YYYYMMDD
    if "-" in s:
        return s.replace("-", "")
    return s


def load_and_enrich(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    basics_fp = data_dir / "stock_basic.csv"
    daily_fp = data_dir / "daily.csv"

    basics = pd.read_csv(basics_fp)
    daily = pd.read_csv(daily_fp)

    use_cols = [
        "ts_code", "trade_date", "open", "high", "low", "close", "vol"
    ]
    for col in use_cols:
        if col not in daily.columns:
            raise RuntimeError(f"日线缺少列: {col}")

    enriched = compute_all_indicators(daily[use_cols].copy())
    # 涨跌幅（若原始数据未提供）
    enriched = enriched.sort_values(["ts_code", "trade_date"]).copy()
    enriched["prev_close"] = enriched.groupby("ts_code")["close"].shift(1)
    enriched["pct_chg"] = (enriched["close"] /
                           enriched["prev_close"] - 1.0) * 100.0
    return basics, enriched


def _generate_trades_for_stock(g: pd.DataFrame, hold_days: int,
                               buy_cost: float, sell_cost: float,
                               start_date: str | None, end_date: str | None) -> List[Dict]:
    g = g.sort_values("trade_date").reset_index(drop=True)
    trades: List[Dict] = []
    n = len(g)
    i = 0
    # 仅在 [start,end] 内出现的信号才考虑（以信号日 t 为准）
    while i < n - 1:  # 至少要有次日数据才能进场
        # 信号日
        if not bool(g.loc[i, "signal"]):
            i += 1
            continue
        signal_date = str(g.loc[i, "trade_date"]).strip()
        if start_date is not None and signal_date < start_date:
            i += 1
            continue
        if end_date is not None and signal_date > end_date:
            break

        entry_idx = i + 1
        if entry_idx >= n:
            break

        entry_date = str(g.loc[entry_idx, "trade_date"]).strip()
        if start_date is not None and entry_date < start_date:
            # 理论上不会发生（因为 entry = signal+1），但为稳妥保留
            i += 1
            continue
        if end_date is not None and entry_date > end_date:
            break

        exit_idx = entry_idx + hold_days - 1
        truncated = False
        if exit_idx >= n:
            exit_idx = n - 1
            truncated = True

        exit_date = str(g.loc[exit_idx, "trade_date"]).strip()
        # 若 exit 超过 end_date，可截断
        if end_date is not None and exit_date > end_date:
            # 将 exit 掐到 end_date 之前的最后一个可用bar
            mask = g["trade_date"] <= int(end_date)
            if mask.any():
                last_idx = mask[mask].index.max()
                if last_idx <= entry_idx:
                    # 没有完整持仓期
                    i = entry_idx + 1
                    continue
                exit_idx = int(last_idx)
                exit_date = str(g.loc[exit_idx, "trade_date"]).strip()
                truncated = True
            else:
                i = entry_idx + 1
                continue

        entry_open = float(g.loc[entry_idx, "open"]) if pd.notna(
            g.loc[entry_idx, "open"]) else np.nan
        exit_close = float(g.loc[exit_idx, "close"]) if pd.notna(
            g.loc[exit_idx, "close"]) else np.nan
        if not (np.isfinite(entry_open) and np.isfinite(exit_close)):
            i = exit_idx + 1
            continue

        gross_return = (exit_close / entry_open) - 1.0
        net_return = (exit_close * (1.0 - sell_cost)) / \
            (entry_open * (1.0 + buy_cost)) - 1.0

        trades.append({
            "ts_code": g.loc[i, "ts_code"],
            "signal_date": signal_date,
            "entry_date": entry_date,
            "entry_open": entry_open,
            "exit_date": exit_date,
            "exit_close": exit_close,
            "holding_days": int(exit_idx - entry_idx + 1),
            "gross_return": gross_return,
            "net_return": net_return,
            "is_truncated": truncated,
        })

        # 不允许重叠：移动到 exit 之后
        i = exit_idx + 1

    return trades


def generate_trades(df_with_signal: pd.DataFrame, hold_days: int,
                    buy_cost_bps: float, sell_cost_bps: float,
                    start_date: str | None, end_date: str | None) -> pd.DataFrame:
    buy_cost = float(buy_cost_bps) / 10000.0
    sell_cost = float(sell_cost_bps) / 10000.0
    all_trades: List[Dict] = []
    for code, g in df_with_signal.groupby("ts_code", sort=False):
        trades = _generate_trades_for_stock(
            g, hold_days, buy_cost, sell_cost, start_date, end_date)
        if trades:
            all_trades.extend(trades)
    if not all_trades:
        return pd.DataFrame(columns=[
            "ts_code", "signal_date", "entry_date", "entry_open",
            "exit_date", "exit_close", "holding_days",
            "gross_return", "net_return", "is_truncated"
        ])
    trades_df = pd.DataFrame(all_trades)
    trades_df = trades_df.sort_values(
        ["entry_date", "ts_code"]).reset_index(drop=True)
    return trades_df


def build_daily_nav(df_with_prices: pd.DataFrame, trades_df: pd.DataFrame,
                    buy_cost_bps: float, sell_cost_bps: float) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(columns=["trade_date", "num_positions", "avg_position_return", "nav"])

    buy_cost = float(buy_cost_bps) / 10000.0
    sell_cost = float(sell_cost_bps) / 10000.0

    # 预备映射：ts_code -> 子表（含 trade_date, open, close）
    df = df_with_prices.sort_values(["ts_code", "trade_date"]).copy()
    keep_cols = ["ts_code", "trade_date", "open", "close"]
    df = df[keep_cols]
    code_to_group: Dict[str, pd.DataFrame] = {
        code: g.reset_index(drop=True) for code, g in df.groupby("ts_code", sort=False)
    }

    # 聚合每日持仓的等权收益
    daily_returns: Dict[str, List[float]] = {}

    for _, row in trades_df.iterrows():
        code = row["ts_code"]
        entry_date = str(row["entry_date"])  # 次日开盘进场日
        exit_date = str(row["exit_date"])    # 持有期最后一日（收盘卖出）

        g = code_to_group.get(code)
        if g is None:
            continue
        # 找到 entry 与 exit 的行号
        idx_map = {str(d): i for i, d in enumerate(
            g["trade_date"].astype(str))}
        if entry_date not in idx_map or exit_date not in idx_map:
            continue
        e_idx = idx_map[entry_date]
        x_idx = idx_map[exit_date]
        if x_idx < e_idx:
            continue

        # 构造该持仓的逐日收益序列
        for j in range(e_idx, x_idx + 1):
            date_j = str(g.loc[j, "trade_date"])  # 当天日
            if j == e_idx:
                # 入场日收益：close/open - 1，并扣除买入成本
                open_j = float(g.loc[j, "open"]) if pd.notna(
                    g.loc[j, "open"]) else np.nan
                close_j = float(g.loc[j, "close"]) if pd.notna(
                    g.loc[j, "close"]) else np.nan
                if not (np.isfinite(open_j) and np.isfinite(close_j)):
                    continue
                r = (close_j / open_j) - 1.0 - buy_cost
            else:
                # 其余日：按收盘对收盘
                close_prev = float(
                    g.loc[j - 1, "close"]) if pd.notna(g.loc[j - 1, "close"]) else np.nan
                close_j = float(g.loc[j, "close"]) if pd.notna(
                    g.loc[j, "close"]) else np.nan
                if not (np.isfinite(close_prev) and np.isfinite(close_j)):
                    continue
                r = (close_j / close_prev) - 1.0
            if j == x_idx:
                # 卖出成本在最后一日扣除
                r -= sell_cost
            daily_returns.setdefault(date_j, []).append(float(r))

    if not daily_returns:
        return pd.DataFrame(columns=["trade_date", "num_positions", "avg_position_return", "nav"])

    # 生成 DataFrame（按日期升序）
    all_dates = sorted(daily_returns.keys())
    rows = []
    nav = 1.0
    for d in all_dates:
        rets = daily_returns[d]
        avg_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0
        nav *= (1.0 + avg_ret)
        rows.append({
            "trade_date": d,
            "num_positions": int(len(rets)),
            "avg_position_return": avg_ret,
            "nav": nav,
        })
    return pd.DataFrame(rows)


def summarize(trades_df: pd.DataFrame, nav_df: pd.DataFrame) -> Dict[str, float | int]:
    stats: Dict[str, float | int] = {}
    num_trades = int(
        trades_df.shape[0]) if trades_df is not None and not trades_df.empty else 0
    stats["trades"] = num_trades
    if num_trades > 0:
        wins = int((trades_df["net_return"] > 0).sum())
        stats["win_rate"] = wins / num_trades
        stats["avg_net_return"] = float(trades_df["net_return"].mean())
        stats["median_net_return"] = float(trades_df["net_return"].median())
    else:
        stats["win_rate"] = 0.0
        stats["avg_net_return"] = 0.0
        stats["median_net_return"] = 0.0

    if nav_df is not None and not nav_df.empty:
        total_ret = float(nav_df.iloc[-1]["nav"]) - 1.0
        stats["portfolio_return"] = total_ret
        # 近似换算年化、回撤
        nav_series = nav_df["nav"].astype(float).values
        peak = -np.inf
        max_dd = 0.0
        for v in nav_series:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        stats["max_drawdown"] = float(max_dd)

        num_days = max(1, len(nav_series))
        # 以 252 交易日估算
        years = num_days / 252.0
        if years > 0:
            cagr = float(nav_series[-1] ** (1.0 / years) - 1.0)
        else:
            cagr = 0.0
        stats["cagr"] = cagr
    else:
        stats["portfolio_return"] = 0.0
        stats["max_drawdown"] = 0.0
        stats["cagr"] = 0.0

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "回测基于筛选策略：J<30, DIFF>0, MA30上移, 成交量为近10日倒数第1或第2或第3, pct_chg∈[-2.5,2.25], 振幅<4%, 收盘偏离知行短期趋势≤1.5%, 且近14日内存在>8%单日涨幅。\n"
            "信号当日（t）满足条件，次日（t+1）开盘买入，持有N日，于第N日收盘卖出。"
        )
    )
    parser.add_argument("--data", type=str, default="data",
                        help="数据目录（包含 stock_basic.csv, daily.csv）")
    parser.add_argument("--start", type=str, default=None,
                        help="回测起始日期（YYYYMMDD 或 YYYY-MM-DD）；默认用数据最早日期")
    parser.add_argument("--end", type=str, default=None,
                        help="回测结束日期（YYYYMMDD 或 YYYY-MM-DD）；默认用数据最晚日期")
    parser.add_argument("--hold-days", type=int, default=5,
                        help="持有交易日数（次日开盘为持仓首日）")
    parser.add_argument("--buy-cost-bps", type=float, default=0.0,
                        help="买入成本（基点，1bp=0.01%），默认0")
    parser.add_argument("--sell-cost-bps", type=float, default=0.0,
                        help="卖出成本（基点，1bp=0.01%），默认0")
    parser.add_argument("--out-trades", type=str, default="backtest_trades.csv",
                        help="交易明细输出CSV")
    parser.add_argument("--out-daily", type=str, default="backtest_nav.csv",
                        help="组合净值输出CSV")
    parser.add_argument("--weight-scheme", type=str, default="equal",
                        choices=["equal", "momentum_heavy", "momentum_tilt"],
                        help="跨模块权重方案：equal=等权；momentum_heavy=动量40/趋势30/量20/风险10")
    parser.add_argument("--norm", type=str, default="zscore",
                        choices=["zscore", "rank"], help="截面规范化方式")
    parser.add_argument("--min-turnover", type=float, default=None,
                        help="成交额门槛（vol*close），用于资格过滤")
    parser.add_argument("--daily-topn", type=int, default=None,
                        help="信号生成后，按每天综合得分选取前N只作为入选标的")
    parser.add_argument("--streak-bonus", type=float, default=0.0,
                        help="信号连续性加分系数（每连续1天叠加该系数）")

    args = parser.parse_args()

    data_dir = Path(args.data)
    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)

    basics, enriched = load_and_enrich(data_dir)
    # 限定全局日期边界，用于每日净值视图
    if start_date is not None:
        enriched = enriched[enriched["trade_date"] >= int(start_date)]
    if end_date is not None:
        enriched = enriched[enriched["trade_date"] <= int(end_date)]
    if enriched.empty:
        print("数据为空或日期范围内无数据。")
        return

    df_sig = add_signal_columns(enriched)
    # 评分并应用资格筛选
    df_scored = add_score_columns(
        df_sig,
        norm=args.norm,
        weight_scheme=args.weight_scheme,
        min_turnover=args.min_turnover,
        streak_bonus=args.streak_bonus,
    )

    # 如配置了 daily-topn，则对每个交易日按 score 选前N 只（仍要求 signal=True）
    if args.daily_topn is not None and int(args.daily_topn) > 0:
        g = df_scored[df_scored["signal"] &
                      df_scored["eligible_liquidity"].fillna(True)].copy()
        g["_rank"] = g.groupby("trade_date")["score"].rank(
            ascending=False, method="first")
        g = g[g["_rank"] <= int(args.daily_topn)]
        g = g.drop(columns=["_rank"])  # 清理
        df_for_backtest = g
    else:
        # 未限制 TopN，则使用所有 signal 且满足流动性资格的样本
        df_for_backtest = df_scored[df_scored["signal"]
                                    & df_scored["eligible_liquidity"].fillna(True)]
    trades = generate_trades(df_for_backtest, hold_days=args.hold_days,
                             buy_cost_bps=args.buy_cost_bps,
                             sell_cost_bps=args.sell_cost_bps,
                             start_date=start_date, end_date=end_date)

    # 附加基本信息
    if not trades.empty:
        trades = trades.merge(
            basics[["ts_code", "name", "industry", "area", "list_date"]],
            on="ts_code", how="left"
        )
        # 列顺序
        cols = [
            "ts_code", "name", "industry", "area",
            "signal_date", "entry_date", "entry_open",
            "exit_date", "exit_close", "holding_days",
            "gross_return", "net_return", "is_truncated", "list_date"
        ]
        trades = trades[cols]

    nav_df = build_daily_nav(
        df_scored, trades, args.buy_cost_bps, args.sell_cost_bps)

    # 输出
    out_trades = Path(args.out_trades)
    out_daily = Path(args.out_daily)
    trades.to_csv(out_trades, index=False, encoding="utf-8-sig")
    nav_df.to_csv(out_daily, index=False, encoding="utf-8-sig")

    stats = summarize(trades, nav_df)
    print(f"交易数: {stats['trades']}")
    print(f"胜率: {stats['win_rate']:.2%}")
    print(f"单笔平均净收益: {stats['avg_net_return']:.2%}")
    print(f"单笔收益中位数: {stats['median_net_return']:.2%}")
    print(f"组合总收益: {stats['portfolio_return']:.2%}")
    print(f"最大回撤: {stats['max_drawdown']:.2%}")
    print(f"年化收益(CAGR): {stats['cagr']:.2%}")
    print(f"已输出交易明细: {out_trades}")
    print(f"已输出净值曲线: {out_daily}")


if __name__ == "__main__":
    main()
