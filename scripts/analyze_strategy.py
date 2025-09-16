import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

try:
    from common.indicators import compute_all_indicators
    from common.env import get_tushare_pro
    from common.strategy import add_signal_columns
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.indicators import compute_all_indicators
    from common.env import get_tushare_pro
    from common.strategy import add_signal_columns


def get_latest_trade_date_from_cal(days_back: int = 30) -> str | None:
    """从交易日历获取最近一个交易日（返回 yyyymmdd 字符串）。
    若无法访问 Tushare（无 token），返回 None。
    """
    try:
        pro = get_tushare_pro()
    except Exception:
        return None
    today = datetime.today()
    start = (today - timedelta(days=days_back)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    cal = cal[cal["is_open"] == 1]
    if cal.empty:
        return None
    # 取区间内的最大交易日（而非依赖返回顺序）
    try:
        latest = int(cal["cal_date"].astype(int).max())
    except Exception:
        latest = str(cal["cal_date"].astype(str).max())
        return latest
    return str(latest)


def analyze(data_dir: Path, prefer_trade_cal: bool = True, specified_date: str | None = None) -> pd.DataFrame:
    basics_fp = data_dir / "stock_basic.csv"
    daily_fp = data_dir / "daily.csv"

    basics = pd.read_csv(basics_fp)
    daily = pd.read_csv(daily_fp)

    # 指标计算
    use_cols = [
        "ts_code", "trade_date", "open", "high", "low", "close", "vol"
    ]
    # 日线数据确保字段存在
    for col in use_cols:
        if col not in daily.columns:
            raise RuntimeError(f"日线缺少列: {col}")

    enriched = compute_all_indicators(daily[use_cols].copy())

    # 计算涨跌幅（若原始数据未提供 pct_chg，则用前一日收盘推算）
    enriched = enriched.sort_values(["ts_code", "trade_date"]).copy()
    enriched["prev_close"] = enriched.groupby("ts_code")["close"].shift(1)
    enriched["pct_chg"] = (enriched["close"] /
                           enriched["prev_close"] - 1.0) * 100
    # 统一由共用策略模块生成信号（含：振幅<4%、偏离≤1.5%、MA30上移等）
    df_sig = add_signal_columns(enriched)

    # 目标交易日：优先使用交易日历的最近交易日；若数据中不存在则回退为数据最大日期
    if enriched.empty:
        return pd.DataFrame()

    last_in_data = enriched["trade_date"].max()
    # 确定目标交易日：优先使用指定日期；否则遵循原有逻辑
    if specified_date is not None:
        target_date = specified_date
    else:
        target_date = last_in_data
        if prefer_trade_cal:
            latest_from_cal = get_latest_trade_date_from_cal(days_back=120)
            if latest_from_cal is not None:
                target_date = latest_from_cal
    # 统一 dtype 以匹配数据列
    if pd.api.types.is_numeric_dtype(enriched["trade_date"].dtype):
        try:
            target_date_norm = int(target_date)
        except Exception:
            target_date_norm = last_in_data
    else:
        target_date_norm = str(target_date)
    if target_date_norm not in set(enriched["trade_date"].unique()):
        # 数据中没有目标交易日，回退为数据最大日期
        target_date_norm = last_in_data

    latest = df_sig[df_sig["trade_date"] == target_date_norm].copy()
    # 直接用共用模块计算的信号
    selected = latest[latest["signal"]].copy()

    # 附加基本信息
    selected = selected.merge(
        basics[["ts_code", "name", "industry", "area", "list_date"]],
        on="ts_code",
        how="left"
    )

    # 输出排序：优先偏离度，其次涨跌幅或 J 从小到大
    if "pct_chg" in selected.columns:
        selected = selected.sort_values(
            ["zx_dev_pct", "pct_chg", "J"], ascending=[True, True, True])
    else:
        selected = selected.sort_values(
            ["zx_dev_pct", "J"], ascending=[True, True])

    selected["trade_date"] = selected["trade_date"].astype(str)
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="按策略筛选A股：J<22，Diff>0，MA30上移，振幅<4%，收盘偏离知行短期趋势≤1.5%，并支持自动设定最新交易日")
    parser.add_argument("--data", type=str, default="data",
                        help="数据目录（包含 stock_basic.csv, daily.csv）")
    parser.add_argument("--out", type=str,
                        default="output.csv", help="筛选结果CSV")
    parser.add_argument("--no-prefer-trade-cal",
                        action="store_true", help="不从交易日历获取最新交易日，改用数据中最大日期")
    parser.add_argument("--date", type=str, default=None,
                        help="指定交易日（YYYYMMDD）。未提供则默认使用最新交易日")
    args = parser.parse_args()

    data_dir = Path(args.data)
    result = analyze(
        data_dir,
        prefer_trade_cal=(not args.no_prefer_trade_cal),
        specified_date=args.date,
    )

    if result.empty:
        print("无符合条件的标的。")
        return

    out_path = Path(args.out)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已输出 {result.shape[0]} 条结果到 {out_path}")


if __name__ == "__main__":
    main()
