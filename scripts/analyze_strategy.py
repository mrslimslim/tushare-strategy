import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

try:
    from common.indicators import compute_all_indicators
    from common.env import get_tushare_pro
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.indicators import compute_all_indicators
    from common.env import get_tushare_pro


def is_ma30_rising(group: pd.DataFrame) -> bool:
    # 至少需要2个有效 ma30 来判断最近两天上移
    ma = group["ma30"].dropna()
    if len(ma) < 2:
        return False
    last2 = ma.tail(2)
    return bool(last2.iloc[-1] > last2.iloc[-2])


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
    return str(cal.iloc[-1]["cal_date"])


def analyze(data_dir: Path, prefer_trade_cal: bool = True) -> pd.DataFrame:
    basics_fp = data_dir / "stock_basic.csv"
    daily_fp = data_dir / "daily.csv"

    basics = pd.read_csv(basics_fp)
    daily = pd.read_csv(daily_fp)

    # 指标计算
    use_cols = [
        "ts_code","trade_date","open","high","low","close","vol"
    ]
    # 日线数据确保字段存在
    for col in use_cols:
        if col not in daily.columns:
            raise RuntimeError(f"日线缺少列: {col}")

    enriched = compute_all_indicators(daily[use_cols].copy())

    # 计算涨跌幅（若原始数据未提供 pct_chg，则用前一日收盘推算）
    enriched = enriched.sort_values(["ts_code", "trade_date"]).copy()
    enriched["prev_close"] = enriched.groupby("ts_code")["close"].shift(1)
    enriched["pct_chg"] = (enriched["close"] / enriched["prev_close"] - 1.0) * 100

    # 目标交易日：优先使用交易日历的最近交易日；若数据中不存在则回退为数据最大日期
    if enriched.empty:
        return pd.DataFrame()

    last_in_data = enriched["trade_date"].max()
    target_date = last_in_data
    if prefer_trade_cal:
        latest_from_cal = get_latest_trade_date_from_cal(days_back=120)
        if latest_from_cal is not None:
            target_date = latest_from_cal
    if target_date not in set(enriched["trade_date"].unique()):
        # 数据中没有目标交易日，回退为数据最大日期
        target_date = last_in_data

    latest = enriched[enriched["trade_date"] == target_date].copy()

    # 条件：J<22 且 日线 DIFF>0 且 MA30 上移 且 成交量 < 10日均量 且 当日涨跌幅在 [-2.5%, +2.25%]
    cond_j = latest["J"] < 22
    cond_diff = latest["diff"] > 0
    cond_vol = latest["vol"].notna() & latest["vol_ma10"].notna() & (latest["vol"] < latest["vol_ma10"])
    cond_pct = latest["pct_chg"].notna() & (latest["pct_chg"] >= -2.5) & (latest["pct_chg"] <= 2.25)
    # MA30 上移需要回看
    rising_codes = set()
    for code, g in enriched.groupby("ts_code"):
        if is_ma30_rising(g.sort_values("trade_date")):
            rising_codes.add(code)
    cond_ma = latest["ts_code"].isin(rising_codes)

    selected = latest[cond_j & cond_diff & cond_ma & cond_vol & cond_pct].copy()

    # 附加基本信息
    selected = selected.merge(
        basics[["ts_code","name","industry","area","list_date"]],
        on="ts_code",
        how="left"
    )

    # 输出排序：按涨跌幅（如有）或 J 从小到大
    if "pct_chg" in selected.columns:
        selected = selected.sort_values(["pct_chg","J"], ascending=[True, True])
    else:
        selected = selected.sort_values(["J"], ascending=True)

    selected["trade_date"] = selected["trade_date"].astype(str)
    return selected


def main():
    parser = argparse.ArgumentParser(description="按策略筛选A股：J<22，Diff>0，MA30上移，并支持自动设定最新交易日")
    parser.add_argument("--data", type=str, default="data", help="数据目录（包含 stock_basic.csv, daily.csv）")
    parser.add_argument("--out", type=str, default="output.csv", help="筛选结果CSV")
    parser.add_argument("--no-prefer-trade-cal", action="store_true", help="不从交易日历获取最新交易日，改用数据中最大日期")
    args = parser.parse_args()

    data_dir = Path(args.data)
    result = analyze(data_dir, prefer_trade_cal=(not args.no_prefer_trade_cal))

    if result.empty:
        print("无符合条件的标的。")
        return

    out_path = Path(args.out)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已输出 {result.shape[0]} 条结果到 {out_path}")


if __name__ == "__main__":
    main()
