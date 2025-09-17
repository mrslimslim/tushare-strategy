import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

try:
    from common.env import get_tushare_pro
except ModuleNotFoundError:  # 直接脚本运行的兜底路径
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.env import get_tushare_pro


OUTPUT_DIR = Path("data")


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_last_trade_date_str(pro, today: datetime | None = None, lookback_days: int = 365) -> str:
    """获取最近一个交易日（上证交易所日历）。
    使用更长回溯窗口以避免系统日期异常导致的偏差。
    """
    today = today or datetime.today()
    start_date = (today - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")
    cal = pro.trade_cal(
        exchange="SSE", start_date=start_date, end_date=end_date)
    cal = cal[cal["is_open"] == 1]
    if cal.empty:
        raise RuntimeError("无法获取最近交易日")
    return str(cal["cal_date"].max())


def _normalize_date_str(date_str: str) -> str:
    """将任意常见日期格式规整为 YYYYMMDD。支持: YYYYMMDD / YYYY-MM-DD / YYYY/MM/DD。
    """
    digits = "".join(ch for ch in str(date_str) if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"非法日期格式: {date_str}. 期望 YYYYMMDD 或 YYYY-MM-DD")
    return digits


def get_last_trade_date_on_or_before(pro, date_str: str | None, lookback_days: int = 365) -> str:
    """获取指定日期当日或之前的最近一个交易日。如果未提供日期，则返回最近交易日。

    参数:
    - date_str: 字符串日期(YYYYMMDD 或 YYYY-MM-DD)。None 表示使用今天。
    - lookback_days: 回溯天数窗口，用于覆盖法定节假日及长假。
    """
    if not date_str:
        return get_last_trade_date_str(pro, lookback_days=lookback_days)

    norm = _normalize_date_str(date_str)
    dt = datetime.strptime(norm, "%Y%m%d")
    start_date = (dt - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_date = norm
    cal = pro.trade_cal(
        exchange="SSE", start_date=start_date, end_date=end_date)
    cal = cal[cal["is_open"] == 1]
    if cal.empty:
        raise RuntimeError(f"在 {start_date} 到 {end_date} 区间内没有开市日，无法确定截止交易日")
    return str(cal["cal_date"].max())


def fetch_a_share_basic(pro) -> pd.DataFrame:
    # 获取所有上市公司基本信息
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,symbol,name,area,industry,list_date,market")
    if df.empty:
        return df
    # 保留主板/中小板/创业板；排除科创板、北交所；排除ST
    allow_markets = {"主板", "中小板"}
    df = df[df["market"].isin(allow_markets)].copy()
    mask_st = df["name"].str.contains("ST", case=False, na=False)
    df = df[~mask_st].copy()
    return df


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i+size] for i in range(0, len(items), size)]


def _get_last_n_trade_dates(pro, end_date: str, n_days: int) -> List[str]:
    start_date = (datetime.strptime(end_date, "%Y%m%d") -
                  timedelta(days=int(n_days * 3))).strftime("%Y%m%d")
    cal = pro.trade_cal(
        exchange="SSE", start_date=start_date, end_date=end_date)
    cal = cal[cal["is_open"] == 1].sort_values("cal_date")
    dates = cal["cal_date"].astype(str).tolist()
    return dates[-n_days:]


def fetch_recent_daily(pro, ts_codes: List[str], end_date: str, n_days: int = 60) -> pd.DataFrame:
    trade_dates = _get_last_n_trade_dates(
        pro, end_date=end_date, n_days=n_days)
    allow_set = set(ts_codes)
    frames: List[pd.DataFrame] = []
    for d in tqdm(trade_dates, desc="下载日线(逐日)"):
        df = pro.daily(trade_date=d)
        if df is None or df.empty:
            continue
        df = df[df["ts_code"].isin(allow_set)]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    daily = pd.concat(frames, ignore_index=True)
    return daily


def run_download(
    days: int = 60,
    token: str | None = None,
    out: str | Path = OUTPUT_DIR,
    end_date: str | None = None,
) -> dict:
    """下载基础与日线数据并保存到目标目录，返回摘要信息。"""
    if days <= 0:
        raise ValueError("days 必须为正整数")

    pro = get_tushare_pro(token=token)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    last_trade_date = get_last_trade_date_on_or_before(pro, end_date)

    basics = fetch_a_share_basic(pro)
    if basics.empty:
        raise RuntimeError("未获取到A股基础信息")

    ts_codes = basics["ts_code"].tolist()
    daily = fetch_recent_daily(
        pro, ts_codes, end_date=last_trade_date, n_days=days)

    basics_fp = out_dir / "stock_basic.csv"
    daily_fp = out_dir / "daily.csv"
    basics.to_csv(basics_fp, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_fp, index=False, encoding="utf-8-sig")

    meta = {
        "last_trade_date": last_trade_date,
        "days": int(days),
        "arg_end_date": end_date,
        "total_stocks": len(ts_codes),
        "daily_rows": int(daily.shape[0]),
        "output_dir": str(out_dir.resolve()),
        "basics_path": str(basics_fp.resolve()),
        "daily_path": str(daily_fp.resolve()),
    }
    pd.Series(meta).to_json(out_dir / "meta.json", force_ascii=False, indent=2)
    return meta


def main():
    parser = argparse.ArgumentParser(description="下载A股近N日日线数据并保存")
    parser.add_argument("--days", type=int, default=60, help="近N个交易日")
    parser.add_argument("--token", type=str, default=None,
                        help="Tushare Token（可选，默认读取 .env TUSHARE_TOKEN）")
    parser.add_argument("--out", type=str,
                        default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--end-date", type=str, default=None,
                        help="下载截止日期(YYYYMMDD)，默认为最近交易日；若为非交易日，将取不晚于该日的最近交易日")
    args = parser.parse_args()

    meta = run_download(
        days=args.days,
        token=args.token,
        out=args.out,
        end_date=args.end_date,
    )

    print(
        f"完成：{meta['total_stocks']}只股票，{meta['daily_rows']}行日线，交易日截至 {meta['last_trade_date']}。数据保存在 {meta['output_dir']}")


if __name__ == "__main__":
    main()
