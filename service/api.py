from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scripts.analyze_strategy import analyze
from scripts.backtest_strategy import run_backtest
from scripts.download_data import run_download


MAX_ANALYZE_PREVIEW = 200
MAX_TRADES_PREVIEW = 200
MAX_NAV_PREVIEW = 300


class DownloadRequest(BaseModel):
    days: int = Field(60, ge=1)
    end_date: Optional[str] = None
    out_dir: str = "data"
    token: Optional[str] = None


class DownloadResponse(BaseModel):
    meta: Dict[str, object]


class AnalyzeRequest(BaseModel):
    data_dir: str = "data"
    prefer_trade_cal: bool = True
    date: Optional[str] = None
    weight_scheme: Literal["equal",
                           "momentum_heavy", "momentum_tilt"] = "equal"
    norm: Literal["zscore", "rank"] = "zscore"
    min_turnover: Optional[float] = Field(default=None, ge=0.0)
    topn: Optional[int] = Field(default=None, ge=1)
    streak_bonus: float = 0.0
    out_path: Optional[str] = None
    limit: Optional[int] = Field(default=MAX_ANALYZE_PREVIEW, ge=1)


class AnalyzeResponse(BaseModel):
    rows: List[Dict[str, object]]
    row_count: int
    truncated: bool
    output_path: Optional[str] = None


class BacktestRequest(BaseModel):
    data_dir: str = "data"
    start: Optional[str] = None
    end: Optional[str] = None
    hold_days: int = Field(5, ge=1)
    buy_cost_bps: float = 0.0
    sell_cost_bps: float = 0.0
    weight_scheme: Literal["equal",
                           "momentum_heavy", "momentum_tilt"] = "equal"
    norm: Literal["zscore", "rank"] = "zscore"
    min_turnover: Optional[float] = Field(default=None, ge=0.0)
    daily_topn: Optional[int] = Field(default=None, ge=1)
    streak_bonus: float = 0.0
    out_trades: Optional[str] = None
    out_daily: Optional[str] = None
    trades_limit: Optional[int] = Field(default=MAX_TRADES_PREVIEW, ge=1)
    nav_limit: Optional[int] = Field(default=MAX_NAV_PREVIEW, ge=1)


class BacktestResponse(BaseModel):
    stats: Dict[str, object]
    trades_preview: List[Dict[str, object]]
    trades_count: int
    trades_truncated: bool
    trades_path: Optional[str] = None
    nav_preview: List[Dict[str, object]]
    nav_count: int
    nav_truncated: bool
    nav_path: Optional[str] = None


app = FastAPI(title="Tushare Strategy Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_parent(fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)


def _preview_records(df: pd.DataFrame, limit: Optional[int]) -> tuple[List[Dict[str, object]], bool]:
    if df is None or df.empty:
        return [], False
    preview_df = df
    truncated = False
    if limit is not None and len(df) > limit:
        preview_df = df.head(limit)
        truncated = True
    # 清洗为 JSON 友好：将 +/-inf 转为 NaN，再把 NaN 转为 None，避免 JSON 序列化报错
    replaced = preview_df.replace([np.inf, -np.inf], np.nan)
    as_object = replaced.astype(object)
    cleaned = as_object.where(pd.notnull(as_object), None)
    # 转换 numpy 标量为原生 Python 类型，防止 json 序列化遇到 numpy 类型

    def _to_builtin(v: object) -> object:
        if isinstance(v, (np.floating, np.integer)):
            try:
                return v.item()
            except Exception:
                return float(v) if isinstance(v, np.floating) else int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        # 处理 Python 原生 float 的 NaN/Inf
        if isinstance(v, float):
            if not np.isfinite(v):
                return None
            return float(v)
        return v
    records = [
        {k: _to_builtin(v) for k, v in row.items()}
        for row in cleaned.to_dict(orient="records")
    ]
    return jsonable_encoder(records), truncated


@app.get("/api/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/data/update", response_model=DownloadResponse)
def update_data(request: DownloadRequest) -> DownloadResponse:
    try:
        meta = run_download(
            days=request.days,
            token=request.token,
            out=request.out_dir,
            end_date=request.end_date,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DownloadResponse(meta=jsonable_encoder(meta))


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        df = analyze(
            Path(request.data_dir),
            prefer_trade_cal=request.prefer_trade_cal,
            specified_date=request.date,
            weight_scheme=request.weight_scheme,
            norm=request.norm,
            min_turnover=request.min_turnover,
            topn=request.topn,
            streak_bonus=request.streak_bonus,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_path: Optional[str] = None
    if request.out_path:
        output_path = str(Path(request.out_path).resolve())
        _ensure_parent(Path(output_path))
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    rows, truncated = _preview_records(df, request.limit)
    return AnalyzeResponse(
        rows=rows,
        row_count=int(df.shape[0]) if df is not None else 0,
        truncated=truncated,
        output_path=output_path,
    )


@app.post("/api/backtest", response_model=BacktestResponse)
def backtest_endpoint(request: BacktestRequest) -> BacktestResponse:
    try:
        trades, nav_df, stats = run_backtest(
            data=request.data_dir,
            start=request.start,
            end=request.end,
            hold_days=request.hold_days,
            buy_cost_bps=request.buy_cost_bps,
            sell_cost_bps=request.sell_cost_bps,
            weight_scheme=request.weight_scheme,
            norm=request.norm,
            min_turnover=request.min_turnover,
            daily_topn=request.daily_topn,
            streak_bonus=request.streak_bonus,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    trades_path: Optional[str] = None
    nav_path: Optional[str] = None
    if request.out_trades:
        trades_path = str(Path(request.out_trades).resolve())
        _ensure_parent(Path(trades_path))
        trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    if request.out_daily:
        nav_path = str(Path(request.out_daily).resolve())
        _ensure_parent(Path(nav_path))
        nav_df.to_csv(nav_path, index=False, encoding="utf-8-sig")

    trades_preview, trades_truncated = _preview_records(
        trades, request.trades_limit)
    nav_preview, nav_truncated = _preview_records(nav_df, request.nav_limit)

    stats_payload = jsonable_encoder(stats)
    return BacktestResponse(
        stats=stats_payload,
        trades_preview=trades_preview,
        trades_count=int(trades.shape[0]) if trades is not None else 0,
        trades_truncated=trades_truncated,
        trades_path=trades_path,
        nav_preview=nav_preview,
        nav_count=int(nav_df.shape[0]) if nav_df is not None else 0,
        nav_truncated=nav_truncated,
        nav_path=nav_path,
    )
