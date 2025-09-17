# A股选股策略（Tushare）

策略：收市最后一日信号，筛选条件：

- 日线 KDJ 的 J 值 < 22
- 排除：科创板、北交所、ST
- 日线 MACD 的 DIFF > 0（12/26）
- 日线 MA30 上移（最近三个有效点单调递增）

## 环境准备

1. 安装依赖

```bash
pip install -r requirements.txt
```

1. 在项目根目录创建 `.env`

```dotenv
TUSHARE_TOKEN=你的tushare_token
```

## 数据下载（近N个交易日）

```bash
python scripts/download_data.py --days 60 --out data
```

输出：

- `data/stock_basic.csv`
- `data/daily.csv`
- `data/meta.json`

## 运行分析

```bash
python scripts/analyze_strategy.py --data data --out output.csv
```

输出：`output.csv`

## 推荐参数（当前窗口回测最佳实践）

回测区间：2025-03-20 至 2025-09-16（180 交易日左右）。

筛选/回测建议：

```bash
# 分析（打分+Top10），成交额门槛 5e6，轻度动量倾斜
python scripts/analyze_strategy.py \
  --data data \
  --weight-scheme momentum_tilt \
  --norm zscore \
  --min-turnover 5000000 \
  --topn 10 \
  --out output.csv

# 回测（持有5日，按日Top10）
python scripts/backtest_strategy.py \
  --data data \
  --start 20250320 --end 20250916 \
  --hold-days 5 \
  --daily-topn 10 \
  --weight-scheme momentum_tilt \
  --norm zscore \
  --min-turnover 5000000 \
  --out-trades backtest_trades.csv \
  --out-daily backtest_nav.csv
```

可选优化：

- 若更保守，可将 `--daily-topn` 调整为 8；或提高风险模块权重。
- `--streak-bonus` 默认为 0；本窗口下使用 0.05/0.10 与 0 基本一致。

## 说明

- 指标计算见 `common/indicators.py`
- 环境与 token 加载见 `common/env.py`
- 如需改变 MA30 上移的判定逻辑，可在 `is_ma30_rising` 中调整。

## Web 服务 & 前端看板

项目新增 FastAPI 服务与 Vue3 + Vite 看板，支持通过浏览器执行数据更新、筛选分析与回测。

### 启动后端服务

```bash
# 建议先进入虚拟环境并安装依赖
pip install -r requirements.txt

# 启动 FastAPI（默认监听 8000）
uvicorn service.api:app --reload
```

### 启动前端（需 Node.js ≥ 18）

```bash
cd frontend
npm install
VITE_API_BASE=http://localhost:8000 npm run dev
```

浏览器访问命令行输出的本地地址即可。若后端端口或地址有调整，请同步修改 `VITE_API_BASE`。
