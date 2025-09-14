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
2. 在项目根目录创建 `.env`
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

## 说明
- 指标计算见 `common/indicators.py`
- 环境与 token 加载见 `common/env.py`
- 如需改变 MA30 上移的判定逻辑，可在 `is_ma30_rising` 中调整。
