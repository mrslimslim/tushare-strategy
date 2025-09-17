<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { getJson, postJson } from './api/client.js'

const apiBase = ref(import.meta.env.VITE_API_BASE || 'http://localhost:8000')

const health = ref(null)
const healthError = ref('')

const dataForm = reactive({
  days: 60,
  endDate: '',
  outDir: 'data',
  token: ''
})
const dataLoading = ref(false)
const dataError = ref('')
const dataResult = ref(null)

const analyzeForm = reactive({
  dataDir: 'data',
  preferTradeCal: true,
  date: '',
  weightScheme: 'equal',
  norm: 'zscore',
  minTurnover: '',
  topn: 20,
  streakBonus: 0,
  outPath: 'output.csv',
  limit: 100
})
const analyzeLoading = ref(false)
const analyzeError = ref('')
const analyzeResult = ref(null)

const backtestForm = reactive({
  dataDir: 'data',
  start: '',
  end: '',
  holdDays: 5,
  buyCost: 0,
  sellCost: 0,
  weightScheme: 'equal',
  norm: 'zscore',
  minTurnover: '',
  dailyTopn: '',
  streakBonus: 0,
  outTrades: 'backtest_trades.csv',
  outDaily: 'backtest_nav.csv',
  tradesLimit: 100,
  navLimit: 200
})
const backtestLoading = ref(false)
const backtestError = ref('')
const backtestResult = ref(null)

const analyzeColumns = computed(() => {
  if (!analyzeResult.value || !analyzeResult.value.rows || analyzeResult.value.rows.length === 0) {
    return []
  }
  return Object.keys(analyzeResult.value.rows[0])
})

const tradesColumns = computed(() => {
  if (!backtestResult.value || !backtestResult.value.trades_preview || backtestResult.value.trades_preview.length === 0) {
    return []
  }
  return Object.keys(backtestResult.value.trades_preview[0])
})

const navColumns = computed(() => {
  if (!backtestResult.value || !backtestResult.value.nav_preview || backtestResult.value.nav_preview.length === 0) {
    return []
  }
  return Object.keys(backtestResult.value.nav_preview[0])
})

function nullableText(value) {
  return value === '' ? null : value
}

function nullableNumber(value) {
  if (value === '' || value === null || value === undefined) {
    return null
  }
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return null
  }
  return parsed
}

async function runDataUpdate() {
  dataLoading.value = true
  dataError.value = ''
  dataResult.value = null
  try {
    const payload = {
      days: Number(dataForm.days),
      end_date: nullableText(dataForm.endDate),
      out_dir: dataForm.outDir || 'data',
      token: nullableText(dataForm.token)
    }
    const res = await postJson('/api/data/update', payload)
    dataResult.value = res
  } catch (err) {
    dataError.value = err.message || String(err)
  } finally {
    dataLoading.value = false
  }
}

async function runAnalyze() {
  analyzeLoading.value = true
  analyzeError.value = ''
  analyzeResult.value = null
  try {
    const payload = {
      data_dir: analyzeForm.dataDir || 'data',
      prefer_trade_cal: analyzeForm.preferTradeCal,
      date: nullableText(analyzeForm.date),
      weight_scheme: analyzeForm.weightScheme,
      norm: analyzeForm.norm,
      min_turnover: nullableNumber(analyzeForm.minTurnover),
      topn: nullableNumber(analyzeForm.topn),
      streak_bonus: Number(analyzeForm.streakBonus) || 0,
      out_path: nullableText(analyzeForm.outPath),
      limit: nullableNumber(analyzeForm.limit) || undefined
    }
    const res = await postJson('/api/analyze', payload)
    analyzeResult.value = res
  } catch (err) {
    analyzeError.value = err.message || String(err)
  } finally {
    analyzeLoading.value = false
  }
}

async function runBacktest() {
  backtestLoading.value = true
  backtestError.value = ''
  backtestResult.value = null
  try {
    const payload = {
      data_dir: backtestForm.dataDir || 'data',
      start: nullableText(backtestForm.start),
      end: nullableText(backtestForm.end),
      hold_days: Number(backtestForm.holdDays) || 1,
      buy_cost_bps: Number(backtestForm.buyCost) || 0,
      sell_cost_bps: Number(backtestForm.sellCost) || 0,
      weight_scheme: backtestForm.weightScheme,
      norm: backtestForm.norm,
      min_turnover: nullableNumber(backtestForm.minTurnover),
      daily_topn: nullableNumber(backtestForm.dailyTopn),
      streak_bonus: Number(backtestForm.streakBonus) || 0,
      out_trades: nullableText(backtestForm.outTrades),
      out_daily: nullableText(backtestForm.outDaily),
      trades_limit: nullableNumber(backtestForm.tradesLimit) || undefined,
      nav_limit: nullableNumber(backtestForm.navLimit) || undefined
    }
    const res = await postJson('/api/backtest', payload)
    backtestResult.value = res
  } catch (err) {
    backtestError.value = err.message || String(err)
  } finally {
    backtestLoading.value = false
  }
}

onMounted(async () => {
  try {
    health.value = await getJson('/api/health')
  } catch (err) {
    healthError.value = err.message || String(err)
  }
})
</script>

<template>
  <main class="page">
    <header class="page__header">
      <h1>策略服务控制台</h1>
      <p>API Base: <code>{{ apiBase }}</code></p>
      <p v-if="health">服务状态：<span class="status-ok">{{ health.status }}</span></p>
      <p v-else-if="healthError" class="status-error">{{ healthError }}</p>
    </header>

    <section class="card">
      <h2>数据更新</h2>
      <form class="form" @submit.prevent="runDataUpdate">
        <div class="form__row">
          <label>近 N 个交易日
            <input type="number" min="1" v-model.number="dataForm.days" required />
          </label>
          <label>截止日期(可选)
            <input type="text" v-model="dataForm.endDate" placeholder="YYYYMMDD" />
          </label>
          <label>输出目录
            <input type="text" v-model="dataForm.outDir" />
          </label>
        </div>
        <div class="form__row">
          <label>Tushare Token (可选)
            <input type="password" v-model="dataForm.token" autocomplete="off" />
          </label>
        </div>
        <button type="submit" :disabled="dataLoading">{{ dataLoading ? '更新中…' : '执行更新' }}</button>
      </form>
      <p v-if="dataError" class="status-error">{{ dataError }}</p>
      <div v-if="dataResult" class="result">
        <h3>摘要</h3>
        <ul>
          <li v-for="(value, key) in dataResult.meta" :key="key"><strong>{{ key }}:</strong> {{ value }}</li>
        </ul>
      </div>
    </section>

    <section class="card">
      <h2>筛选分析</h2>
      <form class="form" @submit.prevent="runAnalyze">
        <div class="form__row">
          <label>数据目录
            <input type="text" v-model="analyzeForm.dataDir" />
          </label>
          <label>交易日 (可选)
            <input type="text" v-model="analyzeForm.date" placeholder="YYYYMMDD" />
          </label>
          <label>Top N (可选)
            <input type="number" min="1" v-model.number="analyzeForm.topn" />
          </label>
        </div>
        <div class="form__row">
          <label>权重方案
            <select v-model="analyzeForm.weightScheme">
              <option value="equal">equal</option>
              <option value="momentum_heavy">momentum_heavy</option>
              <option value="momentum_tilt">momentum_tilt</option>
            </select>
          </label>
          <label>标准化
            <select v-model="analyzeForm.norm">
              <option value="zscore">zscore</option>
              <option value="rank">rank</option>
            </select>
          </label>
          <label>成交额门槛
            <input type="number" min="0" step="any" v-model="analyzeForm.minTurnover" />
          </label>
        </div>
        <div class="form__row">
          <label>连续性加分
            <input type="number" step="any" v-model.number="analyzeForm.streakBonus" />
          </label>
          <label>输出 CSV (可选)
            <input type="text" v-model="analyzeForm.outPath" />
          </label>
          <label>展示行数
            <input type="number" min="1" v-model.number="analyzeForm.limit" />
          </label>
        </div>
        <label class="checkbox">
          <input type="checkbox" v-model="analyzeForm.preferTradeCal" /> 优先交易日历
        </label>
        <button type="submit" :disabled="analyzeLoading">{{ analyzeLoading ? '计算中…' : '执行分析' }}</button>
      </form>
      <p v-if="analyzeError" class="status-error">{{ analyzeError }}</p>
      <div v-if="analyzeResult" class="result">
        <h3>结果 ({{ analyzeResult.row_count }} 条{{ analyzeResult.truncated ? '，已截断展示' : '' }})</h3>
        <p v-if="analyzeResult.output_path">已保存：<code>{{ analyzeResult.output_path }}</code></p>
        <div class="table-wrapper" v-if="analyzeColumns.length">
          <table>
            <thead>
              <tr>
                <th v-for="col in analyzeColumns" :key="col">{{ col }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, idx) in analyzeResult.rows" :key="idx">
                <td v-for="col in analyzeColumns" :key="col">{{ row[col] }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p v-else>无符合条件的标的。</p>
      </div>
    </section>

    <section class="card">
      <h2>回测</h2>
      <form class="form" @submit.prevent="runBacktest">
        <div class="form__row">
          <label>数据目录
            <input type="text" v-model="backtestForm.dataDir" />
          </label>
          <label>起始日期
            <input type="text" v-model="backtestForm.start" placeholder="YYYYMMDD" />
          </label>
          <label>结束日期
            <input type="text" v-model="backtestForm.end" placeholder="YYYYMMDD" />
          </label>
        </div>
        <div class="form__row">
          <label>持有天数
            <input type="number" min="1" v-model.number="backtestForm.holdDays" />
          </label>
          <label>买入成本(bp)
            <input type="number" step="any" v-model.number="backtestForm.buyCost" />
          </label>
          <label>卖出成本(bp)
            <input type="number" step="any" v-model.number="backtestForm.sellCost" />
          </label>
        </div>
        <div class="form__row">
          <label>权重方案
            <select v-model="backtestForm.weightScheme">
              <option value="equal">equal</option>
              <option value="momentum_heavy">momentum_heavy</option>
              <option value="momentum_tilt">momentum_tilt</option>
            </select>
          </label>
          <label>标准化
            <select v-model="backtestForm.norm">
              <option value="zscore">zscore</option>
              <option value="rank">rank</option>
            </select>
          </label>
          <label>成交额门槛
            <input type="number" min="0" step="any" v-model="backtestForm.minTurnover" />
          </label>
        </div>
        <div class="form__row">
          <label>每日 TopN
            <input type="number" min="1" v-model.number="backtestForm.dailyTopn" />
          </label>
          <label>连续性加分
            <input type="number" step="any" v-model.number="backtestForm.streakBonus" />
          </label>
          <label>展示交易条数
            <input type="number" min="1" v-model.number="backtestForm.tradesLimit" />
          </label>
        </div>
        <div class="form__row">
          <label>展示净值节点
            <input type="number" min="1" v-model.number="backtestForm.navLimit" />
          </label>
          <label>交易 CSV (可选)
            <input type="text" v-model="backtestForm.outTrades" />
          </label>
          <label>净值 CSV (可选)
            <input type="text" v-model="backtestForm.outDaily" />
          </label>
        </div>
        <button type="submit" :disabled="backtestLoading">{{ backtestLoading ? '回测中…' : '执行回测' }}</button>
      </form>
      <p v-if="backtestError" class="status-error">{{ backtestError }}</p>
      <div v-if="backtestResult" class="result">
        <h3>统计</h3>
        <ul class="stats-list">
          <li v-for="(value, key) in backtestResult.stats" :key="key"><strong>{{ key }}:</strong> {{ value }}</li>
        </ul>
        <p v-if="backtestResult.trades_path">交易明细：<code>{{ backtestResult.trades_path }}</code></p>
        <p v-if="backtestResult.nav_path">净值曲线：<code>{{ backtestResult.nav_path }}</code></p>
        <div class="table-wrapper" v-if="tradesColumns.length">
          <h4>交易预览 ({{ backtestResult.trades_count }} 条{{ backtestResult.trades_truncated ? '，已截断展示' : '' }})</h4>
          <table>
            <thead>
              <tr>
                <th v-for="col in tradesColumns" :key="col">{{ col }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, idx) in backtestResult.trades_preview" :key="idx">
                <td v-for="col in tradesColumns" :key="col">{{ row[col] }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="table-wrapper" v-if="navColumns.length">
          <h4>净值预览 ({{ backtestResult.nav_count }} 条{{ backtestResult.nav_truncated ? '，已截断展示' : '' }})</h4>
          <table>
            <thead>
              <tr>
                <th v-for="col in navColumns" :key="col">{{ col }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(row, idx) in backtestResult.nav_preview" :key="idx">
                <td v-for="col in navColumns" :key="col">{{ row[col] }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
</template>

<style scoped>
.page {
  max-width: 1080px;
  margin: 0 auto;
  padding: 2rem 1rem 4rem;
  font-family: "SF Pro", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  color: #1f2933;
}

.page__header {
  margin-bottom: 1.5rem;
}

.page__header h1 {
  margin: 0 0 0.5rem;
  font-size: 1.8rem;
}

.card {
  background: #ffffff;
  border: 1px solid #d9e2ec;
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
}

.card h2 {
  margin-top: 0;
  font-size: 1.3rem;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.form__row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.form label {
  display: flex;
  flex-direction: column;
  font-size: 0.95rem;
  min-width: 180px;
  flex: 1;
  gap: 0.35rem;
}

.form input,
.form select {
  padding: 0.45rem 0.6rem;
  border: 1px solid #cbd2d9;
  border-radius: 6px;
  font-size: 0.95rem;
}

.form input:focus,
.form select:focus {
  outline: none;
  border-color: #4c7ef3;
  box-shadow: 0 0 0 2px rgba(76, 126, 243, 0.15);
}

.checkbox {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.95rem;
}

button {
  align-self: flex-start;
  padding: 0.55rem 1.4rem;
  border-radius: 8px;
  border: none;
  background: linear-gradient(135deg, #4c7ef3, #5a67d8);
  color: #ffffff;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.1s ease, box-shadow 0.2s ease;
}

button:hover:enabled {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(76, 126, 243, 0.25);
}

button:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.result {
  margin-top: 1rem;
  background: #f8fafc;
  border-radius: 10px;
  padding: 1rem;
  border: 1px solid #e4ebf5;
}

.table-wrapper {
  overflow-x: auto;
  margin-top: 1rem;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

table th,
table td {
  border: 1px solid #d9e2ec;
  padding: 0.4rem 0.6rem;
  text-align: left;
  white-space: nowrap;
}

table thead {
  background: #edf2fb;
}

.status-ok {
  color: #0f9d58;
  font-weight: 600;
}

.status-error {
  color: #d93025;
  font-weight: 600;
}

.stats-list {
  list-style: none;
  padding-left: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1.5rem;
  margin: 0 0 1rem;
}

.stats-list li {
  font-size: 0.92rem;
}

code {
  background: rgba(15, 23, 42, 0.06);
  padding: 0.1rem 0.3rem;
  border-radius: 4px;
}

@media (max-width: 768px) {
  .form__row {
    flex-direction: column;
  }
}
</style>
