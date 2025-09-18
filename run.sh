#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PORT="${API_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UVICORN_BIN="${UVICORN_BIN:-uvicorn}"
FRONTEND_DIR="${ROOT_DIR}/frontend"

if [ -z "${VIRTUAL_ENV:-}" ] && [ ! -d "${ROOT_DIR}/.venv" ]; then
  echo "[run.sh] Warning: no active virtualenv detected; consider running 'python -m venv .venv && source .venv/bin/activate'" >&2
fi

if ! command -v "${UVICORN_BIN}" >/dev/null 2>&1; then
  echo "[run.sh] uvicorn 未找到，准备安装 Python 依赖..." >&2
  "${PYTHON_BIN}" -m pip install --upgrade pip >/dev/null
  "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt"
fi

if [ ! -d "${FRONTEND_DIR}" ]; then
  echo "[run.sh] 未找到前端目录 ${FRONTEND_DIR}" >&2
  exit 1
fi

if [ ! -d "${FRONTEND_DIR}/node_modules" ]; then
  echo "[run.sh] 正在为前端安装依赖 (npm install)..." >&2
  (cd "${FRONTEND_DIR}" && npm install)
fi

if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node -v | sed 's/v\([0-9]*\).*/\1/')"
  if [ "${NODE_MAJOR}" -lt 18 ]; then
    echo "[run.sh] Warning: 检测到 Node.js 版本 < 18，Vite dev server 可能无法正常运行。" >&2
  fi
else
  echo "[run.sh] Warning: 未检测到 Node.js，请先安装 Node.js >= 18。" >&2
fi

cleanup() {
  echo "\n[run.sh] 正在停止服务..." >&2
  if [ -n "${API_PID:-}" ] && kill -0 "${API_PID}" >/dev/null 2>&1; then
    kill "${API_PID}" >/dev/null 2>&1 || true
  fi
  if [ -n "${FRONT_PID:-}" ] && kill -0 "${FRONT_PID}" >/dev/null 2>&1; then
    kill "${FRONT_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "${ROOT_DIR}"

echo "[run.sh] 启动后端: ${UVICORN_BIN} service.api:app --host 0.0.0.0 --port ${API_PORT} --reload" >&2
"${UVICORN_BIN}" service.api:app --host 0.0.0.0 --port "${API_PORT}" --reload &
API_PID=$!

sleep 1

echo "[run.sh] 启动前端: npm run dev -- --host --port ${FRONTEND_PORT} (VITE_API_BASE=http://localhost:${API_PORT})" >&2
VITE_API_BASE="http://localhost:${API_PORT}" npm --prefix "${FRONTEND_DIR}" run dev -- --host --port "${FRONTEND_PORT}" &
FRONT_PID=$!

wait
