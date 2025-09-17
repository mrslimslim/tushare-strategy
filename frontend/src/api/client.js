const defaultBase = 'http://localhost:8000'

function joinUrl(base, path) {
  if (path.startsWith('http')) {
    return path
  }
  const normalizedBase = base.endsWith('/') ? base.slice(0, -1) : base
  return `${normalizedBase}${path.startsWith('/') ? '' : '/'}${path}`
}

export async function postJson(path, payload) {
  const apiBase = import.meta.env.VITE_API_BASE || defaultBase
  const url = joinUrl(apiBase, path)
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  })
  const text = await response.text()
  if (!response.ok) {
    let message = text
    try {
      const parsed = JSON.parse(text)
      message = parsed.detail || parsed.message || text
    } catch (err) {
      // ignore parse errors and fall back to raw text
    }
    throw new Error(message || '请求失败')
  }
  return text ? JSON.parse(text) : null
}

export async function getJson(path) {
  const apiBase = import.meta.env.VITE_API_BASE || defaultBase
  const url = joinUrl(apiBase, path)
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`请求失败: ${response.status}`)
  }
  return response.json()
}
