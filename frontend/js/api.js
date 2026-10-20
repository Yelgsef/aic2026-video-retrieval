// Calls the gateway's `POST /api/query` endpoint. Port of src/lib/queryApi.ts.

/**
 * @param {string} server
 * @param {'KIS'|'TRAKE'|'QA'} tab
 * @param {string} query
 * @param {boolean} queryTransformer Whether Qwen must transform the query.
 * @returns {Promise<{ok: true, data: unknown} | {ok: false, error: string}>}
 */
export async function runQuery(server, tab, query, queryTransformer = false) {
  try {
    const response = await fetch(`${server}/api/query`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ type: tab, query, query_transformer: queryTransformer }),
    })
    const body = await response.json().catch(() => null)

    if (!response.ok) {
      const message =
        body && typeof body === 'object' && typeof body.error === 'string'
          ? body.error
          : body && typeof body === 'object' && typeof body.detail === 'string'
            ? body.detail
          : `Backend returned HTTP ${response.status}.`
      return { ok: false, error: message }
    }

    const data = body && typeof body === 'object' && 'data' in body ? body.data : body
    return { ok: true, data }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Could not reach the backend.' }
  }
}

export async function runQueryStream(server, tab, query, {
  queryTransformer = false,
  rerankerEnabled = true,
  onStart,
  onRanked,
  onFinalStart,
  onBatch,
  onTrakeRow,
  onTrakeRejected,
} = {}) {
  const collected = []
  const trakeRows = []
  let previewItems = []
  let finalBatchesStarted = false
  try {
    const response = await fetch(`${server}/api/query/stream`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/x-ndjson' },
      body: JSON.stringify({
        type: tab,
        query,
        query_transformer: queryTransformer,
        reranker_enabled: rerankerEnabled,
      }),
    })
    if (!response.ok || !response.body) {
      const payload = await response.json().catch(() => null)
      return { ok: false, error: payload?.error ?? payload?.detail ?? `Backend returned HTTP ${response.status}.` }
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let metadata = {}
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.trim()) continue
        const message = JSON.parse(line)
        if (message.event === 'start') onStart?.()
        if (message.event === 'ranked') {
          previewItems = Array.isArray(message.items) ? message.items : []
          onRanked?.(previewItems, message)
        }
        if (message.event === 'batch') {
          if (!finalBatchesStarted) {
            finalBatchesStarted = true
            onFinalStart?.()
          }
          collected.push(...message.items)
          onBatch?.(message.items, collected.length)
        }
        if (message.event === 'trake_row') {
          trakeRows.push(message.row)
          onTrakeRow?.(message.row, message.rank)
        }
        if (message.event === 'trake_rejected') onTrakeRejected?.(message.row, message.rank)
        if (message.event === 'complete') metadata = message.data ?? {}
        if (message.event === 'error') return { ok: false, error: message.error ?? 'Query pipeline failed.' }
      }
      if (done) break
    }
    return {
      ok: true,
      data: { ...metadata, results: collected.length ? collected : previewItems, trake_rows: trakeRows },
    }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : 'Could not reach the backend.' }
  }
}
