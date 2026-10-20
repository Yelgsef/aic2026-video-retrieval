// Sends QA requests to the Python backend. The backend calls Google AI
// Studio; 9Router is reserved in Settings for a later implementation.

export async function askGeminiWithImages(server, settings, question, images) {
  const provider = settings.provider
  const apiKey = provider === '9router' ? settings.router9ApiKey : settings.aiStudioApiKey
  if (provider === '9router') return { ok: false, error: '9Router is not implemented yet. Select AI Studio in Settings.' }
  if (!settings.model.trim()) return { ok: false, error: 'LLM model is not configured.' }

  try {
    const response = await fetch(`${server}/api/qa/answer`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        provider,
        api_key: apiKey,
        model: settings.model,
        system_prompt: settings.systemPrompt,
        question,
        images: images.map((image) => ({ data_url: image.dataUrl, label: image.label })),
      }),
      signal: AbortSignal.timeout(65_000),
    })
    const payload = await response.json().catch(() => null)
    if (!response.ok) {
      const fallback = response.status === 404
        ? 'The QA answer endpoint is unavailable. Restart/update the Python backend, then try again.'
        : `Backend returned HTTP ${response.status}.`
      return { ok: false, error: payload?.error ?? payload?.detail ?? fallback }
    }
    if (!payload?.text) return { ok: false, error: 'The LLM returned an empty response.' }
    return { ok: true, text: payload.text }
  } catch (error) {
    return { ok: false, error: `Could not reach the QA backend: ${error instanceof Error ? error.message : 'network error'}` }
  }
}

export async function listAiStudioModels(server, apiKey, provider = 'ai_studio') {
  if (!server) return { ok: false, error: 'Configure and select a QA backend server first.' }
  try {
    const response = await fetch(`${server}/api/qa/models`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ provider, api_key: apiKey }),
      signal: AbortSignal.timeout(35_000),
    })
    const payload = await response.json().catch(() => null)
    if (!response.ok) {
      const fallback = response.status === 404
        ? 'The QA model-list endpoint is unavailable. Restart/update the Python backend, then try again.'
        : `Backend returned HTTP ${response.status}.`
      return { ok: false, error: payload?.error ?? payload?.detail ?? fallback }
    }
    return { ok: true, models: Array.isArray(payload?.models) ? payload.models : [] }
  } catch (error) {
    return { ok: false, error: `Could not load models: ${error instanceof Error ? error.message : 'network error'}` }
  }
}
