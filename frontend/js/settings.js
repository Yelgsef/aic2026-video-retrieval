// Backend server list + LLM settings, persisted to localStorage.
// Direct port of src/stores/settings.ts (Pinia) without the framework.

const SERVERS_KEY = 'query-workbench-servers'
const SELECTED_SERVERS_KEY = 'query-workbench-selected-servers'
const LLM_SETTINGS_KEY = 'query-workbench-llm-settings'
const RANKING_SETTINGS_KEY = 'query-workbench-ranking-settings'

export const DEFAULT_RANKING_SETTINGS = {
  exactOcrWeight: 5,
  exactMetadataWeight: 3,
  weightingMode: 'global',
  rerankerEnabled: true,
  similarFilterEnabled: true,
  similarFrameThreshold: 0.92,
  similarFrameFilterK: 100,
}

function defaultServer() {
  return window.location.origin
}

export const DEFAULT_LLM_SETTINGS = {
  provider: '9router',
  aiStudioApiKey: '',
  router9ApiKey: '',
  model: 'ag/gemini-3.8-flash-medium',
  systemPrompt:
    'Bạn là trợ lý hỏi đáp hình ảnh cho hệ thống truy xuất video. Bạn nhận được một keyframe ' +
    'và một câu hỏi. Hãy trả lời ngắn gọn, trực tiếp bằng tiếng Việt, chỉ dựa trên nội dung ' +
    'thực sự nhìn thấy trong frame. Không suy đoán thông tin ngoài ảnh. Nếu frame không đủ ' +
    'bằng chứng để trả lời, hãy nói rõ: "Không đủ thông tin trong frame này."',
  maxImages: 5,
}

export const WORKSPACE_TABS = ['KIS', 'TRAKE', 'QA', 'CHECK']

function loadJson(key, fallback) {
  try {
    const stored = localStorage.getItem(key)
    if (!stored) return fallback
    return JSON.parse(stored)
  } catch {
    return fallback
  }
}

function loadServers() {
  const stored = loadJson(SERVERS_KEY, [defaultServer()])
  const servers = Array.isArray(stored) ? stored.filter((item) => typeof item === 'string') : []
  // An HTTPS tunnel page cannot call the old HTTP localhost default because
  // of mixed-content rules (and localhost would mean the viewer's machine).
  if (window.location.protocol === 'https:' && !servers.includes(defaultServer())) servers.unshift(defaultServer())
  return servers
}

function loadSelectedServers() {
  const stored = loadJson(SELECTED_SERVERS_KEY, {})
  const fallback = defaultServer()
  const selected = (value) => window.location.protocol === 'https:' && value?.startsWith('http://localhost') ? fallback : (value ?? fallback)
  return {
    KIS: selected(stored.KIS),
    TRAKE: selected(stored.TRAKE),
    QA: selected(stored.QA),
    CHECK: selected(stored.CHECK),
  }
}

function loadRankingSettings() {
  const stored = loadJson(RANKING_SETTINGS_KEY, {})
  return { ...DEFAULT_RANKING_SETTINGS, ...stored }
}

function loadLlmSettings() {
  const stored = loadJson(LLM_SETTINGS_KEY, {})
  const merged = { ...DEFAULT_LLM_SETTINGS, ...stored }
  // Preserve an existing AI Studio key from the previous single-provider UI.
  if (!merged.aiStudioApiKey && typeof stored.apiKey === 'string') merged.aiStudioApiKey = stored.apiKey
  return merged
}

class SettingsStore {
  constructor() {
    this.servers = loadServers()
    this.selectedServers = loadSelectedServers()
    this.llm = loadLlmSettings()
    this.ranking = loadRankingSettings()
  }

  get isLlmConfigured() {
    return Boolean(this.llm.model.trim() && ['ai_studio', '9router'].includes(this.llm.provider))
  }

  persistServers() {
    localStorage.setItem(SERVERS_KEY, JSON.stringify(this.servers))
  }

  persistSelectedServers() {
    localStorage.setItem(SELECTED_SERVERS_KEY, JSON.stringify(this.selectedServers))
  }

  persistRanking() {
    localStorage.setItem(RANKING_SETTINGS_KEY, JSON.stringify(this.ranking))
  }

  persistLlm() {
    try {
      localStorage.setItem(LLM_SETTINGS_KEY, JSON.stringify(this.llm))
      return true
    } catch {
      return false
    }
  }

  isApiKeyPersisted(field) {
    const stored = loadJson(LLM_SETTINGS_KEY, {})
    return typeof stored[field] === 'string' && stored[field] === this.llm[field]
  }

  /** @returns {string | null} error message, or null on success */
  addServer(url) {
    const value = url.trim().replace(/\/$/, '')
    if (!value) return 'Enter a server URL.'

    try {
      const parsed = new URL(value)
      if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error()
    } catch {
      return 'Use a valid URL beginning with http:// or https://.'
    }

    if (this.servers.includes(value)) return 'This server is already in your list.'

    this.servers.push(value)
    this.persistServers()
    return null
  }

  removeServer(server) {
    this.servers = this.servers.filter((item) => item !== server)
    this.persistServers()
    for (const tab of WORKSPACE_TABS) {
      if (this.selectedServers[tab] === server) this.selectedServers[tab] = ''
    }
    this.persistSelectedServers()
  }

  setSelectedServer(tab, server) {
    this.selectedServers[tab] = server
    this.persistSelectedServers()
  }

  resetLlmSettings() {
    this.llm = { ...DEFAULT_LLM_SETTINGS }
    this.persistLlm()
  }

  resetRankingSettings() {
    this.ranking = { ...DEFAULT_RANKING_SETTINGS }
    this.persistRanking()
  }
}

export const settings = new SettingsStore()
