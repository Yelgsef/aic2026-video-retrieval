// Wires together tabs, search forms, results rendering and settings.
// Plain-JS port of App.vue + SimpleQueryView/QaView/SettingsPanel/
// ResultsList/CandidateCard/SearchForm.

import { runQuery, runQueryStream } from './api.js?v=20260828-ranking-controls'
import { parseQueryResult } from './parseResult.js?v=20260828-ranking-controls'
import { settings } from './settings.js?v=20260828-ranking-controls'
import { extractFrame } from './frameExtractor.js?v=20260828-ranking-controls'
import { askGeminiWithImages, listAiStudioModels } from './geminiClient.js?v=20260828-ranking-controls'
import { appendGridCandidates, renderGrid, renderGridSkeleton, updateGridAnswer, updateGridMeta } from './resultGrid.js?v=20260828-ranking-controls'
import { setupDetailModal } from './detailModal.js?v=20260907-frame-export'
import { appendRejectedTrakeRow, appendTrakeStreamRow, finishTrakeStream, renderTrakeStreamStart } from './trakeRows.js?v=20260828-ranking-controls'
import { setupVideoFrameChecker } from './videoFrameChecker.js?v=20260907-combined-check'

const svgNS = 'http://www.w3.org/2000/svg'

function el(tag, className, children) {
  const node = document.createElement(tag)
  if (className) node.className = className
  for (const child of children ?? []) {
    if (child == null) continue
    node.append(child instanceof Node ? child : document.createTextNode(String(child)))
  }
  return node
}

function candidateKey(candidate) {
  return `${candidate.videoId}:${candidate.bestFrameId ?? candidate.sceneIndex ?? 0}`
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function timeLabel(candidate) {
  const { startTime, endTime } = candidate
  if (startTime === undefined) return null
  return endTime !== undefined ? `${formatTime(startTime)} \u2013 ${formatTime(endTime)}` : formatTime(startTime)
}

function queryUsesQwen(event) {
  // Pressing Enter has no submitter in some browsers; Qwen is the explicit
  // default for keyboard submission, while the second button is pure mode.
  return event.submitter?.dataset.queryMode !== 'plain'
}

function setQueryButtonsDisabled(panel, disabled) {
  panel.querySelectorAll('.search-form__submit').forEach((button) => {
    button.disabled = disabled
  })
}

/* ---------------- Candidate card ---------------- */

function renderCandidateCard(candidate, { selectable, selected, onToggle } = {}) {
  const card = el('div', `candidate${selectable ? ' candidate--selectable' : ''}`)
  if (selectable) {
    card.addEventListener('click', () => onToggle?.(candidate))
  }

  const thumbBtn = el('button', 'candidate__thumb')
  thumbBtn.type = 'button'
  const placeholderSvg = document.createElementNS(svgNS, 'svg')
  placeholderSvg.setAttribute('viewBox', '0 0 24 24')
  placeholderSvg.setAttribute('fill', 'none')
  placeholderSvg.setAttribute('stroke', 'currentColor')
  placeholderSvg.setAttribute('stroke-width', '1.8')
  const path = document.createElementNS(svgNS, 'path')
  path.setAttribute('d', 'M8 5v14l11-7z')
  placeholderSvg.append(path)
  thumbBtn.append(placeholderSvg)

  let previewLoaded = false
  thumbBtn.addEventListener('click', async (event) => {
    event.stopPropagation()
    if (previewLoaded) return
    previewLoaded = true
    thumbBtn.replaceChildren(el('span', 'candidate__thumb-label', ['Loading\u2026']))
    try {
      const seekTime = candidate.startTime ?? 0
      const frame = await extractFrame(candidate.videoId, seekTime)
      const img = document.createElement('img')
      img.src = frame.dataUrl
      img.alt = ''
      thumbBtn.replaceChildren(img)
    } catch (error) {
      previewLoaded = false
      const message = error instanceof Error ? error.message : 'Could not load preview.'
      thumbBtn.replaceChildren(el('span', 'candidate__thumb-label', [message]))
    }
  })

  const body = el('div', 'candidate__body')
  const row = el('div', 'candidate__row')

  if (selectable) {
    const checkbox = el('span', `candidate__checkbox${selected ? ' candidate__checkbox--checked' : ''}`)
    if (selected) {
      const checkSvg = document.createElementNS(svgNS, 'svg')
      checkSvg.setAttribute('viewBox', '0 0 24 24')
      checkSvg.setAttribute('fill', 'none')
      checkSvg.setAttribute('stroke', 'currentColor')
      checkSvg.setAttribute('stroke-width', '3')
      const checkPath = document.createElementNS(svgNS, 'path')
      checkPath.setAttribute('d', 'm5 12 5 5 9-9')
      checkSvg.append(checkPath)
      checkbox.append(checkSvg)
    }
    row.append(checkbox)
  }

  row.append(el('p', 'candidate__video-id', [candidate.videoId]))
  if (candidate.sceneIndex !== undefined) {
    row.append(el('span', 'candidate__meta', [`scene #${candidate.sceneIndex}`]))
  }
  const time = timeLabel(candidate)
  if (time) row.append(el('span', 'candidate__meta', [time]))
  if (candidate.score !== undefined) {
    row.append(el('span', 'candidate__score', [candidate.score.toFixed(3)]))
  }
  body.append(row)

  if (candidate.sceneContext) {
    body.append(el('p', 'candidate__context', [candidate.sceneContext]))
  } else if (candidate.ocrText) {
    body.append(el('p', 'candidate__context', [`OCR: ${candidate.ocrText}`]))
  } else if (candidate.asrText) {
    body.append(el('p', 'candidate__context', [`ASR: ${candidate.asrText}`]))
  }

  card.append(thumbBtn, body)
  return card
}

/* ---------------- Results list ---------------- */

function renderResults(container, result, server, query, { selectable, selectedKeys, onToggle } = {}) {
  container.replaceChildren()
  container.hidden = false

  const meta = el('p', 'results__meta', [
    'Response from ',
    el('strong', null, [server]),
    ` for "${query}".`,
  ])
  container.append(meta)

  if (result.candidates.length) {
    const list = el('div', 'results__list')
    for (const candidate of result.candidates) {
      const selected = selectable && selectedKeys?.includes(candidateKey(candidate))
      list.append(renderCandidateCard(candidate, { selectable, selected, onToggle }))
    }
    container.append(list)
  } else {
    const details = el('details', 'results__raw')
    const summary = el('summary', null, ['No structured candidates recognized \u2014 raw response'])
    const pre = el('pre', null, [JSON.stringify(result.raw, null, 2)])
    details.append(summary, pre)
    container.append(details)
  }
}

/* ---------------- Search panel (KIS / TRAKE) ---------------- */

function populateServerSelect(select, tab) {
  select.replaceChildren()
  const placeholder = document.createElement('option')
  placeholder.value = ''
  placeholder.textContent = settings.servers.length ? 'Choose server' : 'No server configured'
  select.append(placeholder)
  for (const server of settings.servers) {
    const option = document.createElement('option')
    option.value = server
    option.textContent = server
    select.append(option)
  }
  select.disabled = settings.servers.length === 0
  select.value = settings.selectedServers[tab] ?? ''
  select.addEventListener('change', () => settings.setSelectedServer(tab, select.value))
}

function setupSimplePanel(tab) {
  const panel = document.getElementById(`panel-${tab}`)
  const form = panel.querySelector('.search-form')
  const input = panel.querySelector('.search-form__input')
  const select = panel.querySelector('.search-form__select')
  const submitBtn = panel.querySelector('.search-form__submit[data-query-mode="qwen"]')
  const submitLabel = submitBtn.querySelector('.search-form__submit-label')
  const errorBox = panel.querySelector('.error-box')
  const resultsBox = panel.querySelector('.results')

  populateServerSelect(select, tab)

  function updateSubmitState() {
    setQueryButtonsDisabled(panel, !input.value.trim() || !select.value)
  }
  input.addEventListener('input', updateSubmitState)
  select.addEventListener('change', updateSubmitState)
  updateSubmitState()

  form.addEventListener('submit', async (event) => {
    event.preventDefault()
    const trimmed = input.value.trim()
    if (!trimmed || !select.value) return

    const useQwen = queryUsesQwen(event)
    setQueryButtonsDisabled(panel, true)
    submitLabel.textContent = useQwen ? 'Searching with Qwen...' : 'Searching...'
    errorBox.hidden = true
    resultsBox.hidden = true

    const response = await runQuery(select.value, tab, trimmed, useQwen)

    if (!response.ok) {
      errorBox.hidden = false
      errorBox.textContent = response.error
    } else {
      const parsed = parseQueryResult(response.data)
      renderResults(resultsBox, parsed, select.value, trimmed)
    }

    submitLabel.textContent = 'Search with Qwen'
    setQueryButtonsDisabled(panel, false)
    updateSubmitState()
  })

  return { refreshServerSelect: () => populateServerSelect(select, tab) }
}

/* ---------------- KIS panel: 10x10 preview grid + detail modal ---------------- */

function setupKisPanel(modal) {
  const tab = 'KIS'
  const panel = document.getElementById(`panel-${tab}`)
  const form = panel.querySelector('.search-form')
  const input = panel.querySelector('.search-form__input')
  const select = panel.querySelector('.search-form__select')
  const submitBtn = panel.querySelector('.search-form__submit[data-query-mode="qwen"]')
  const submitLabel = submitBtn.querySelector('.search-form__submit-label')
  const errorBox = panel.querySelector('.error-box')
  const metaEl = panel.querySelector('.grid-meta')
  const gridEl = panel.querySelector('.result-grid')
  const rerankerEnabledInput = panel.querySelector('#kis-reranker-enabled')
  rerankerEnabledInput.checked = settings.ranking.rerankerEnabled !== false
  rerankerEnabledInput.addEventListener('change', () => {
    settings.ranking.rerankerEnabled = rerankerEnabledInput.checked
    settings.persistRanking()
  })

  populateServerSelect(select, tab)

  function updateSubmitState() {
    setQueryButtonsDisabled(panel, !input.value.trim() || !select.value)
  }
  input.addEventListener('input', updateSubmitState)
  select.addEventListener('change', updateSubmitState)
  updateSubmitState()

  form.addEventListener('submit', async (event) => {
    event.preventDefault()
    const trimmed = input.value.trim()
    if (!trimmed || !select.value) return

    const useQwen = queryUsesQwen(event)
    setQueryButtonsDisabled(panel, true)
    submitLabel.textContent = useQwen ? 'Searching with Qwen...' : 'Searching...'
    errorBox.hidden = true
    renderGridSkeleton(metaEl, gridEl, select.value, trimmed)

    const response = await runQueryStream(select.value, tab, trimmed, {
      queryTransformer: useQwen,
      rerankerEnabled: settings.ranking.rerankerEnabled,
      onRanked: (items, info) => {
        const preview = parseQueryResult({ results: items })
        if (!preview.candidates.length) return
        renderGrid(metaEl, gridEl, preview, select.value, trimmed, {
          onOpen: (candidate, rank) => modal.open(candidate, rank, {
            server: select.value,
            allowSubmission: true,
          }),
        })
        updateGridMeta(
          metaEl,
          select.value,
          trimmed,
          preview.candidates.length,
          info.final ? 'Weighted ranking' : 'Weighted preview — reranking in progress',
        )
      },
      onFinalStart: () => {
        renderGridSkeleton(metaEl, gridEl, select.value, trimmed)
      },
      onBatch: (items, received) => {
        const batch = parseQueryResult({ results: items }).candidates
        appendGridCandidates(gridEl, batch, received - batch.length + 1, {
          server: select.value,
          onOpen: (candidate, rank) => modal.open(candidate, rank, { server: select.value, allowSubmission: true }),
        })
        updateGridMeta(metaEl, select.value, trimmed, received)
      },
    })

    if (!response.ok) {
      errorBox.hidden = false
      errorBox.textContent = response.error
      gridEl.hidden = true
      metaEl.hidden = true
    } else {
      const parsed = parseQueryResult(response.data)
      gridEl.querySelectorAll('.grid-skeleton').forEach((node) => node.remove())
      if (parsed.candidates.length) updateGridMeta(
        metaEl, select.value, trimmed, parsed.candidates.length,
        parsed.queryTransformerUsed ? 'Qwen' : 'Pure',
      )
      else renderGrid(metaEl, gridEl, parsed, select.value, trimmed, {
        onOpen: (candidate, rank) => modal.open(candidate, rank, { server: select.value, allowSubmission: true }),
      })
    }

    submitLabel.textContent = 'Search with Qwen'
    setQueryButtonsDisabled(panel, false)
    updateSubmitState()
  })

  return { refreshServerSelect: () => populateServerSelect(select, tab) }
}

/* ---------------- QA panel ---------------- */

function setupQaPanel(modal) {
  const panel = document.getElementById('panel-QA')
  const form = panel.querySelector('.search-form')
  const input = panel.querySelector('.search-form__input')
  const select = panel.querySelector('.search-form__select')
  const submitBtn = panel.querySelector('.search-form__submit[data-query-mode="qwen"]')
  const submitLabel = submitBtn.querySelector('.search-form__submit-label')
  const errorBox = panel.querySelector('.error-box')
  const metaEl = panel.querySelector('.grid-meta')
  const gridEl = panel.querySelector('.result-grid')

  const askBox = document.getElementById('qa-ask')
  const askCount = document.getElementById('qa-ask__count')
  const askWarning = document.getElementById('qa-ask__llm-warning')
  const askButton = document.getElementById('qa-ask__button')
  const askError = document.getElementById('qa-ask__error')
  const askAnswer = document.getElementById('qa-ask__answer')

  populateServerSelect(select, 'QA')

  function updateSubmitState() {
    setQueryButtonsDisabled(panel, !input.value.trim() || !select.value)
  }
  input.addEventListener('input', updateSubmitState)
  select.addEventListener('change', updateSubmitState)
  updateSubmitState()

  let lastResult = null
  let lastQuery = ''
  const frameAnswers = new Map()
  let llmRunning = false

  function frameAnswerKey(candidate) {
    return `${candidate.videoId}:${candidate.bestFrameId ?? candidate.representativeFrameIds?.[0] ?? 0}`
  }

  function effectiveCandidates() {
    if (!lastResult) return []
    const { candidates } = lastResult
    return candidates.slice(0, Math.max(1, settings.llm.maxImages))
  }

  function renderAskState() {
    if (!lastResult || !lastResult.candidates.length) {
      askBox.hidden = true
      return
    }
    askBox.hidden = false
    const count = `Using top ${effectiveCandidates().length} scene(s)`
    askCount.textContent = `${count} \u2014 max ${settings.llm.maxImages} image(s) sent, configurable in Settings.`
    askWarning.hidden = settings.isLlmConfigured
    askButton.disabled = !settings.isLlmConfigured
  }

  function rerenderResults() {
    renderGrid(metaEl, gridEl, lastResult, select.value, lastQuery, {
      onOpen: (candidate, rank) => modal.open(candidate, rank, {
        answer: frameAnswers.get(frameAnswerKey(candidate)) || lastResult.answer?.text,
        server: select.value,
      }),
      answers: frameAnswers,
    })
  }

  function candidateSystemPrompt(candidate) {
    const evidence = [
      `Video ID: ${candidate.videoId}`,
      `Frame ID: ${candidate.bestFrameId ?? candidate.representativeFrameIds?.[0] ?? 0}`,
      `Scene context: ${candidate.sceneContext || '(empty)'}`,
      `OCR: ${candidate.ocrText || '(empty)'}`,
      `ASR: ${candidate.asrText || '(empty)'}`,
    ].join('\n')
    return `${settings.llm.systemPrompt.trim()}\n\nRetrieved metadata for this frame:\n${evidence}`.trim()
  }

  async function frameDataUrl(candidate) {
    const frameId = candidate.bestFrameId ?? candidate.representativeFrameIds?.[0] ?? 0
    const response = await fetch(`${select.value}/api/frame/${encodeURIComponent(candidate.videoId)}/${frameId}?width=1280`)
    if (!response.ok) throw new Error(`Could not load frame ${frameId} (HTTP ${response.status}).`)
    const blob = await response.blob()
    const bitmap = await createImageBitmap(blob)
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      bitmap.close()
      throw new Error('Could not decode frame image.')
    }
    ctx.drawImage(bitmap, 0, 0)
    bitmap.close()
    return canvas.toDataURL('image/jpeg', 0.85)
  }

  async function askAllFrames() {
    if (llmRunning) return
    if (!settings.isLlmConfigured) {
      askError.hidden = false
      askError.textContent = 'Select AI Studio and configure its model in Settings first.'
      return
    }
    if (!lastQuery) return
    const candidates = effectiveCandidates()
    if (!candidates.length) return

    llmRunning = true
    askButton.disabled = true
    askError.hidden = true
    askAnswer.hidden = true
    try {
      for (const [index, candidate] of candidates.entries()) {
        askButton.textContent = `Asking LLM for frame ${index + 1}/${candidates.length}…`
        let answer
        try {
          const dataUrl = await frameDataUrl(candidate)
          const llmSettings = { ...settings.llm, systemPrompt: candidateSystemPrompt(candidate) }
          const result = await askGeminiWithImages(select.value, llmSettings, lastQuery, [
            { dataUrl, label: `${candidate.videoId}, frame ${candidate.bestFrameId ?? 0}` },
          ])
          answer = result.ok ? result.text : `LLM error: ${result.error}`
        } catch (error) {
          answer = `LLM error: ${error instanceof Error ? error.message : 'Could not process frame.'}`
        }
        frameAnswers.set(frameAnswerKey(candidate), answer)
        updateGridAnswer(gridEl, candidate, answer)
        askAnswer.hidden = false
        askAnswer.textContent = `Đã trả lời ${frameAnswers.size}/${candidates.length} frame.`
      }
    } finally {
      llmRunning = false
      askButton.disabled = !settings.isLlmConfigured
      askButton.textContent = 'Retry LLM for these frames'
    }
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault()
    const trimmed = input.value.trim()
    if (!trimmed || !select.value) return

    const useQwen = queryUsesQwen(event)
    setQueryButtonsDisabled(panel, true)
    submitLabel.textContent = useQwen ? 'Searching with Qwen...' : 'Searching...'
    errorBox.hidden = true
    renderGridSkeleton(metaEl, gridEl, select.value, trimmed)
    askAnswer.hidden = true
    askError.hidden = true
    frameAnswers.clear()

    const response = await runQueryStream(select.value, 'QA', trimmed, {
      queryTransformer: useQwen,
      onBatch: (items, received) => {
        const batch = parseQueryResult({ results: items }).candidates
        appendGridCandidates(gridEl, batch, received - batch.length + 1, {
          server: select.value,
          answers: frameAnswers,
          onOpen: (candidate, rank) => modal.open(candidate, rank, {
            answer: frameAnswers.get(frameAnswerKey(candidate)),
            server: select.value,
          }),
        })
        updateGridMeta(metaEl, select.value, trimmed, received)
      },
    })

    if (!response.ok) {
      errorBox.hidden = false
      errorBox.textContent = response.error
      lastResult = null
      gridEl.hidden = true
      metaEl.hidden = true
    } else {
      lastResult = parseQueryResult(response.data)
      lastQuery = trimmed
      gridEl.querySelectorAll('.grid-skeleton').forEach((node) => node.remove())
      if (lastResult.candidates.length) updateGridMeta(
        metaEl, select.value, trimmed, lastResult.candidates.length,
        lastResult.queryTransformerUsed ? 'Qwen' : 'Pure',
      )
      else rerenderResults()
    }
    renderAskState()

    if (response.ok && settings.isLlmConfigured && lastResult?.candidates.length) {
      submitLabel.textContent = 'Answering...'
      await askAllFrames()
    }

    submitLabel.textContent = 'Search with Qwen'
    setQueryButtonsDisabled(panel, false)
    updateSubmitState()
  })

  askButton.addEventListener('click', askAllFrames)

  return { refreshServerSelect: () => populateServerSelect(select, 'QA') }
}

function setupTrakePanel(modal) {
  const tab = 'TRAKE'
  const panel = document.getElementById('panel-TRAKE')
  const form = panel.querySelector('.search-form')
  const input = panel.querySelector('.search-form__input')
  const select = panel.querySelector('.search-form__select')
  const submitBtn = panel.querySelector('.search-form__submit[data-query-mode="qwen"]')
  const submitLabel = submitBtn.querySelector('.search-form__submit-label')
  const errorBox = panel.querySelector('.error-box')
  const resultsBox = panel.querySelector('.results')
  populateServerSelect(select, tab)
  const update = () => { setQueryButtonsDisabled(panel, !input.value.trim() || !select.value) }
  input.addEventListener('input', update)
  select.addEventListener('change', update)
  update()
  form.addEventListener('submit', async (event) => {
    event.preventDefault()
    const query = input.value.trim()
    if (!query || !select.value) return
    const useQwen = queryUsesQwen(event)
    setQueryButtonsDisabled(panel, true)
    submitLabel.textContent = useQwen ? 'Searching with Qwen...' : 'Searching...'
    errorBox.hidden = true
    renderTrakeStreamStart(resultsBox, query)
    const response = await runQueryStream(select.value, tab, query, {
      queryTransformer: useQwen,
      onTrakeRow: (row, rank) => appendTrakeStreamRow(resultsBox, row, rank, select.value, (candidate, eventIndex) => {
        modal.open(candidate, eventIndex, { rankLabel: `T${rank} — Event #${eventIndex}`, server: select.value })
      }),
      onTrakeRejected: (row, rank) => appendRejectedTrakeRow(resultsBox, row, rank, select.value, (candidate, eventIndex) => {
        modal.open(candidate, eventIndex, { rankLabel: `Bị loại — Event #${eventIndex}`, server: select.value })
      }),
    })
    if (!response.ok) {
      errorBox.hidden = false
      errorBox.textContent = response.error
      resultsBox.hidden = true
    } else {
      finishTrakeStream(resultsBox, response.data.trake_rows?.length ?? 0)
    }
    submitLabel.textContent = 'Search with Qwen'
    setQueryButtonsDisabled(panel, false)
    update()
  })
  return { refreshServerSelect: () => populateServerSelect(select, tab) }
}

/* ---------------- Settings panel ---------------- */

function setupSettingsPanel(onServersChanged) {
  const serverForm = document.getElementById('server-form')
  const serverUrlInput = document.getElementById('server-url')
  const serverError = document.getElementById('server-error')
  const serverList = document.getElementById('server-list')
  const serverEmpty = document.getElementById('server-empty')
  const serversCount = document.getElementById('servers-count')

  function renderServerList() {
    serverList.replaceChildren()
    serversCount.textContent = String(settings.servers.length).padStart(2, '0')
    serverEmpty.hidden = settings.servers.length > 0

    settings.servers.forEach((server, index) => {
      const item = el('div', 'server-item')
      item.append(el('span', 'server-item__index', [String(index + 1).padStart(2, '0')]))
      item.append(el('span', 'server-item__url', [server]))

      const removeBtn = el('button', 'server-item__remove')
      removeBtn.type = 'button'
      removeBtn.setAttribute('aria-label', `Remove ${server}`)
      const svg = document.createElementNS(svgNS, 'svg')
      svg.setAttribute('viewBox', '0 0 24 24')
      svg.setAttribute('fill', 'none')
      svg.setAttribute('stroke', 'currentColor')
      svg.setAttribute('stroke-width', '1.8')
      const path = document.createElementNS(svgNS, 'path')
      path.setAttribute('d', 'M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13M10 11v5m4-5v5')
      svg.append(path)
      removeBtn.append(svg)
      removeBtn.addEventListener('click', () => {
        settings.removeServer(server)
        renderServerList()
        onServersChanged()
      })

      item.append(removeBtn)
      serverList.append(item)
    })
  }

  serverForm.addEventListener('submit', (event) => {
    event.preventDefault()
    const error = settings.addServer(serverUrlInput.value)
    serverError.textContent = error ?? ''
    if (!error) {
      serverUrlInput.value = ''
      renderServerList()
      onServersChanged()
    }
  })
  serverUrlInput.addEventListener('input', () => {
    serverError.textContent = ''
  })

  renderServerList()

  // LLM settings
  const providerInputs = document.querySelectorAll('input[name="llm-provider"]')
  const aiStudioKeyInput = document.getElementById('llm-ai-studio-api-key')
  const router9KeyInput = document.getElementById('llm-9router-api-key')
  const aiStudioKeySave = document.getElementById('llm-ai-studio-api-key-save')
  const router9KeySave = document.getElementById('llm-9router-api-key-save')
  const aiStudioKeyStatus = document.getElementById('llm-ai-studio-api-key-status')
  const router9KeyStatus = document.getElementById('llm-9router-api-key-status')
  const modelInput = document.getElementById('llm-model')
  const modelRefresh = document.getElementById('llm-model-refresh')
  const modelStatus = document.getElementById('llm-model-status')
  const maxImagesInput = document.getElementById('llm-max-images')
  const systemPromptInput = document.getElementById('llm-system-prompt')
  const resetBtn = document.getElementById('llm-reset')

  function renderLlmFields() {
    providerInputs.forEach((input) => { input.checked = input.value === settings.llm.provider })
    aiStudioKeyInput.value = settings.llm.aiStudioApiKey
    router9KeyInput.value = settings.llm.router9ApiKey
    modelInput.replaceChildren()
    const currentModel = document.createElement('option')
    currentModel.value = settings.llm.model
    currentModel.textContent = settings.llm.model
    modelInput.append(currentModel)
    modelInput.value = settings.llm.model
    maxImagesInput.value = String(settings.llm.maxImages)
    systemPromptInput.value = settings.llm.systemPrompt
    aiStudioKeyStatus.textContent = settings.llm.aiStudioApiKey ? 'API key đã được đọc từ localStorage của website này.' : 'Chưa có key trong localStorage; backend sẽ dùng AI_STUDIO_API_KEY nếu đã cấu hình.'
    router9KeyStatus.textContent = settings.llm.router9ApiKey ? 'API key đã được đọc từ localStorage của website này.' : 'Chưa có key trong localStorage; backend sẽ dùng NINEROUTER_KEY nếu đã cấu hình.'
    if (settings.llm.aiStudioApiKey) modelStatus.textContent = 'AI Studio API key đã lưu. Bấm Load models để tải danh sách.'
  }

  providerInputs.forEach((input) => input.addEventListener('change', () => {
    // The UI uses checkboxes as requested, but they behave as a mutually
    // exclusive provider selector and never allow an ambiguous empty state.
    if (!input.checked) {
      input.checked = true
      return
    }
    providerInputs.forEach((other) => { other.checked = other === input })
    settings.llm.provider = input.value
    settings.persistLlm()
  }))
  aiStudioKeyInput.addEventListener('input', () => { aiStudioKeyStatus.textContent = 'Chưa lưu thay đổi.' })
  router9KeyInput.addEventListener('input', () => { router9KeyStatus.textContent = 'Chưa lưu thay đổi.' })
  function saveApiKey(field, value, status) {
    settings.llm[field] = value.trim()
    const writeSucceeded = settings.persistLlm()
    if (!settings.llm[field]) {
      status.textContent = 'Đã xóa key khỏi localStorage.'
    } else if (writeSucceeded && settings.isApiKeyPersisted(field)) {
      status.textContent = `Đã lưu và xác nhận trong localStorage của ${window.location.origin}.`
    } else {
      // Do not keep using an unpersisted browser key silently. An empty key in
      // the API request makes the Python backend use its .env fallback.
      settings.llm[field] = ''
      status.textContent = 'Lưu localStorage thất bại. Hệ thống sẽ dùng key fallback từ .env của backend.'
    }
    if (field === 'aiStudioApiKey' && settings.llm[field]) {
      modelStatus.textContent = 'AI Studio API key đã lưu. Bấm Load models để tải danh sách.'
    }
  }
  aiStudioKeySave.addEventListener('click', () => saveApiKey('aiStudioApiKey', aiStudioKeyInput.value, aiStudioKeyStatus))
  router9KeySave.addEventListener('click', () => saveApiKey('router9ApiKey', router9KeyInput.value, router9KeyStatus))
  modelInput.addEventListener('input', () => {
    settings.llm.model = modelInput.value
    settings.persistLlm()
  })
  modelRefresh.addEventListener('click', async () => {
    // Save the value currently visible in the field so Load models also works
    // immediately after pasting a new key.
    const provider = settings.llm.provider
    const key = provider === '9router' ? router9KeyInput.value : aiStudioKeyInput.value
    const keyField = provider === '9router' ? 'router9ApiKey' : 'aiStudioApiKey'
    const keyStatus = provider === '9router' ? router9KeyStatus : aiStudioKeyStatus
    saveApiKey(keyField, key, keyStatus)
    modelRefresh.disabled = true
    modelRefresh.textContent = 'Loading...'
    modelStatus.textContent = `Loading models from ${provider === '9router' ? '9Router' : 'Google AI Studio'}...`
    const selectedServer = settings.selectedServers.QA
    const server = settings.servers.includes(selectedServer) ? selectedServer : (settings.servers[0] || '')
    const result = await listAiStudioModels(server, settings.llm[keyField], provider)
    if (!result.ok) {
      modelStatus.textContent = result.error
    } else if (!result.models.length) {
      modelStatus.textContent = 'No models supporting generateContent were returned.'
    } else {
      modelInput.replaceChildren()
      for (const model of result.models) {
        const option = document.createElement('option')
        option.value = model.id
        option.textContent = model.name === model.id ? model.id : `${model.name} — ${model.id}`
        option.title = model.description || model.id
        modelInput.append(option)
      }
      const available = result.models.some((model) => model.id === settings.llm.model)
      settings.llm.model = available ? settings.llm.model : result.models[0].id
      modelInput.value = settings.llm.model
      settings.persistLlm()
      modelStatus.textContent = `${result.models.length} generateContent model(s) available.`
    }
    modelRefresh.disabled = false
    modelRefresh.textContent = 'Reload models'
  })
  maxImagesInput.addEventListener('input', () => {
    const value = Number(maxImagesInput.value)
    settings.llm.maxImages = Number.isFinite(value) ? value : settings.llm.maxImages
    settings.persistLlm()
  })
  systemPromptInput.addEventListener('input', () => {
    settings.llm.systemPrompt = systemPromptInput.value
    settings.persistLlm()
  })
  resetBtn.addEventListener('click', () => {
    settings.resetLlmSettings()
    renderLlmFields()
  })

  renderLlmFields()
}

/* ---------------- Tabs ---------------- */

function setupTabs() {
  const buttons = document.querySelectorAll('.tabs__btn')
  const panels = document.querySelectorAll('.panel')

  function activate(tab) {
    buttons.forEach((btn) => {
      const isActive = btn.dataset.tab === tab
      btn.classList.toggle('is-active', isActive)
      if (isActive) btn.setAttribute('aria-current', 'page')
      else btn.removeAttribute('aria-current')
    })
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.panel !== tab
    })
  }

  buttons.forEach((btn) => {
    btn.addEventListener('click', () => activate(btn.dataset.tab))
  })

  activate('KIS')
}

/* ---------------- Bootstrap ---------------- */

function main() {
  setupTabs()
  const modal = setupDetailModal()
  const kis = setupKisPanel(modal)
  const trake = setupTrakePanel(modal)
  const qa = setupQaPanel(modal)
  const checker = setupVideoFrameChecker({ modal, populateServerSelect })
  setupSettingsPanel(() => {
    kis.refreshServerSelect()
    trake.refreshServerSelect()
    qa.refreshServerSelect()
    checker.refreshServerSelect()
  })
}

document.addEventListener('DOMContentLoaded', main)
