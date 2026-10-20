// KIS detail modal: video (left panel = info, right = video, per user's
// request "1 popup modal để xem video và thông tin bên trái"). This is the
// only place in the app that ever requests /videos/{video_id}.mp4 — the
// grid only shows single-frame JPEG thumbnails from /api/frame/.

import { candidateFrameId } from './resultGrid.js?v=20260828-ranking-controls'

function formatTime(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function fact(dl, label, value) {
  if (value === undefined || value === null || value === '') return
  const dt = document.createElement('dt')
  dt.textContent = label
  const dd = document.createElement('dd')
  dd.textContent = value
  dl.append(dt, dd)
}

export function buildKisSubmission(videoId, centerFrame, frameCount) {
  const start = Math.max(0, centerFrame - 49)
  const end = Math.min(frameCount - 1, centerFrame + 50)
  const lines = []
  for (let frameId = start; frameId <= end && lines.length < 100; frameId += 1) {
    lines.push(`${videoId},${frameId}`)
  }
  return { csv: `\uFEFF${lines.join('\r\n')}\r\n`, count: lines.length, start, end }
}

export function frameImageUrl(server, videoId, frameId, width = 1280) {
  const base = String(server ?? '').replace(/\/$/, '')
  return `${base}/api/frame/${encodeURIComponent(videoId)}/${frameId}?width=${width}`
}

export function frameImageFilename(videoId, frameId) {
  return `${videoId}_${frameId}.jpg`
}

export function setupDetailModal() {
  const overlay = document.getElementById('detail-modal')
  const closeBtn = document.getElementById('modal-close')
  const rankEl = document.getElementById('modal-rank')
  const videoIdEl = document.getElementById('modal-video-id')
  const factsEl = document.getElementById('modal-facts')
  const contextEl = document.getElementById('modal-context')
  const ocrBlock = document.getElementById('modal-ocr-block')
  const ocrEl = document.getElementById('modal-ocr')
  const asrBlock = document.getElementById('modal-asr-block')
  const asrEl = document.getElementById('modal-asr')
  const answerBlock = document.getElementById('modal-answer-block')
  const answerEl = document.getElementById('modal-answer')
  const videoEl = document.getElementById('modal-video-el')
  const currentFrameEl = document.getElementById('modal-current-frame')
  const fpsEl = document.getElementById('modal-fps')
  const frameInput = document.getElementById('modal-frame-input')
  const frameLoad = document.getElementById('modal-frame-load')
  const frameStatus = document.getElementById('modal-frame-status')
  const frameDownload = document.getElementById('modal-frame-download')
  const submissionDownload = document.getElementById('modal-submission-download')
  let metadata = null
  let openToken = 0
  let activeVideoId = ''
  let activeServer = ''
  let allowSubmission = false
  let allowFrameDownload = false
  let frameDownloadBusy = false
  let playbackReady = false

  function setPlaybackReady(ready) {
    playbackReady = ready
    videoEl.controls = ready
  }

  function seekAndUnlock(timeSeconds, token, statusText, onReady) {
    if (token !== openToken || videoEl.readyState < 1) return
    setPlaybackReady(false)
    const duration = Number.isFinite(videoEl.duration) ? videoEl.duration : timeSeconds
    const target = Math.min(Math.max(timeSeconds, 0), Math.max(duration - 0.05, 0))
    let finished = false
    let timeoutId = null
    const cleanup = () => {
      if (timeoutId !== null) clearTimeout(timeoutId)
      videoEl.removeEventListener('seeked', finish)
      videoEl.removeEventListener('loadeddata', finish)
      videoEl.removeEventListener('error', fail)
    }
    const finish = () => {
      if (finished || token !== openToken) return
      finished = true
      cleanup()
      setPlaybackReady(true)
      updateCurrentFrame()
      updateDownloadVisibility()
      frameStatus.textContent = statusText
      onReady?.()
    }
    const fail = () => {
      if (finished || token !== openToken) return
      finished = true
      cleanup()
      setPlaybackReady(false)
      frameLoad.disabled = true
      frameDownload.hidden = true
      submissionDownload.hidden = true
      frameStatus.textContent = 'Không thể tải hoặc seek tới frame được yêu cầu.'
    }

    videoEl.addEventListener('error', fail, { once: true })
    timeoutId = setTimeout(fail, 15_000)

    if (Math.abs(videoEl.currentTime - target) < 0.001) {
      if (videoEl.readyState >= 2) finish()
      else videoEl.addEventListener('loadeddata', finish, { once: true })
      return
    }
    videoEl.addEventListener('seeked', finish, { once: true })
    videoEl.currentTime = target
  }

  function updateCurrentFrame() {
    currentFrameEl.textContent = metadata?.fps ? String(Math.max(0, Math.floor(videoEl.currentTime * metadata.fps + 1e-6))) : '—'
  }

  function currentFrameId() {
    return metadata?.fps ? Math.max(0, Math.floor(videoEl.currentTime * metadata.fps + 1e-6)) : null
  }

  function updateDownloadVisibility() {
    frameDownload.hidden = !(allowFrameDownload && playbackReady && metadata?.fps && videoEl.paused)
    frameDownload.disabled = frameDownloadBusy
    submissionDownload.hidden = !(allowSubmission && playbackReady && metadata?.fps && videoEl.paused)
  }

  async function downloadFrame() {
    const frameId = currentFrameId()
    if (frameId === null || !activeVideoId || frameDownloadBusy) return
    const token = openToken
    frameDownloadBusy = true
    updateDownloadVisibility()
    frameStatus.textContent = `Đang xuất frame ${frameId}…`
    try {
      const response = await fetch(frameImageUrl(activeServer, activeVideoId, frameId))
      if (!response.ok) {
        const body = await response.json().catch(() => null)
        throw new Error(body?.detail ?? `HTTP ${response.status}`)
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = frameImageFilename(activeVideoId, frameId)
      anchor.click()
      setTimeout(() => URL.revokeObjectURL(url), 0)
      if (token === openToken) frameStatus.textContent = `Đã tải ${anchor.download}.`
    } catch (error) {
      if (token === openToken) {
        frameStatus.textContent = `Không thể xuất frame ${frameId}: ${error instanceof Error ? error.message : 'Lỗi không xác định'}`
      }
    } finally {
      if (token === openToken) {
        frameDownloadBusy = false
        updateDownloadVisibility()
      }
    }
  }

  function downloadSubmission() {
    const center = currentFrameId()
    if (center === null || !metadata?.frame_count || !activeVideoId) return
    const submission = buildKisSubmission(activeVideoId, center, metadata.frame_count)
    const blob = new Blob([submission.csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'ans.csv'
    anchor.click()
    setTimeout(() => URL.revokeObjectURL(url), 0)
    frameStatus.textContent = `Đã tải ans.csv gồm ${submission.count} frame (${submission.start}–${submission.end}).`
  }

  function loadFrame() {
    if (!metadata?.fps) return
    const frameId = Number(frameInput.value)
    if (!Number.isInteger(frameId) || frameId < 0 || frameId >= metadata.frame_count) {
      frameStatus.textContent = `Frame ID phải nằm trong khoảng 0–${Math.max(0, metadata.frame_count - 1)}.`
      return
    }
    videoEl.pause()
    frameStatus.textContent = `Đang tải frame ${frameId}…`
    seekAndUnlock(frameId / metadata.fps, openToken, `Đã tải frame ${frameId}.`)
  }

  function close() {
    openToken += 1
    metadata = null
    allowSubmission = false
    allowFrameDownload = false
    frameDownloadBusy = false
    frameDownload.hidden = true
    submissionDownload.hidden = true
    overlay.hidden = true
    videoEl.pause()
    setPlaybackReady(false)
    videoEl.removeAttribute('src')
    videoEl.load()
  }

  function open(candidate, rank, options = {}) {
    const token = ++openToken
    activeVideoId = candidate.videoId
    activeServer = String(options.server ?? '').replace(/\/$/, '')
    allowSubmission = options.allowSubmission === true
    allowFrameDownload = options.allowFrameDownload === true
    frameDownloadBusy = false
    frameDownload.hidden = true
    submissionDownload.hidden = true
    rankEl.textContent = options.rankLabel ?? `Result #${rank}`
    videoIdEl.textContent = candidate.videoId

    factsEl.replaceChildren()
    fact(factsEl, 'Frame id', candidateFrameId(candidate))
    fact(factsEl, 'Scene', candidate.sceneIndex !== undefined ? `#${candidate.sceneIndex}` : undefined)
    fact(
      factsEl,
      'Time range',
      candidate.startTime !== undefined
        ? candidate.endTime !== undefined
          ? `${formatTime(candidate.startTime)} \u2013 ${formatTime(candidate.endTime)}`
          : formatTime(candidate.startTime)
        : undefined,
    )
    fact(factsEl, 'Score', candidate.score !== undefined ? candidate.score.toFixed(4) : undefined)

    contextEl.textContent = candidate.sceneContext || '\u2014'

    ocrBlock.hidden = !candidate.ocrText
    if (candidate.ocrText) ocrEl.textContent = candidate.ocrText

    asrBlock.hidden = !candidate.asrText
    if (candidate.asrText) asrEl.textContent = candidate.asrText

    answerBlock.hidden = !options.answer
    if (options.answer) answerEl.textContent = options.answer

    // Only now — on explicit user intent to view the clip — does the
    // browser request the actual video file.
    const server = activeServer
    const initialFrameId = candidateFrameId(candidate)
    metadata = null
    currentFrameEl.textContent = '—'
    fpsEl.textContent = 'FPS: loading…'
    frameInput.value = String(initialFrameId)
    frameInput.removeAttribute('max')
    frameLoad.disabled = true
    frameStatus.textContent = `Đang tải frame ${initialFrameId}…`

    setPlaybackReady(false)
    videoEl.preload = 'metadata'
    videoEl.src = `${server}/videos/${encodeURIComponent(candidate.videoId)}.mp4`
    videoEl.addEventListener('error', () => {
      if (token !== openToken) return
      setPlaybackReady(false)
      frameLoad.disabled = true
      frameDownload.hidden = true
      submissionDownload.hidden = true
      frameStatus.textContent = 'Không thể tải video từ backend đã chọn.'
    }, { once: true })

    overlay.hidden = false

    const metadataPath = `/api/videos/${encodeURIComponent(candidate.videoId)}/metadata`
    const fetchMetadata = async () => {
      const urls = [...new Set([`${server}${metadataPath}`, metadataPath])]
      let lastError = null
      for (const url of urls) {
        try {
          const response = await fetch(url)
          const body = await response.json().catch(() => null)
          if (!response.ok) throw new Error(body?.detail ?? `HTTP ${response.status}`)
          return body
        } catch (error) {
          lastError = error
        }
      }
      throw lastError ?? new Error('Metadata request failed')
    }
    const metadataRequest = options.metadata ? Promise.resolve(options.metadata) : fetchMetadata()
    metadataRequest
      .then((value) => {
        if (token !== openToken) return
        metadata = value
        fpsEl.textContent = `FPS: ${metadata.fps}`
        frameInput.max = String(Math.max(0, metadata.frame_count - 1))
        updateDownloadVisibility()
        const seek = () => {
          seekAndUnlock(
            initialFrameId / metadata.fps,
            token,
            `Đã tải frame ${initialFrameId}.`,
            () => { frameLoad.disabled = false },
          )
        }
        if (videoEl.readyState >= 1) seek()
        else videoEl.addEventListener('loadedmetadata', seek, { once: true })
      })
      .catch((error) => {
        if (token !== openToken) return
        fpsEl.textContent = 'FPS: unavailable'
        if (options.metadata) {
          setPlaybackReady(false)
          frameLoad.disabled = true
          frameStatus.textContent = `Không thể dùng metadata đã kiểm tra: ${error.message}`
          return
        }
        const seekFallback = () => {
          seekAndUnlock(
            candidate.startTime ?? 0,
            token,
            `Không tải được metadata; đang dùng thời gian scene. ${error.message}`,
          )
        }
        if (videoEl.readyState >= 1) seekFallback()
        else videoEl.addEventListener('loadedmetadata', seekFallback, { once: true })
      })
  }

  videoEl.addEventListener('timeupdate', updateCurrentFrame)
  videoEl.addEventListener('pause', () => {
    updateCurrentFrame()
    updateDownloadVisibility()
  })
  videoEl.addEventListener('play', () => {
    if (!playbackReady) {
      videoEl.pause()
      return
    }
    updateDownloadVisibility()
  })
  videoEl.addEventListener('seeked', updateCurrentFrame)
  frameLoad.addEventListener('click', loadFrame)
  frameDownload.addEventListener('click', downloadFrame)
  submissionDownload.addEventListener('click', downloadSubmission)
  frameInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') loadFrame()
  })

  closeBtn.addEventListener('click', close)
  overlay.addEventListener('click', (event) => {
    if (event.target === overlay) close()
  })
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !overlay.hidden) close()
  })

  return { open, close }
}
