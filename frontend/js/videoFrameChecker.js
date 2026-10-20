const VIDEO_ID_RE = /^[A-Za-z0-9_-]+$/

export class VideoFrameCheckError extends Error {}

export function normalizeVideoId(value) {
  const videoId = String(value ?? '').trim()
  if (!videoId) throw new VideoFrameCheckError('Nhập Video ID.')
  if (!VIDEO_ID_RE.test(videoId)) {
    throw new VideoFrameCheckError('Video ID chỉ được chứa chữ, số, dấu gạch ngang và gạch dưới.')
  }
  return videoId
}

export function parseFrameId(value) {
  const raw = String(value ?? '').trim()
  if (!raw) throw new VideoFrameCheckError('Nhập Frame ID.')
  const frameId = Number(raw)
  if (!Number.isInteger(frameId) || frameId < 0) {
    throw new VideoFrameCheckError('Frame ID phải là số nguyên không âm.')
  }
  return frameId
}

export function parseVideoFrameCoordinate(videoValue, frameValue) {
  const rawVideo = String(videoValue ?? '').trim().replace(/^\uFEFF/, '')
  if (!rawVideo.includes(',')) {
    return { videoId: normalizeVideoId(rawVideo), frameId: parseFrameId(frameValue), combined: false }
  }
  const parts = rawVideo.split(',')
  if (parts.length !== 2) {
    throw new VideoFrameCheckError('Định dạng phải là Video ID,Frame ID, ví dụ L26_V035,5100.')
  }
  return {
    videoId: normalizeVideoId(parts[0]),
    frameId: parseFrameId(parts[1]),
    combined: true,
  }
}

export function validateVideoMetadata(value, frameId) {
  if (!value || typeof value !== 'object') {
    throw new VideoFrameCheckError('Backend trả về metadata video không hợp lệ.')
  }
  const fps = Number(value.fps)
  const frameCount = Number(value.frame_count)
  if (!Number.isFinite(fps) || fps <= 0 || !Number.isInteger(frameCount) || frameCount <= 0) {
    throw new VideoFrameCheckError('Metadata video thiếu FPS hoặc số lượng frame hợp lệ.')
  }
  if (frameId >= frameCount) {
    throw new VideoFrameCheckError(`Frame ID phải nằm trong khoảng 0–${frameCount - 1}.`)
  }
  return { ...value, fps, frame_count: frameCount }
}

export async function fetchVideoMetadata(server, videoId, { fetchImpl = fetch, signal } = {}) {
  const base = String(server ?? '').replace(/\/$/, '')
  if (!base) throw new VideoFrameCheckError('Chọn backend server.')
  let response
  try {
    response = await fetchImpl(`${base}/api/videos/${encodeURIComponent(videoId)}/metadata`, { signal })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new VideoFrameCheckError(error instanceof Error ? error.message : 'Không thể kết nối backend.')
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = body?.detail ?? body?.error
    throw new VideoFrameCheckError(detail || `Backend trả về HTTP ${response.status}.`)
  }
  if (!body) throw new VideoFrameCheckError('Backend trả về metadata video không hợp lệ.')
  return body
}

export function createVideoFrameCheckController({ fetchImpl = fetch, openModal }) {
  let requestToken = 0
  let activeController = null

  return {
    cancel() {
      requestToken += 1
      activeController?.abort()
      activeController = null
    },

    async check({ server, videoId: rawVideoId, frameId: rawFrameId }) {
      const { videoId, frameId, combined } = parseVideoFrameCoordinate(rawVideoId, rawFrameId)
      const token = ++requestToken
      activeController?.abort()
      const controller = new AbortController()
      activeController = controller

      try {
        const rawMetadata = await fetchVideoMetadata(server, videoId, {
          fetchImpl,
          signal: controller.signal,
        })
        if (token !== requestToken) return { ok: false, stale: true }
        const metadata = validateVideoMetadata(rawMetadata, frameId)
        openModal(
          {
            videoId,
            bestFrameId: frameId,
            startTime: frameId / metadata.fps,
          },
          1,
          {
            server: String(server).replace(/\/$/, ''),
            rankLabel: 'Frame check',
            allowSubmission: true,
            allowFrameDownload: true,
            metadata,
          },
        )
        return { ok: true, videoId, frameId, combined, metadata }
      } catch (error) {
        if (token !== requestToken || error?.name === 'AbortError') return { ok: false, stale: true }
        throw error
      } finally {
        if (token === requestToken) activeController = null
      }
    },
  }
}

export function setupVideoFrameChecker({ modal, populateServerSelect }) {
  const panel = document.getElementById('panel-CHECK')
  const form = document.getElementById('video-frame-check-form')
  const videoInput = document.getElementById('check-video-id')
  const frameInput = document.getElementById('check-frame-id')
  const serverSelect = document.getElementById('check-server')
  const submit = document.getElementById('check-submit')
  const status = document.getElementById('check-status')
  const controller = createVideoFrameCheckController({ openModal: modal.open })
  let loading = false

  function refreshServerSelect() {
    populateServerSelect(serverSelect, 'CHECK')
    updateSubmitState()
  }

  function updateSubmitState() {
    submit.disabled = loading
  }

  function showStatus(message, isError = false) {
    status.textContent = message
    status.classList.toggle('video-check__status--error', isError)
    status.hidden = !message
  }

  for (const input of [videoInput, frameInput, serverSelect]) {
    input.addEventListener('input', updateSubmitState)
    input.addEventListener('change', updateSubmitState)
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault()
    if (loading) return
    loading = true
    updateSubmitState()
    submit.textContent = 'Checking…'
    showStatus('Đang kiểm tra video và frame…')
    try {
      const result = await controller.check({
        server: serverSelect.value,
        videoId: videoInput.value,
        frameId: frameInput.value,
      })
      if (!result.stale) {
        videoInput.value = result.videoId
        frameInput.value = String(result.frameId)
        showStatus(`Đã mở ${result.videoId}, frame ${result.frameId}.`)
      }
    } catch (error) {
      showStatus(error instanceof Error ? error.message : 'Không thể kiểm tra video và frame.', true)
    } finally {
      loading = false
      submit.textContent = 'Open frame'
      updateSubmitState()
    }
  })

  populateServerSelect(serverSelect, 'CHECK')
  updateSubmitState()
  return { refreshServerSelect, cancel: () => controller.cancel(), panel }
}
