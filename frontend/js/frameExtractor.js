// Extracts a still frame from a video file served at `/videos/{videoId}.mp4`
// (served directly by the FastAPI backend from `data/videos/` with HTTP
// Range support — see `backend/routers/videos.py`) by seeking a hidden
// <video> element and drawing the current frame onto a <canvas>, then
// exporting it as a base64 JPEG data URL. Port of
// src/composables/useFrameExtractor.ts.

const FRAME_TIMEOUT_MS = 15_000

function videoUrlFor(videoId) {
  return `/videos/${videoId}.mp4`
}

async function loadVideo(src) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'auto'
    video.muted = true
    video.playsInline = true
    video.crossOrigin = 'anonymous'
    video.src = src

    const timeout = setTimeout(() => reject(new Error(`Timed out loading video: ${src}`)), FRAME_TIMEOUT_MS)

    video.addEventListener(
      'loadedmetadata',
      () => {
        clearTimeout(timeout)
        resolve(video)
      },
      { once: true },
    )
    video.addEventListener(
      'error',
      () => {
        clearTimeout(timeout)
        reject(new Error(`Failed to load video: ${src}`))
      },
      { once: true },
    )
  })
}

async function seekTo(video, timeSeconds) {
  return new Promise((resolve, reject) => {
    const target = Math.min(Math.max(timeSeconds, 0), Math.max(video.duration - 0.05, 0))
    const timeout = setTimeout(() => reject(new Error('Timed out seeking video frame.')), FRAME_TIMEOUT_MS)

    video.addEventListener(
      'seeked',
      () => {
        clearTimeout(timeout)
        resolve()
      },
      { once: true },
    )
    video.currentTime = target
  })
}

/**
 * Extracts a single frame at `timeSeconds` from `/videos/{videoId}.mp4`.
 * @returns {Promise<{videoId: string, timeSeconds: number, dataUrl: string}>}
 */
export async function extractFrame(videoId, timeSeconds) {
  const video = await loadVideo(videoUrlFor(videoId))
  try {
    await seekTo(video, timeSeconds)

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas 2D context is not available.')

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85)

    return { videoId, timeSeconds, dataUrl }
  } finally {
    video.src = ''
    video.load()
  }
}

/** Assumes fps=30 when only a frame index is available. */
export const DEFAULT_FPS = 30

export function frameIdToSeconds(frameId, fps = DEFAULT_FPS) {
  return frameId / fps
}
