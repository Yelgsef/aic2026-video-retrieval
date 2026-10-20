// KIS result grid: 10x10 preview cells (video_id + frame_id + thumbnail).
// Thumbnails are single JPEG frames decoded server-side at
// GET /api/frame/{video_id}/{frame_id} (see backend/routers/frame.py) — the
// backend never streams a whole video just to produce a grid preview.
// Clicking a cell opens the detail modal (see detailModal.js), which is the
// only place that ever requests /videos/{video_id}.mp4.

const GRID_THUMB_WIDTH = 240

// Keep thumbnail decoding/network work bounded to one frame at a time. The
// query response contains IDs and metadata only; this queue resolves each
// `/api/frame/{video_id}/{frame_id}` request before starting the next one.
class FrameThumbnailQueue {
  constructor() {
    this.pending = []
    this.running = false
    this.generation = 0
  }

  reset() {
    this.pending = []
    this.generation += 1
  }

  enqueue(img, url, onLoad, onError) {
    this.pending.push({ img, url, onLoad, onError, generation: this.generation })
    this.drain()
  }

  drain() {
    if (this.running) return
    this.running = true
    const run = async () => {
      try {
        while (this.pending.length) {
          const job = this.pending.shift()
          if (!job || job.generation !== this.generation) continue
          await new Promise((resolve) => {
            let settled = false
            const finish = (callback) => {
              if (settled) return
              settled = true
              try {
                callback()
              } finally {
                resolve()
              }
            }
            fetch(job.url)
              .then((response) => {
                if (!response.ok) throw new Error(`HTTP ${response.status}`)
                return response.blob()
              })
              .then((blob) => {
                const objectUrl = URL.createObjectURL(blob)
                job.img.addEventListener('load', () => {
                  URL.revokeObjectURL(objectUrl)
                  finish(job.onLoad)
                }, { once: true })
                job.img.addEventListener('error', () => {
                  URL.revokeObjectURL(objectUrl)
                  finish(job.onError)
                }, { once: true })
                job.img.src = objectUrl
              })
              .catch(() => finish(job.onError))
          })
        }
      } finally {
        this.running = false
        // A new candidate can be appended while the previous image is
        // finishing. Drain again so it is not stranded after the loop exits.
        if (this.pending.length) this.drain()
      }
    }
    void run()
  }
}

const thumbnailQueues = new WeakMap()

function thumbnailQueueFor(gridEl) {
  let queue = thumbnailQueues.get(gridEl)
  if (!queue) {
    queue = new FrameThumbnailQueue()
    thumbnailQueues.set(gridEl, queue)
  }
  return queue
}

function el(tag, className, children) {
  const node = document.createElement(tag)
  if (className) node.className = className
  for (const child of children ?? []) {
    if (child == null) continue
    node.append(child instanceof Node ? child : document.createTextNode(String(child)))
  }
  return node
}

/** Best-effort frame id for a candidate: prefer best_frame_id, fall back to the first representative frame. */
export function candidateFrameId(candidate) {
  if (candidate.bestFrameId !== undefined) return candidate.bestFrameId
  if (candidate.representativeFrameIds?.length) return candidate.representativeFrameIds[0]
  return 0
}

export function frameThumbUrl(server, candidate, width = GRID_THUMB_WIDTH) {
  const frameId = candidateFrameId(candidate)
  return `${server}/api/frame/${encodeURIComponent(candidate.videoId)}/${frameId}?width=${width}`
}

function renderCell(candidate, rank, { server, onOpen, answer, thumbnailQueue }) {
  const cell = el('button', 'grid-cell')
  cell.type = 'button'
  cell.title = `${candidate.videoId} — frame ${candidateFrameId(candidate)}`

  const img = document.createElement('img')
  // The queue, rather than the browser's eager/lazy scheduler, controls the
  // order in which thumbnails are requested.
  img.loading = 'eager'
  img.decoding = 'async'
  img.alt = `${candidate.videoId} frame ${candidateFrameId(candidate)}`

  const placeholder = el('span', 'grid-cell__placeholder')
  const spinner = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  spinner.setAttribute('viewBox', '0 0 24 24')
  spinner.setAttribute('fill', 'none')
  spinner.setAttribute('stroke', 'currentColor')
  spinner.setAttribute('stroke-width', '1.8')
  const spinnerPath = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  spinnerPath.setAttribute('d', 'M8 5v14l11-7z')
  spinner.append(spinnerPath)
  placeholder.append(spinner)

  thumbnailQueue.enqueue(
    img,
    frameThumbUrl(server, candidate),
    () => placeholder.remove(),
    () => {
      placeholder.replaceChildren(document.createTextNode('N/A'))
    },
  )

  const label = el('span', 'grid-cell__label', [
    el('span', 'grid-cell__label-video', [candidate.videoId]),
    el('span', 'grid-cell__label-frame', [`frame ${candidateFrameId(candidate)}`]),
  ])

  cell.append(img, placeholder, el('span', 'grid-cell__rank', [String(rank)]), label)
  if (candidate.score !== undefined) {
    cell.append(el('span', 'grid-cell__score', [candidate.score.toFixed(3)]))
  }

  cell.addEventListener('click', () => onOpen(candidate, rank))
  if (answer === undefined) return cell

  const item = el('article', 'grid-item')
  item.append(cell)
  const answerEl = el('p', `grid-item__answer${answer ? '' : ' grid-item__answer--empty'}`, [
    answer || 'Chưa có câu trả lời từ LLM.',
  ])
  answerEl.dataset.candidateKey = `${candidate.videoId}:${candidateFrameId(candidate)}`
  item.append(answerEl)
  return item
}

export function updateGridAnswer(gridEl, candidate, answer) {
  const key = `${candidate.videoId}:${candidateFrameId(candidate)}`
  const answerEl = [...gridEl.querySelectorAll('.grid-item__answer')]
    .find((node) => node.dataset.candidateKey === key)
  if (!answerEl) return
  answerEl.textContent = answer
  answerEl.classList.toggle('grid-item__answer--empty', !answer)
}

function createSkeleton() {
  const skeleton = el('div', 'grid-skeleton')
  skeleton.setAttribute('aria-hidden', 'true')
  return skeleton
}

export function renderGridSkeleton(metaEl, gridEl, server, query, count = 10) {
  thumbnailQueueFor(gridEl).reset()
  metaEl.hidden = false
  metaEl.replaceChildren('Searching ', el('strong', null, [server]), ` for "${query}"…`)
  gridEl.replaceChildren()
  gridEl.hidden = false
  for (let index = 0; index < count; index += 1) {
    gridEl.append(createSkeleton())
  }
}

export function appendGridCandidates(gridEl, candidates, startRank, { server, onOpen, answers } = {}) {
  const thumbnailQueue = thumbnailQueueFor(gridEl)
  gridEl.querySelectorAll('.grid-skeleton').forEach((node) => node.remove())
  candidates.forEach((candidate, index) => {
    const answer = answers instanceof Map ? answers.get(`${candidate.videoId}:${candidateFrameId(candidate)}`) ?? '' : undefined
    gridEl.append(renderCell(candidate, startRank + index, {
      server, onOpen, answer, thumbnailQueue,
    }))
  })
  for (let index = 0; index < 5; index += 1) gridEl.append(createSkeleton())
}

export function updateGridMeta(metaEl, server, query, total, mode = '') {
  const suffix = mode ? ` — ${mode}` : ''
  metaEl.replaceChildren('Response from ', el('strong', null, [server]), ` for "${query}" — ${total} result(s)${suffix}.`)
}

/** @param {HTMLElement} metaEl @param {HTMLElement} gridEl */
export function renderGrid(metaEl, gridEl, result, server, query, { onOpen, answers } = {}) {
  thumbnailQueueFor(gridEl).reset()
  metaEl.hidden = false
  metaEl.replaceChildren(
    'Response from ',
    el('strong', null, [server]),
    ` for "${query}" \u2014 ${result.candidates.length} result(s).`,
  )

  gridEl.replaceChildren()
  gridEl.hidden = false

  if (!result.candidates.length) {
    metaEl.replaceChildren('No structured candidates recognized in the response from ', el('strong', null, [server]), ` for "${query}".`)
    gridEl.hidden = true
    const details = el('details', 'results__raw')
    details.append(el('summary', null, ['Raw response']), el('pre', null, [JSON.stringify(result.raw, null, 2)]))
    gridEl.hidden = false
    gridEl.replaceChildren(details)
    return
  }

  result.candidates.forEach((candidate, index) => {
    const answer = answers instanceof Map ? answers.get(`${candidate.videoId}:${candidateFrameId(candidate)}`) ?? '' : undefined
    gridEl.append(renderCell(candidate, index + 1, {
      server, onOpen, answer, thumbnailQueue: thumbnailQueueFor(gridEl),
    }))
  })
}
