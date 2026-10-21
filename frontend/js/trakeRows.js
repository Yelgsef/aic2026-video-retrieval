import { parseCandidate } from './parseResult.js?v=20260828-ranking-controls'
import { candidateFrameId, frameThumbUrl } from './resultGrid.js?v=20260828-ranking-controls'

function el(tag, className, children) {
  const node = document.createElement(tag)
  if (className) node.className = className
  for (const child of children ?? []) node.append(child instanceof Node ? child : document.createTextNode(String(child)))
  return node
}

function eventCandidates(rawEvent) {
  const list = rawEvent?.results ?? rawEvent?.candidates ?? rawEvent?.items ?? []
  return Array.isArray(list) ? list.map(parseCandidate).filter(Boolean) : []
}

export function parseTrakeRows(raw) {
  const data = raw?.data ?? raw
  const events = data?.task_plan?.events ?? []
  const groups = Array.isArray(data?.event_results) ? data.event_results.map(eventCandidates) : []
  const videos = new Map()

  groups.forEach((candidates, eventIndex) => {
    candidates.forEach((candidate) => {
      const perEvent = videos.get(candidate.videoId) ?? Array.from({ length: groups.length }, () => [])
      perEvent[eventIndex].push(candidate)
      videos.set(candidate.videoId, perEvent)
    })
  })

  // Independent per-event top hits can point backwards in time. Keep only
  // the strongest sequence whose frame ids increase with the event order.
  function bestOrderedFrames(perEvent) {
    if (perEvent.some((items) => !items.length)) return null
    let paths = perEvent[0].map((candidate) => ({ frames: [candidate], score: candidate.score ?? 0 }))
    for (const candidates of perEvent.slice(1)) {
      const next = []
      for (const candidate of candidates) {
        const prior = paths
          .filter((path) => candidateFrameId(path.frames.at(-1)) < candidateFrameId(candidate))
          .sort((a, b) => b.score - a.score)[0]
        if (prior) next.push({ frames: [...prior.frames, candidate], score: prior.score + (candidate.score ?? 0) })
      }
      paths = next
      if (!paths.length) return null
    }
    return paths.sort((a, b) => b.score - a.score)[0]
  }

  const rows = []
  videos.forEach((perEvent, videoId) => {
    const sequence = bestOrderedFrames(perEvent)
    if (sequence) rows.push({ videoId, ...sequence })
  })

  return {
    events,
    rows: rows.sort((a, b) => b.score - a.score),
  }
}

export function renderTrakeRows(container, raw, server, query, onOpen) {
  const { events, rows } = parseTrakeRows(raw)
  container.replaceChildren()
  container.hidden = false
  container.append(el('p', 'results__meta', [`${rows.length} video(s) match all ${events.length} ordered event(s) for “${query}”.`]))

  if (!rows.length) {
    container.append(el('p', 'trake-empty', ['No video contains candidates for every ordered event.']))
    return
  }

  rows.forEach((row, rowIndex) => {
    const section = el('section', 'trake-row')
    section.append(el('h2', 'trake-row__title', [`#${rowIndex + 1}  ${row.videoId}`]))
    const frames = el('div', 'trake-row__frames')
    row.frames.forEach((candidate, eventIndex) => {
      const button = el('button', 'trake-frame')
      button.type = 'button'
      const img = document.createElement('img')
      img.loading = 'lazy'
      img.alt = `${row.videoId}, event ${eventIndex + 1}, frame ${candidateFrameId(candidate)}`
      img.src = frameThumbUrl(server, candidate)
      button.append(img, el('span', 'trake-frame__event', [`${eventIndex + 1}`]), el('span', 'trake-frame__label', [events[eventIndex] ?? `Event ${eventIndex + 1}`]))
      button.addEventListener('click', () => onOpen(candidate, eventIndex + 1, row.videoId))
      frames.append(button)
    })
    section.append(frames)
    container.append(section)
  })
}

function renderStreamRow(container, rawRow, rank, server, onOpen, rejected = false) {
  const candidates = (rawRow.frames ?? []).map(parseCandidate).filter(Boolean)
  const videoId = rawRow.video_id ?? rawRow.videoId ?? candidates[0]?.videoId ?? ''
  const section = el('section', 'trake-row')
  const title = el('h2', 'trake-row__title', [`${rejected ? 'R' : 'T'}${rank}  ${videoId}`])
  if (rejected) title.append(el('span', 'trake-row__rejected-tag', ['Bị loại']))
  section.append(title)
  if (rejected && rawRow.rejection_reason) section.append(el('p', 'trake-row__reason', [rawRow.rejection_reason]))
  const frames = el('div', 'trake-row__frames')
  candidates.forEach((candidate, eventIndex) => {
    const button = el('button', 'trake-frame')
    button.type = 'button'
    const img = document.createElement('img')
    img.loading = 'lazy'
    img.alt = `${videoId}, event ${eventIndex + 1}, frame ${candidateFrameId(candidate)}`
    img.src = frameThumbUrl(server, candidate)
    button.append(
      img,
      el('span', 'trake-frame__event', [`${eventIndex + 1}`]),
      el('span', 'trake-frame__label', [`Event ${eventIndex + 1} — frame ${candidateFrameId(candidate)}`]),
    )
    button.addEventListener('click', () => onOpen(candidate, eventIndex + 1, videoId))
    frames.append(button)
  })
  section.append(frames)
  container.append(section)
}

export function renderTrakeStreamStart(container, query) {
  container.hidden = false
  container.replaceChildren(el('p', 'results__meta', [`Đang tìm Top 10 video cho “${query}”…`]))
  for (let index = 0; index < 3; index += 1) {
    container.append(el('div', 'trake-row-skeleton'))
  }
}

export function appendTrakeStreamRow(container, row, rank, server, onOpen) {
  container.querySelectorAll('.trake-row-skeleton').forEach((node) => node.remove())
  renderStreamRow(container, row, rank, server, onOpen)
  if (rank < 10) container.append(el('div', 'trake-row-skeleton'))
  const meta = container.querySelector('.results__meta')
  if (meta) meta.textContent = `Đã tìm thấy ${rank}/10 video — đang tiếp tục…`
}

export function appendRejectedTrakeRow(container, row, rank, server, onOpen) {
  let heading = container.querySelector('.trake-rejected-heading')
  if (!heading) {
    heading = el('h2', 'trake-rejected-heading', ['Ứng viên bị loại'])
    container.append(heading)
  }
  renderStreamRow(container, row, rank, server, onOpen, true)
}

export function finishTrakeStream(container, total, queryTransformerUsed = false) {
  container.querySelectorAll('.trake-row-skeleton').forEach((node) => node.remove())
  const meta = container.querySelector('.results__meta')
  if (meta) meta.textContent = `Top ${total} video có chuỗi sự kiện hợp lệ.${queryTransformerUsed ? ' Đã mở rộng query bằng Qwen.' : ''}`
  if (!total) container.append(el('p', 'trake-empty', ['Không tìm thấy video có đủ các sự kiện theo đúng thứ tự.']))
}
