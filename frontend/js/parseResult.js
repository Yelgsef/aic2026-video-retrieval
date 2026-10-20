// Best-effort, defensive parser for the raw JSON returned by the gateway's
// `/api/query` endpoint. Port of src/lib/parseQueryResult.ts.

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asString(value) {
  return typeof value === 'string' && value.trim() ? value : undefined
}

function asNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function asNumberArray(value) {
  return Array.isArray(value) ? value.filter((item) => typeof item === 'number') : []
}

/**
 * Exported so TRAKE row-building (trakeRows.js) can parse each raw event's
 * `results` array item-by-item without going through `findCandidateList`.
 * @returns {import('./types.js').QueryCandidate | null}
 */
export function parseCandidate(item) {
  if (!isRecord(item)) return null

  const videoId = asString(item.video_id) ?? asString(item.videoId)
  if (!videoId) return null

  return {
    videoId,
    sceneIndex: asNumber(item.scene_index) ?? asNumber(item.sceneIndex),
    startTime: asNumber(item.start_time) ?? asNumber(item.startTime),
    endTime: asNumber(item.end_time) ?? asNumber(item.endTime),
    bestFrameId: asNumber(item.best_frame_id) ?? asNumber(item.bestFrameId) ?? asNumber(item.frame_id),
    representativeFrameIds: asNumberArray(item.representative_frame_ids ?? item.representativeFrameIds),
    score: asNumber(item.fusion_score) ?? asNumber(item.reranker_score) ?? asNumber(item.score) ?? asNumber(item.semantic_score),
    sceneContext: asString(item.scene_context) ?? asString(item.sceneContext),
    ocrText: Array.isArray(item.ocr_text) ? item.ocr_text.filter((v) => typeof v === 'string').join(' ') : asString(item.ocr_text),
    asrText: asString(item.asr_text) ?? asString(item.asrText),
  }
}

function findCandidateList(node, depth = 0) {
  if (depth > 4 || !isRecord(node)) return null

  for (const key of ['results', 'candidates', 'items']) {
    const value = node[key]
    if (Array.isArray(value) && value.length && isRecord(value[0])) {
      return value
    }
  }

  for (const key of ['retrieval', 'result', 'data']) {
    const nested = findCandidateList(node[key], depth + 1)
    if (nested) return nested
  }

  return null
}

function findAnswer(node, depth = 0) {
  if (depth > 4 || !isRecord(node)) return null

  const answerNode = node.answer
  if (isRecord(answerNode)) {
    const text = asString(answerNode.answer) ?? asString(answerNode.text)
    if (text) {
      return {
        text,
        confidence: asNumber(answerNode.confidence),
        videoId: asString(answerNode.video_id) ?? asString(answerNode.videoId),
        frameId: asNumber(answerNode.frame_id) ?? asNumber(answerNode.frameId),
        evidence: asString(answerNode.evidence_vi) ?? asString(answerNode.evidence),
      }
    }
  } else if (typeof answerNode === 'string' && answerNode.trim()) {
    return { text: answerNode }
  }

  for (const key of ['result', 'data']) {
    const nested = findAnswer(node[key], depth + 1)
    if (nested) return nested
  }

  return null
}

/** @returns {{raw: unknown, candidates: import('./types.js').QueryCandidate[], answer: object|null}} */
export function parseQueryResult(raw) {
  const list = findCandidateList(raw) ?? []
  const candidates = list.map(parseCandidate).filter((candidate) => candidate !== null)

  const source = isRecord(raw) ? raw : {}

  return {
    raw,
    candidates,
    answer: findAnswer(raw),
    queryTransformerUsed: source.query_transformer_used === true,
    queryExpansion: isRecord(source.query_expansion) ? source.query_expansion : null,
  }
}
