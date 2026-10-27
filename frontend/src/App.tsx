import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api'
import GraphView from './components/GraphView'
import LeftSidebar from './components/LeftSidebar'
import RightSidebar from './components/RightSidebar'
import type {
  Dataset,
  Dim,
  DistancesResponse,
  Metric,
  Method,
  Neighbor,
} from './types'
import { METRIC_LABELS } from './types'

export default function App() {
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [enabledIds, setEnabledIds] = useState<Set<string>>(new Set())
  const [coords, setCoords] = useState<Record<string, number[]> | null>(null)
  const [notes, setNotes] = useState<string[]>([])
  const [dim, setDim] = useState<Dim>(2)
  const [method, setMethod] = useState<Method>('pca')
  const [metric, setMetric] = useState<Metric>('cosine_distance')
  const [showLines, setShowLines] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const [k, setK] = useState(5)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [hoverId, setHoverId] = useState<string | null>(null)
  const [distances, setDistances] = useState<DistancesResponse | null>(null)
  const [neighbors, setNeighbors] = useState<Neighbor[]>([])
  const [search, setSearch] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [hint, setHint] = useState<string | null>(null)
  // the graph re-emits its hint on every rebuild, which would pin it on
  // screen forever — show each distinct hint once, then auto-hide
  const seenHints = useRef<Set<string>>(new Set())
  const pushHint = useCallback((h: string | null) => {
    if (!h) {
      setHint(null)
      return
    }
    if (seenHints.current.has(h)) return
    seenHints.current.add(h)
    setHint(h)
    window.setTimeout(() => setHint((cur) => (cur === h ? null : cur)), 6000)
  }, [])
  const [busy, setBusy] = useState<string | null>(null)
  const [resetKey, setResetKey] = useState(0)
  const [focusKey, setFocusKey] = useState(0)
  const [dotMaxPx, setDotMaxPx] = useState(32)

  const vectors = useMemo(() => dataset?.vectors ?? [], [dataset])

  // ---- dataset loading -------------------------------------------------
  const handleDataset = useCallback((ds: Dataset) => {
    setDataset(ds)
    // start clean: nothing enabled, nothing selected — the user builds the
    // graph up progressively (point → line → triangle → full graph)
    setEnabledIds(new Set())
    setSelectedId(null)
    setNeighbors([])
    setDistances(null)
    setCoords(null)
    setError(null)
    setBusy(null)
  }, [])

  // ---- projection (all vectors, re-run on method/dim/metric change) ----
  useEffect(() => {
    if (!dataset) return
    let cancelled = false
    setBusy('project')
    setCoords(null)
    api
      .project(dataset.vectors, method, dim, metric)
      .then((res) => {
        if (cancelled) return
        setCoords(res.coords)
        setNotes(res.notes)
      })
      .catch((err) => !cancelled && setError(`Projection failed: ${err.message}`))
      .finally(() => !cancelled && setBusy(null))
    return () => {
      cancelled = true
    }
  }, [dataset, method, dim, metric])

  // ---- true distances among enabled vectors ----------------------------
  useEffect(() => {
    if (!dataset || enabledIds.size < 2) {
      setDistances(null)
      return
    }
    let cancelled = false
    api
      .distances(dataset.vectors, [...enabledIds], metric)
      .then((res) => !cancelled && setDistances(res))
      .catch((err) => !cancelled && setError(`Distance calculation failed: ${err.message}`))
    return () => {
      cancelled = true
    }
  }, [dataset, enabledIds, metric])

  // ---- nearest neighbours of the selection -----------------------------
  useEffect(() => {
    if (!dataset || !selectedId) {
      setNeighbors([])
      return
    }
    let cancelled = false
    api
      .neighbors(dataset.vectors, selectedId, k, metric)
      .then((res) => !cancelled && setNeighbors(res.neighbors))
      .catch((err) => !cancelled && setError(`Neighbour lookup failed: ${err.message}`))
    return () => {
      cancelled = true
    }
  }, [dataset, selectedId, k, metric])

  const distanceLookup = useMemo(() => {
    const map = new Map<string, number>()
    if (!distances) return map
    for (let i = 0; i < distances.ids.length; i++) {
      for (let j = i + 1; j < distances.ids.length; j++) {
        const a = distances.ids[i]
        const b = distances.ids[j]
        map.set(a < b ? `${a}|${b}` : `${b}|${a}`, distances.matrix[i][j])
      }
    }
    return map
  }, [distances])

  const neighborIds = useMemo(() => new Set(neighbors.map((n) => n.id)), [neighbors])
  // 1, 2, 3 badges for the three closest neighbours — only when there is more
  // than one pair, per the click-to-inspect numbering behaviour
  const neighborRanks = useMemo(() => {
    const m = new Map<string, number>()
    if (neighbors.length >= 2) neighbors.slice(0, 3).forEach((n, i) => m.set(n.id, i + 1))
    return m
  }, [neighbors])
  const selected = useMemo(
    () => (selectedId ? (vectors.find((v) => v.id === selectedId) ?? null) : null),
    [selectedId, vectors],
  )

  // ---- handlers --------------------------------------------------------
  const toggle = useCallback((id: string) => {
    setEnabledIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  const toggleMany = useCallback((ids: string[], enabled: boolean) => {
    setEnabledIds((prev) => {
      const next = new Set(prev)
      for (const id of ids) {
        if (enabled) next.add(id)
        else next.delete(id)
      }
      return next
    })
  }, [])

  const select = useCallback((id: string | null) => {
    setSelectedId(id)
    if (id) {
      // a selected vector must be visible in the graph
      setEnabledIds((prev) => (prev.has(id) ? prev : new Set(prev).add(id)))
    }
  }, [])

  return (
    <div className="app">
      <LeftSidebar
        dataset={dataset}
        vectors={vectors}
        enabledIds={enabledIds}
        selectedId={selectedId}
        search={search}
        onSearch={setSearch}
        onDataset={handleDataset}
        onToggle={toggle}
        onToggleMany={toggleMany}
        onSelect={select}
        onError={setError}
      />

      <main className="center">
        <div className="toolbar">
          <div className="segmented">
            <button className={dim === 2 ? 'seg active' : 'seg'} onClick={() => setDim(2)}>
              2D
            </button>
            <button className={dim === 3 ? 'seg active' : 'seg'} onClick={() => setDim(3)}>
              3D
            </button>
          </div>

          <label className="tool">
            <span>Project</span>
            <select value={method} onChange={(e) => setMethod(e.target.value as Method)}>
              <option value="pca">PCA</option>
              <option value="mds">MDS</option>
              <option value="umap">UMAP</option>
            </select>
          </label>

          <label className="tool">
            <span>Metric</span>
            <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
              {(Object.keys(METRIC_LABELS) as Metric[]).map((m) => (
                <option key={m} value={m}>
                  {METRIC_LABELS[m]}
                </option>
              ))}
            </select>
          </label>

          <button
            className={showLines ? 'chip active' : 'chip'}
            onClick={() => setShowLines((v) => !v)}
          >
            Lines
          </button>
          <button
            className={showLabels ? 'chip active' : 'chip'}
            onClick={() => setShowLabels((v) => !v)}
          >
            Distance labels
          </button>

          <label className="tool">
            <span>Highlight</span>
            <select value={k} onChange={(e) => setK(Number(e.target.value))}>
              {[3, 5, 10, 15].map((n) => (
                <option key={n} value={n}>
                  {n} neighbours
                </option>
              ))}
            </select>
          </label>

          <label className="tool" title="Maximum on-screen size of dots when zoomed in">
            <span>Stars</span>
            <input
              className="range"
              type="range"
              min={6}
              max={40}
              step={1}
              value={dotMaxPx}
              onChange={(e) => setDotMaxPx(Number(e.target.value))}
            />
          </label>

          <button className="chip" onClick={() => setResetKey((n) => n + 1)}>
            Reset view
          </button>
          <button
            className="chip"
            disabled={!selectedId}
            title="Zoom the camera onto the selected vector and its nearest neighbours"
            onClick={() => setFocusKey((n) => n + 1)}
          >
            Focus group
          </button>

          <div className="toolbar-status">
            {busy === 'project' && <span className="pulse">projecting…</span>}
            {enabledIds.size > 0 && <span>{enabledIds.size} visible</span>}
          </div>
        </div>

        <div className="graph-wrap">
          <GraphView
            vectors={vectors}
            enabledIds={enabledIds}
            coords={coords}
            dim={dim}
            showLines={showLines}
            showLabels={showLabels}
            selectedId={selectedId}
            hoverId={hoverId}
            neighborIds={neighborIds}
            neighborRanks={neighborRanks}
            distanceLookup={distanceLookup}
            onSelect={select}
            onHover={setHoverId}
            resetKey={resetKey}
            focusKey={focusKey}
            dotMaxPx={dotMaxPx}
            onRequestFocus={() => setFocusKey((n) => n + 1)}
            onHint={pushHint}
          />

          {!dataset && (
            <div className="overlay">
              <div className="overlay-card">
                <h2>Vector Visualizer</h2>
                <p>
                  Connect a vector database (Qdrant, Pinecone, Chroma, pgvector), upload embeddings, or load the
                  demo dataset from the left panel.
                </p>
                <p className="tiny-note">
                  Toggle vectors to build the graph: one point → a line with its distance → a triangle → a full
                  similarity graph.
                </p>
              </div>
            </div>
          )}

          {dataset && !coords && !error && (
            <div className="overlay soft">
              <div className="overlay-card">Computing {method.toUpperCase()} projection…</div>
            </div>
          )}

          {dataset && coords && enabledIds.size === 0 && !error && (
            <div className="overlay soft">
              <div className="overlay-card">
                <h2>Enable vectors to build the graph</h2>
                <p>
                  Tick vectors in the left panel — one point → a line with its distance → a triangle → a full
                  similarity graph.
                </p>
                <button
                  className="btn"
                  onClick={() => setEnabledIds(new Set(vectors.map((v) => v.id)))}
                  disabled={!vectors.length}
                >
                  Show all {vectors.length} vectors
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="error-banner" onClick={() => setError(null)}>
              {error} <span className="dismiss">dismiss</span>
            </div>
          )}

          {hint && <div className="graph-hint">{hint}</div>}

          <div className="legend">
            <span>
              <i className="dot sel" /> selected
            </span>
            <span>
              <i className="dot nb" /> nearest neighbours (numbered 1–3 on click)
            </span>
            <span>line label = true {METRIC_LABELS[metric].toLowerCase()} · positions ≈ projection</span>
          </div>
        </div>
      </main>

      <RightSidebar
        dataset={dataset}
        selected={selected}
        hoverId={hoverId}
        enabledIds={enabledIds}
        distances={distances}
        coords={coords}
        neighbors={neighbors}
        metric={metric}
        notes={notes}
        dim={dim}
        onSelect={select}
      />
    </div>
  )
}
