import { useMemo } from 'react'
import type { Dataset, DistancesResponse, Metric, Neighbor, Vector } from '../types'
import { METRIC_LABELS } from '../types'

interface Props {
  dataset: Dataset | null
  selected: Vector | null
  hoverId: string | null
  enabledIds: Set<string>
  distances: DistancesResponse | null
  coords: Record<string, number[]> | null
  neighbors: Neighbor[]
  metric: Metric
  notes: string[]
  dim: 2 | 3
  onSelect: (id: string) => void
}

function fmt(x: number): string {
  const ax = Math.abs(x)
  if (ax >= 1000) return x.toFixed(0)
  if (ax >= 1) return x.toFixed(3)
  return x.toFixed(4)
}

function visualDistance(
  a: string,
  b: string,
  coords: Record<string, number[]> | null,
): number | null {
  const ca = coords?.[a]
  const cb = coords?.[b]
  if (!ca || !cb) return null
  const n = Math.min(ca.length, cb.length)
  let sum = 0
  for (let i = 0; i < n; i++) sum += (ca[i] - cb[i]) ** 2
  return Math.sqrt(sum)
}

export default function RightSidebar(props: Props) {
  const { selected, distances, coords, metric, neighbors } = props

  const rows = useMemo(() => {
    if (!selected || !distances) return []
    const idx = distances.ids.indexOf(selected.id)
    if (idx < 0) return []
    const descending = metric === 'cosine_similarity'
    return distances.ids
      .map((id, i) => ({
        id,
        actual: distances.matrix[idx][i],
        cosine: distances.cosine[idx][i],
        visual: visualDistance(selected.id, id, coords),
      }))
      .filter((r) => r.id !== selected.id && props.enabledIds.has(r.id))
      .sort((a, b) => (descending ? b.actual - a.actual : a.actual - b.actual))
      .slice(0, 40)
  }, [selected, distances, coords, metric, props.enabledIds])

  return (
    <aside className="sidebar right">
      <div className="panel-title">Inspector</div>

      {!selected && (
        <div className="hint">
          Click a point in the graph — or a row in the left list — to inspect its embedding, metadata and distances
          to every visible vector.
        </div>
      )}

      {selected && (
        <>
          <div className="selected-head">
            <div className="selected-id">{selected.id}</div>
            <div className="selected-dims">
              {selected.values.length}-D embedding · {props.enabledIds.has(selected.id) ? 'visible' : 'hidden'}
            </div>
          </div>

          <div className="section">
            <div className="section-title">Metadata</div>
            {Object.keys(selected.metadata ?? {}).length === 0 ? (
              <div className="hint">No metadata.</div>
            ) : (
              <table className="kv">
                <tbody>
                  {Object.entries(selected.metadata).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{typeof v === 'object' ? JSON.stringify(v) : String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="section">
            <div className="section-title">Embedding values (first 10)</div>
            <div className="values">
              {selected.values.slice(0, 10).map((v, i) => (
                <span key={i}>{fmt(v)}</span>
              ))}
              {selected.values.length > 10 && <span>… +{selected.values.length - 10} more</span>}
            </div>
          </div>

          <div className="section">
            <div className="section-title">
              Distances to visible vectors <em>({METRIC_LABELS[metric]})</em>
            </div>
            <p className="tiny-note">
              <b>Actual</b> = computed on the original {selected.values.length}-D embeddings.{' '}
              <b>Visual</b> = straight-line distance in the {props.dim}D projection — approximate.
            </p>
            {rows.length === 0 ? (
              <div className="hint">
                {distances ? 'Enable other vectors to compare distances.' : 'Computing distances…'}
              </div>
            ) : (
              <table className="dist-table">
                <thead>
                  <tr>
                    <th>Vector</th>
                    <th>Actual</th>
                    <th>Visual</th>
                    <th>cos sim</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.id} onClick={() => props.onSelect(r.id)}>
                      <td className="truncate" title={r.id}>
                        {r.id}
                      </td>
                      <td>{fmt(r.actual)}</td>
                      <td className="approx">{r.visual === null ? '—' : fmt(r.visual)}</td>
                      <td>{fmt(r.cosine)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <div className="section">
            <div className="section-title">Nearest neighbours (highlighted)</div>
            {neighbors.length === 0 ? (
              <div className="hint">Computing…</div>
            ) : (
              <ul className="neighbor-list">
                {neighbors.map((nb, i) => (
                  <li key={nb.id}>
                    <button className="neighbor-btn" onClick={() => props.onSelect(nb.id)}>
                      <span className="rank">{i + 1}</span>
                      <span className="truncate" title={nb.id}>
                        {nb.id}
                      </span>
                      <span className="nb-dist">{fmt(nb.distance)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </>
      )}

      <div className="section notes">
        <div className="section-title">Projection</div>
        <ul>
          {props.notes.length > 0 ? props.notes.map((n, i) => <li key={i}>{n}</li>) : <li className="hint">—</li>}
        </ul>
        {props.dataset && (
          <div className="hint">Source: {props.dataset.name}</div>
        )}
      </div>
    </aside>
  )
}
