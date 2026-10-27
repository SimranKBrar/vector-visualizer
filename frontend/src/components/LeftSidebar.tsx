import { useMemo, useRef, useState } from 'react'
import type { Dataset, DbType, Vector } from '../types'
import { api } from '../api'

interface Props {
  dataset: Dataset | null
  vectors: Vector[]
  enabledIds: Set<string>
  selectedId: string | null
  search: string
  onSearch: (s: string) => void
  onDataset: (ds: Dataset) => void
  onToggle: (id: string) => void
  onToggleMany: (ids: string[], enabled: boolean) => void
  onSelect: (id: string) => void
  onError: (msg: string | null) => void
}

type Tab = 'connect' | 'upload' | 'demo'

interface FieldSpec {
  key: string
  label: string
  placeholder?: string
  secret?: boolean
  required?: boolean
}

const DB_FIELDS: Record<DbType, { config: FieldSpec[]; collection: FieldSpec }> = {
  qdrant: {
    config: [
      { key: 'url', label: 'URL', placeholder: 'http://localhost:6333', required: true },
      { key: 'api_key', label: 'API key (optional)', secret: true },
    ],
    collection: { key: 'collection', label: 'Collection', placeholder: 'my-collection', required: true },
  },
  pinecone: {
    config: [
      { key: 'host', label: 'Index host', placeholder: 'idx-xxx.svc.aped-4627-b74a.pinecone.io', required: true },
      { key: 'api_key', label: 'API key', secret: true, required: true },
      { key: 'namespace', label: 'Namespace (optional)' },
    ],
    collection: { key: 'collection', label: 'Index name', placeholder: 'my-index', required: true },
  },
  chroma: {
    config: [
      { key: 'url', label: 'URL', placeholder: 'http://localhost:8000', required: true },
      { key: 'tenant', label: 'Tenant', placeholder: 'default0' },
      { key: 'database', label: 'Database', placeholder: 'default_database' },
    ],
    collection: { key: 'collection', label: 'Collection', placeholder: 'my-collection', required: true },
  },
  pgvector: {
    config: [
      {
        key: 'dsn',
        label: 'Postgres DSN',
        placeholder: 'postgresql://user:pass@localhost:5432/db',
        required: true,
      },
      { key: 'id_column', label: 'ID column', placeholder: 'id' },
      { key: 'embedding_column', label: 'Embedding column', placeholder: 'embedding' },
    ],
    collection: { key: 'collection', label: 'Table', placeholder: 'public.documents', required: true },
  },
}

export default function LeftSidebar(props: Props) {
  const [tab, setTab] = useState<Tab>('demo')
  const [dbType, setDbType] = useState<DbType>('qdrant')
  const [values, setValues] = useState<Record<string, string>>({})
  const [limit, setLimit] = useState(500)
  const [busy, setBusy] = useState<string | null>(null)
  const [collections, setCollections] = useState<string[]>([])
  const [status, setStatus] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const specs = DB_FIELDS[dbType]
  const filtered = useMemo(() => {
    const q = props.search.trim().toLowerCase()
    if (!q) return props.vectors
    return props.vectors.filter((v) => {
      if (v.id.toLowerCase().includes(q)) return true
      return JSON.stringify(v.metadata ?? {}).toLowerCase().includes(q)
    })
  }, [props.vectors, props.search])

  const shown = filtered.slice(0, 400)

  const configFor = (type: DbType): Record<string, unknown> => {
    const out: Record<string, unknown> = {}
    for (const f of DB_FIELDS[type].config) {
      const v = (values[`${type}.${f.key}`] ?? '').trim()
      if (v) out[f.key] = v
      else if (f.required) throw new Error(`Missing: ${f.label}`)
    }
    return out
  }

  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(label)
    props.onError(null)
    try {
      await fn()
    } catch (err) {
      props.onError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  const handleConnect = () =>
    run('connect', async () => {
      const config = configFor(dbType)
      const collection = (values[`${dbType}.${specs.collection.key}`] ?? '').trim()
      if (!collection) throw new Error(`Missing: ${specs.collection.label}`)
      const ds = await api.connect(dbType, config, collection, limit)
      setCollections([])
      props.onDataset(ds)
      setStatus(`Loaded ${ds.vectors.length} vectors from ${ds.name}`)
    })

  const handleList = () =>
    run('list', async () => {
      const config = configFor(dbType)
      const { collections: cols } = await api.collections(dbType, config)
      setCollections(cols)
      setStatus(cols.length ? `Found ${cols.length} collections` : 'No collections found')
    })

  const handleFile = (file: File) =>
    run('upload', async () => {
      const ds = await api.upload(file)
      props.onDataset(ds)
      setStatus(`Loaded ${ds.vectors.length} vectors from ${file.name}`)
    })

  const handleDemo = () =>
    run('demo', async () => {
      const ds = await api.demo()
      props.onDataset(ds)
      setStatus(`Loaded ${ds.vectors.length} demo vectors`)
    })

  const visibleIds = shown.map((v) => v.id)
  const allShownOn = visibleIds.every((id) => props.enabledIds.has(id))

  return (
    <aside className="sidebar left">
      <div className="brand">
        <span className="brand-dot" />
        Vector Visualizer
      </div>

      <div className="tabs">
        {(['connect', 'upload', 'demo'] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? 'tab active' : 'tab'} onClick={() => setTab(t)}>
            {t === 'connect' ? 'Connect' : t === 'upload' ? 'Upload' : 'Demo'}
          </button>
        ))}
      </div>

      {tab === 'connect' && (
        <div className="panel">
          <label className="field">
            <span>Database</span>
            <select
              value={dbType}
              onChange={(e) => {
                setDbType(e.target.value as DbType)
                setCollections([])
              }}
            >
              <option value="qdrant">Qdrant</option>
              <option value="pinecone">Pinecone</option>
              <option value="chroma">Chroma</option>
              <option value="pgvector">pgvector (Postgres)</option>
            </select>
          </label>

          {specs.config.map((f) => (
            <label className="field" key={f.key}>
              <span>
                {f.label}
                {f.required ? ' *' : ''}
              </span>
              <input
                type={f.secret ? 'password' : 'text'}
                placeholder={f.placeholder}
                value={values[`${dbType}.${f.key}`] ?? ''}
                onChange={(e) => setValues((v) => ({ ...v, [`${dbType}.${f.key}`]: e.target.value }))}
              />
            </label>
          ))}

          <label className="field">
            <span>
              {specs.collection.label} *
              {collections.length > 0 && (
                <select
                  className="inline-select"
                  value=""
                  onChange={(e) =>
                    setValues((v) => ({ ...v, [`${dbType}.${specs.collection.key}`]: e.target.value }))
                  }
                >
                  <option value="">pick from list…</option>
                  {collections.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              )}
            </span>
            <input
              placeholder={specs.collection.placeholder}
              value={values[`${dbType}.${specs.collection.key}`] ?? ''}
              onChange={(e) => setValues((v) => ({ ...v, [`${dbType}.${specs.collection.key}`]: e.target.value }))}
            />
          </label>

          <label className="field">
            <span>Max vectors</span>
            <input
              type="number"
              min={1}
              max={20000}
              value={limit}
              onChange={(e) => setLimit(Math.max(1, Number(e.target.value) || 1))}
            />
          </label>

          <div className="row">
            <button className="btn" onClick={handleConnect} disabled={busy !== null}>
              {busy === 'connect' ? 'Connecting…' : 'Connect'}
            </button>
            <button className="btn ghost" onClick={handleList} disabled={busy !== null}>
              {busy === 'list' ? 'Listing…' : 'List collections'}
            </button>
          </div>
        </div>
      )}

      {tab === 'upload' && (
        <div className="panel">
          <p className="hint">
            Upload <code>.json</code>, <code>.jsonl</code> or <code>.csv</code>. Each record needs an id and a list
            of numbers (<code>embedding</code>, <code>values</code>, …); extra fields become metadata.
          </p>
          <input
            ref={fileRef}
            type="file"
            accept=".json,.jsonl,.ndjson,.csv"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) handleFile(f)
              e.target.value = ''
            }}
          />
          <div className="row">
            <button className="btn" onClick={handleDemo} disabled={busy !== null}>
              {busy === 'demo' ? 'Loading…' : 'Load demo dataset'}
            </button>
          </div>
        </div>
      )}

      {tab === 'demo' && (
        <div className="panel">
          <p className="hint">
            No database handy? Load a generated dataset of 36 vectors (3 semantic clusters, 32 dimensions) to try
            the full workflow.
          </p>
          <button className="btn" onClick={handleDemo} disabled={busy !== null}>
            {busy === 'demo' ? 'Loading…' : 'Load demo dataset'}
          </button>
        </div>
      )}

      {status && <div className="status">{status}</div>}

      <div className="vectors-section">
        <div className="vectors-head">
          <span>
            Vectors <em>{props.vectors.length ? `${filtered.length}/${props.vectors.length}` : ''}</em>
          </span>
          <div className="mini-buttons">
            <button
              className="mini"
              onClick={() => props.onToggleMany(visibleIds, !allShownOn)}
              disabled={!shown.length}
            >
              {allShownOn ? 'Hide shown' : 'Show shown'}
            </button>
            <button className="mini" onClick={() => props.onToggleMany(props.vectors.map((v) => v.id), true)}>
              All
            </button>
            <button className="mini" onClick={() => props.onToggleMany(props.vectors.map((v) => v.id), false)}>
              None
            </button>
          </div>
        </div>

        <input
          className="search"
          placeholder="Search id or metadata…"
          value={props.search}
          onChange={(e) => props.onSearch(e.target.value)}
        />

        <ul className="vector-list">
          {shown.map((v) => {
            const on = props.enabledIds.has(v.id)
            const selected = props.selectedId === v.id
            const named = (v.metadata?.label ?? v.metadata?.title) as string | undefined
            return (
              <li key={v.id} className={selected ? 'vector-item selected' : 'vector-item'}>
                <input
                  type="checkbox"
                  checked={on}
                  onChange={() => props.onToggle(v.id)}
                  onClick={(e) => e.stopPropagation()}
                />
                <button className="vector-btn" onClick={() => props.onSelect(v.id)}>
                  <span className="vector-id">{v.id}</span>
                  {named && <span className="vector-meta">{named}</span>}
                </button>
              </li>
            )
          })}
          {!shown.length && <li className="empty">No vectors match.</li>}
          {filtered.length > shown.length && (
            <li className="empty">…{filtered.length - shown.length} more (narrow the search)</li>
          )}
        </ul>
      </div>
    </aside>
  )
}
