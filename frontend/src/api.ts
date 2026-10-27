import type {
  Dataset,
  DbType,
  DistancesResponse,
  Metric,
  Method,
  NeighborsResponse,
  ProjectResponse,
  Vector,
} from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* keep status text */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export const api = {
  health: () => request<{ status: string }>('/api/health'),

  demo: () => request<Dataset>('/api/demo'),

  upload: async (file: File): Promise<Dataset> => {
    const form = new FormData()
    form.append('file', file)
    return request<Dataset>('/api/upload', { method: 'POST', body: form })
  },

  collections: (type: DbType, config: Record<string, unknown>) =>
    post<{ collections: string[] }>('/api/collections', { type, config }),

  connect: (type: DbType, config: Record<string, unknown>, collection: string, limit: number) =>
    post<Dataset>('/api/connect', { type, config, collection, limit }),

  project: (vectors: Vector[], method: Method, dim: 2 | 3, metric: Metric) =>
    post<ProjectResponse>('/api/project', { vectors, method, dim, metric }),

  distances: (vectors: Vector[], ids: string[], metric: Metric) =>
    post<DistancesResponse>('/api/distances', { vectors, ids, metric }),

  neighbors: (vectors: Vector[], id: string, k: number, metric: Metric) =>
    post<NeighborsResponse>('/api/neighbors', { vectors, id, k, metric }),
}
