export interface Vector {
  id: string
  values: number[]
  metadata: Record<string, unknown>
}

export interface Dataset {
  name: string
  vectors: Vector[]
}

export type Metric = 'cosine_similarity' | 'cosine_distance' | 'euclidean'
export type Dim = 2 | 3
export type Method = 'pca' | 'mds' | 'umap'
export type DbType = 'qdrant' | 'pinecone' | 'chroma' | 'pgvector'

export interface ProjectResponse {
  coords: Record<string, number[]>
  method: string
  dim: number
  notes: string[]
}

export interface DistancesResponse {
  ids: string[]
  metric: Metric
  matrix: number[][]
  cosine: number[][]
}

export interface Neighbor {
  id: string
  distance: number
  metric: Metric
  cosine_similarity: number
}

export interface NeighborsResponse {
  id: string
  metric: Metric
  neighbors: Neighbor[]
}

export const METRIC_LABELS: Record<Metric, string> = {
  cosine_similarity: 'Cosine similarity',
  cosine_distance: 'Cosine distance',
  euclidean: 'Euclidean distance',
}
