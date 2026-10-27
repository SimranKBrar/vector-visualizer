from typing import Any, Literal

from pydantic import BaseModel, Field


class Vector(BaseModel):
    id: str
    values: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    name: str
    vectors: list[Vector]


class ConnectRequest(BaseModel):
    type: Literal["qdrant", "pinecone", "chroma", "pgvector"]
    config: dict[str, Any] = Field(default_factory=dict)
    collection: str | None = None
    limit: int = 500


class CollectionsRequest(BaseModel):
    type: Literal["qdrant", "pinecone", "chroma", "pgvector"]
    config: dict[str, Any] = Field(default_factory=dict)


class ProjectRequest(BaseModel):
    vectors: list[Vector]
    method: Literal["pca", "mds", "umap"] = "pca"
    dim: Literal[2, 3] = 2
    metric: Literal["cosine_similarity", "cosine_distance", "euclidean"] = "cosine_distance"
    n_neighbors: int = 15


class ProjectResponse(BaseModel):
    coords: dict[str, list[float]]
    method: str
    dim: int
    notes: list[str] = []


class DistancesRequest(BaseModel):
    vectors: list[Vector]
    ids: list[str] | None = None  # restrict to these ids; None = all
    metric: Literal["cosine_similarity", "cosine_distance", "euclidean"] = "cosine_distance"


class DistancesResponse(BaseModel):
    ids: list[str]
    metric: str
    matrix: list[list[float]]  # value of `metric` between ids[i] and ids[j]
    cosine: list[list[float]]  # cosine similarity matrix (always provided)


class NeighborsRequest(BaseModel):
    vectors: list[Vector]
    id: str
    k: int = 5
    metric: Literal["cosine_similarity", "cosine_distance", "euclidean"] = "cosine_distance"


class Neighbor(BaseModel):
    id: str
    distance: float
    metric: str
    cosine_similarity: float


class NeighborsResponse(BaseModel):
    id: str
    metric: str
    neighbors: list[Neighbor]
