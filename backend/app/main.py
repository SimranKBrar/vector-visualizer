"""Vector Visualizer API server."""

from __future__ import annotations

import threading

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .db import ConnectorError, fetch_vectors, list_collections
from .demo import build_demo
from .math_ops import METRICS, compute_distances, compute_neighbors, project
from .models import (
    CollectionsRequest,
    ConnectRequest,
    Dataset,
    DistancesRequest,
    DistancesResponse,
    NeighborsRequest,
    NeighborsResponse,
    ProjectRequest,
    ProjectResponse,
)
from .parsing import parse_file

app = FastAPI(title="Vector Visualizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def _warm_umap() -> None:
    """Import umap in the background so the first UMAP request isn't slow."""

    def run() -> None:
        try:
            import umap  # noqa: F401
        except Exception:  # pragma: no cover - best effort
            pass

    threading.Thread(target=run, daemon=True).start()


@app.on_event("startup")
def on_startup() -> None:
    _warm_umap()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/demo")
def demo() -> Dataset:
    return Dataset(name="Demo dataset (3 clusters)", vectors=build_demo())


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> Dataset:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 50 MB).")
    try:
        vectors = parse_file(file.filename or "upload", content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Dataset(name=file.filename or "upload", vectors=vectors)


@app.post("/api/connect")
def connect(req: ConnectRequest) -> Dataset:
    if not req.collection:
        raise HTTPException(400, "A collection/table/index name is required.")
    try:
        vectors = fetch_vectors(req.type, req.config, req.collection, req.limit)
    except ConnectorError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"{req.type} connection failed: {exc}") from exc
    return Dataset(name=f"{req.type}:{req.collection}", vectors=vectors)


@app.post("/api/collections")
def collections(req: CollectionsRequest) -> dict:
    try:
        return {"collections": list_collections(req.type, req.config)}
    except ConnectorError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"{req.type} connection failed: {exc}") from exc


@app.post("/api/project")
def project_endpoint(req: ProjectRequest) -> ProjectResponse:
    if not req.vectors:
        raise HTTPException(400, "No vectors supplied.")
    try:
        coords, notes = project(req.vectors, req.method, req.dim, req.metric, req.n_neighbors)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Projection failed: {exc}") from exc
    return ProjectResponse(coords=coords, method=req.method, dim=req.dim, notes=notes)


@app.post("/api/distances")
def distances(req: DistancesRequest) -> DistancesResponse:
    if req.metric not in METRICS:
        raise HTTPException(400, f"Unknown metric '{req.metric}'.")
    try:
        return compute_distances(req.vectors, req.ids, req.metric)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/neighbors")
def neighbors(req: NeighborsRequest) -> NeighborsResponse:
    if req.metric not in METRICS:
        raise HTTPException(400, f"Unknown metric '{req.metric}'.")
    try:
        return compute_neighbors(req.vectors, req.id, req.k, req.metric)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
