"""Vector database connectors implemented over their HTTP APIs (plus psycopg for pgvector)."""

from __future__ import annotations

from typing import Any

import httpx

from .models import Vector


class ConnectorError(RuntimeError):
    pass


def _require(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if value in (None, ""):
        raise ConnectorError(f"Missing required config field '{key}'.")
    return value


def _clean_url(url: str) -> str:
    return str(url).rstrip("/")


# ---------------------------------------------------------------- qdrant --

def qdrant_collections(config: dict[str, Any]) -> list[str]:
    url = _clean_url(_require(config, "url"))
    headers = {"api-key": config["api_key"]} if config.get("api_key") else {}
    resp = httpx.get(f"{url}/collections", headers=headers, timeout=30)
    resp.raise_for_status()
    return [c["name"] for c in resp.json().get("collections", [])]


def qdrant_fetch(config: dict[str, Any], collection: str, limit: int) -> list[Vector]:
    url = _clean_url(_require(config, "url"))
    headers = {"api-key": config.get("api_key", "")}
    resp = httpx.post(
        f"{url}/collections/{collection}/points/scroll",
        headers=headers,
        json={"limit": limit, "with_vector": True, "with_payload": True},
        timeout=60,
    )
    resp.raise_for_status()
    points = resp.json().get("points", [])
    vectors: list[Vector] = []
    for point in points:
        vec = point.get("vector")
        if isinstance(vec, dict):  # named vectors -> take the first
            vec = next((v for v in vec.values() if isinstance(v, list)), None)
        if not isinstance(vec, list):
            continue
        payload = point.get("payload") or {}
        metadata = {k: v for k, v in payload.items() if k != "$values"}
        vectors.append(Vector(id=str(point.get("id")), values=[float(x) for x in vec], metadata=metadata))
    if not vectors:
        raise ConnectorError(f"No vectors returned from Qdrant collection '{collection}'.")
    return vectors


# --------------------------------------------------------------- pinecone --

def pinecone_collections(config: dict[str, Any]) -> list[str]:
    api_key = _require(config, "api_key")
    resp = httpx.get(
        "https://api.pinecone.io/indexes",
        headers={"Api-Key": api_key, "X-Pinecone-API-Version": "2025-04"},
        timeout=30,
    )
    resp.raise_for_status()
    return [idx.get("name", "") for idx in resp.json().get("indexes", []) if idx.get("name")]


def pinecone_fetch(config: dict[str, Any], collection: str, limit: int) -> list[Vector]:
    host = _clean_url(_require(config, "host"))
    api_key = _require(config, "api_key")
    namespace = config.get("namespace") or ""
    headers = {"Api-Key": api_key, "Content-Type": "application/json"}

    ids: list[str] = []
    token: str | None = None
    while len(ids) < limit:
        params: dict[str, Any] = {"limit": min(100, limit - len(ids))}
        if namespace:
            params["namespace"] = namespace
        if token:
            params["paginationToken"] = token
        resp = httpx.get(f"{host}/vectors/list", headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        for item in body.get("vectors", []):
            if item.get("id") is not None:
                ids.append(str(item["id"]))
        token = (body.get("pagination") or {}).get("next")
        if not token or not body.get("vectors"):
            break
    if not ids:
        raise ConnectorError("Pinecone index returned no vector IDs (check namespace).")

    vectors: list[Vector] = []
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        payload: dict[str, Any] = {"ids": batch}
        if namespace:
            payload["namespace"] = namespace
        resp = httpx.post(f"{host}/vectors/fetch", headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        fetched = resp.json().get("vectors") or {}
        if isinstance(fetched, dict):
            for vid, item in fetched.items():
                values = item.get("values")
                if not values:
                    continue
                vectors.append(
                    Vector(
                        id=str(vid),
                        values=[float(x) for x in values],
                        metadata=item.get("metadata") or {},
                    )
                )
    if not vectors:
        raise ConnectorError("Pinecone fetch returned no vectors with values.")
    return vectors


# ----------------------------------------------------------------- chroma --

def _chroma_get(url: str, collection: str, config: dict[str, Any], limit: int) -> dict[str, Any]:
    tenant = config.get("tenant") or "default0"
    database = config.get("database") or "default_database"
    headers = {}
    if config.get("api_key"):
        headers["X-Chroma-Token"] = config["api_key"]

    # try v2 API first, then legacy v1
    base_v2 = f"{_clean_url(url)}/api/v2/tenants/{tenant}/databases/{database}/collections"
    try:
        coll_resp = httpx.get(f"{base_v2}/{collection}", headers=headers, timeout=30)
        coll_resp.raise_for_status()
        coll_id = coll_resp.json().get("id") or collection
        resp = httpx.post(
            f"{base_v2}/{coll_id}/get",
            headers=headers,
            json={"include": ["metadatas", "documents", "embeddings"], "limit": limit},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError:
        base_v1 = f"{_clean_url(url)}/api/v1/collections/{collection}"
        resp = httpx.post(
            f"{base_v1}/get",
            headers=headers,
            json={"include": ["metadatas", "documents", "embeddings"], "limit": limit},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()


def chroma_collections(config: dict[str, Any]) -> list[str]:
    url = _clean_url(_require(config, "url"))
    tenant = config.get("tenant") or "default0"
    database = config.get("database") or "default_database"
    resp = httpx.get(f"{url}/api/v2/tenants/{tenant}/databases/{database}/collections", timeout=30)
    if resp.status_code >= 400:  # legacy server
        resp = httpx.get(f"{url}/api/v1/collections", timeout=30)
    resp.raise_for_status()
    items = resp.json()
    if isinstance(items, dict):
        items = items.get("collections", [])
    return [c.get("name", "") for c in items if c.get("name")]


def chroma_fetch(config: dict[str, Any], collection: str, limit: int) -> list[Vector]:
    url = _clean_url(_require(config, "url"))
    body = _chroma_get(url, collection, config, limit)
    ids = body.get("ids") or []
    embeddings = body.get("embeddings")
    metadatas = body.get("metadatas") or []
    documents = body.get("documents") or []

    if embeddings is None:
        raise ConnectorError("Chroma collection returned no embeddings.")
    # chroma may return {"_type": "dense", "data": [...]} or base64 in newer formats
    if isinstance(embeddings, dict):
        embeddings = embeddings.get("data") or embeddings.get("embeddings")
    if embeddings and isinstance(embeddings[0], (bytes, str)):
        raise ConnectorError("Chroma embeddings returned in binary/base64 format are not supported yet.")

    vectors: list[Vector] = []
    for i, vid in enumerate(ids):
        values = embeddings[i] if i < len(embeddings) else None
        if values is None:
            continue
        meta = dict(metadatas[i]) if i < len(metadatas) and metadatas[i] else {}
        if i < len(documents) and documents[i]:
            meta.setdefault("document", documents[i])
        vectors.append(Vector(id=str(vid), values=[float(x) for x in values], metadata=meta))
    if not vectors:
        raise ConnectorError(f"No vectors returned from Chroma collection '{collection}'.")
    return vectors


# --------------------------------------------------------------- pgvector --

def pgvector_collections(config: dict[str, Any]) -> list[str]:
    import psycopg

    dsn = _require(config, "dsn")
    with psycopg.connect(dsn, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT table_schema || '.' || table_name
                FROM information_schema.columns
                WHERE udt_name = 'vector'
                ORDER BY 1
                """
            )
            rows = [r[0] for r in cur.fetchall()]
    if not rows:
        with psycopg.connect(dsn, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_schema || '.' || table_name FROM information_schema.tables "
                    "WHERE table_schema NOT IN ('pg_catalog','information_schema') ORDER BY 1"
                )
                rows = [r[0] for r in cur.fetchall()]
    return rows


def pgvector_fetch(config: dict[str, Any], collection: str, limit: int) -> list[Vector]:
    import psycopg
    from psycopg import sql

    dsn = _require(config, "dsn")
    table = collection or _require(config, "table")
    id_col = config.get("id_column") or "id"
    emb_col = config.get("embedding_column") or "embedding"

    query = config.get("query")
    with psycopg.connect(dsn, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            if query:
                cur.execute(query)
            else:
                cur.execute(
                    sql.SQL("SELECT * FROM {} LIMIT {}").format(
                        sql.Identifier(table), sql.Literal(int(limit))
                    )
                )
            columns = [d.name for d in cur.description]
            if emb_col not in columns:
                candidates = [c for c in columns if c.lower() in ("embedding", "vector", "embeddings", "values")]
                if not candidates:
                    raise ConnectorError(
                        f"Embedding column '{emb_col}' not found. Columns: {columns}. "
                        "Set 'embedding_column' in the connection config."
                    )
                emb_col = candidates[0]
            if id_col not in columns:
                id_col = columns[0]

            vectors: list[Vector] = []
            for row in cur.fetchall():
                record = dict(zip(columns, row))
                raw = record.get(emb_col)
                if raw is None:
                    continue
                if isinstance(raw, str):
                    raw = raw.strip().strip("[]")
                    values = [float(x) for x in raw.split(",") if x.strip()]
                else:
                    values = [float(x) for x in raw]
                metadata: dict[str, Any] = {}
                for key, value in record.items():
                    if key in (id_col, emb_col) or value is None:
                        continue
                    if hasattr(value, "isoformat"):
                        value = value.isoformat()
                    elif not isinstance(value, (str, int, float, bool, dict, list)):
                        value = str(value)
                    metadata[key] = value
                vectors.append(Vector(id=str(record.get(id_col)), values=values, metadata=metadata))
    if not vectors:
        raise ConnectorError(f"No vectors returned from table '{table}'.")
    return vectors


# ------------------------------------------------------------------- api --

def list_collections(type_: str, config: dict[str, Any]) -> list[str]:
    try:
        if type_ == "qdrant":
            return qdrant_collections(config)
        if type_ == "pinecone":
            return pinecone_collections(config)
        if type_ == "chroma":
            return chroma_collections(config)
        if type_ == "pgvector":
            return pgvector_collections(config)
    except httpx.HTTPStatusError as exc:
        raise ConnectorError(
            f"{type_} responded {exc.response.status_code}: {exc.response.text[:300]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ConnectorError(f"Could not reach {type_}: {exc}") from exc
    raise ConnectorError(f"Unknown database type '{type_}'.")


def fetch_vectors(type_: str, config: dict[str, Any], collection: str, limit: int) -> list[Vector]:
    try:
        if type_ == "qdrant":
            return qdrant_fetch(config, collection, limit)
        if type_ == "pinecone":
            return pinecone_fetch(config, collection, limit)
        if type_ == "chroma":
            return chroma_fetch(config, collection, limit)
        if type_ == "pgvector":
            return pgvector_fetch(config, collection, limit)
    except httpx.HTTPStatusError as exc:
        raise ConnectorError(
            f"{type_} responded {exc.response.status_code}: {exc.response.text[:300]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise ConnectorError(f"Could not reach {type_}: {exc}") from exc
    raise ConnectorError(f"Unknown database type '{type_}'.")


__all__ = ["ConnectorError", "fetch_vectors", "list_collections"]
