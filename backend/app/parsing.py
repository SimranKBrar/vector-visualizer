"""Parse uploaded JSON / JSONL / CSV files into Vector records."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from .models import Vector

ID_KEYS = ("id", "_id", "vector_id", "vectorId", "pk", "key", "name", "doc_id")
VALUE_KEYS = ("values", "embedding", "vector", "embeddings", "arr", "contents", "floats")


def _is_number_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def _extract_record(record: dict[str, Any], index: int) -> Vector:
    vid: str | None = None
    for key in ID_KEYS:
        if key in record and record[key] is not None:
            vid = str(record[key])
            break

    values: list[float] | None = None
    for key in VALUE_KEYS:
        if _is_number_list(record.get(key)):
            values = [float(v) for v in record[key]]
            break
    if values is None:
        for key, value in record.items():
            if key in ID_KEYS:
                continue
            if _is_number_list(value):
                values = [float(v) for v in value]
                break

    if values is None:
        raise ValueError(
            f"Record {index}: no list of numbers found. "
            f"Expected one of {VALUE_KEYS} or a numeric list field."
        )

    if vid is None:
        vid = f"vector-{index}"

    metadata: dict[str, Any] = {}
    if isinstance(record.get("metadata"), dict):
        metadata.update(record["metadata"])
    for key, value in record.items():
        if key in ID_KEYS or key in VALUE_KEYS or key == "metadata":
            continue
        if key in record and _is_number_list(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        elif isinstance(value, (dict, list)):
            metadata[key] = value

    return Vector(id=vid, values=values, metadata=metadata)


def _records_from_json(data: Any) -> list[dict[str, Any]]:
    """Normalise the many shapes JSON payloads arrive in."""
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return data
        # bare list of numbers = a single vector
        if _is_number_list(data):
            return [{"id": "vector-0", "values": data}]
        raise ValueError("JSON array must contain objects with an embedding each.")

    if isinstance(data, dict):
        # Pinecone fetch style: {"vectors": {"id": {"values": [...], "metadata": {}}}}
        for key in ("vectors", "points", "data", "items", "results", "embeddings", "records"):
            inner = data.get(key)
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                return inner
            if isinstance(inner, dict):
                out = []
                for vid, payload in inner.items():
                    if isinstance(payload, dict):
                        rec = dict(payload)
                        rec.setdefault("id", vid)
                        out.append(rec)
                if out:
                    return out

        # parallel arrays: {"ids": [...], "embeddings": [[...], ...]}
        ids = data.get("ids")
        embeddings = data.get("embeddings") or data.get("vectors")
        if isinstance(ids, list) and isinstance(embeddings, list):
            return [
                {"id": str(vid), "values": vec}
                for vid, vec in zip(ids, embeddings)
                if _is_number_list(vec)
            ]

        if _is_number_list(data.get("values")) or _is_number_list(data.get("embedding")):
            return [data]

    raise ValueError("Unrecognised JSON structure: expected a list of records.")


def _parse_json_or_jsonl(text: str) -> list[Vector]:
    stripped = text.lstrip()
    try:
        data = json.loads(text)
        return [_extract_record(rec, i) for i, rec in enumerate(_records_from_json(data))]
    except json.JSONDecodeError:
        pass

    # JSONL fallback
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_no + 1} is not valid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"Line {line_no + 1} must be a JSON object.")
        records.append(obj)
    if not records:
        raise ValueError("File is empty.")
    return [_extract_record(rec, i) for i, rec in enumerate(records)]


def _looks_numeric(values: list[str]) -> bool:
    checked = [v for v in values if v not in ("", None)]
    if not checked:
        return False
    for v in checked[:20]:
        try:
            float(v)
        except ValueError:
            return False
    return True


def _parse_csv(text: str) -> list[Vector]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row.")

    columns = list(reader.fieldnames)
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has no data rows.")

    id_col = next(
        (c for c in columns if c.lower() in {k.lower() for k in ID_KEYS}),
        columns[0],
    )

    # an embedding column holding a JSON list, e.g. "[0.1, 0.2, ...]"
    list_col = None
    for col in columns:
        if col == id_col:
            continue
        sample = (rows[0].get(col) or "").strip()
        if sample.startswith("["):
            list_col = col
            break

    numeric_cols = [
        col for col in columns if col != id_col and _looks_numeric([row.get(col, "") for row in rows])
    ]

    vectors: list[Vector] = []
    for i, row in enumerate(rows):
        if list_col:
            try:
                values = [float(v) for v in json.loads(row[list_col])]
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Row {i + 1}: invalid embedding in column '{list_col}': {exc}") from exc
        else:
            if not numeric_cols:
                raise ValueError(
                    "No numeric embedding columns found. "
                    "Provide numeric columns or an embedding column containing a JSON list."
                )
            values = [float(row[col] or 0) for col in numeric_cols]

        metadata = {
            col: (row[col] if row[col] not in ("", None) else None)
            for col in columns
            if col != id_col and col != list_col and col not in numeric_cols
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        vectors.append(Vector(id=str(row.get(id_col) or f"vector-{i}"), values=values, metadata=metadata))

    return vectors


def parse_file(filename: str, content: bytes) -> list[Vector]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not decode file as UTF-8: {exc}") from exc

    name = (filename or "").lower()
    if name.endswith(".csv"):
        vectors = _parse_csv(text)
    elif name.endswith((".json", ".jsonl", ".ndjson")):
        vectors = _parse_json_or_jsonl(text)
    else:
        try:
            vectors = _parse_json_or_jsonl(text)
        except ValueError:
            vectors = _parse_csv(text)

    if not vectors:
        raise ValueError("No vectors found in file.")

    seen: set[str] = set()
    for i, vec in enumerate(vectors):
        if vec.id in seen:
            vec.id = f"{vec.id}-{i}"
        seen.add(vec.id)
        if len(vec.values) == 0:
            raise ValueError(f"Vector '{vec.id}' has an empty embedding.")

    return vectors
