"""Distance metrics and dimensionality reduction."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from .models import DistancesResponse, Neighbor, NeighborsResponse, Vector

METRICS = ("cosine_similarity", "cosine_distance", "euclidean")


def to_matrix(vectors: list[Vector]) -> np.ndarray:
    if not vectors:
        raise ValueError("No vectors supplied.")
    dims = {len(v.values) for v in vectors}
    if len(dims) > 1:
        raise ValueError(f"Inconsistent embedding dimensions found: {sorted(dims)}.")
    return np.asarray([v.values for v in vectors], dtype=np.float64)


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    sim = sklearn_cosine_similarity(X)
    return np.clip(sim, -1.0, 1.0)


def distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine_similarity":
        return cosine_sim_matrix(X)
    if metric == "cosine_distance":
        return 1.0 - cosine_sim_matrix(X)
    if metric == "euclidean":
        return pairwise_distances(X, metric="euclidean")
    raise ValueError(f"Unknown metric '{metric}'. Choose from {METRICS}.")


def compute_distances(vectors: list[Vector], ids: list[str] | None, metric: str) -> DistancesResponse:
    by_id = {v.id: v for v in vectors}
    selected = [by_id[i] for i in ids] if ids else list(vectors)
    if not selected:
        raise ValueError("No vectors selected.")
    X = to_matrix(selected)
    return DistancesResponse(
        ids=[v.id for v in selected],
        metric=metric,
        matrix=np.round(distance_matrix(X, metric), 6).tolist(),
        cosine=np.round(cosine_sim_matrix(X), 6).tolist(),
    )


def compute_neighbors(
    vectors: list[Vector], target_id: str, k: int, metric: str
) -> NeighborsResponse:
    by_id = {v.id: v for v in vectors}
    if target_id not in by_id:
        raise ValueError(f"Unknown vector id '{target_id}'.")
    X = to_matrix(vectors)
    target = by_id[target_id]
    x = np.asarray(target.values, dtype=np.float64).reshape(1, -1)

    sims = np.clip(sklearn_cosine_similarity(x, X)[0], -1.0, 1.0)
    if metric == "cosine_similarity":
        d = 1.0 - sims
        higher_is_better = True
    elif metric == "cosine_distance":
        d = 1.0 - sims
        higher_is_better = False
    else:
        d = np.sqrt(((X - x) ** 2).sum(axis=1))
        higher_is_better = False

    order = np.argsort(-d if higher_is_better else d)
    order = [i for i in order if vectors[i].id != target_id][: max(1, k)]

    neighbors = [
        Neighbor(
            id=vectors[i].id,
            distance=float(round(d[i], 6)),
            metric=metric,
            cosine_similarity=float(round(sims[i], 6)),
        )
        for i in order
    ]
    return NeighborsResponse(id=target_id, metric=metric, neighbors=neighbors)


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------

def classical_mds(D: np.ndarray, dim: int) -> np.ndarray:
    """Classical (Torgerson) multidimensional scaling on a distance matrix."""
    n = D.shape[0]
    D2 = D**2
    C = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * C @ D2 @ C
    # symmetric eigendecomposition, descending eigenvalues
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    k = max(1, min(dim, n - 1, int((eigvals > 1e-12).sum()) or 1))
    coords = eigvecs[:, :k] * np.sqrt(np.clip(eigvals[:k], 0, None))
    if k < dim:
        coords = np.hstack([coords, np.zeros((n, dim - k))])
    return coords


def _scale(coords: np.ndarray) -> np.ndarray:
    """Center and scale to a ~[-1, 1] box so the frontend can frame it."""
    if coords.size == 0:
        return coords
    coords = coords - coords.mean(axis=0, keepdims=True)
    maxabs = np.abs(coords).max()
    if maxabs > 1e-12:
        coords = coords / maxabs
    return coords


def project(
    vectors: list[Vector],
    method: str,
    dim: int,
    metric: str,
    n_neighbors: int = 15,
) -> tuple[dict[str, list[float]], list[str]]:
    X = to_matrix(vectors)
    n, d = X.shape
    notes: list[str] = []

    if n == 1:
        coords = np.zeros((1, dim))
        notes.append("Single vector placed at the origin.")
        return {vectors[0].id: coords[0].tolist()}, notes

    if method == "pca":
        n_comp = max(1, min(dim, n - 1, d))
        if n_comp < dim:
            notes.append(f"PCA limited to {n_comp} component(s) for this dataset.")
        model = PCA(n_components=n_comp, random_state=0)
        coords = model.fit_transform(X)
        explained = model.explained_variance_ratio_.sum()
        notes.append(f"PCA retained {explained * 100:.1f}% of variance.")
        if n_comp < dim:
            coords = np.hstack([coords, np.zeros((n, dim - n_comp))])
        method_used = "pca"

    elif method == "umap":
        if n < 8:
            notes.append("Too few vectors for UMAP (<8); used classical MDS instead.")
            D = distance_matrix(X, metric if metric != "cosine_similarity" else "cosine_distance")
            coords = classical_mds(D, dim)
            method_used = "classical-mds"
        else:
            import umap  # lazy: numba JIT is slow on first import

            umap_metric = "cosine" if metric.startswith("cosine") else "euclidean"
            reducer = umap.UMAP(
                n_components=dim,
                n_neighbors=max(2, min(n_neighbors, n - 1)),
                metric=umap_metric,
                random_state=42,
                min_dist=0.1,
            )
            coords = reducer.fit_transform(X)
            notes.append(f"UMAP ({umap_metric}); local structure is approximate.")
            method_used = "umap"

    elif method == "mds":
        if n > 400:
            notes.append(
                f"MDS is O(n^2) with n={n}; used classical MDS for speed."
            )
            D = distance_matrix(X, metric if metric != "cosine_similarity" else "cosine_distance")
            coords = classical_mds(D, dim)
            method_used = "classical-mds"
        else:
            D = distance_matrix(X, metric if metric != "cosine_similarity" else "cosine_distance")
            model = MDS(
                n_components=dim,
                metric=True,
                dissimilarity="precomputed",
                normalized_stress="auto",
                n_init=4,
                max_iter=300,
                random_state=42,
            )
            coords = model.fit_transform(D)
            method_used = "mds"
    else:
        raise ValueError(f"Unknown projection method '{method}'.")

    coords = _scale(np.asarray(coords, dtype=np.float64))
    return {v.id: np.round(coords[i], 6).tolist() for i, v in enumerate(vectors)}, [
        f"Method: {method_used}",
        *notes,
        "Positions are an approximation — use the true embedding distances in the panel.",
    ]
