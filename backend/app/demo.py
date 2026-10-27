"""Built-in demo dataset so the app works without any credentials."""

from __future__ import annotations

import numpy as np

from .models import Vector

CLUSTERS = {
    "protein": ("Alpha helix", "Beta sheet", "Signal peptide", "Transmembrane domain",
                "Globin fold", "Kinase domain", "Zinc finger", "EGF repeat",
                "Immunoglobulin fold", "Leucine zipper", "Coiled coil", "Disulfide bridge"),
    "document": ("Introduction", "Methods", "Results", "Discussion", "Related work",
                 "Conclusion", "Appendix", "Benchmark",
                 "Ablation study", "Hyperparameters", "Dataset statistics", "Ethics statement"),
    "image": ("Sunset over mountains", "Street at night", "Portrait of a chef", "Ocean waves",
              "Forest path", "City skyline", "Desert dunes", "Snowy cabin",
              "Autumn leaves", "Rainy window", "Neon sign", "Bird in flight"),
}


def build_demo(dim: int = 32, seed: int = 7) -> list[Vector]:
    rng = np.random.default_rng(seed)
    vectors: list[Vector] = []
    for cluster, labels in CLUSTERS.items():
        centre = rng.normal(0.0, 1.0, dim)
        for i, label in enumerate(labels):
            noise = rng.normal(0.0, 0.35, dim)
            values = centre + noise + rng.normal(0.0, 0.05, dim) * i
            values = values / (np.linalg.norm(values) or 1.0)
            vectors.append(
                Vector(
                    id=f"{cluster}-{i:02d}",
                    values=[round(float(v), 6) for v in values],
                    metadata={
                        "label": label,
                        "cluster": cluster,
                        "source": "demo",
                        "tokens": int(rng.integers(20, 400)),
                    },
                )
            )
    return vectors
