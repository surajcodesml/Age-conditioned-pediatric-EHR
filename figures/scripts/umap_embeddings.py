#!/usr/bin/env python3
"""FIGURE A10: umap_embeddings (appendix, low priority).

High-dimensional embeddings already exist under outputs/umap/.
Age bands in the NPZ are pediatric paper bins: <1, 1-5, 6-11, 12-17.
If umap-learn is unavailable, fall back to PCA-2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paper_style import REPO, apply_style, label_panel, save_fig  # noqa: E402

NPZ = REPO / "outputs/umap/umap_embeddings.npz"

BAND_ORDER = ["<1", "1-5", "6-11", "12-17"]
BAND_LABEL = {"<1": "<1", "1-5": "1–5", "6-11": "6–11", "12-17": "12–17"}
BAND_COLORS = {
    "<1": "#1B4F72",
    "1-5": "#148F77",
    "6-11": "#B9770E",
    "12-17": "#6C3483",
}


def _embed_2d(H: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap  # type: ignore

        Z = umap.UMAP(
            n_components=2, metric="cosine", n_neighbors=30, min_dist=0.1, random_state=0
        ).fit_transform(H)
        return Z, "UMAP"
    except Exception:
        # Standardize then PCA for a stable exploratory view
        H0 = H - H.mean(axis=0, keepdims=True)
        std = H0.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        H0 = H0 / std
        Z = PCA(n_components=2, random_state=0).fit_transform(H0)
        return Z, "PCA (umap-learn unavailable)"


def main() -> None:
    apply_style()
    data = np.load(NPZ, allow_pickle=True)
    bands = np.asarray(data["age_band"]).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, key, title, letter in zip(
        axes,
        ["H__Vanilla", "H__Kernel_age"],
        ["Vanilla", "Age-kernel"],
        "AB",
    ):
        H = np.asarray(data[key], dtype=np.float64)
        Z, method = _embed_2d(H)
        for b in BAND_ORDER:
            m = bands == b
            if not np.any(m):
                continue
            ax.scatter(
                Z[m, 0],
                Z[m, 1],
                s=6,
                alpha=0.55,
                c=BAND_COLORS[b],
                label=BAND_LABEL[b],
                linewidths=0,
            )
        ax.set_title(f"{title} — {method}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        label_panel(ax, letter)
        if letter == "A":
            ax.legend(frameon=False, fontsize=6.5, markerscale=2.5, loc="best")

    fig.suptitle("PIC representation embeddings by age band", fontsize=9.5, y=1.02)
    fig.tight_layout()
    save_fig(fig, "umap_embeddings")


if __name__ == "__main__":
    main()
