#!/usr/bin/env python3
"""Visualize MERT/CultureMERT embeddings with t-SNE and UMAP."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.manifold import TSNE


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
MODELS = ["mert_95m", "culturemert_95m"]
NUM_LAYERS = {"mert_95m": 13, "culturemert_95m": 13, "mert_330m": 25}


def load_embeddings(model_name: str, split: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    emb_path = os.path.join(EMBEDDINGS_DIR, model_name, f"{split}_layer{layer:02d}.npy")
    lab_path = os.path.join(EMBEDDINGS_DIR, model_name, f"{split}_labels.npy")
    if not os.path.exists(emb_path) or not os.path.exists(lab_path):
        raise FileNotFoundError(f"Missing embeddings for {model_name} {split} layer {layer}.")
    return np.load(emb_path), np.load(lab_path, allow_pickle=True)


def make_tsne(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    perplexity = max(2, min(30, (len(embeddings) - 1) // 3))
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": seed,
        "init": "pca",
        "learning_rate": "auto",
    }
    try:
        return TSNE(max_iter=1000, **kwargs).fit_transform(embeddings)
    except TypeError:
        return TSNE(n_iter=1000, **kwargs).fit_transform(embeddings)


def make_umap(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    n_neighbors = max(2, min(15, len(embeddings) - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings)


def plot_2d(points: np.ndarray, labels: np.ndarray, title: str, save_path: str) -> None:
    plt.figure(figsize=(12, 9))
    unique_labels = sorted(set(labels))
    palette = sns.color_palette("husl", len(unique_labels))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(points[mask, 0], points[mask, 1], s=24, alpha=0.65, color=palette[idx], label=label)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_layer_grid(model_name: str) -> None:
    n_layers = NUM_LAYERS[model_name]
    layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in layers:
        layers.append(n_layers - 1)

    cols = (len(layers) + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(5.5 * cols, 9))
    axes = axes.flatten()

    for ax_idx, layer_idx in enumerate(layers):
        embeddings, labels = load_embeddings(model_name, "train", layer_idx)
        points = make_tsne(embeddings)
        unique_labels = sorted(set(labels))
        palette = sns.color_palette("husl", len(unique_labels))
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            axes[ax_idx].scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.55, color=palette[idx])
        axes[ax_idx].set_title(f"Layer {layer_idx}")
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

    for ax_idx in range(len(layers), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(f"{model_name}: layer-wise t-SNE", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"{model_name}_tsne_layer_grid.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_side_by_side() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, model_name in zip(axes, MODELS):
        last_layer = NUM_LAYERS[model_name] - 1
        embeddings, labels = load_embeddings(model_name, "train", last_layer)
        points = make_tsne(embeddings)
        unique_labels = sorted(set(labels))
        palette = sns.color_palette("husl", len(unique_labels))
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.6, color=palette[idx], label=label)
        ax.set_title(model_name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=2)
    fig.suptitle("MERT-95M vs CultureMERT-95M: last-layer t-SNE")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "comparison_tsne_last_layer.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    for model_name in MODELS:
        print(f"\nVisualizing {model_name}")
        last_layer = NUM_LAYERS[model_name] - 1
        embeddings, labels = load_embeddings(model_name, "train", last_layer)

        plot_2d(
            make_tsne(embeddings),
            labels,
            f"{model_name}: last layer t-SNE",
            os.path.join(FIGURES_DIR, f"{model_name}_tsne_last_layer.png"),
        )
        plot_2d(
            make_umap(embeddings),
            labels,
            f"{model_name}: last layer UMAP",
            os.path.join(FIGURES_DIR, f"{model_name}_umap_last_layer.png"),
        )
        plot_layer_grid(model_name)

    plot_side_by_side()


if __name__ == "__main__":
    main()
