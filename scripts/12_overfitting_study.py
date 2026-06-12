#!/usr/bin/env python3
"""Study and reduce classifier overfitting on frozen MERT embeddings.

This follow-up keeps the same five ragas, 30-second chunks and five track-wise
folds as the main benchmark. It compares the earlier 256-unit neural head with
smaller and more strongly regularized heads, and plots the embeddings before
classification using PCA, t-SNE and UMAP.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
    top_k_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


RANDOM_SEED = 42
LINEAR_C_VALUES = (0.001, 0.01, 0.1, 1.0)
REGULARIZATION_CONFIGS = (
    {
        "name": "old_256_d01_lr1e3",
        "hidden_dim": 256,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "lr": 1e-3,
        "noise_std": 0.0,
    },
    {
        "name": "old_256_d03_lr1e3",
        "hidden_dim": 256,
        "dropout": 0.3,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "lr": 1e-3,
        "noise_std": 0.0,
    },
    {
        "name": "old_256_d01_lr3e4",
        "hidden_dim": 256,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "lr": 3e-4,
        "noise_std": 0.0,
    },
    {
        "name": "old_256_d03_lr3e4",
        "hidden_dim": 256,
        "dropout": 0.3,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "lr": 3e-4,
        "noise_std": 0.0,
    },
    {
        "name": "small_64",
        "hidden_dim": 64,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "lr": 1e-3,
        "noise_std": 0.0,
    },
    {
        "name": "small_dropout",
        "hidden_dim": 64,
        "dropout": 0.5,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "lr": 1e-3,
        "noise_std": 0.0,
    },
    {
        "name": "strong_decay",
        "hidden_dim": 64,
        "dropout": 0.5,
        "weight_decay": 0.1,
        "label_smoothing": 0.0,
        "lr": 1e-3,
        "noise_std": 0.0,
    },
    {
        "name": "label_smoothing",
        "hidden_dim": 64,
        "dropout": 0.5,
        "weight_decay": 0.1,
        "label_smoothing": 0.1,
        "lr": 1e-3,
        "noise_std": 0.0,
    },
    {
        "name": "lower_lr",
        "hidden_dim": 64,
        "dropout": 0.5,
        "weight_decay": 0.1,
        "label_smoothing": 0.1,
        "lr": 3e-4,
        "noise_std": 0.0,
    },
    {
        "name": "embedding_noise",
        "hidden_dim": 64,
        "dropout": 0.5,
        "weight_decay": 0.1,
        "label_smoothing": 0.1,
        "lr": 3e-4,
        "noise_std": 0.05,
    },
)


def load_crossval_module():
    path = Path(__file__).with_name("11_crossval_benchmark.py")
    if not path.exists():
        raise RuntimeError(f"Required helper is missing: {path}")
    spec = importlib.util.spec_from_file_location("crossval_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CV = load_crossval_module()
BASE = CV.BASE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="saraga_carnatic")
    parser.add_argument("--kaggle-data-root", required=True)
    parser.add_argument("--base-output-dir", default="/kaggle/working/mert_raga_5fold")
    parser.add_argument("--output-dir", default="/kaggle/working/mert_raga_overfit")
    parser.add_argument("--num-ragas", type=int, default=5)
    parser.add_argument("--tracks-per-raga", type=int, default=5)
    parser.add_argument("--exclude-raga", action="append", default=[])
    parser.add_argument("--segments-per-track", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--force-clips", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def make_probe(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2500,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def prediction_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
        "top_3_accuracy": float(
            top_k_accuracy_score(
                labels,
                probabilities,
                k=min(3, num_classes),
                labels=np.arange(num_classes),
            )
        ),
    }


def aggregate_tracks(
    probabilities: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    track_probabilities = []
    track_labels = []
    for track in sorted(set(tracks)):
        mask = tracks == track
        track_probabilities.append(probabilities[mask].mean(axis=0))
        track_labels.append(Counter(labels[mask]).most_common(1)[0][0])
    return np.stack(track_probabilities), np.asarray(track_labels)


class RegularizedHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def train_candidate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    config: dict[str, Any],
    epochs: int,
    patience: int,
    seed: int,
    class_weights: np.ndarray | None = None,
) -> tuple[RegularizedHead, StandardScaler, list[dict[str, float]], dict[str, float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x).astype(np.float32)
    val_scaled = scaler.transform(val_x).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(train_scaled),
        torch.from_numpy(train_y.astype(np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(16, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedHead(
        train_x.shape[1],
        int(config["hidden_dim"]),
        num_classes,
        float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(
        weight=(
            torch.from_numpy(class_weights.astype(np.float32)).to(device)
            if class_weights is not None
            else None
        ),
        label_smoothing=float(config["label_smoothing"])
    )
    val_values = torch.from_numpy(val_scaled).to(device)
    val_targets = torch.from_numpy(val_y.astype(np.int64)).to(device)

    history = []
    best_rank = None
    best_state = None
    best_epoch = 0
    stale = 0
    noise_std = float(config["noise_std"])
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_predictions = []
        train_labels = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if noise_std:
                batch_x = batch_x + torch.randn_like(batch_x) * noise_std
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch_y)
            train_predictions.extend(logits.argmax(1).detach().cpu().numpy())
            train_labels.extend(batch_y.detach().cpu().numpy())

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_values)
            val_loss = float(criterion(val_logits, val_targets).item())
            val_predictions = val_logits.argmax(1).cpu().numpy()
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / len(dataset),
            "val_loss": val_loss,
            "train_accuracy": float(
                accuracy_score(train_labels, train_predictions)
            ),
            "val_accuracy": float(accuracy_score(val_y, val_predictions)),
            "val_macro_f1": float(
                f1_score(
                    val_y,
                    val_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
        }
        history.append(row)
        gap = row["train_accuracy"] - row["val_accuracy"]
        rank = (
            row["val_macro_f1"],
            row["val_accuracy"],
            -val_loss,
            -gap,
            -epoch,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("Classifier training did not complete.")
    model.load_state_dict(best_state)
    model.eval()
    best = history[best_epoch - 1]
    return (
        model,
        scaler,
        history,
        {
            "best_epoch": best_epoch,
            "train_loss": best["train_loss"],
            "val_loss": best["val_loss"],
            "train_accuracy": best["train_accuracy"],
            "val_accuracy": best["val_accuracy"],
            "val_macro_f1": best["val_macro_f1"],
            "accuracy_gap": best["train_accuracy"] - best["val_accuracy"],
        },
    )


def refit_candidate(
    values: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    config: dict[str, Any],
    epochs: int,
    seed: int,
    class_weights: np.ndarray | None = None,
) -> tuple[RegularizedHead, StandardScaler]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(scaled),
        torch.from_numpy(labels.astype(np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(16, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegularizedHead(
        values.shape[1],
        int(config["hidden_dim"]),
        num_classes,
        float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(
        weight=(
            torch.from_numpy(class_weights.astype(np.float32)).to(device)
            if class_weights is not None
            else None
        ),
        label_smoothing=float(config["label_smoothing"])
    )
    noise_std = float(config["noise_std"])
    for _ in range(max(1, epochs)):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if noise_std:
                batch_x = batch_x + torch.randn_like(batch_x) * noise_std
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval(), scaler


def head_probabilities(
    model: RegularizedHead,
    scaler: StandardScaler,
    values: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    scaled = scaler.transform(values).astype(np.float32)
    with torch.inference_mode():
        return torch.softmax(
            model(torch.from_numpy(scaled).to(device)),
            dim=1,
        ).cpu().numpy()


def choose_layer_and_linear_c(
    embeddings: np.ndarray,
    encoded: np.ndarray,
    tracks: np.ndarray,
    folds: list[dict[str, list[dict[str, str]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tuning_rows = []
    selections = []
    for fold_index, fold in enumerate(folds):
        masks = CV.masks_for_fold(fold, tracks)
        best = None
        for layer in range(embeddings.shape[0]):
            for c_value in LINEAR_C_VALUES:
                probe = make_probe(c_value)
                probe.fit(embeddings[layer, masks["train"]], encoded[masks["train"]])
                pred = probe.predict(embeddings[layer, masks["val"]])
                row = {
                    "fold": fold_index,
                    "layer": layer,
                    "C": c_value,
                    "val_accuracy": float(
                        accuracy_score(encoded[masks["val"]], pred)
                    ),
                    "val_macro_f1": float(
                        f1_score(
                            encoded[masks["val"]],
                            pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
                tuning_rows.append(row)
                rank = (
                    row["val_macro_f1"],
                    row["val_accuracy"],
                    -abs(math.log10(c_value)),
                    -layer,
                )
                if best is None or rank > best["rank"]:
                    best = {**row, "rank": rank}
        assert best is not None
        best.pop("rank")
        selections.append(best)
    return tuning_rows, selections


def representative_layer(tuning_rows: list[dict[str, Any]]) -> int:
    frame = pd.DataFrame(tuning_rows)
    best_c_per_fold_layer = (
        frame.sort_values(
            ["fold", "layer", "val_macro_f1", "val_accuracy"],
            ascending=[True, True, False, False],
        )
        .groupby(["fold", "layer"], as_index=False)
        .first()
    )
    summary = (
        best_c_per_fold_layer.groupby("layer")
        .agg(
            macro_f1=("val_macro_f1", "mean"),
            accuracy=("val_accuracy", "mean"),
        )
        .reset_index()
        .sort_values(
            ["macro_f1", "accuracy", "layer"],
            ascending=[False, False, True],
        )
    )
    return int(summary.iloc[0]["layer"])


def track_mean_embeddings(
    values: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = []
    track_labels = []
    track_ids = []
    for track in sorted(set(tracks)):
        mask = tracks == track
        means.append(values[mask].mean(axis=0))
        track_labels.append(Counter(labels[mask]).most_common(1)[0][0])
        track_ids.append(track)
    return (
        np.stack(means),
        np.asarray(track_labels, dtype=object),
        np.asarray(track_ids, dtype=object),
    )


def plot_embedding_clusters(
    model_key: str,
    values: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
    layer: int,
    figures_dir: Path,
) -> dict[str, float]:
    import umap

    track_values, track_labels, _ = track_mean_embeddings(
        values,
        labels,
        tracks,
    )
    scaled = StandardScaler().fit_transform(track_values)
    projections = {
        "PCA": PCA(n_components=2).fit_transform(scaled),
        "t-SNE": TSNE(
            n_components=2,
            perplexity=5,
            init="pca",
            learning_rate="auto",
            random_state=RANDOM_SEED,
        ).fit_transform(scaled),
        "UMAP": umap.UMAP(
            n_components=2,
            n_neighbors=5,
            min_dist=0.2,
            random_state=RANDOM_SEED,
        ).fit_transform(scaled),
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (name, points) in zip(axes, projections.items()):
        frame = pd.DataFrame(
            {"x": points[:, 0], "y": points[:, 1], "raga": track_labels}
        )
        sns.scatterplot(
            data=frame,
            x="x",
            y="y",
            hue="raga",
            s=85,
            alpha=0.85,
            ax=ax,
            legend=name == "UMAP",
        )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        if name != "UMAP" and ax.get_legend() is not None:
            ax.get_legend().remove()
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(
        f"{model_key}: track-mean embeddings before classifier, layer {layer}"
    )
    plt.tight_layout()
    path = figures_dir / f"{model_key}_embedding_clusters.png"
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return {
        "model": model_key,
        "layer": layer,
        "num_tracks": len(track_values),
        "silhouette": float(silhouette_score(scaled, track_labels)),
        "davies_bouldin": float(
            davies_bouldin_score(scaled, track_labels)
        ),
    }


def plot_training_curves(
    model_key: str,
    old_histories: list[dict[str, Any]],
    selected_histories: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row_index, (heading, histories) in enumerate(
        (
            ("Earlier 256-unit head", old_histories),
            ("Selected regularized head", selected_histories),
        )
    ):
        for record in histories:
            frame = pd.DataFrame(record["history"])
            label = f"fold {record['fold'] + 1}"
            axes[row_index, 0].plot(
                frame["epoch"],
                frame["train_loss"],
                alpha=0.75,
                label=f"{label} train",
            )
            axes[row_index, 0].plot(
                frame["epoch"],
                frame["val_loss"],
                linestyle="--",
                alpha=0.75,
                label=f"{label} val",
            )
            axes[row_index, 1].plot(
                frame["epoch"],
                frame["train_accuracy"],
                alpha=0.75,
            )
            axes[row_index, 1].plot(
                frame["epoch"],
                frame["val_accuracy"],
                linestyle="--",
                alpha=0.75,
            )
        axes[row_index, 0].set_title(
            f"{heading} loss: solid train, dashed validation"
        )
        axes[row_index, 1].set_title(
            f"{heading} accuracy: solid train, dashed validation"
        )
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Cross-entropy loss")
    axes[1, 0].set_ylabel("Cross-entropy loss")
    axes[0, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0, 1.02)
    axes[1, 1].set_ylim(0, 1.02)
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{model_key}: classifier training across five folds")
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"{model_key}_regularized_training_curves.png",
        dpi=170,
    )
    plt.close()


def plot_confusion(
    model_key: str,
    classifier: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    figures_dir: Path,
) -> None:
    matrix = confusion_matrix(
        labels,
        probabilities.argmax(axis=1),
        labels=np.arange(len(classes)),
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=35,
        colorbar=False,
    )
    ax.set_title(f"{model_key}: {classifier}, five-fold test predictions")
    plt.tight_layout()
    plt.savefig(
        figures_dir / f"{model_key}_{classifier}_confusion.png",
        dpi=170,
    )
    plt.close()


def run_model_study(
    model_key: str,
    data: dict[str, np.ndarray],
    folds: list[dict[str, list[dict[str, str]]]],
    args: argparse.Namespace,
    figures_dir: Path,
) -> dict[str, Any]:
    embeddings = data["embeddings"]
    labels = data["labels"]
    tracks = data["tracks"]
    encoder = LabelEncoder().fit(labels)
    encoded = encoder.transform(labels)
    num_classes = len(encoder.classes_)

    layer_tuning, layer_selections = choose_layer_and_linear_c(
        embeddings,
        encoded,
        tracks,
        folds,
    )
    display_layer = representative_layer(layer_tuning)
    cluster_metrics = plot_embedding_clusters(
        model_key,
        embeddings[display_layer],
        labels,
        tracks,
        display_layer,
        figures_dir,
    )

    linear_oof = np.full((len(labels), num_classes), np.nan, dtype=np.float32)
    old_oof = np.full_like(linear_oof, np.nan)
    regularized_oof = np.full_like(linear_oof, np.nan)
    tuning_rows = []
    fold_rows = []
    old_histories = []
    selected_histories = []

    for fold_index, fold in enumerate(folds):
        print(f"\n{model_key}: regularization fold {fold_index + 1}/5")
        masks = CV.masks_for_fold(fold, tracks)
        selection = layer_selections[fold_index]
        layer = int(selection["layer"])
        c_value = float(selection["C"])
        trainval = masks["train"] | masks["val"]

        probe = make_probe(c_value)
        probe.fit(embeddings[layer, trainval], encoded[trainval])
        linear_oof[masks["test"]] = probe.predict_proba(
            embeddings[layer, masks["test"]]
        )

        best_candidate = None
        old_candidate = None
        for config_index, config in enumerate(REGULARIZATION_CONFIGS):
            model, scaler, history, best = train_candidate(
                embeddings[layer, masks["train"]],
                encoded[masks["train"]],
                embeddings[layer, masks["val"]],
                encoded[masks["val"]],
                num_classes,
                config,
                args.epochs,
                args.patience,
                RANDOM_SEED + fold_index * 100 + config_index,
            )
            row = {
                "model": model_key,
                "fold": fold_index,
                "layer": layer,
                **config,
                **best,
                "start_train_loss": history[0]["train_loss"],
                "final_train_loss": history[-1]["train_loss"],
                "final_val_loss": history[-1]["val_loss"],
            }
            tuning_rows.append(row)
            rank = (
                best["val_macro_f1"],
                best["val_accuracy"],
                -best["val_loss"],
                -best["accuracy_gap"],
                -int(config["hidden_dim"]),
            )
            record = {
                "rank": rank,
                "config": config,
                "best": best,
                "history": history,
            }
            if config["name"].startswith("old_256"):
                if old_candidate is None or rank > old_candidate["rank"]:
                    old_candidate = record
            elif best_candidate is None or rank > best_candidate["rank"]:
                best_candidate = record
            del model, scaler

        assert best_candidate is not None and old_candidate is not None
        old_model, old_scaler = refit_candidate(
            embeddings[layer, trainval],
            encoded[trainval],
            num_classes,
            old_candidate["config"],
            int(old_candidate["best"]["best_epoch"]),
            RANDOM_SEED + 1000 + fold_index,
        )
        old_oof[masks["test"]] = head_probabilities(
            old_model,
            old_scaler,
            embeddings[layer, masks["test"]],
        )
        selected_model, selected_scaler = refit_candidate(
            embeddings[layer, trainval],
            encoded[trainval],
            num_classes,
            best_candidate["config"],
            int(best_candidate["best"]["best_epoch"]),
            RANDOM_SEED + 2000 + fold_index,
        )
        regularized_oof[masks["test"]] = head_probabilities(
            selected_model,
            selected_scaler,
            embeddings[layer, masks["test"]],
        )
        old_histories.append(
            {
                "fold": fold_index,
                "config": old_candidate["config"],
                "history": old_candidate["history"],
            }
        )
        selected_histories.append(
            {
                "fold": fold_index,
                "config": best_candidate["config"],
                "history": best_candidate["history"],
            }
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "layer": layer,
                "linear_C": c_value,
                "old_config": old_candidate["config"]["name"],
                "selected_config": best_candidate["config"]["name"],
                **{
                    f"old_{key}": value
                    for key, value in old_candidate["best"].items()
                },
                **{
                    f"selected_{key}": value
                    for key, value in best_candidate["best"].items()
                },
            }
        )
        del old_model, old_scaler, selected_model, selected_scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if (
        np.isnan(linear_oof).any()
        or np.isnan(old_oof).any()
        or np.isnan(regularized_oof).any()
    ):
        raise RuntimeError("One or more folds did not produce predictions.")

    results = {}
    for name, probabilities in (
        ("linear", linear_oof),
        ("old_256", old_oof),
        ("regularized", regularized_oof),
    ):
        clip = prediction_metrics(encoded, probabilities, num_classes)
        track_probability, track_labels = aggregate_tracks(
            probabilities,
            encoded,
            tracks,
        )
        track = prediction_metrics(
            track_labels,
            track_probability,
            num_classes,
        )
        track["num_tracks"] = len(track_labels)
        results[name] = {"clip": clip, "track": track}
        plot_confusion(
            model_key,
            name,
            encoded,
            probabilities,
            encoder.classes_,
            figures_dir,
        )

    plot_training_curves(
        model_key,
        old_histories,
        selected_histories,
        figures_dir,
    )
    return {
        "classes": list(encoder.classes_),
        "display_layer": display_layer,
        "cluster_metrics": cluster_metrics,
        "layer_tuning": layer_tuning,
        "classifier_tuning": tuning_rows,
        "fold_results": fold_rows,
        "old_histories": old_histories,
        "selected_histories": selected_histories,
        "results": results,
    }


def cluster_reading(score: float) -> str:
    if score < 0:
        return "The raga groups overlap strongly; the embedding space does not form useful clusters."
    if score < 0.1:
        return "There is only weak separation and substantial overlap between ragas."
    if score < 0.25:
        return "There is some separation, but the clusters are not clean."
    return "The raga groups show reasonably clear separation."


def write_report(
    path: Path,
    args: argparse.Namespace,
    selected: dict[str, list[dict[str, str]]],
    studies: dict[str, Any],
) -> None:
    lines = [
        "# Overfitting And Embedding Report",
        "",
        "## Dataset And Split",
        "",
        f"- Ragas: {', '.join(selected)}",
        f"- Original recordings: {args.tracks_per_raga} per raga",
        f"- Chunk size: {args.segment_seconds:.0f} seconds",
        f"- Chunks per recording: {args.segments_per_track}",
        "- Each fold: 3 training, 1 validation and 1 test recording per raga",
        "- Five folds, so every recording is tested once",
        "- All chunks from one recording stay in the same split",
        "",
        "## Frozen Models",
        "",
        "- `m-a-p/MERT-v1-95M`",
        "- `ntua-slp/CultureMERT-95M`",
        "- Both backbones remain unchanged",
        "- Mean pooling over time produces one 768-dimensional embedding per chunk",
        "",
        "## Variables Tried",
        "",
        "| Name | Hidden units | Dropout | Weight decay | Label smoothing | Learning rate | Embedding noise |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for config in REGULARIZATION_CONFIGS:
        lines.append(
            f"| {config['name']} | {config['hidden_dim']} | "
            f"{config['dropout']} | {config['weight_decay']} | "
            f"{config['label_smoothing']} | {config['lr']} | "
            f"{config['noise_std']} |"
        )
    lines.extend(
        [
            "",
            f"- Maximum epochs: {args.epochs}",
            f"- Early-stopping patience: {args.patience}",
            f"- Linear C values: {list(LINEAR_C_VALUES)}",
            "",
            "## Five-Fold Test Results",
            "",
            "| Model | Classifier | Clip accuracy | Track accuracy | Track macro F1 | Track top-3 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    labels = {
        "linear": "Logistic regression",
        "old_256": "Earlier 256-unit head",
        "regularized": "Selected regularized head",
    }
    for model_key, study in studies.items():
        for classifier, label in labels.items():
            result = study["results"][classifier]
            lines.append(
                f"| {model_key} | {label} | "
                f"{result['clip']['accuracy']:.3f} | "
                f"{result['track']['accuracy']:.3f} | "
                f"{result['track']['macro_f1']:.3f} | "
                f"{result['track']['top_3_accuracy']:.3f} |"
            )

    lines.extend(["", "## Training And Overfitting", ""])
    for model_key, study in studies.items():
        frame = pd.DataFrame(study["fold_results"])
        lines.extend(
            [
                f"### {model_key}",
                "",
                f"- Earlier head mean training accuracy: "
                f"{frame['old_train_accuracy'].mean():.3f}",
                f"- Earlier head mean validation accuracy: "
                f"{frame['old_val_accuracy'].mean():.3f}",
                f"- Earlier head mean train-validation gap: "
                f"{frame['old_accuracy_gap'].mean():.3f}",
                f"- Earlier head mean training loss: "
                f"{frame['old_train_loss'].mean():.3f}",
                f"- Earlier head mean validation loss: "
                f"{frame['old_val_loss'].mean():.3f}",
                f"- Earlier settings selected by fold: "
                f"{dict(Counter(frame['old_config']))}",
                "",
                f"- Regularized head mean training accuracy: "
                f"{frame['selected_train_accuracy'].mean():.3f}",
                f"- Regularized head mean validation accuracy: "
                f"{frame['selected_val_accuracy'].mean():.3f}",
                f"- Regularized head mean train-validation gap: "
                f"{frame['selected_accuracy_gap'].mean():.3f}",
                f"- Regularized head mean training loss: "
                f"{frame['selected_train_loss'].mean():.3f}",
                f"- Regularized head mean validation loss: "
                f"{frame['selected_val_loss'].mean():.3f}",
                f"- Regularized settings selected by fold: "
                f"{dict(Counter(frame['selected_config']))}",
                "",
            ]
        )

    lines.extend(["## Embeddings Before The Classifier", ""])
    for model_key, study in studies.items():
        cluster = study["cluster_metrics"]
        lines.extend(
            [
                f"### {model_key}",
                "",
                f"- Plotted hidden-state layer: {cluster['layer']}",
                f"- Track-level silhouette score: {cluster['silhouette']:.3f}",
                f"- Track-level Davies-Bouldin score: "
                f"{cluster['davies_bouldin']:.3f}",
                f"- Reading: {cluster_reading(cluster['silhouette'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Conclusion",
            "",
            "The experiment did not malfunction. Both models produced embeddings and "
            "all five folds completed. The main question is whether stronger "
            "regularization reduces the train-validation gap and improves unseen-track "
            "accuracy. If the gap becomes smaller but test accuracy remains near 20%, "
            "then the current frozen embeddings and small number of recordings are the "
            "main limitation rather than only the classifier size.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def package_results(output_dir: Path) -> Path:
    path = output_dir / "overfitting_embedding_report.zip"
    path.unlink(missing_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in (
            output_dir / "REPORT.md",
            output_dir / "metrics",
            output_dir / "figures",
        ):
            if item.is_file():
                archive.write(item, item.relative_to(output_dir).as_posix())
            elif item.is_dir():
                for file in sorted(item.rglob("*")):
                    if file.is_file():
                        archive.write(
                            file,
                            file.relative_to(output_dir).as_posix(),
                        )
    return path


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle P100 or T4 GPU and rerun.")
    if args.tracks_per_raga != 5:
        raise RuntimeError("This study expects five recordings/folds.")
    if args.segment_seconds != 30:
        raise RuntimeError("Keep the chunk size at 30 seconds for comparison.")

    sns.set_theme(style="whitegrid")
    base_output = Path(args.base_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    base_config = CV.configuration(args)
    cache_id = CV.fingerprint(base_config)
    cache_root = base_output / "cache" / cache_id
    clips_dir = cache_root / "clips"
    embedding_dir = cache_root / "embeddings"

    data_root = Path(args.kaggle_data_root).resolve()
    rows = BASE.collect_kaggle_tracks(data_root)
    selected = BASE.choose_balanced_tracks(
        rows,
        args.num_ragas,
        args.tracks_per_raga,
        args.exclude_raga,
    )
    folds = CV.build_folds(selected)
    manifest = CV.prepare_all_clips(
        selected,
        clips_dir,
        args.segment_seconds,
        args.segments_per_track,
        args.force_clips,
    )
    CV.validate_clips(manifest, selected, args)
    print(f"Using cache directory: {cache_root}")
    print(f"Prepared clips: {len(manifest)}")

    embeddings = {}
    for model_key, hf_name in CV.MODELS.items():
        embeddings[model_key] = CV.extract_all_embeddings(
            model_key,
            hf_name,
            manifest,
            embedding_dir,
            args.batch_size,
            args.force_embeddings,
        )

    studies = {}
    for model_key, data in embeddings.items():
        studies[model_key] = run_model_study(
            model_key,
            data,
            folds,
            args,
            figures_dir,
        )
        json_dump(metrics_dir / f"{model_key}_study.json", studies[model_key])
        pd.DataFrame(studies[model_key]["layer_tuning"]).to_csv(
            metrics_dir / f"{model_key}_layer_tuning.csv",
            index=False,
        )
        pd.DataFrame(studies[model_key]["classifier_tuning"]).to_csv(
            metrics_dir / f"{model_key}_classifier_tuning.csv",
            index=False,
        )
        pd.DataFrame(studies[model_key]["fold_results"]).to_csv(
            metrics_dir / f"{model_key}_fold_results.csv",
            index=False,
        )

    summary_rows = []
    cluster_rows = []
    for model_key, study in studies.items():
        cluster_rows.append(study["cluster_metrics"])
        for classifier, result in study["results"].items():
            summary_rows.append(
                {
                    "model": model_key,
                    "classifier": classifier,
                    **{
                        f"clip_{key}": value
                        for key, value in result["clip"].items()
                    },
                    **{
                        f"track_{key}": value
                        for key, value in result["track"].items()
                    },
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        metrics_dir / "regularization_summary.csv",
        index=False,
    )
    pd.DataFrame(cluster_rows).to_csv(
        metrics_dir / "cluster_metrics.csv",
        index=False,
    )
    json_dump(
        metrics_dir / "experiment_variables.json",
        {
            "chunk_seconds": args.segment_seconds,
            "chunks_per_recording": args.segments_per_track,
            "tracks_per_raga": args.tracks_per_raga,
            "folds": 5,
            "linear_C_values": LINEAR_C_VALUES,
            "regularization_configs": REGULARIZATION_CONFIGS,
            "epochs": args.epochs,
            "patience": args.patience,
        },
    )
    write_report(output_dir / "REPORT.md", args, selected, studies)
    archive = package_results(output_dir)
    print("\nOverfitting study completed.")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Report: {output_dir / 'REPORT.md'}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
