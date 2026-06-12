#!/usr/bin/env python3
"""Final expanded-data benchmark with a completely frozen MERT backbone.

This experiment uses all available recordings (up to seven per class) for the
five best-supported genuine ragas. It keeps MERT frozen, selects an embedding
layer with validation data, and compares a linear probe with one fixed,
regularized neural classifier. Original recordings, not clips, define folds.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


RANDOM_SEED = 42
MODEL_KEY = "mert_95m"
MODEL_NAME = "m-a-p/MERT-v1-95M"
LINEAR_C_VALUES = (0.001, 0.01, 0.1, 1.0)
HEAD_CONFIG = {
    "name": "fixed_regularized_head",
    "hidden_dim": 64,
    "dropout": 0.5,
    "weight_decay": 0.1,
    "label_smoothing": 0.1,
    "lr": 3e-4,
    "noise_std": 0.05,
}


def load_module(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    if not path.exists():
        raise RuntimeError(f"Required helper is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CV = load_module("11_crossval_benchmark.py", "crossval_benchmark")
STUDY = load_module("12_overfitting_study.py", "overfitting_study")
BASE = CV.BASE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle-data-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/mert_final_expanded",
    )
    parser.add_argument("--num-ragas", type=int, default=5)
    parser.add_argument("--minimum-tracks", type=int, default=5)
    parser.add_argument("--maximum-tracks", type=int, default=7)
    parser.add_argument("--exclude-raga", action="append", default=[])
    parser.add_argument("--segments-per-track", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--head-patience", type=int, default=5)
    parser.add_argument("--force-clips", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def select_expanded_tracks(
    rows: list[dict[str, str]],
    num_ragas: int,
    minimum_tracks: int,
    maximum_tracks: int,
    excluded_ragas: list[str],
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, int],
]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    excluded = {
        BASE.normalized_label(name)
        for name in excluded_ragas
    }
    for row in rows:
        if BASE.normalized_label(row["raga"]) not in excluded:
            grouped[row["raga"]].append(row)
    counts = {
        raga: len(items)
        for raga, items in sorted(
            grouped.items(),
            key=lambda item: (-len(item[1]), item[0].lower()),
        )
    }
    eligible = [
        (raga, items)
        for raga, items in grouped.items()
        if len(items) >= minimum_tracks
    ]
    eligible.sort(key=lambda item: (-len(item[1]), item[0].lower()))
    if len(eligible) < num_ragas:
        raise RuntimeError(
            f"Only {len(eligible)} ragas have at least "
            f"{minimum_tracks} tracks. Counts: {counts}"
        )
    selected = {}
    for raga, items in eligible[:num_ragas]:
        ordered = sorted(items, key=lambda item: item["track_id"])
        selected[raga] = ordered[:maximum_tracks]
    return selected, counts


def build_folds(
    selected: dict[str, list[dict[str, str]]],
) -> list[dict[str, list[dict[str, str]]]]:
    rows = [
        row
        for raga_rows in selected.values()
        for row in raga_rows
    ]
    labels = np.asarray([row["raga"] for row in rows], dtype=object)
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    folds = []
    all_test_tracks = []
    placeholder = np.zeros(len(rows))
    for fold_index, (trainval_index, test_index) in enumerate(
        splitter.split(placeholder, labels)
    ):
        trainval = [rows[index] for index in trainval_index]
        test = [rows[index] for index in test_index]
        by_raga: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in trainval:
            by_raga[row["raga"]].append(row)
        validation = []
        train = []
        for raga in sorted(selected):
            candidates = sorted(
                by_raga[raga],
                key=lambda row: row["track_id"],
            )
            validation_index = fold_index % len(candidates)
            validation.append(candidates[validation_index])
            train.extend(
                row
                for index, row in enumerate(candidates)
                if index != validation_index
            )
        fold = {"train": train, "val": validation, "test": test}
        expected = set(selected)
        ids = {
            split: {row["track_id"] for row in split_rows}
            for split, split_rows in fold.items()
        }
        if not (
            ids["train"].isdisjoint(ids["val"])
            and ids["train"].isdisjoint(ids["test"])
            and ids["val"].isdisjoint(ids["test"])
        ):
            raise RuntimeError(f"Fold {fold_index} has track leakage.")
        for split, split_rows in fold.items():
            if {row["raga"] for row in split_rows} != expected:
                raise RuntimeError(
                    f"Fold {fold_index} {split} lost a raga."
                )
        all_test_tracks.extend(row["track_id"] for row in test)
        folds.append(fold)
    expected_tracks = {
        row["track_id"]
        for raga_rows in selected.values()
        for row in raga_rows
    }
    if Counter(all_test_tracks) != Counter(
        {track: 1 for track in expected_tracks}
    ):
        raise RuntimeError(
            "Every expanded recording must be tested exactly once."
        )
    return folds


def validate_manifest(
    manifest: list[dict[str, Any]],
    selected: dict[str, list[dict[str, str]]],
    segments_per_track: int,
    segment_seconds: float,
) -> None:
    expected_tracks = sum(len(items) for items in selected.values())
    expected_clips = expected_tracks * segments_per_track
    if len(manifest) != expected_clips:
        raise RuntimeError(
            f"Expected {expected_clips} clips, found {len(manifest)}."
        )
    per_track = Counter(item["track_id"] for item in manifest)
    if set(per_track.values()) != {segments_per_track}:
        raise RuntimeError(f"Unequal clips per track: {per_track}")
    expected_frames = int(round(segment_seconds * CV.SAMPLE_RATE))
    for item in manifest:
        path = Path(item["path"])
        if not path.exists():
            raise RuntimeError(f"Cached clip is missing: {path}")
        info = sf.info(path)
        if (
            info.samplerate != CV.SAMPLE_RATE
            or info.frames != expected_frames
        ):
            raise RuntimeError(
                f"Invalid cached clip {path}: {info.samplerate} Hz, "
                f"{info.frames} frames. Expected {CV.SAMPLE_RATE} Hz and "
                f"{expected_frames} frames."
            )


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


def select_layer(
    embeddings: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
    masks: dict[str, np.ndarray],
) -> tuple[int, float, list[dict[str, Any]]]:
    rows = []
    best = None
    for layer in range(embeddings.shape[0]):
        for c_value in LINEAR_C_VALUES:
            probe = make_probe(c_value)
            probe.fit(
                embeddings[layer, masks["train"]],
                labels[masks["train"]],
            )
            probabilities = probe.predict_proba(
                embeddings[layer, masks["val"]]
            )
            clip_score = STUDY.prediction_metrics(
                labels[masks["val"]],
                probabilities,
                len(np.unique(labels)),
            )
            track_probabilities, track_labels = STUDY.aggregate_tracks(
                probabilities,
                labels[masks["val"]],
                tracks[masks["val"]],
            )
            track_score = STUDY.prediction_metrics(
                track_labels,
                track_probabilities,
                len(np.unique(labels)),
            )
            row = {
                "layer": layer,
                "C": c_value,
                "val_clip_accuracy": clip_score["accuracy"],
                "val_clip_macro_f1": clip_score["macro_f1"],
                "val_track_accuracy": track_score["accuracy"],
                "val_track_macro_f1": track_score["macro_f1"],
            }
            rows.append(row)
            rank = (
                track_score["macro_f1"],
                track_score["accuracy"],
                clip_score["macro_f1"],
                clip_score["accuracy"],
                -abs(math.log10(c_value)),
                -layer,
            )
            if best is None or rank > best["rank"]:
                best = {**row, "rank": rank}
    if best is None:
        raise RuntimeError("Layer selection failed.")
    return int(best["layer"]), float(best["C"]), rows


def track_masks(
    fold: dict[str, list[dict[str, str]]],
    tracks: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        split: np.isin(
            tracks,
            [row["track_id"] for row in rows],
        )
        for split, rows in fold.items()
    }


def balanced_class_weights(
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        raise RuntimeError("A training split lost a class.")
    return len(labels) / (num_classes * counts)


def plot_distribution(
    selected: dict[str, list[dict[str, str]]],
    path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "raga": list(selected),
            "tracks": [len(selected[raga]) for raga in selected],
        }
    )
    plt.figure(figsize=(9, 4.5))
    sns.barplot(
        data=frame,
        x="raga",
        y="tracks",
        hue="raga",
        legend=False,
        palette="deep",
    )
    plt.ylim(0, max(frame["tracks"]) + 1)
    plt.title("Expanded original recordings per raga")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_confusion(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    title: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(
        labels,
        probabilities.argmax(axis=1),
        labels=np.arange(len(classes)),
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(
        matrix,
        display_labels=classes,
    ).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=35,
        colorbar=False,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_layers(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    frame = pd.DataFrame(rows)
    best = (
        frame.sort_values(
            [
                "fold",
                "layer",
                "val_track_macro_f1",
                "val_track_accuracy",
                "val_clip_macro_f1",
            ],
            ascending=[True, True, False, False, False],
        )
        .groupby(["fold", "layer"], as_index=False)
        .first()
    )
    summary = (
        best.groupby("layer")
        .agg(
            val_macro_f1=("val_track_macro_f1", "mean"),
            val_accuracy=("val_track_accuracy", "mean"),
        )
        .reset_index()
    )
    plt.figure(figsize=(9, 4.5))
    plt.plot(
        summary["layer"],
        summary["val_macro_f1"],
        marker="o",
        label="Macro F1",
    )
    plt.plot(
        summary["layer"],
        summary["val_accuracy"],
        marker="s",
        label="Accuracy",
    )
    plt.xlabel("MERT hidden-state level")
    plt.ylabel("Mean validation score")
    plt.title("Expanded benchmark: layer selection")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
    layer: int,
    path: Path,
) -> dict[str, float]:
    import umap

    values, track_labels, _ = STUDY.track_mean_embeddings(
        embeddings[layer],
        labels,
        tracks,
    )
    standardized = StandardScaler().fit_transform(values)
    projections = (
        ("PCA", PCA(n_components=2).fit_transform(standardized)),
        (
            "t-SNE",
            TSNE(
                n_components=2,
                perplexity=6,
                init="pca",
                learning_rate="auto",
                random_state=RANDOM_SEED,
            ).fit_transform(standardized),
        ),
        (
            "UMAP",
            umap.UMAP(
                n_components=2,
                n_neighbors=6,
                min_dist=0.2,
                random_state=RANDOM_SEED,
            ).fit_transform(standardized),
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for index, (name, points) in enumerate(projections):
        frame = pd.DataFrame(
            {
                "x": points[:, 0],
                "y": points[:, 1],
                "raga": track_labels,
            }
        )
        sns.scatterplot(
            data=frame,
            x="x",
            y="y",
            hue="raga",
            s=80,
            alpha=0.85,
            ax=axes[index],
            legend=index == 2,
        )
        axes[index].set_title(name)
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        if index != 2 and axes[index].get_legend() is not None:
            axes[index].get_legend().remove()
    axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(
        f"Frozen MERT embeddings before classifier, layer {layer}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    encoded = LabelEncoder().fit_transform(track_labels)
    return {
        "layer": layer,
        "silhouette": float(
            STUDY.silhouette_score(standardized, encoded)
        ),
        "davies_bouldin": float(
            STUDY.davies_bouldin_score(standardized, encoded)
        ),
    }


def plot_head_training(
    histories: list[dict[str, Any]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for record in histories:
        frame = pd.DataFrame(record["history"])
        label = f"fold {record['fold'] + 1}"
        axes[0].plot(
            frame["epoch"],
            frame["train_loss"],
            label=f"{label} train",
            alpha=0.75,
        )
        axes[0].plot(
            frame["epoch"],
            frame["val_loss"],
            linestyle="--",
            label=f"{label} val",
            alpha=0.75,
        )
        axes[1].plot(
            frame["epoch"],
            frame["train_accuracy"],
            alpha=0.75,
        )
        axes[1].plot(
            frame["epoch"],
            frame["val_accuracy"],
            linestyle="--",
            alpha=0.75,
        )
    axes[0].set_title("Loss: solid train, dashed validation")
    axes[1].set_title("Accuracy: solid train, dashed validation")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Fixed regularized classifier across expanded folds")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def write_report(
    path: Path,
    args: argparse.Namespace,
    selected: dict[str, list[dict[str, str]]],
    overall: dict[str, Any],
    fold_frame: pd.DataFrame,
    cluster: dict[str, float],
    selected_layers: Counter,
) -> None:
    total_tracks = sum(len(items) for items in selected.values())
    lines = [
        "# Final Expanded Frozen-MERT Raga Benchmark",
        "",
        "## Final Model Choice",
        "",
        "MERT remains completely frozen. The experiment does not unfreeze "
        "additional transformer layers because the matched control showed no "
        "benefit from training layer 12.",
        "",
        "## Expanded Dataset",
        "",
        f"- Total original recordings: {total_tracks}",
        f"- Previous experiment: 25 recordings",
        f"- Increase: {total_tracks - 25} recordings "
        f"({100 * (total_tracks / 25 - 1):.1f}%)",
        f"- Four {args.segment_seconds:.0f}-second clips per recording",
        "- Five track-wise folds; every recording is tested exactly once",
        "- One validation recording per raga inside each outer fold",
        "",
        "| Raga | Original recordings |",
        "|---|---:|",
    ]
    for raga, rows in selected.items():
        lines.append(f"| {raga} | {len(rows)} |")
    lines.extend(
        [
            "",
            "## Models Evaluated",
            "",
            "1. Layer-wise logistic regression with validation-selected layer "
            "and C value. Selection uses validation recording-level macro F1.",
            "2. One fixed regularized neural classifier on the same selected "
            "layer: `768 -> 64 -> 5`, dropout 0.5, weight decay 0.1, label "
            "smoothing 0.1, learning rate 3e-4 and embedding noise 0.05.",
            "",
            "No neural-head hyperparameter sweep is performed on the expanded "
            "data. This reduces the risk of choosing a lucky configuration.",
            "",
            "## Overall Out-Of-Fold Results",
            "",
            "| Classifier | Clip accuracy | Track accuracy | Track macro F1 | "
            "Track top-3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, result in overall.items():
        lines.append(
            f"| {name} | {result['clip']['accuracy']:.3f} | "
            f"{result['track']['accuracy']:.3f} | "
            f"{result['track']['macro_f1']:.3f} | "
            f"{result['track']['top_3_accuracy']:.3f} |"
        )
    lines.extend(
        [
            "",
            "- Earlier optimized frozen MERT result on 25 recordings: 0.360",
            "- The old and expanded scores are not directly identical tests "
            "because the recording set has changed.",
            "",
            "## Fold And Overfitting Summary",
            "",
            f"- Selected layers by fold: {dict(selected_layers)}",
            f"- Fixed-head mean training accuracy: "
            f"{fold_frame['head_train_accuracy'].mean():.3f}",
            f"- Fixed-head mean validation accuracy: "
            f"{fold_frame['head_val_accuracy'].mean():.3f}",
            f"- Mean train-validation gap: "
            f"{fold_frame['head_accuracy_gap'].mean():.3f}",
            "",
            "## Embeddings",
            "",
            f"- Displayed layer: {cluster['layer']}",
            f"- Silhouette score: {cluster['silhouette']:.3f}",
            f"- Davies-Bouldin score: {cluster['davies_bouldin']:.3f}",
            "",
            "Negative or near-zero silhouette means that recordings of the "
            "same raga do not form clean frozen-embedding clusters.",
            "",
            "## Final Interpretation",
            "",
            "This is the final model for the current dataset: frozen MERT with "
            "a small external classifier. If performance remains modest, the "
            "next bottleneck is the number and diversity of original "
            "recordings, not a need to unfreeze more MERT layers. A high "
            "accuracy cannot be guaranteed from 31 recordings, but this "
            "protocol gives the most defensible estimate obtained in the "
            "project.",
            "",
            "The archive also contains out-of-fold predictions and the small "
            "fold-specific classifier checkpoints. MERT itself is not copied "
            "because it remains the unchanged public pretrained model.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def package(output_dir: Path) -> Path:
    path = output_dir / "final_expanded_frozen_mert_report.zip"
    path.unlink(missing_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in (
            output_dir / "REPORT.md",
            output_dir / "metrics",
            output_dir / "figures",
            output_dir / "models",
        ):
            if item.is_file():
                archive.write(
                    item,
                    item.relative_to(output_dir).as_posix(),
                )
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
        raise RuntimeError("Enable a Kaggle P100 or T4 GPU.")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    sns.set_theme(style="whitegrid")

    output_dir = Path(args.output_dir).resolve()
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    rows = BASE.collect_kaggle_tracks(
        Path(args.kaggle_data_root).resolve()
    )
    selected, all_counts = select_expanded_tracks(
        rows,
        args.num_ragas,
        args.minimum_tracks,
        args.maximum_tracks,
        args.exclude_raga,
    )
    folds = build_folds(selected)
    config = {
        "experiment": "expanded_frozen_mert_v1",
        "selected_tracks": {
            raga: [row["track_id"] for row in raga_rows]
            for raga, raga_rows in selected.items()
        },
        "segment_seconds": args.segment_seconds,
        "segments_per_track": args.segments_per_track,
        "model": MODEL_NAME,
    }
    cache_id = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = output_dir / "cache" / cache_id
    clips_dir = cache_dir / "clips"
    embedding_dir = cache_dir / "embeddings"
    manifest = CV.prepare_all_clips(
        selected,
        clips_dir,
        args.segment_seconds,
        args.segments_per_track,
        args.force_clips,
    )
    validate_manifest(
        manifest,
        selected,
        args.segments_per_track,
        args.segment_seconds,
    )
    print("Expanded preflight")
    print("All raga counts:", all_counts)
    print(
        "Selected:",
        {raga: len(items) for raga, items in selected.items()},
    )
    print("Total recordings:", sum(len(items) for items in selected.values()))
    print("Total clips:", len(manifest))

    data = CV.extract_all_embeddings(
        MODEL_KEY,
        MODEL_NAME,
        manifest,
        embedding_dir,
        args.batch_size,
        args.force_embeddings,
    )
    embeddings = data["embeddings"]
    string_labels = data["labels"]
    track_ids = data["tracks"]
    encoder = LabelEncoder().fit(string_labels)
    labels = encoder.transform(string_labels)
    num_classes = len(encoder.classes_)

    linear_oof = np.full(
        (len(labels), num_classes),
        np.nan,
        dtype=np.float32,
    )
    head_oof = np.full_like(linear_oof, np.nan)
    layer_rows = []
    fold_rows = []
    selected_layer_counts: Counter[int] = Counter()
    selected_histories = []
    for fold_index, fold in enumerate(folds):
        print(f"\nExpanded fold {fold_index + 1}/5")
        masks = track_masks(fold, track_ids)
        layer, c_value, tuning = select_layer(
            embeddings,
            labels,
            track_ids,
            masks,
        )
        selected_layer_counts[layer] += 1
        for row in tuning:
            layer_rows.append({"fold": fold_index, **row})
        trainval = masks["train"] | masks["val"]
        probe = make_probe(c_value)
        probe.fit(
            embeddings[layer, trainval],
            labels[trainval],
        )
        joblib.dump(
            {
                "model": probe,
                "selected_layer": layer,
                "classes": encoder.classes_,
            },
            models_dir / f"fold_{fold_index + 1}_logistic.joblib",
        )
        linear_oof[masks["test"]] = probe.predict_proba(
            embeddings[layer, masks["test"]]
        )

        model, scaler, history, best = STUDY.train_candidate(
            embeddings[layer, masks["train"]],
            labels[masks["train"]],
            embeddings[layer, masks["val"]],
            labels[masks["val"]],
            num_classes,
            HEAD_CONFIG,
            args.head_epochs,
            args.head_patience,
            RANDOM_SEED + fold_index,
            balanced_class_weights(
                labels[masks["train"]],
                num_classes,
            ),
        )
        del model, scaler
        final_model, final_scaler = STUDY.refit_candidate(
            embeddings[layer, trainval],
            labels[trainval],
            num_classes,
            HEAD_CONFIG,
            int(best["best_epoch"]),
            RANDOM_SEED + 100 + fold_index,
            balanced_class_weights(
                labels[trainval],
                num_classes,
            ),
        )
        head_oof[masks["test"]] = STUDY.head_probabilities(
            final_model,
            final_scaler,
            embeddings[layer, masks["test"]],
        )
        torch.save(
            {
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in final_model.state_dict().items()
                },
                "selected_layer": layer,
                "classes": encoder.classes_.tolist(),
                "config": HEAD_CONFIG,
                "input_dim": int(embeddings.shape[2]),
                "num_classes": num_classes,
            },
            models_dir / f"fold_{fold_index + 1}_regularized_head.pt",
        )
        joblib.dump(
            final_scaler,
            models_dir / f"fold_{fold_index + 1}_head_scaler.joblib",
        )
        del final_model, final_scaler
        selected_histories.append(
            {"fold": fold_index, "history": history}
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "selected_layer": layer,
                "selected_C": c_value,
                "train_tracks": len(fold["train"]),
                "val_tracks": len(fold["val"]),
                "test_tracks": len(fold["test"]),
                "head_best_epoch": best["best_epoch"],
                "head_train_accuracy": best["train_accuracy"],
                "head_val_accuracy": best["val_accuracy"],
                "head_accuracy_gap": best["accuracy_gap"],
                "head_train_loss": best["train_loss"],
                "head_val_loss": best["val_loss"],
            }
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if np.isnan(linear_oof).any() or np.isnan(head_oof).any():
        raise RuntimeError("One or more expanded folds are incomplete.")

    overall = {}
    for name, probabilities in (
        ("Logistic regression", linear_oof),
        ("Fixed regularized head", head_oof),
    ):
        clip = STUDY.prediction_metrics(
            labels,
            probabilities,
            num_classes,
        )
        track_probabilities, track_labels = STUDY.aggregate_tracks(
            probabilities,
            labels,
            track_ids,
        )
        track = STUDY.prediction_metrics(
            track_labels,
            track_probabilities,
            num_classes,
        )
        track["num_tracks"] = len(track_labels)
        overall[name] = {"clip": clip, "track": track}

    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(metrics_dir / "fold_results.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(
        metrics_dir / "layer_tuning.csv",
        index=False,
    )
    json_dump(metrics_dir / "overall_results.json", overall)
    np.savez_compressed(
        metrics_dir / "clip_oof_predictions.npz",
        labels=labels,
        tracks=track_ids,
        classes=encoder.classes_,
        linear_probabilities=linear_oof,
        head_probabilities=head_oof,
    )
    prediction_rows = []
    ordered_tracks = np.asarray(sorted(set(track_ids)), dtype=object)
    for classifier, probabilities in (
        ("Logistic regression", linear_oof),
        ("Fixed regularized head", head_oof),
    ):
        track_probabilities, track_labels = STUDY.aggregate_tracks(
            probabilities,
            labels,
            track_ids,
        )
        for index, track_id in enumerate(ordered_tracks):
            predicted = int(track_probabilities[index].argmax())
            prediction_rows.append(
                {
                    "classifier": classifier,
                    "track_id": track_id,
                    "true_raga": encoder.classes_[track_labels[index]],
                    "predicted_raga": encoder.classes_[predicted],
                    "correct": predicted == track_labels[index],
                    "confidence": float(
                        track_probabilities[index, predicted]
                    ),
                }
            )
    pd.DataFrame(prediction_rows).to_csv(
        metrics_dir / "track_oof_predictions.csv",
        index=False,
    )
    json_dump(metrics_dir / "selected_tracks.json", selected)
    json_dump(metrics_dir / "folds.json", folds)
    json_dump(
        metrics_dir / "experiment_variables.json",
        {
            **config,
            "cache_id": cache_id,
            "all_raga_counts": all_counts,
            "selected_raga_counts": {
                raga: len(items)
                for raga, items in selected.items()
            },
            "linear_C_values": LINEAR_C_VALUES,
            "head_config": HEAD_CONFIG,
            "head_epochs": args.head_epochs,
            "head_patience": args.head_patience,
            "random_seed": RANDOM_SEED,
        },
    )

    plot_distribution(
        selected,
        figures_dir / "expanded_dataset_distribution.png",
    )
    plot_layers(
        layer_rows,
        figures_dir / "expanded_layer_performance.png",
    )
    plot_confusion(
        labels,
        linear_oof,
        encoder.classes_,
        "Expanded frozen MERT: logistic regression",
        figures_dir / "expanded_linear_confusion.png",
    )
    plot_confusion(
        labels,
        head_oof,
        encoder.classes_,
        "Expanded frozen MERT: fixed regularized head",
        figures_dir / "expanded_head_confusion.png",
    )
    display_layer = selected_layer_counts.most_common(1)[0][0]
    cluster = plot_clusters(
        embeddings,
        string_labels,
        track_ids,
        display_layer,
        figures_dir / "expanded_embedding_clusters.png",
    )
    json_dump(metrics_dir / "cluster_metrics.json", cluster)
    plot_head_training(
        selected_histories,
        figures_dir / "expanded_head_training_curves.png",
    )
    write_report(
        output_dir / "REPORT.md",
        args,
        selected,
        overall,
        fold_frame,
        cluster,
        selected_layer_counts,
    )
    archive = package(output_dir)
    print("\nFinal expanded frozen benchmark completed.")
    print(json.dumps(overall, indent=2))
    print(f"Report: {output_dir / 'REPORT.md'}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
