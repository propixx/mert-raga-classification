#!/usr/bin/env python3
"""Five-fold, track-wise MERT vs CultureMERT benchmark for Saraga Carnatic.

This is the main experiment after the small single-split pilot. It extracts
embeddings once for 25 balanced source recordings, then rotates every recording
through an unseen test fold. The MERT backbones remain frozen; only a linear
probe and a small external neural classifier are trained.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
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
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


MODELS = {
    "mert_95m": "m-a-p/MERT-v1-95M",
    "culturemert_95m": "ntua-slp/CultureMERT-95M",
}
SAMPLE_RATE = 24000
RANDOM_SEED = 42
LINEAR_C_VALUES = (0.01, 0.1, 1.0, 10.0)
HEAD_CONFIGS = (
    {"lr": 1e-3, "dropout": 0.1},
    {"lr": 1e-3, "dropout": 0.3},
    {"lr": 3e-4, "dropout": 0.1},
    {"lr": 3e-4, "dropout": 0.3},
)


def load_base_module() -> ModuleType:
    path = Path(__file__).with_name("10_balanced_benchmark.py")
    if not path.exists():
        raise RuntimeError(
            f"Required helper script is missing: {path}. Keep scripts 10 and 11 together."
        )
    spec = importlib.util.spec_from_file_location("balanced_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="saraga_carnatic")
    parser.add_argument("--data-home", default="data/raw/saraga_crossval")
    parser.add_argument("--kaggle-data-root", default=None)
    parser.add_argument("--output-dir", default="crossval_outputs")
    parser.add_argument("--num-ragas", type=int, default=5)
    parser.add_argument("--tracks-per-raga", type=int, default=5)
    parser.add_argument("--exclude-raga", action="append", default=[])
    parser.add_argument("--segments-per-track", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--head-patience", type=int, default=6)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-clips", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "five_fold_track_cross_validation_v1",
        "dataset": args.dataset,
        "source_mode": "kaggle_input" if args.kaggle_data_root else "mirdata",
        "num_ragas": args.num_ragas,
        "tracks_per_raga": args.tracks_per_raga,
        "excluded_ragas": sorted(args.exclude_raga),
        "segments_per_track": args.segments_per_track,
        "segment_seconds": args.segment_seconds,
        "sample_rate": SAMPLE_RATE,
        "folds": args.tracks_per_raga,
        "random_seed": RANDOM_SEED,
        "models": MODELS,
        "linear_c_values": LINEAR_C_VALUES,
        "head_configs": HEAD_CONFIGS,
    }


def fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def build_folds(
    selected: dict[str, list[dict[str, str]]],
) -> list[dict[str, list[dict[str, str]]]]:
    fold_count = len(next(iter(selected.values())))
    if fold_count < 5 or any(len(rows) != fold_count for rows in selected.values()):
        raise RuntimeError("Cross-validation requires the same five tracks for every raga.")

    folds = []
    all_test_ids: list[str] = []
    for fold_index in range(fold_count):
        split = {"train": [], "val": [], "test": []}
        for _, tracks in selected.items():
            test_index = fold_index
            val_index = (fold_index + 1) % fold_count
            for index, row in enumerate(tracks):
                if index == test_index:
                    split["test"].append(row)
                    all_test_ids.append(row["track_id"])
                elif index == val_index:
                    split["val"].append(row)
                else:
                    split["train"].append(row)

        ids = {name: {row["track_id"] for row in rows} for name, rows in split.items()}
        assert ids["train"].isdisjoint(ids["val"])
        assert ids["train"].isdisjoint(ids["test"])
        assert ids["val"].isdisjoint(ids["test"])
        expected_classes = set(selected)
        for rows in split.values():
            if {row["raga"] for row in rows} != expected_classes:
                raise RuntimeError(f"Fold {fold_index} lost a raga.")
        folds.append(split)

    expected_ids = {
        row["track_id"] for tracks in selected.values() for row in tracks
    }
    if Counter(all_test_ids) != Counter({track_id: 1 for track_id in expected_ids}):
        raise RuntimeError("Every source track must appear in the test set exactly once.")
    return folds


def prepare_all_clips(
    selected: dict[str, list[dict[str, str]]],
    clips_dir: Path,
    segment_seconds: float,
    segments_per_track: int,
    force: bool,
) -> list[dict[str, Any]]:
    manifest_path = clips_dir / "all_clips.json"
    if manifest_path.exists() and not force:
        print(f"Using cached clips: {manifest_path}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    if force and clips_dir.exists():
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    samples_needed = int(round(segment_seconds * SAMPLE_RATE))
    manifest: list[dict[str, Any]] = []
    failures = []

    rows = [row for tracks in selected.values() for row in tracks]
    for row in tqdm(rows, desc="Creating 30-second clips"):
        try:
            duration = float(librosa.get_duration(path=row["audio_path"]))
            offsets = BASE.segment_offsets(
                duration,
                segment_seconds,
                max(segments_per_track * 3, segments_per_track),
            )
            saved = 0
            for offset in offsets:
                audio, _ = librosa.load(
                    row["audio_path"],
                    sr=SAMPLE_RATE,
                    mono=True,
                    offset=float(offset),
                    duration=segment_seconds,
                )
                if len(audio) < samples_needed:
                    audio = np.pad(audio, (0, samples_needed - len(audio)))
                else:
                    audio = audio[:samples_needed]
                if float(np.max(np.abs(audio))) < 0.01:
                    continue

                path = clips_dir / f"{BASE.safe_name(row['track_id'])}_s{saved:02d}.wav"
                sf.write(path, audio, SAMPLE_RATE)
                manifest.append(
                    {
                        "path": str(path.resolve()),
                        "track_id": row["track_id"],
                        "raga": row["raga"],
                        "offset_seconds": float(offset),
                    }
                )
                saved += 1
                if saved == segments_per_track:
                    break
            if saved != segments_per_track:
                failures.append({**row, "clips_found": saved})
        except Exception as exc:
            failures.append({**row, "clips_found": 0, "error": str(exc)})

    if failures:
        json_dump(clips_dir / "clip_failures.json", failures)
        raise RuntimeError(
            "Some tracks could not provide four valid clips. "
            f"See {clips_dir / 'clip_failures.json'}."
        )
    json_dump(manifest_path, manifest)
    return manifest


def validate_clips(
    manifest: list[dict[str, Any]],
    selected: dict[str, list[dict[str, str]]],
    args: argparse.Namespace,
) -> None:
    expected_tracks = args.num_ragas * args.tracks_per_raga
    expected_clips = expected_tracks * args.segments_per_track
    if len(manifest) != expected_clips:
        raise RuntimeError(f"Expected {expected_clips} clips, found {len(manifest)}.")

    per_track = Counter(item["track_id"] for item in manifest)
    per_raga = Counter(item["raga"] for item in manifest)
    if set(per_track.values()) != {args.segments_per_track}:
        raise RuntimeError(f"Unequal clips per track: {per_track}")
    expected_per_raga = args.tracks_per_raga * args.segments_per_track
    if set(per_raga) != set(selected) or set(per_raga.values()) != {expected_per_raga}:
        raise RuntimeError(f"Unequal clips per raga: {per_raga}")

    expected_frames = int(round(args.segment_seconds * SAMPLE_RATE))
    for item in manifest:
        info = sf.info(item["path"])
        if info.samplerate != SAMPLE_RATE or info.frames != expected_frames:
            raise RuntimeError(
                f"Invalid clip {item['path']}: {info.samplerate} Hz, {info.frames} frames."
            )


def plot_dataset(selected: dict[str, list[dict[str, str]]], manifest: list[dict[str, Any]], path: Path) -> None:
    counts = Counter(item["raga"] for item in manifest)
    frame = pd.DataFrame(
        {
            "raga": list(counts),
            "clips": [counts[name] for name in counts],
            "tracks": [len(selected[name]) for name in counts],
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=frame, x="raga", y="tracks", ax=axes[0], color="#4878A8")
    sns.barplot(data=frame, x="raga", y="clips", ax=axes[1], color="#D65F5F")
    axes[0].set_title("Original recordings per raga")
    axes[1].set_title("30-second clips per raga")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def extract_all_embeddings(
    model_key: str,
    hf_name: str,
    manifest: list[dict[str, Any]],
    cache_dir: Path,
    batch_size: int,
    force: bool,
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model_key}_all.npz"
    progress_path = cache_dir / f"{model_key}_progress.npz"
    if cache_path.exists() and not force:
        print(f"Using cached embeddings: {cache_path}")
        with np.load(cache_path, allow_pickle=True) as cached:
            return {
                key: cached[key].copy()
                for key in ("embeddings", "labels", "tracks")
            }
    if force:
        progress_path.unlink(missing_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Loading {hf_name} on {device} with {dtype}")
    try:
        extractor = AutoFeatureExtractor.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            hf_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).eval().to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {hf_name}. Enable Kaggle Internet and rerun. {exc}"
        ) from exc

    layer_batches: list[list[np.ndarray]] | None = None
    start_index = 0
    if progress_path.exists():
        with np.load(progress_path) as progress:
            partial = progress["embeddings"].copy()
            start_index = int(progress["completed"])
        if (
            partial.ndim != 3
            or partial.shape[0] != 13
            or partial.shape[1] != start_index
            or start_index > len(manifest)
        ):
            raise RuntimeError(
                f"Invalid embedding progress file: {progress_path}. "
                "Delete it or rerun with --force-embeddings."
            )
        layer_batches = [[partial[layer]] for layer in range(partial.shape[0])]
        print(f"Resuming {model_key} after {start_index}/{len(manifest)} clips.")

    for start in tqdm(
        range(start_index, len(manifest), batch_size),
        desc=f"{model_key} embeddings",
    ):
        batch_items = manifest[start : start + batch_size]
        waveforms = BASE.load_audio_batch(batch_items, extractor.sampling_rate)
        inputs = extractor(
            waveforms,
            sampling_rate=extractor.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        model_inputs: dict[str, Any] = {
            "input_values": inputs["input_values"].to(device=device, dtype=dtype),
            "output_hidden_states": True,
        }
        if inputs.get("attention_mask") is not None:
            model_inputs["attention_mask"] = inputs["attention_mask"].to(device)
        try:
            with torch.inference_mode():
                hidden_states = model(**model_inputs).hidden_states
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "GPU memory was exhausted. Use batch size 1; completed model caches are reused."
            ) from exc
        if hidden_states is None or len(hidden_states) != 13:
            raise RuntimeError(f"{hf_name} did not return 13 hidden-state levels.")
        if layer_batches is None:
            layer_batches = [[] for _ in hidden_states]
        for layer, hidden in enumerate(hidden_states):
            layer_batches[layer].append(
                hidden.float().mean(dim=1).cpu().numpy().astype(np.float32)
            )
        completed = min(start + batch_size, len(manifest))
        if completed % 10 == 0 or completed == len(manifest):
            partial = np.stack(
                [np.concatenate(chunks, axis=0) for chunks in layer_batches],
                axis=0,
            )
            np.savez_compressed(
                progress_path,
                embeddings=partial,
                completed=np.array(completed),
            )
            print(f"Checkpointed {model_key}: {completed}/{len(manifest)} clips")

    if layer_batches is None:
        raise RuntimeError("No embeddings were extracted.")
    output = {
        "embeddings": np.stack(
            [np.concatenate(chunks, axis=0) for chunks in layer_batches],
            axis=0,
        ),
        "labels": np.array([item["raga"] for item in manifest], dtype=object),
        "tracks": np.array([item["track_id"] for item in manifest], dtype=object),
    }
    np.savez_compressed(cache_path, **output)
    progress_path.unlink(missing_ok=True)
    print(f"Saved embeddings: {cache_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


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


def metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
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
    encoded_labels: np.ndarray,
    tracks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped_probabilities = []
    grouped_labels = []
    grouped_tracks = []
    for track in sorted(set(tracks)):
        mask = tracks == track
        grouped_probabilities.append(probabilities[mask].mean(axis=0))
        grouped_labels.append(Counter(encoded_labels[mask]).most_common(1)[0][0])
        grouped_tracks.append(track)
    return (
        np.stack(grouped_probabilities),
        np.asarray(grouped_labels),
        np.asarray(grouped_tracks, dtype=object),
    )


class ExternalHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def train_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    hidden_dim: int,
    lr: float,
    dropout: float,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[ExternalHead, StandardScaler, list[dict[str, float]], dict[str, float]]:
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
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(16, len(dataset)),
        shuffle=True,
        generator=generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExternalHead(train_x.shape[1], hidden_dim, num_classes, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    val_tensor = torch.from_numpy(val_scaled).to(device)

    history = []
    best_rank = None
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        train_pred = []
        train_true = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_y)
            train_pred.extend(logits.argmax(1).detach().cpu().numpy())
            train_true.extend(batch_y.detach().cpu().numpy())

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_tensor)
            val_prob = torch.softmax(val_logits, dim=1).cpu().numpy()
        val_pred = val_prob.argmax(axis=1)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(dataset),
            "train_accuracy": float(accuracy_score(train_true, train_pred)),
            "val_accuracy": float(accuracy_score(val_y, val_pred)),
            "val_macro_f1": float(
                f1_score(val_y, val_pred, average="macro", zero_division=0)
            ),
        }
        history.append(row)
        rank = (row["val_macro_f1"], row["val_accuracy"], -epoch)
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

    if best_state is None or best_rank is None:
        raise RuntimeError("External-head training did not complete.")
    model.load_state_dict(best_state)
    model.eval()
    best = {
        "best_epoch": best_epoch,
        "val_macro_f1": float(best_rank[0]),
        "val_accuracy": float(best_rank[1]),
        "best_train_accuracy": float(history[best_epoch - 1]["train_accuracy"]),
    }
    return model, scaler, history, best


def head_probabilities(
    model: ExternalHead,
    scaler: StandardScaler,
    values: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    scaled = scaler.transform(values).astype(np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(scaled).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def retrain_head(
    values: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    hidden_dim: int,
    lr: float,
    dropout: float,
    epochs: int,
    seed: int,
) -> tuple[ExternalHead, StandardScaler]:
    """Refit the chosen head on train+validation for the selected epoch count."""
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
    model = ExternalHead(values.shape[1], hidden_dim, num_classes, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(max(1, epochs)):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval(), scaler


def masks_for_fold(
    fold: dict[str, list[dict[str, str]]],
    tracks: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        split: np.isin(tracks, [row["track_id"] for row in rows])
        for split, rows in fold.items()
    }


def plot_confusion(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: np.ndarray,
    title: str,
    path: Path,
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=35,
        colorbar=False,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_layer_summary(rows: list[dict[str, Any]], model_key: str, path: Path) -> None:
    frame = pd.DataFrame(rows)
    best_per_fold_layer = (
        frame.sort_values(
            ["fold", "layer", "val_macro_f1", "val_accuracy"],
            ascending=[True, True, False, False],
        )
        .groupby(["fold", "layer"], as_index=False)
        .first()
    )
    summary = (
        best_per_fold_layer.groupby("layer")
        .agg(
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", "std"),
            val_accuracy_mean=("val_accuracy", "mean"),
        )
        .reset_index()
    )
    plt.figure(figsize=(9, 4.8))
    plt.errorbar(
        summary["layer"],
        summary["val_macro_f1_mean"],
        yerr=summary["val_macro_f1_std"].fillna(0),
        marker="o",
        capsize=3,
        label="validation macro F1",
    )
    plt.plot(
        summary["layer"],
        summary["val_accuracy_mean"],
        "s-",
        label="validation accuracy",
    )
    plt.xlabel("Hidden-state layer")
    plt.ylabel("Mean score across five folds")
    plt.title(f"{model_key}: layer-wise frozen probe")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_head_tuning(rows: list[dict[str, Any]], model_key: str, path: Path) -> None:
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["lr", "dropout"])["val_macro_f1"]
        .mean()
        .reset_index()
        .pivot(index="dropout", columns="lr", values="val_macro_f1")
    )
    plt.figure(figsize=(6.5, 4.5))
    sns.heatmap(summary, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title(f"{model_key}: external-head validation macro F1")
    plt.xlabel("Learning rate")
    plt.ylabel("Dropout")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def evaluate_model(
    model_key: str,
    data: dict[str, np.ndarray],
    folds: list[dict[str, list[dict[str, str]]]],
    args: argparse.Namespace,
    figures_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    embeddings = data["embeddings"]
    labels = data["labels"]
    tracks = data["tracks"]
    encoder = LabelEncoder().fit(labels)
    encoded = encoder.transform(labels)
    num_classes = len(encoder.classes_)
    sample_count = len(labels)

    linear_oof = np.full((sample_count, num_classes), np.nan, dtype=np.float32)
    neural_oof = np.full((sample_count, num_classes), np.nan, dtype=np.float32)
    layer_tuning_rows = []
    head_tuning_rows = []
    fold_results = []
    selected_head_histories = []

    for fold_index, fold in enumerate(folds):
        print(f"\n{model_key}: fold {fold_index + 1}/{len(folds)}")
        masks = masks_for_fold(fold, tracks)
        train_y, val_y, test_y = (
            encoded[masks["train"]],
            encoded[masks["val"]],
            encoded[masks["test"]],
        )

        best_linear = None
        for layer in range(embeddings.shape[0]):
            for c_value in LINEAR_C_VALUES:
                probe = make_probe(c_value)
                probe.fit(embeddings[layer, masks["train"]], train_y)
                val_pred = probe.predict(embeddings[layer, masks["val"]])
                row = {
                    "fold": fold_index,
                    "layer": layer,
                    "C": c_value,
                    "val_accuracy": float(accuracy_score(val_y, val_pred)),
                    "val_macro_f1": float(
                        f1_score(val_y, val_pred, average="macro", zero_division=0)
                    ),
                }
                layer_tuning_rows.append(row)
                rank = (
                    row["val_macro_f1"],
                    row["val_accuracy"],
                    -abs(math.log10(c_value)),
                    -layer,
                )
                if best_linear is None or rank > best_linear["rank"]:
                    best_linear = {**row, "rank": rank}

        assert best_linear is not None
        layer = int(best_linear["layer"])
        c_value = float(best_linear["C"])
        trainval_mask = masks["train"] | masks["val"]
        final_probe = make_probe(c_value)
        final_probe.fit(embeddings[layer, trainval_mask], encoded[trainval_mask])
        linear_prob = final_probe.predict_proba(embeddings[layer, masks["test"]])
        linear_oof[masks["test"]] = linear_prob
        linear_fold_metrics = metrics(
            test_y,
            linear_prob.argmax(axis=1),
            linear_prob,
            num_classes,
        )
        joblib.dump(
            final_probe,
            models_dir / f"{model_key}_fold{fold_index}_linear.joblib",
        )

        best_head = None
        for config_index, config in enumerate(HEAD_CONFIGS):
            model, scaler, history, validation = train_head(
                embeddings[layer, masks["train"]],
                train_y,
                embeddings[layer, masks["val"]],
                val_y,
                num_classes,
                args.head_hidden_dim,
                config["lr"],
                config["dropout"],
                args.head_epochs,
                args.head_patience,
                RANDOM_SEED + fold_index * 10 + config_index,
            )
            row = {
                "fold": fold_index,
                "layer": layer,
                "lr": config["lr"],
                "dropout": config["dropout"],
                **validation,
            }
            head_tuning_rows.append(row)
            rank = (
                row["val_macro_f1"],
                row["val_accuracy"],
                -row["dropout"],
                -abs(math.log10(row["lr"]) + 3),
            )
            if best_head is None or rank > best_head["rank"]:
                best_head = {
                    "rank": rank,
                    "model": model,
                    "scaler": scaler,
                    "history": history,
                    "config": config,
                    "validation": validation,
                }
            else:
                del model

        assert best_head is not None
        selected_epoch = int(best_head["validation"]["best_epoch"])
        final_head, final_scaler = retrain_head(
            embeddings[layer, trainval_mask],
            encoded[trainval_mask],
            num_classes,
            args.head_hidden_dim,
            best_head["config"]["lr"],
            best_head["config"]["dropout"],
            selected_epoch,
            RANDOM_SEED + 1000 + fold_index,
        )
        neural_prob = head_probabilities(
            final_head,
            final_scaler,
            embeddings[layer, masks["test"]],
        )
        neural_oof[masks["test"]] = neural_prob
        neural_fold_metrics = metrics(
            test_y,
            neural_prob.argmax(axis=1),
            neural_prob,
            num_classes,
        )
        selected_head_histories.append(
            {
                "fold": fold_index,
                "history": best_head["history"],
                "config": best_head["config"],
            }
        )
        joblib.dump(
            final_scaler,
            models_dir / f"{model_key}_fold{fold_index}_head_scaler.joblib",
        )
        torch.save(
            {
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in final_head.state_dict().items()
                },
                "input_dim": embeddings.shape[-1],
                "hidden_dim": args.head_hidden_dim,
                "num_classes": num_classes,
                "dropout": best_head["config"]["dropout"],
                "lr": best_head["config"]["lr"],
                "layer": layer,
                "validation": best_head["validation"],
                "refit_on_train_and_validation": True,
            },
            models_dir / f"{model_key}_fold{fold_index}_external_head.pt",
        )

        fold_results.append(
            {
                "fold": fold_index,
                "selected_layer": layer,
                "selected_C": c_value,
                "selected_head_lr": best_head["config"]["lr"],
                "selected_head_dropout": best_head["config"]["dropout"],
                "head_best_epoch": best_head["validation"]["best_epoch"],
                "head_train_accuracy_at_best": best_head["validation"][
                    "best_train_accuracy"
                ],
                "head_val_accuracy": best_head["validation"]["val_accuracy"],
                "head_val_macro_f1": best_head["validation"]["val_macro_f1"],
                "linear_test": linear_fold_metrics,
                "external_head_test": neural_fold_metrics,
            }
        )
        del best_head["model"], final_head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if np.isnan(linear_oof).any() or np.isnan(neural_oof).any():
        raise RuntimeError("Cross-validation did not generate one prediction per clip.")

    linear_pred = linear_oof.argmax(axis=1)
    neural_pred = neural_oof.argmax(axis=1)
    linear_clip = metrics(encoded, linear_pred, linear_oof, num_classes)
    neural_clip = metrics(encoded, neural_pred, neural_oof, num_classes)
    linear_track_prob, track_labels, track_ids = aggregate_tracks(
        linear_oof, encoded, tracks
    )
    neural_track_prob, neural_track_labels, _ = aggregate_tracks(
        neural_oof, encoded, tracks
    )
    linear_track = metrics(
        track_labels,
        linear_track_prob.argmax(axis=1),
        linear_track_prob,
        num_classes,
    )
    neural_track = metrics(
        neural_track_labels,
        neural_track_prob.argmax(axis=1),
        neural_track_prob,
        num_classes,
    )
    linear_track["num_tracks"] = len(track_ids)
    neural_track["num_tracks"] = len(track_ids)

    selected_layers = [row["selected_layer"] for row in fold_results]
    display_layer = Counter(selected_layers).most_common(1)[0][0]
    scaled = StandardScaler().fit_transform(embeddings[display_layer])
    cluster = {
        "display_layer": int(display_layer),
        "silhouette": float(silhouette_score(scaled, encoded)),
        "davies_bouldin": float(davies_bouldin_score(scaled, encoded)),
    }

    plot_confusion(
        encoded,
        linear_pred,
        encoder.classes_,
        f"{model_key}: five-fold out-of-fold linear probe",
        figures_dir / f"{model_key}_crossval_linear_confusion.png",
    )
    plot_confusion(
        encoded,
        neural_pred,
        encoder.classes_,
        f"{model_key}: five-fold out-of-fold external head",
        figures_dir / f"{model_key}_crossval_external_head_confusion.png",
    )
    plot_layer_summary(
        layer_tuning_rows,
        model_key,
        figures_dir / f"{model_key}_crossval_layer_performance.png",
    )
    plot_head_tuning(
        head_tuning_rows,
        model_key,
        figures_dir / f"{model_key}_head_hyperparameters.png",
    )
    BASE.plot_tsne(
        embeddings[display_layer],
        labels,
        f"{model_key}: all clips, layer {display_layer}",
        figures_dir / f"{model_key}_crossval_tsne.png",
    )
    joblib.dump(encoder, models_dir / f"{model_key}_label_encoder.joblib")

    return {
        "classes": list(encoder.classes_),
        "fold_results": fold_results,
        "layer_tuning": layer_tuning_rows,
        "head_tuning": head_tuning_rows,
        "selected_head_histories": selected_head_histories,
        "linear": {"clip_metrics": linear_clip, "track_metrics": linear_track},
        "external_head": {
            "clip_metrics": neural_clip,
            "track_metrics": neural_track,
        },
        "cluster_metrics": cluster,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def write_report(
    path: Path,
    args: argparse.Namespace,
    selected: dict[str, list[dict[str, str]]],
    results: dict[str, Any],
) -> None:
    lines = [
        "# Five-Fold MERT vs CultureMERT Raga Benchmark",
        "",
        "## Question",
        "",
        "Do frozen CultureMERT representations transfer better than general MERT "
        "representations to a small Carnatic raga-classification problem?",
        "",
        "## Method",
        "",
        f"- {args.num_ragas} ragas and {args.tracks_per_raga} original recordings per raga",
        f"- {args.segments_per_track} clips of {args.segment_seconds:.0f} seconds from each recording",
        f"- {args.tracks_per_raga}-fold rotation: 3 train, 1 validation and 1 test track per raga",
        "- Every original recording is unseen test data exactly once",
        "- MERT and CultureMERT remain frozen",
        "- Linear-probe sweep: all 13 layers and C in 0.01, 0.1, 1 and 10",
        "- External-head sweep: dropout 0.1/0.3 and learning rate 1e-3/3e-4",
        "",
        "Selected ragas: " + ", ".join(selected),
        "",
        "## Overall Out-of-Fold Results",
        "",
        "| Model | Classifier | Clip accuracy | Clip macro F1 | Clip top-3 | Track accuracy | Track macro F1 | Track top-3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, result in results.items():
        for key, label in (
            ("linear", "Linear probe"),
            ("external_head", "External neural head"),
        ):
            clip = result[key]["clip_metrics"]
            track = result[key]["track_metrics"]
            lines.append(
                f"| {model_key} | {label} | {clip['accuracy']:.3f} | "
                f"{clip['macro_f1']:.3f} | {clip['top_3_accuracy']:.3f} | "
                f"{track['accuracy']:.3f} | {track['macro_f1']:.3f} | "
                f"{track['top_3_accuracy']:.3f} |"
            )

    lines.extend(["", "## Fold Stability", ""])
    for model_key, result in results.items():
        for classifier_key, label in (
            ("linear_test", "linear"),
            ("external_head_test", "external head"),
        ):
            values = [
                row[classifier_key]["accuracy"] for row in result["fold_results"]
            ]
            mean, std = mean_std(values)
            lines.append(
                f"- {model_key} {label}: fold clip accuracy {mean:.3f} ± {std:.3f}"
            )

    lines.extend(["", "## Layer And Hyperparameter Observations", ""])
    for model_key, result in results.items():
        layer_counts = Counter(
            row["selected_layer"] for row in result["fold_results"]
        )
        c_counts = Counter(row["selected_C"] for row in result["fold_results"])
        head_counts = Counter(
            (row["selected_head_lr"], row["selected_head_dropout"])
            for row in result["fold_results"]
        )
        train_mean = np.mean(
            [
                row["head_train_accuracy_at_best"]
                for row in result["fold_results"]
            ]
        )
        val_mean = np.mean(
            [row["head_val_accuracy"] for row in result["fold_results"]]
        )
        lines.extend(
            [
                f"### {model_key}",
                "",
                f"- Selected layers across folds: {dict(layer_counts)}",
                f"- Selected linear C values: {dict(c_counts)}",
                f"- Selected external-head (learning rate, dropout): {dict(head_counts)}",
                f"- External head mean train accuracy at selected epoch: {train_mean:.3f}",
                f"- External head mean validation accuracy: {val_mean:.3f}",
                "",
            ]
        )

    mert = results["mert_95m"]["linear"]["track_metrics"]["accuracy"]
    culture = results["culturemert_95m"]["linear"]["track_metrics"]["accuracy"]
    lines.extend(
        [
            "## Reading The Result",
            "",
            f"CultureMERT minus MERT track-level linear-probe accuracy is "
            f"**{culture - mert:+.3f}**.",
            "",
            "This cross-validation result is more trustworthy than the earlier single "
            "split because all 25 recordings contribute to testing. It is still a small "
            "study: clips from the same recording are related observations, so track-level "
            "metrics are the main result.",
            "",
            "The external head can fit the training embeddings more strongly than the "
            "linear probe. A large train-validation gap is evidence of overfitting, not "
            "evidence that the frozen backbone improved. Because the backbones are never "
            "updated, this experiment also avoids catastrophic forgetting, but it cannot "
            "adapt their representations to raga-specific phrases.",
            "",
            "## Limitations And Next Step",
            "",
            "- Five recordings per raga are still too few for a broad conclusion.",
            "- Thirty seconds gives useful context but not a complete raga development.",
            "- Performer, composition, tonic and recording conditions may remain confounds.",
            "- The next research step is to add more independently recorded tracks before "
            "attempting partial backbone fine-tuning.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def package_results(output_dir: Path) -> Path:
    archive_path = output_dir / "mert_culturemert_5fold_results.zip"
    archive_path.unlink(missing_ok=True)
    include = [
        output_dir / "REPORT.md",
        output_dir / "figures",
        output_dir / "metrics",
        output_dir / "models",
    ]
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in include:
            if item.is_file():
                archive.write(item, item.relative_to(output_dir).as_posix())
            elif item.is_dir():
                for path in sorted(item.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(output_dir).as_posix())
    return archive_path


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    sns.set_theme(style="whitegrid")

    if args.tracks_per_raga != 5:
        raise RuntimeError("This experiment is designed for exactly five folds/tracks.")
    if args.segment_seconds < 30:
        raise RuntimeError("Use 30-second clips for the main experiment.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A Kaggle GPU is required. Enable a P100 or T4 accelerator and rerun."
        )

    output_dir = Path(args.output_dir).resolve()
    config = configuration(args)
    cache_id = fingerprint(config)
    cache_root = output_dir / "cache" / cache_id
    clips_dir = cache_root / "clips"
    embedding_dir = cache_root / "embeddings"
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    models_dir = output_dir / "models"
    for directory in (figures_dir, metrics_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cache fingerprint: {cache_id}")
    json_dump(metrics_dir / "experiment_config.json", {**config, "fingerprint": cache_id})

    if args.kaggle_data_root:
        data_root = Path(args.kaggle_data_root).resolve()
        source_counts = BASE.kaggle_file_counts(data_root)
        rows = BASE.collect_kaggle_tracks(data_root)
    else:
        data_home = Path(args.data_home).resolve()
        data_home.mkdir(parents=True, exist_ok=True)
        dataset = BASE.mirdata.initialize(args.dataset, data_home=str(data_home))
        if not args.skip_download:
            dataset.download()
        dataset.validate()
        rows = BASE.collect_tracks(dataset, args.dataset)
        source_counts = {"audio_files": len(rows), "metadata_files": len(rows)}

    selected = BASE.choose_balanced_tracks(
        rows,
        args.num_ragas,
        args.tracks_per_raga,
        args.exclude_raga,
    )
    folds = build_folds(selected)
    print("Selected ragas:")
    for raga, tracks in selected.items():
        print(f"  {raga}: {len(tracks)} tracks")
    print(
        f"Source scan: {source_counts['audio_files']} audio, "
        f"{source_counts['metadata_files']} metadata, {len(rows)} labeled tracks"
    )
    json_dump(metrics_dir / "selected_tracks.json", selected)
    json_dump(metrics_dir / "crossval_folds.json", folds)

    manifest = prepare_all_clips(
        selected,
        clips_dir,
        args.segment_seconds,
        args.segments_per_track,
        args.force_clips,
    )
    validate_clips(manifest, selected, args)
    json_dump(metrics_dir / "clip_manifest.json", manifest)
    plot_dataset(selected, manifest, figures_dir / "dataset_distribution.png")

    embeddings = {}
    for model_key, hf_name in MODELS.items():
        embeddings[model_key] = extract_all_embeddings(
            model_key,
            hf_name,
            manifest,
            embedding_dir,
            args.batch_size,
            args.force_embeddings,
        )

    results = {}
    for model_key, data in embeddings.items():
        results[model_key] = evaluate_model(
            model_key,
            data,
            folds,
            args,
            figures_dir,
            models_dir,
        )
        json_dump(metrics_dir / f"{model_key}_results.json", results[model_key])

    json_dump(metrics_dir / "crossval_results.json", results)
    summary_rows = []
    for model_key, result in results.items():
        for classifier in ("linear", "external_head"):
            summary_rows.append(
                {
                    "model": model_key,
                    "classifier": classifier,
                    **{
                        f"clip_{key}": value
                        for key, value in result[classifier]["clip_metrics"].items()
                    },
                    **{
                        f"track_{key}": value
                        for key, value in result[classifier]["track_metrics"].items()
                    },
                    **result["cluster_metrics"],
                }
            )
        pd.DataFrame(result["layer_tuning"]).to_csv(
            metrics_dir / f"{model_key}_layer_tuning.csv", index=False
        )
        pd.DataFrame(result["head_tuning"]).to_csv(
            metrics_dir / f"{model_key}_head_tuning.csv", index=False
        )
        pd.DataFrame(result["fold_results"]).to_json(
            metrics_dir / f"{model_key}_fold_results.json",
            orient="records",
            indent=2,
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(metrics_dir / "crossval_summary.csv", index=False)
    write_report(output_dir / "REPORT.md", args, selected, results)
    archive = package_results(output_dir)

    print("\nFive-fold experiment completed.")
    print(summary.to_string(index=False))
    print(f"Report: {output_dir / 'REPORT.md'}")
    print(f"Results: {archive}")


if __name__ == "__main__":
    main()
