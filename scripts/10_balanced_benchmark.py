#!/usr/bin/env python3
"""Balanced, leakage-safe MERT vs CultureMERT benchmark for Saraga.

This script is intended for a Colab/Kaggle T4 run. It keeps the experiment
small enough for a project-selection task while producing meaningful outputs:

- balanced raga selection
- train/validation/test split by original track
- embeddings from all 13 layers of both 95M models
- layer-wise linear probes
- a small external neural classifier with early stopping
- clip-level and track-level top-1/top-3 metrics
- macro F1, confusion matrices, t-SNE, silhouette and Davies-Bouldin scores
- a generated Markdown report and a downloadable result zip
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import librosa
import matplotlib.pyplot as plt
import mirdata
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
    top_k_accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


MODELS = {
    "mert_95m": "m-a-p/MERT-v1-95M",
    "culturemert_95m": "ntua-slp/CultureMERT-95M",
}
SAMPLE_RATE = 24000
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="saraga_carnatic", choices=["saraga_carnatic", "saraga_hindustani"])
    parser.add_argument("--data-home", default="data/raw/saraga_benchmark")
    parser.add_argument(
        "--kaggle-data-root",
        default=None,
        help="Attached Kaggle Saraga root containing numbered folders with JSON and MP3 files",
    )
    parser.add_argument("--output-dir", default="benchmark_outputs")
    parser.add_argument("--num-ragas", type=int, default=6)
    parser.add_argument("--tracks-per-raga", type=int, default=6)
    parser.add_argument("--segments-per-track", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-epochs", type=int, default=100)
    parser.add_argument("--head-patience", type=int, default=12)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-clips", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "source_mode": "kaggle_input" if getattr(args, "kaggle_data_root", None) else "mirdata",
        "num_ragas": args.num_ragas,
        "tracks_per_raga": args.tracks_per_raga,
        "segments_per_track": args.segments_per_track,
        "segment_seconds": args.segment_seconds,
        "sample_rate": SAMPLE_RATE,
        "random_seed": RANDOM_SEED,
        "models": MODELS,
    }


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def metadata_name(value: Any) -> str | None:
    """Read a human-readable label from Saraga's nested metadata fields."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("name", "title", "label", "raaga", "raag"):
            if value.get(key):
                return str(value[key]).strip()
        for nested in value.values():
            found = metadata_name(nested)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = metadata_name(item)
            if found:
                return found
    return None


def track_raga(track: Any, dataset_name: str) -> str | None:
    metadata = getattr(track, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    keys = ("raaga", "raga", "raag") if dataset_name == "saraga_carnatic" else ("raags", "raag", "raga")
    for key in keys:
        name = metadata_name(metadata.get(key))
        if name:
            return name
    return None


def collect_tracks(dataset: Any, dataset_name: str) -> list[dict[str, str]]:
    rows = []
    for track_id, track in tqdm(dataset.load_tracks().items(), desc="Reading Saraga metadata"):
        try:
            raga = track_raga(track, dataset_name)
            audio_path = getattr(track, "audio_path", None)
            if raga and audio_path and os.path.exists(audio_path):
                rows.append(
                    {
                        "track_id": str(track_id),
                        "raga": raga,
                        "audio_path": os.path.abspath(audio_path),
                    }
                )
        except Exception as exc:
            print(f"Skipping metadata for {track_id}: {exc}")
    return rows


def collect_kaggle_tracks(data_root: Path) -> list[dict[str, str]]:
    """Read the compact Kaggle Saraga copy without mirdata's full archive."""
    if not data_root.exists():
        raise RuntimeError(f"Attached Kaggle Saraga directory does not exist: {data_root}")

    rows = []
    metadata_paths = []
    for root, _, files in os.walk(data_root, followlinks=True):
        for filename in files:
            if filename.lower().endswith(".json"):
                metadata_paths.append(Path(root) / filename)
    metadata_paths.sort()

    for metadata_path in tqdm(metadata_paths, desc="Reading attached Kaggle Saraga metadata"):
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            raga = None
            for key in ("raaga", "raga", "raag"):
                raga = metadata_name(metadata.get(key))
                if raga:
                    break
            if not raga:
                continue

            audio_candidates = sorted(
                path
                for path in metadata_path.parent.iterdir()
                if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".flac", ".m4a"}
            )
            if not audio_candidates:
                continue

            exact_stem = [
                path
                for path in audio_candidates
                if path.stem == metadata_path.stem
                or path.name == f"{metadata_path.stem}.mp3.mp3"
            ]
            audio_path = exact_stem[0] if exact_stem else audio_candidates[0]
            relative = metadata_path.relative_to(data_root).with_suffix("")
            rows.append(
                {
                    "track_id": safe_name(relative.as_posix()),
                    "raga": raga,
                    "audio_path": str(audio_path.resolve()),
                }
            )
        except Exception as exc:
            print(f"Skipping attached metadata {metadata_path}: {exc}")

    if not rows:
        raise RuntimeError(
            f"No labeled JSON/MP3 track pairs were found under {data_root}. "
            "Attach the Kaggle dataset 'Saraga Carnatic Music Dataset' and point "
            "--kaggle-data-root to its carnatic directory."
        )
    return rows


def choose_balanced_tracks(
    rows: list[dict[str, str]],
    num_ragas: int,
    tracks_per_raga: int,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["raga"]].append(row)

    eligible = [(raga, tracks) for raga, tracks in grouped.items() if len(tracks) >= tracks_per_raga]
    eligible.sort(key=lambda item: (-len(item[1]), item[0].lower()))
    selected = eligible[:num_ragas]
    if len(selected) < num_ragas:
        counts = sorted(((name, len(items)) for name, items in grouped.items()), key=lambda item: -item[1])
        raise RuntimeError(
            f"Only {len(selected)} ragas have at least {tracks_per_raga} tracks. "
            f"Reduce --num-ragas or --tracks-per-raga. Top counts: {counts[:15]}"
        )

    rng = random.Random(RANDOM_SEED)
    output = {}
    for raga, tracks in selected:
        tracks = sorted(tracks, key=lambda item: item["track_id"])
        output[raga] = rng.sample(tracks, tracks_per_raga)
    return output


def split_tracks(selected: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    """Guarantee every raga appears in all splits and never split a track."""
    splits = {"train": [], "val": [], "test": []}
    for raga, tracks in selected.items():
        if len(tracks) < 4:
            raise ValueError(f"{raga} needs at least four tracks for a three-way split.")
        splits["val"].append(tracks[-2])
        splits["test"].append(tracks[-1])
        splits["train"].extend(tracks[:-2])

    split_track_ids = {name: {row["track_id"] for row in rows} for name, rows in splits.items()}
    assert split_track_ids["train"].isdisjoint(split_track_ids["val"])
    assert split_track_ids["train"].isdisjoint(split_track_ids["test"])
    assert split_track_ids["val"].isdisjoint(split_track_ids["test"])
    return splits


def safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


def segment_offsets(duration: float, segment_seconds: float, count: int) -> np.ndarray:
    usable_start = min(15.0, max(0.0, duration * 0.05))
    usable_end = max(usable_start, duration - segment_seconds - min(15.0, duration * 0.05))
    if usable_end <= usable_start:
        return np.array([0.0])
    return np.linspace(usable_start, usable_end, count)


def prepare_clips(
    splits: dict[str, list[dict[str, str]]],
    clips_dir: Path,
    segment_seconds: float,
    segments_per_track: int,
    force: bool,
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = clips_dir / "manifests.json"
    if manifest_path.exists() and not force:
        with manifest_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    if clips_dir.exists() and force:
        shutil.rmtree(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    samples_needed = int(round(segment_seconds * SAMPLE_RATE))
    failures = []

    for split_name, tracks in splits.items():
        split_dir = clips_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for row in tqdm(tracks, desc=f"Creating {split_name} clips"):
            try:
                duration = float(librosa.get_duration(path=row["audio_path"]))
                candidate_count = max(segments_per_track * 3, segments_per_track)
                offsets = segment_offsets(duration, segment_seconds, candidate_count)
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

                    filename = f"{safe_name(row['track_id'])}_s{saved:02d}.wav"
                    path = split_dir / filename
                    sf.write(path, audio, SAMPLE_RATE)
                    manifests[split_name].append(
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
                if saved < segments_per_track:
                    failures.append(
                        {
                            "split": split_name,
                            "track_id": row["track_id"],
                            "raga": row["raga"],
                            "audio_path": row["audio_path"],
                            "clips_found": saved,
                            "clips_required": segments_per_track,
                        }
                    )
            except Exception as exc:
                failures.append(
                    {
                        "split": split_name,
                        "track_id": row["track_id"],
                        "raga": row["raga"],
                        "audio_path": row["audio_path"],
                        "error": str(exc),
                        "clips_found": 0,
                        "clips_required": segments_per_track,
                    }
                )

    if failures:
        json_dump(clips_dir / "clip_failures.json", failures)
        preview = "\n".join(
            f"  {item['split']} / {item['raga']} / {item['track_id']}: "
            f"{item['clips_found']}/{item['clips_required']} clips"
            for item in failures[:10]
        )
        raise RuntimeError(
            "Some source tracks could not provide enough non-silent 30-second clips. "
            "Details were saved to clip_failures.json.\n"
            f"{preview}"
        )

    json_dump(manifest_path, manifests)
    return manifests


def plot_dataset_summary(manifests: dict[str, list[dict[str, Any]]], path: Path) -> None:
    rows = []
    for split, items in manifests.items():
        for item in items:
            rows.append({"split": split, "raga": item["raga"]})
    frame = pd.DataFrame(rows)
    plt.figure(figsize=(11, 5))
    sns.countplot(data=frame, x="raga", hue="split", order=sorted(frame["raga"].unique()))
    plt.xticks(rotation=35, ha="right")
    plt.title("Balanced Saraga benchmark: clips per raga and split")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def load_audio_batch(items: list[dict[str, Any]], sample_rate: int) -> list[np.ndarray]:
    waveforms = []
    for item in items:
        audio, _ = librosa.load(item["path"], sr=sample_rate, mono=True)
        waveforms.append(audio.astype(np.float32))
    return waveforms


def extract_embeddings(
    model_key: str,
    hf_name: str,
    manifests: dict[str, list[dict[str, Any]]],
    cache_dir: Path,
    batch_size: int,
    force: bool,
) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output: dict[str, np.ndarray] = {}
    missing_splits = []
    for split_name, items in manifests.items():
        cache_path = cache_dir / f"{model_key}_{split_name}.npz"
        if cache_path.exists() and not force:
            print(f"Using cached embeddings: {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            output[f"{split_name}_embeddings"] = cached["embeddings"]
            output[f"{split_name}_labels"] = cached["labels"]
            output[f"{split_name}_tracks"] = cached["tracks"]
        else:
            missing_splits.append((split_name, items, cache_path))

    if not missing_splits:
        return output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"\nLoading {hf_name} on {device} ({dtype})")
    try:
        extractor = AutoFeatureExtractor.from_pretrained(hf_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(hf_name, trust_remote_code=True, torch_dtype=dtype)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download or load {hf_name}. In Kaggle, enable Internet in "
            f"Notebook settings and rerun. Original error: {exc}"
        ) from exc
    model.eval().to(device)

    for split_name, items, cache_path in missing_splits:
        layer_batches: list[list[np.ndarray]] | None = None
        for start in tqdm(range(0, len(items), batch_size), desc=f"{model_key} {split_name}"):
            batch_items = items[start : start + batch_size]
            waveforms = load_audio_batch(batch_items, extractor.sampling_rate)
            inputs = extractor(
                waveforms,
                sampling_rate=extractor.sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            input_values = inputs["input_values"].to(device=device, dtype=dtype)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            try:
                with torch.inference_mode():
                    model_inputs = {"input_values": input_values, "output_hidden_states": True}
                    if attention_mask is not None:
                        model_inputs["attention_mask"] = attention_mask
                    hidden_states = model(**model_inputs).hidden_states
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU memory was exhausted during 30-second embedding extraction. "
                    "Rerun with --batch-size 1; completed split caches will be reused."
                ) from exc

            if hidden_states is None or len(hidden_states) != 13:
                raise RuntimeError(
                    f"{hf_name} returned "
                    f"{0 if hidden_states is None else len(hidden_states)} hidden states; "
                    "this benchmark expects 13."
                )
            if hidden_states[-1].shape[-1] != 768:
                raise RuntimeError(
                    f"{hf_name} returned hidden dimension {hidden_states[-1].shape[-1]}; "
                    "this benchmark expects 768."
                )

            if layer_batches is None:
                layer_batches = [[] for _ in hidden_states]
            for layer_idx, hidden in enumerate(hidden_states):
                pooled = hidden.float().mean(dim=1).cpu().numpy().astype(np.float32)
                layer_batches[layer_idx].append(pooled)

        if layer_batches is None:
            raise RuntimeError(f"No embeddings extracted for {model_key}/{split_name}")
        split_embeddings = np.stack(
            [np.concatenate(chunks, axis=0) for chunks in layer_batches],
            axis=0,
        )
        split_labels = np.array([item["raga"] for item in items], dtype=object)
        split_tracks = np.array([item["track_id"] for item in items], dtype=object)
        np.savez_compressed(
            cache_path,
            embeddings=split_embeddings,
            labels=split_labels,
            tracks=split_tracks,
        )
        print(f"Saved completed split cache: {cache_path}")
        output[f"{split_name}_embeddings"] = split_embeddings
        output[f"{split_name}_labels"] = split_labels
        output[f"{split_name}_tracks"] = split_tracks

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def make_probe() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2500,
                    solver="lbfgs",
                ),
            ),
        ]
    )


class ExternalClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def prediction_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    top_k = min(3, num_classes)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "top_3_accuracy": float(
            top_k_accuracy_score(labels, probabilities, k=top_k, labels=np.arange(num_classes))
        ),
    }


def track_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    track_ids: np.ndarray,
    encoder: LabelEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    grouped_probs = []
    grouped_labels = []
    for track_id in sorted(set(track_ids)):
        mask = track_ids == track_id
        grouped_probs.append(probabilities[mask].mean(axis=0))
        label_values = labels[mask]
        grouped_labels.append(encoder.transform([Counter(label_values).most_common(1)[0][0]])[0])
    return np.stack(grouped_probs), np.array(grouped_labels)


def train_external_classifier(
    model_key: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
    figures_dir: Path,
    models_dir: Path,
) -> tuple[ExternalClassifier, StandardScaler, list[dict[str, float]], int]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x).astype(np.float32)
    val_scaled = scaler.transform(val_x).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(train_scaled),
        torch.from_numpy(train_y.astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(16, len(train_dataset)),
        shuffle=True,
        generator=generator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    model = ExternalClassifier(
        input_dim=train_x.shape[1],
        hidden_dim=args.head_hidden_dim,
        num_classes=num_classes,
        dropout=args.head_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.head_lr,
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()
    val_tensor = torch.from_numpy(val_scaled).to(device)

    history = []
    best_rank = None
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.head_epochs + 1):
        model.train()
        loss_sum = 0.0
        train_predictions = []
        train_labels = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch_y)
            train_predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())
            train_labels.extend(batch_y.detach().cpu().numpy())

        model.eval()
        with torch.inference_mode():
            val_logits = model(val_tensor)
            val_predictions = val_logits.argmax(dim=1).cpu().numpy()

        train_accuracy = accuracy_score(train_labels, train_predictions)
        val_accuracy = accuracy_score(val_y, val_predictions)
        val_macro_f1 = f1_score(val_y, val_predictions, average="macro", zero_division=0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / len(train_dataset),
                "train_accuracy": float(train_accuracy),
                "val_accuracy": float(val_accuracy),
                "val_macro_f1": float(val_macro_f1),
            }
        )

        rank = (val_macro_f1, val_accuracy, -epoch)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.head_patience:
                break

    if best_state is None:
        raise RuntimeError("The external classifier did not complete a training epoch.")
    model.load_state_dict(best_state)
    model.to(device).eval()

    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="train loss")
    axes[0].axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training loss")
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="train accuracy")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="validation accuracy")
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], label="validation macro F1")
    axes[1].axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    fig.suptitle(f"{model_key}: external classifier training")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{model_key}_external_head_training.png", dpi=160)
    plt.close()

    joblib.dump(scaler, models_dir / f"{model_key}_external_head_scaler.joblib")
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": train_x.shape[1],
            "hidden_dim": args.head_hidden_dim,
            "num_classes": num_classes,
            "dropout": args.head_dropout,
            "best_epoch": best_epoch,
        },
        models_dir / f"{model_key}_external_head.pt",
    )
    return model, scaler, history, best_epoch


def neural_probabilities(
    model: ExternalClassifier,
    scaler: StandardScaler,
    embeddings: np.ndarray,
) -> np.ndarray:
    device = next(model.parameters()).device
    scaled = scaler.transform(embeddings).astype(np.float32)
    with torch.inference_mode():
        logits = model(torch.from_numpy(scaled).to(device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def plot_layer_curve(model_key: str, layer_rows: list[dict[str, Any]], best_layer: int, path: Path) -> None:
    frame = pd.DataFrame(layer_rows)
    plt.figure(figsize=(9, 4.5))
    plt.plot(frame["layer"], frame["val_accuracy"], "o-", label="validation accuracy")
    plt.plot(frame["layer"], frame["val_macro_f1"], "s-", label="validation macro F1")
    plt.axvline(best_layer, color="black", linestyle="--", alpha=0.55, label=f"selected layer {best_layer}")
    plt.xlabel("Hidden-state layer")
    plt.ylabel("Score")
    plt.title(f"{model_key}: layer-wise frozen probe")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    perplexity = max(3, min(20, (len(embeddings) - 1) // 3))
    scaled = StandardScaler().fit_transform(embeddings)
    points = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_SEED,
    ).fit_transform(scaled)
    frame = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "raga": labels})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=frame, x="x", y="y", hue="raga", s=55, alpha=0.8)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def evaluate_model(
    model_key: str,
    data: dict[str, np.ndarray],
    figures_dir: Path,
    models_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    encoder = LabelEncoder().fit(data["train_labels"])
    y_train = encoder.transform(data["train_labels"])
    y_val = encoder.transform(data["val_labels"])
    y_test = encoder.transform(data["test_labels"])
    num_classes = len(encoder.classes_)

    layer_rows = []
    best = None
    for layer_idx in range(data["train_embeddings"].shape[0]):
        probe = make_probe()
        probe.fit(data["train_embeddings"][layer_idx], y_train)
        val_pred = probe.predict(data["val_embeddings"][layer_idx])
        row = {
            "layer": layer_idx,
            "val_accuracy": float(accuracy_score(y_val, val_pred)),
            "val_macro_f1": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
        }
        layer_rows.append(row)
        rank = (row["val_macro_f1"], row["val_accuracy"], -layer_idx)
        if best is None or rank > best["rank"]:
            best = {"rank": rank, "layer": layer_idx}

    assert best is not None
    best_layer = int(best["layer"])
    plot_layer_curve(
        model_key,
        layer_rows,
        best_layer,
        figures_dir / f"{model_key}_layer_curve.png",
    )

    x_trainval = np.concatenate(
        [data["train_embeddings"][best_layer], data["val_embeddings"][best_layer]],
        axis=0,
    )
    y_trainval = np.concatenate([y_train, y_val])
    final_probe = make_probe()
    final_probe.fit(x_trainval, y_trainval)
    test_x = data["test_embeddings"][best_layer]
    test_pred = final_probe.predict(test_x)
    test_prob = final_probe.predict_proba(test_x)
    linear_clip_metrics = prediction_metrics(y_test, test_pred, test_prob, num_classes)

    track_prob, track_labels = track_probabilities(
        test_prob,
        data["test_labels"],
        data["test_tracks"],
        encoder,
    )
    track_pred = track_prob.argmax(axis=1)
    linear_track_metrics = prediction_metrics(
        track_labels,
        track_pred,
        track_prob,
        num_classes,
    )
    linear_track_metrics["num_test_tracks"] = int(len(track_labels))

    scaled_test = StandardScaler().fit_transform(test_x)
    cluster_metrics = {
        "silhouette": float(silhouette_score(scaled_test, y_test)),
        "davies_bouldin": float(davies_bouldin_score(scaled_test, y_test)),
    }

    def save_confusion(predictions: np.ndarray, classifier_name: str) -> None:
        cm = confusion_matrix(y_test, predictions, labels=np.arange(num_classes))
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=35,
            colorbar=False,
        )
        ax.set_title(
            f"{model_key}: {classifier_name} test confusion matrix, layer {best_layer}"
        )
        plt.tight_layout()
        plt.savefig(
            figures_dir / f"{model_key}_{classifier_name}_confusion_matrix.png",
            dpi=160,
        )
        plt.close()

    save_confusion(test_pred, "linear")

    plot_tsne(
        test_x,
        data["test_labels"],
        f"{model_key}: test embeddings, layer {best_layer}",
        figures_dir / f"{model_key}_tsne.png",
    )

    joblib.dump(final_probe, models_dir / f"{model_key}_probe.joblib")
    joblib.dump(encoder, models_dir / f"{model_key}_labels.joblib")

    external_head, external_scaler, head_history, best_epoch = train_external_classifier(
        model_key=model_key,
        train_x=data["train_embeddings"][best_layer],
        train_y=y_train,
        val_x=data["val_embeddings"][best_layer],
        val_y=y_val,
        num_classes=num_classes,
        args=args,
        figures_dir=figures_dir,
        models_dir=models_dir,
    )
    neural_test_prob = neural_probabilities(external_head, external_scaler, test_x)
    neural_test_pred = neural_test_prob.argmax(axis=1)
    neural_clip_metrics = prediction_metrics(
        y_test,
        neural_test_pred,
        neural_test_prob,
        num_classes,
    )
    neural_track_prob, neural_track_labels = track_probabilities(
        neural_test_prob,
        data["test_labels"],
        data["test_tracks"],
        encoder,
    )
    neural_track_pred = neural_track_prob.argmax(axis=1)
    neural_track_metrics = prediction_metrics(
        neural_track_labels,
        neural_track_pred,
        neural_track_prob,
        num_classes,
    )
    neural_track_metrics["num_test_tracks"] = int(len(neural_track_labels))
    save_confusion(neural_test_pred, "external_head")

    del external_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best_layer": best_layer,
        "classes": list(encoder.classes_),
        "cluster_metrics": cluster_metrics,
        "layer_results": layer_rows,
        "linear": {
            "clip_metrics": linear_clip_metrics,
            "track_metrics": linear_track_metrics,
            "classification_report": classification_report(
                y_test,
                test_pred,
                target_names=encoder.classes_,
                zero_division=0,
                output_dict=True,
            ),
        },
        "external_head": {
            "best_epoch": best_epoch,
            "history": head_history,
            "clip_metrics": neural_clip_metrics,
            "track_metrics": neural_track_metrics,
            "classification_report": classification_report(
                y_test,
                neural_test_pred,
                target_names=encoder.classes_,
                zero_division=0,
                output_dict=True,
            ),
        },
    }


def write_report(
    path: Path,
    args: argparse.Namespace,
    manifests: dict[str, list[dict[str, Any]]],
    results: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Balanced MERT vs CultureMERT Benchmark",
        "",
        "This run uses a deliberately small but leakage-safe Saraga benchmark.",
        "Every test clip comes from a source track that is absent from training and validation.",
        "",
        "## Setup",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Ragas: {args.num_ragas}",
        f"- Source tracks per raga: {args.tracks_per_raga}",
        f"- Clips per source track: up to {args.segments_per_track}",
        f"- Clip duration: {args.segment_seconds:.1f} seconds at 24 kHz",
        f"- Train/validation/test clips: {len(manifests['train'])}/{len(manifests['val'])}/{len(manifests['test'])}",
        "- Split unit: original track, not clip",
        "- Backbones: frozen; no MERT or CultureMERT weights are updated",
        "- Linear probe: selected on validation macro F1, then fitted on train + validation",
        "- External head: trained on train embeddings with validation-based early stopping",
        "",
        "## Results",
        "",
        "| Model | Classifier | Best layer | Clip accuracy | Clip balanced accuracy | Clip macro F1 | Clip top-3 | Track accuracy | Track balanced accuracy | Track macro F1 | Track top-3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, result in results.items():
        for classifier_key, classifier_label in (
            ("linear", "Linear probe"),
            ("external_head", "External neural head"),
        ):
            classifier = result[classifier_key]
            clip = classifier["clip_metrics"]
            track = classifier["track_metrics"]
            lines.append(
                f"| {model_key} | {classifier_label} | {result['best_layer']} | "
                f"{clip['accuracy']:.3f} | {clip['balanced_accuracy']:.3f} | "
                f"{clip['macro_f1']:.3f} | {clip['top_3_accuracy']:.3f} | "
                f"{track['accuracy']:.3f} | {track['balanced_accuracy']:.3f} | "
                f"{track['macro_f1']:.3f} | {track['top_3_accuracy']:.3f} |"
            )

    culture = results["culturemert_95m"]["linear"]["clip_metrics"]["accuracy"]
    base = results["mert_95m"]["linear"]["clip_metrics"]["accuracy"]
    delta = culture - base
    lines.extend(
        [
            "",
            "## Embedding Structure",
            "",
            "| Model | Silhouette (higher is better) | Davies-Bouldin (lower is better) |",
            "|---|---:|---:|",
        ]
    )
    for model_key, result in results.items():
        cluster = result["cluster_metrics"]
        lines.append(
            f"| {model_key} | {cluster['silhouette']:.3f} | "
            f"{cluster['davies_bouldin']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Short Reading",
            "",
            f"For the linear probe, CultureMERT minus MERT clip-level accuracy is **{delta:+.3f}**.",
            "",
            "This is a compact benchmark, so the result should be treated as evidence from this split rather than a final claim about all Indian classical music. "
            "The most important methodological choice is the track-level split, which prevents different chunks of the same recording from appearing in train and test.",
            "",
            "## Limitations And Possible Drawbacks",
            "",
            "- Only six recordings per raga are used, so results may change with a different track selection.",
            "- A 30-second excerpt gives more melodic context than an 8-second excerpt, but a complete raga performance develops over much longer periods.",
            "- The external neural head has more parameters than the linear probe and can overfit this small dataset. Its training and validation curves should be checked.",
            "- The backbones are frozen, so this experiment does not show whether full fine-tuning would improve accuracy or damage general musical representations.",
            "- Recording conditions, performer identity, tonic and instrumentation can still influence predictions even with a track-level split.",
            "",
            "## Reference Comparison",
            "",
            "The related `ritgit24/MERT` project reports results on a much larger Hindustani setup. "
            "This notebook borrows its useful ideas (top-k accuracy and embedding-cluster metrics) but keeps 24 kHz MERT input and track-level leakage protection.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def package_results(output_dir: Path, archive_name: str) -> Path:
    """Package only submission-sized outputs, not cached audio/embeddings."""
    archive_path = output_dir / archive_name
    if archive_path.exists():
        archive_path.unlink()
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
                for path in item.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(output_dir).as_posix())
    return archive_path


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    sns.set_theme(style="whitegrid")

    output_dir = Path(args.output_dir).resolve()
    config = experiment_config(args)
    fingerprint = config_fingerprint(config)
    cache_root = output_dir / "cache" / fingerprint
    clips_dir = cache_root / "clips"
    cache_dir = cache_root / "embeddings"
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    models_dir = output_dir / "models"
    for path in (figures_dir, metrics_dir, models_dir):
        path.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif args.segment_seconds >= 30:
        raise RuntimeError(
            "A GPU is required for this 30-second benchmark. In Kaggle, open "
            "Notebook options, select a T4 or P100 accelerator, and rerun."
        )
    print(f"Experiment cache fingerprint: {fingerprint}")
    json_dump(metrics_dir / "experiment_config.json", {**config, "fingerprint": fingerprint})

    if args.kaggle_data_root:
        if args.dataset != "saraga_carnatic":
            raise RuntimeError("--kaggle-data-root currently supports Saraga Carnatic only.")
        kaggle_data_root = Path(args.kaggle_data_root).resolve()
        print(f"Using attached Kaggle Saraga data: {kaggle_data_root}")
        rows = collect_kaggle_tracks(kaggle_data_root)
    else:
        data_home = Path(args.data_home).resolve()
        data_home.mkdir(parents=True, exist_ok=True)
        dataset = mirdata.initialize(args.dataset, data_home=str(data_home))
        if not args.skip_download:
            print(f"Downloading/validating {args.dataset} at {data_home}")
            try:
                dataset.download()
            except Exception as exc:
                raise RuntimeError(
                    "Saraga could not be downloaded. The current Carnatic archive is too "
                    "large for Kaggle's working disk. Attach the Kaggle dataset "
                    "'Saraga Carnatic Music Dataset' and use --kaggle-data-root instead. "
                    f"Original error: {exc}"
                ) from exc
        try:
            dataset.validate()
        except Exception as exc:
            raise RuntimeError(
                f"Saraga validation failed at {data_home}. Check that the attached/downloaded "
                f"dataset is complete. "
                f"Original error: {exc}"
            ) from exc
        rows = collect_tracks(dataset, args.dataset)

    print(f"Usable labeled tracks: {len(rows)}")
    selected = choose_balanced_tracks(rows, args.num_ragas, args.tracks_per_raga)
    print("Selected ragas:")
    for raga, tracks in selected.items():
        print(f"  {raga}: {len(tracks)} tracks")

    splits = split_tracks(selected)
    json_dump(metrics_dir / "track_splits.json", splits)
    manifests = prepare_clips(
        splits,
        clips_dir,
        args.segment_seconds,
        args.segments_per_track,
        args.force_clips,
    )
    json_dump(metrics_dir / "clip_manifest.json", manifests)
    plot_dataset_summary(manifests, figures_dir / "dataset_distribution.png")

    split_track_counts = {
        split_name: Counter(row["raga"] for row in split_rows)
        for split_name, split_rows in splits.items()
    }
    for split_name, items in manifests.items():
        counts = Counter(item["raga"] for item in items)
        print(f"{split_name}: {len(items)} clips, {dict(counts)}")
        if set(counts) != set(selected):
            raise RuntimeError(f"{split_name} lost one or more ragas after clipping: {counts}")
        expected = {
            raga: track_count * args.segments_per_track
            for raga, track_count in split_track_counts[split_name].items()
        }
        if dict(counts) != expected:
            raise RuntimeError(
                f"{split_name} is not balanced after clipping. "
                f"Expected {expected}, found {dict(counts)}"
            )
        for item in items:
            info = sf.info(item["path"])
            expected_frames = int(round(args.segment_seconds * SAMPLE_RATE))
            if info.samplerate != SAMPLE_RATE or info.frames != expected_frames:
                raise RuntimeError(
                    f"Invalid processed clip {item['path']}: "
                    f"{info.samplerate} Hz, {info.frames} frames"
                )

    embeddings = {}
    for model_key, hf_name in MODELS.items():
        embeddings[model_key] = extract_embeddings(
            model_key,
            hf_name,
            manifests,
            cache_dir,
            args.batch_size,
            args.force_embeddings,
        )

    results = {}
    for model_key, data in embeddings.items():
        results[model_key] = evaluate_model(
            model_key,
            data,
            figures_dir,
            models_dir,
            args,
        )

    json_dump(metrics_dir / "benchmark_results.json", results)
    summary_rows = []
    for model_key, result in results.items():
        for classifier_key in ("linear", "external_head"):
            classifier = result[classifier_key]
            summary_rows.append(
                {
                    "model": model_key,
                    "classifier": classifier_key,
                    "best_layer": result["best_layer"],
                    **{
                        f"clip_{key}": value
                        for key, value in classifier["clip_metrics"].items()
                    },
                    **{
                        f"track_{key}": value
                        for key, value in classifier["track_metrics"].items()
                    },
                    **result["cluster_metrics"],
                }
            )
    pd.DataFrame(summary_rows).to_csv(metrics_dir / "benchmark_summary.csv", index=False)
    write_report(output_dir / "REPORT.md", args, manifests, results)

    archive_name = (
        "mert_raga_30s_results.zip"
        if args.segment_seconds == 30
        else "mert_raga_benchmark_results.zip"
    )
    archive_path = package_results(output_dir, archive_name)
    print("\nFinished.")
    print(f"Report: {output_dir / 'REPORT.md'}")
    print(f"Results archive: {archive_path}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
