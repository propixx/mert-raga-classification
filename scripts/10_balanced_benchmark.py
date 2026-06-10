#!/usr/bin/env python3
"""Balanced, leakage-safe MERT vs CultureMERT benchmark for Saraga.

This script is intended for a Colab/Kaggle T4 run. It keeps the experiment
small enough for a project-selection task while producing meaningful outputs:

- balanced raga selection
- train/validation/test split by original track
- embeddings from all 13 layers of both 95M models
- layer-wise linear probes
- clip-level and track-level top-1/top-3 metrics
- macro F1, confusion matrices, t-SNE, silhouette and Davies-Bouldin scores
- a generated Markdown report and a downloadable result zip
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--output-dir", default="benchmark_outputs")
    parser.add_argument("--num-ragas", type=int, default=6)
    parser.add_argument("--tracks-per-raga", type=int, default=6)
    parser.add_argument("--segments-per-track", type=int, default=6)
    parser.add_argument("--segment-seconds", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-clips", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


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

    for split_name, tracks in splits.items():
        split_dir = clips_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for row in tqdm(tracks, desc=f"Creating {split_name} clips"):
            try:
                duration = float(librosa.get_duration(path=row["audio_path"]))
                offsets = segment_offsets(duration, segment_seconds, segments_per_track)
                saved = 0
                for offset_idx, offset in enumerate(offsets):
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

                    filename = f"{safe_name(row['track_id'])}_s{offset_idx:02d}.wav"
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
                if saved == 0:
                    print(f"WARNING: no usable clips from {row['audio_path']}")
            except Exception as exc:
                print(f"WARNING: failed to segment {row['audio_path']}: {exc}")

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
    cache_path = cache_dir / f"{model_key}.npz"
    if cache_path.exists() and not force:
        print(f"Using cached embeddings: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        return {key: cached[key] for key in cached.files}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"\nLoading {hf_name} on {device} ({dtype})")
    extractor = AutoFeatureExtractor.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(hf_name, trust_remote_code=True, torch_dtype=dtype)
    model.eval().to(device)

    output: dict[str, np.ndarray] = {}
    for split_name, items in manifests.items():
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

            with torch.inference_mode():
                model_inputs = {"input_values": input_values, "output_hidden_states": True}
                if attention_mask is not None:
                    model_inputs["attention_mask"] = attention_mask
                hidden_states = model(**model_inputs).hidden_states

            if layer_batches is None:
                layer_batches = [[] for _ in hidden_states]
            for layer_idx, hidden in enumerate(hidden_states):
                pooled = hidden.float().mean(dim=1).cpu().numpy().astype(np.float32)
                layer_batches[layer_idx].append(pooled)

        if layer_batches is None:
            raise RuntimeError(f"No embeddings extracted for {model_key}/{split_name}")
        output[f"{split_name}_embeddings"] = np.stack(
            [np.concatenate(chunks, axis=0) for chunks in layer_batches],
            axis=0,
        )
        output[f"{split_name}_labels"] = np.array([item["raga"] for item in items], dtype=object)
        output[f"{split_name}_tracks"] = np.array([item["track_id"] for item in items], dtype=object)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **output)
    del model
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
    top_k = min(3, num_classes)

    clip_metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "macro_f1": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "top_3_accuracy": float(
            top_k_accuracy_score(y_test, test_prob, k=top_k, labels=np.arange(num_classes))
        ),
    }

    track_prob, track_labels = track_probabilities(
        test_prob,
        data["test_labels"],
        data["test_tracks"],
        encoder,
    )
    track_pred = track_prob.argmax(axis=1)
    track_metrics = {
        "accuracy": float(accuracy_score(track_labels, track_pred)),
        "macro_f1": float(f1_score(track_labels, track_pred, average="macro", zero_division=0)),
        "top_3_accuracy": float(
            top_k_accuracy_score(track_labels, track_prob, k=top_k, labels=np.arange(num_classes))
        ),
        "num_test_tracks": int(len(track_labels)),
    }

    scaled_test = StandardScaler().fit_transform(test_x)
    cluster_metrics = {
        "silhouette": float(silhouette_score(scaled_test, y_test)),
        "davies_bouldin": float(davies_bouldin_score(scaled_test, y_test)),
    }

    cm = confusion_matrix(y_test, test_pred, labels=np.arange(num_classes))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=35,
        colorbar=False,
    )
    ax.set_title(f"{model_key}: test confusion matrix, layer {best_layer}")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{model_key}_confusion_matrix.png", dpi=160)
    plt.close()

    plot_tsne(
        test_x,
        data["test_labels"],
        f"{model_key}: test embeddings, layer {best_layer}",
        figures_dir / f"{model_key}_tsne.png",
    )

    joblib.dump(final_probe, models_dir / f"{model_key}_probe.joblib")
    joblib.dump(encoder, models_dir / f"{model_key}_labels.joblib")

    return {
        "best_layer": best_layer,
        "classes": list(encoder.classes_),
        "clip_metrics": clip_metrics,
        "track_metrics": track_metrics,
        "cluster_metrics": cluster_metrics,
        "layer_results": layer_rows,
        "classification_report": classification_report(
            y_test,
            test_pred,
            target_names=encoder.classes_,
            zero_division=0,
            output_dict=True,
        ),
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
        "",
        "## Results",
        "",
        "| Model | Best layer | Clip accuracy | Clip macro F1 | Clip top-3 | Track accuracy | Track top-3 | Silhouette | Davies-Bouldin |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key, result in results.items():
        clip = result["clip_metrics"]
        track = result["track_metrics"]
        cluster = result["cluster_metrics"]
        lines.append(
            f"| {model_key} | {result['best_layer']} | {clip['accuracy']:.3f} | "
            f"{clip['macro_f1']:.3f} | {clip['top_3_accuracy']:.3f} | "
            f"{track['accuracy']:.3f} | {track['top_3_accuracy']:.3f} | "
            f"{cluster['silhouette']:.3f} | {cluster['davies_bouldin']:.3f} |"
        )

    culture = results["culturemert_95m"]["clip_metrics"]["accuracy"]
    base = results["mert_95m"]["clip_metrics"]["accuracy"]
    delta = culture - base
    lines.extend(
        [
            "",
            "## Short Reading",
            "",
            f"CultureMERT minus MERT clip-level accuracy: **{delta:+.3f}**.",
            "",
            "This is a compact benchmark, so the result should be treated as evidence from this split rather than a final claim about all Indian classical music. "
            "The most important methodological choice is the track-level split, which prevents different chunks of the same recording from appearing in train and test.",
            "",
            "## Reference Comparison",
            "",
            "The related `ritgit24/MERT` project reports results on a much larger Hindustani setup. "
            "This notebook borrows its useful ideas (top-k accuracy and embedding-cluster metrics) but keeps 24 kHz MERT input and track-level leakage protection.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def package_results(output_dir: Path) -> Path:
    """Package only submission-sized outputs, not cached audio/embeddings."""
    archive_path = output_dir / "mert_raga_benchmark_results.zip"
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
    clips_dir = output_dir / "clips"
    cache_dir = output_dir / "embedding_cache"
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    models_dir = output_dir / "models"
    for path in (figures_dir, metrics_dir, models_dir):
        path.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_home = Path(args.data_home).resolve()
    data_home.mkdir(parents=True, exist_ok=True)
    dataset = mirdata.initialize(args.dataset, data_home=str(data_home))
    if not args.skip_download:
        print(f"Downloading/validating {args.dataset} at {data_home}")
        dataset.download()
    dataset.validate()

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
    plot_dataset_summary(manifests, figures_dir / "dataset_distribution.png")

    for split_name, items in manifests.items():
        counts = Counter(item["raga"] for item in items)
        print(f"{split_name}: {len(items)} clips, {dict(counts)}")
        if set(counts) != set(selected):
            raise RuntimeError(f"{split_name} lost one or more ragas after clipping: {counts}")

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
        results[model_key] = evaluate_model(model_key, data, figures_dir, models_dir)

    json_dump(metrics_dir / "benchmark_results.json", results)
    summary_rows = []
    for model_key, result in results.items():
        summary_rows.append(
            {
                "model": model_key,
                "best_layer": result["best_layer"],
                **{f"clip_{key}": value for key, value in result["clip_metrics"].items()},
                **{f"track_{key}": value for key, value in result["track_metrics"].items()},
                **result["cluster_metrics"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(metrics_dir / "benchmark_summary.csv", index=False)
    write_report(output_dir / "REPORT.md", args, manifests, results)

    archive_path = package_results(output_dir)
    print("\nFinished.")
    print(f"Report: {output_dir / 'REPORT.md'}")
    print(f"Results archive: {archive_path}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
