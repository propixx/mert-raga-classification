#!/usr/bin/env python3
"""Run frozen or final-layer MERT training on the 30-second benchmark.

The experiment reuses the same five balanced ragas, 25 recordings, 100 clips
and five track-wise folds. In frozen mode, only the external classifier learns.
In last-layer mode, transformer layer 12 learns with the classifier.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import random
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
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
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


RANDOM_SEED = 42
MODEL_NAME = "m-a-p/MERT-v1-95M"
SAMPLE_RATE = 24000


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
    parser.add_argument("--kaggle-data-root", required=True)
    parser.add_argument(
        "--base-output-dir",
        default="/kaggle/working/mert_raga_5fold",
        help="Location of the compatible 30-second clip cache.",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/mert_last_layer_finetune",
    )
    parser.add_argument(
        "--mode",
        choices=("frozen", "last_layer"),
        default="last_layer",
        help="Train only the classifier, or the classifier plus MERT layer 12.",
    )
    parser.add_argument("--num-ragas", type=int, default=5)
    parser.add_argument("--tracks-per-raga", type=int, default=5)
    parser.add_argument("--exclude-raga", action="append", default=[])
    parser.add_argument("--segments-per-track", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--baseline-track-accuracy", type=float, default=0.36)
    parser.add_argument("--baseline-track-f1", type=float, default=0.319)
    parser.add_argument("--baseline-train-accuracy", type=float, default=0.777)
    parser.add_argument("--baseline-val-accuracy", type=float, default=0.490)
    parser.add_argument("--force-clips", action="store_true")
    parser.add_argument("--force-folds", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


class AudioClipDataset(Dataset):
    def __init__(
        self,
        manifest: list[dict[str, Any]],
        indices: np.ndarray,
        encoded_labels: np.ndarray,
    ):
        self.manifest = manifest
        self.indices = np.asarray(indices, dtype=int)
        self.encoded_labels = encoded_labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        index = int(self.indices[position])
        item = self.manifest[index]
        waveform, sample_rate = sf.read(
            item["path"],
            dtype="float32",
            always_2d=False,
        )
        if sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                f"Unexpected sample rate for {item['path']}: {sample_rate}"
            )
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return {
            "waveform": waveform,
            "label": int(self.encoded_labels[index]),
            "track_id": item["track_id"],
            "index": index,
        }


class AudioCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = self.feature_extractor(
            [row["waveform"] for row in rows],
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_tensors="pt",
        )
        return {
            "input_values": inputs["input_values"],
            "attention_mask": inputs.get("attention_mask"),
            "labels": torch.tensor(
                [row["label"] for row in rows],
                dtype=torch.long,
            ),
            "tracks": [row["track_id"] for row in rows],
            "indices": np.asarray([row["index"] for row in rows], dtype=int),
        }


class LastLayerMERTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        mode: str = "last_layer",
    ):
        super().__init__()
        self.mode = mode
        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        encoder_layers = getattr(
            getattr(self.backbone, "encoder", None),
            "layers",
            None,
        )
        if encoder_layers is None or len(encoder_layers) != 12:
            raise RuntimeError(
                "Expected MERT-95M to expose twelve encoder layers."
            )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.last_layer = encoder_layers[-1]
        if mode == "last_layer":
            for parameter in self.last_layer.parameters():
                parameter.requires_grad = True
        elif mode != "frozen":
            raise ValueError(f"Unknown training mode: {mode}")

        width = int(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def set_training_mode(self) -> None:
        self.train()
        self.backbone.eval()
        if self.mode == "last_layer":
            self.last_layer.train()
        else:
            self.last_layer.eval()
        self.classifier.train()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_embedding: bool = False,
    ):
        kwargs: dict[str, Any] = {"input_values": input_values}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        hidden = self.backbone(**kwargs).last_hidden_state
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        if return_embedding:
            return logits, pooled
        return logits


def trainable_state(model: LastLayerMERTClassifier) -> dict[str, Any]:
    return {
        "last_layer": {
            name: value.detach().cpu().clone()
            for name, value in model.last_layer.state_dict().items()
        },
        "classifier": {
            name: value.detach().cpu().clone()
            for name, value in model.classifier.state_dict().items()
        },
    }


def load_trainable_state(
    model: LastLayerMERTClassifier,
    state: dict[str, Any],
) -> None:
    model.last_layer.load_state_dict(state["last_layer"])
    model.classifier.load_state_dict(state["classifier"])


def aggregate_tracks(
    probabilities: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_probabilities = []
    output_labels = []
    output_tracks = []
    for track in sorted(set(tracks)):
        mask = tracks == track
        output_probabilities.append(probabilities[mask].mean(axis=0))
        output_labels.append(Counter(labels[mask]).most_common(1)[0][0])
        output_tracks.append(track)
    return (
        np.stack(output_probabilities),
        np.asarray(output_labels),
        np.asarray(output_tracks, dtype=object),
    )


def metrics(
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


def evaluate(
    model: LastLayerMERTClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_embeddings: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    count = 0
    probabilities = []
    labels = []
    tracks = []
    indices = []
    embeddings = []
    with torch.inference_mode():
        for batch in loader:
            input_values = batch["input_values"].to(
                device,
                non_blocking=True,
            )
            attention_mask = batch["attention_mask"]
            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    device,
                    non_blocking=True,
                )
            targets = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(
                    input_values,
                    attention_mask,
                    return_embedding=return_embeddings,
                )
                if return_embeddings:
                    logits, pooled = output
                    embeddings.append(
                        pooled.float().cpu().numpy().astype(np.float32)
                    )
                else:
                    logits = output
                loss = criterion(logits, targets)
            total_loss += float(loss.item()) * len(targets)
            count += len(targets)
            probabilities.append(
                torch.softmax(logits.float(), dim=1).cpu().numpy()
            )
            labels.append(targets.cpu().numpy())
            tracks.extend(batch["tracks"])
            indices.append(batch["indices"])
    output = {
        "loss": total_loss / max(1, count),
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels),
        "tracks": np.asarray(tracks, dtype=object),
        "indices": np.concatenate(indices),
    }
    if return_embeddings:
        output["embeddings"] = np.concatenate(embeddings)
    clip = metrics(output["labels"], output["probabilities"], model.classifier[-1].out_features)
    track_probabilities, track_labels, track_ids = aggregate_tracks(
        output["probabilities"],
        output["labels"],
        output["tracks"],
    )
    output["clip_metrics"] = clip
    output["track_metrics"] = metrics(
        track_labels,
        track_probabilities,
        model.classifier[-1].out_features,
    )
    output["track_ids"] = track_ids
    return output


def make_loader(
    manifest: list[dict[str, Any]],
    indices: np.ndarray,
    labels: np.ndarray,
    collator: AudioCollator,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = AudioClipDataset(manifest, indices, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
    )


def parameter_drift(
    initial: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, initial_value in initial.items():
        current_value = current[name].detach().cpu().float()
        initial_value = initial_value.detach().cpu().float()
        numerator += float(torch.sum((current_value - initial_value) ** 2))
        denominator += float(torch.sum(initial_value**2))
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1e-12)


def cosine_similarity_rows(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    numerator = np.sum(before * after, axis=1)
    denominator = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def train_fold(
    fold_index: int,
    fold: dict[str, list[dict[str, str]]],
    manifest: list[dict[str, Any]],
    encoded_labels: np.ndarray,
    track_ids: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    fold_dir = output_dir / "checkpoints" / f"fold_{fold_index + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    completed_path = fold_dir / "completed_predictions.npz"
    summary_path = fold_dir / "summary.json"
    if completed_path.exists() and summary_path.exists() and not args.force_folds:
        print(f"Fold {fold_index + 1}: using completed checkpoint.")
        with np.load(completed_path, allow_pickle=True) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        return {
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "arrays": arrays,
        }

    seed_everything(RANDOM_SEED + fold_index)
    device = torch.device("cuda")
    extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    collator = AudioCollator(extractor)
    masks = CV.masks_for_fold(fold, track_ids)
    index_map = {
        split: np.flatnonzero(mask)
        for split, mask in masks.items()
    }
    loaders = {
        split: make_loader(
            manifest,
            indices,
            encoded_labels,
            collator,
            args.batch_size,
            split == "train",
            RANDOM_SEED + fold_index,
        )
        for split, indices in index_map.items()
    }

    print(f"Fold {fold_index + 1}: loading {MODEL_NAME}")
    model = LastLayerMERTClassifier(
        num_classes,
        args.hidden_dim,
        args.dropout,
        args.mode,
    ).to(device)
    initial_last_layer = {
        name: value.detach().cpu().clone()
        for name, value in model.last_layer.state_dict().items()
    }
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Fold {fold_index + 1}: trainable "
        f"{trainable / 1e6:.2f}M/{total / 1e6:.2f}M parameters"
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )
    parameter_groups = []
    if args.mode == "last_layer":
        parameter_groups.append(
            {
                "params": model.last_layer.parameters(),
                "lr": args.backbone_lr,
                "name": "mert_last_layer",
            }
        )
    parameter_groups.append(
        {
            "params": model.classifier.parameters(),
            "lr": args.classifier_lr,
            "name": "classifier",
        }
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    print(f"Fold {fold_index + 1}: measuring pretrained test embeddings.")
    before_test = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        return_embeddings=True,
    )

    checkpoint_path = fold_dir / "last_epoch.pt"
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_rank = None
    best_state = None
    best_epoch = 0
    stale = 0
    if checkpoint_path.exists() and not args.force_folds:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_trainable_state(model, checkpoint["trainable_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_rank = tuple(checkpoint["best_rank"])
        best_state = checkpoint["best_state"]
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(
            f"Fold {fold_index + 1}: resuming from epoch "
            f"{start_epoch}/{args.epochs}."
        )

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.set_training_mode()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_count = 0
        train_predictions = []
        train_targets = []
        for step, batch in enumerate(
            tqdm(
                loaders["train"],
                desc=f"Fold {fold_index + 1} epoch {epoch}",
                leave=False,
            ),
            start=1,
        ):
            input_values = batch["input_values"].to(
                device,
                non_blocking=True,
            )
            attention_mask = batch["attention_mask"]
            if attention_mask is not None:
                attention_mask = attention_mask.to(
                    device,
                    non_blocking=True,
                )
            targets = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(input_values, attention_mask)
                raw_loss = criterion(logits, targets)
                loss = raw_loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            should_step = (
                step % args.gradient_accumulation == 0
                or step == len(loaders["train"])
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss += float(raw_loss.item()) * len(targets)
            train_count += len(targets)
            train_predictions.extend(
                logits.detach().argmax(1).cpu().numpy()
            )
            train_targets.extend(targets.cpu().numpy())

        scheduler.step()
        validation = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_count),
            "train_accuracy": float(
                accuracy_score(train_targets, train_predictions)
            ),
            "val_loss": validation["loss"],
            "val_clip_accuracy": validation["clip_metrics"]["accuracy"],
            "val_clip_macro_f1": validation["clip_metrics"]["macro_f1"],
            "val_track_accuracy": validation["track_metrics"]["accuracy"],
            "val_track_macro_f1": validation["track_metrics"]["macro_f1"],
            "backbone_lr": (
                optimizer.param_groups[0]["lr"]
                if args.mode == "last_layer"
                else 0.0
            ),
            "classifier_lr": (
                optimizer.param_groups[-1]["lr"]
            ),
        }
        history.append(row)
        rank = (
            row["val_track_macro_f1"],
            row["val_track_accuracy"],
            row["val_clip_macro_f1"],
            -row["val_loss"],
            -epoch,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_state = trainable_state(model)
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        print(
            f"Fold {fold_index + 1} epoch {epoch}: "
            f"train acc={row['train_accuracy']:.3f}, "
            f"val track acc={row['val_track_accuracy']:.3f}, "
            f"val F1={row['val_track_macro_f1']:.3f}, "
            f"val loss={row['val_loss']:.3f}"
        )
        torch.save(
            {
                "epoch": epoch,
                "trainable_state": trainable_state(model),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best_rank": best_rank,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "stale": stale,
            },
            checkpoint_path,
        )
        pd.DataFrame(history).to_csv(
            fold_dir / "history.csv",
            index=False,
        )
        if stale >= args.patience:
            print(
                f"Fold {fold_index + 1}: early stopping after epoch {epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold_index + 1} did not train.")
    load_trainable_state(model, best_state)
    torch.save(best_state, fold_dir / "best_trainable_state.pt")

    train_result = evaluate(
        model,
        loaders["train"],
        criterion,
        device,
    )
    val_result = evaluate(
        model,
        loaders["val"],
        criterion,
        device,
    )
    test_result = evaluate(
        model,
        loaders["test"],
        criterion,
        device,
        return_embeddings=True,
    )
    if not np.array_equal(
        before_test["indices"],
        test_result["indices"],
    ):
        raise RuntimeError(
            "Pretrained and fine-tuned test embeddings lost alignment."
        )
    drift = parameter_drift(
        initial_last_layer,
        model.last_layer.state_dict(),
    )
    cosine = cosine_similarity_rows(
        before_test["embeddings"],
        test_result["embeddings"],
    )
    elapsed = time.perf_counter() - started
    summary = {
        "fold": fold_index,
        "best_epoch": best_epoch,
        "train_clip": train_result["clip_metrics"],
        "train_track": train_result["track_metrics"],
        "train_loss": train_result["loss"],
        "val_clip": val_result["clip_metrics"],
        "val_track": val_result["track_metrics"],
        "val_loss": val_result["loss"],
        "test_clip": test_result["clip_metrics"],
        "test_track": test_result["track_metrics"],
        "test_loss": test_result["loss"],
        "train_val_track_accuracy_gap": (
            train_result["track_metrics"]["accuracy"]
            - val_result["track_metrics"]["accuracy"]
        ),
        "relative_last_layer_weight_drift": drift,
        "test_embedding_cosine_mean": float(cosine.mean()),
        "test_embedding_cosine_std": float(cosine.std()),
        "elapsed_minutes": elapsed / 60.0,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }
    arrays = {
        "indices": test_result["indices"],
        "labels": test_result["labels"],
        "tracks": test_result["tracks"],
        "probabilities": test_result["probabilities"],
        "before_embeddings": before_test["embeddings"],
        "after_embeddings": test_result["embeddings"],
    }
    np.savez_compressed(completed_path, **arrays)
    json_dump(summary_path, summary)
    checkpoint_path.unlink(missing_ok=True)

    del model, optimizer, scaler, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return {"summary": summary, "arrays": arrays}


def track_mean_embeddings(
    values: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    output_labels = []
    for track in sorted(set(tracks)):
        mask = tracks == track
        means.append(values[mask].mean(axis=0))
        output_labels.append(Counter(labels[mask]).most_common(1)[0][0])
    return np.stack(means), np.asarray(output_labels)


def plot_embedding_comparison(
    before: np.ndarray,
    after: np.ndarray,
    labels: np.ndarray,
    tracks: np.ndarray,
    classes: np.ndarray,
    path: Path,
    mode: str,
) -> dict[str, float]:
    import umap

    before_track, track_labels = track_mean_embeddings(
        before,
        labels,
        tracks,
    )
    after_track, _ = track_mean_embeddings(after, labels, tracks)
    encoded = LabelEncoder().fit(classes).transform(track_labels)
    outputs: dict[str, float] = {}
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    after_name = (
        "Fine-tuned"
        if mode == "last_layer"
        else "Frozen after classifier training"
    )
    for row, (name, values) in enumerate(
        (("Pretrained", before_track), (after_name, after_track))
    ):
        standardized = (
            values - values.mean(axis=0, keepdims=True)
        ) / np.maximum(values.std(axis=0, keepdims=True), 1e-8)
        projections = (
            ("PCA", PCA(n_components=2).fit_transform(standardized)),
            (
                "t-SNE",
                TSNE(
                    n_components=2,
                    perplexity=5,
                    init="pca",
                    learning_rate="auto",
                    random_state=RANDOM_SEED,
                ).fit_transform(standardized),
            ),
            (
                "UMAP",
                umap.UMAP(
                    n_components=2,
                    n_neighbors=5,
                    min_dist=0.2,
                    random_state=RANDOM_SEED,
                ).fit_transform(standardized),
            ),
        )
        metric_prefix = "pretrained" if row == 0 else "after"
        outputs[f"{metric_prefix}_silhouette"] = float(
            silhouette_score(standardized, encoded)
        )
        outputs[f"{metric_prefix}_davies_bouldin"] = float(
            davies_bouldin_score(standardized, encoded)
        )
        for column, (method, points) in enumerate(projections):
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
                ax=axes[row, column],
                legend=row == 0 and column == 2,
            )
            axes[row, column].set_title(f"{name}: {method}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if not (row == 0 and column == 2):
                legend = axes[row, column].get_legend()
                if legend is not None:
                    legend.remove()
    axes[0, 2].legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(
        "Out-of-fold MERT embeddings before and after last-layer fine-tuning"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return outputs


def plot_curves(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for fold_index in range(5):
        history_path = (
            output_dir
            / "checkpoints"
            / f"fold_{fold_index + 1}"
            / "history.csv"
        )
        frame = pd.read_csv(history_path)
        label = f"fold {fold_index + 1}"
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
            frame["val_track_accuracy"],
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
    fig.suptitle("MERT final-layer fine-tuning across five folds")
    plt.tight_layout()
    plt.savefig(
        output_dir / "figures" / "finetuning_training_curves.png",
        dpi=170,
    )
    plt.close()


def plot_confusion(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    path: Path,
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
    ax.set_title("MERT last-layer fine-tuning: five-fold test clips")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def decision_text(
    track_accuracy: float,
    baseline: float,
    gap: float,
) -> str:
    if track_accuracy >= 0.40 and track_accuracy > baseline:
        if gap <= 0.30:
            return (
                "Partial fine-tuning is promising: it reached at least 40% "
                "and did not show an extreme train-validation gap. The next "
                "step is to expand the dataset and repeat partial fine-tuning."
            )
        return (
            "Accuracy crossed 40%, but the train-validation gap is still "
            "large. Treat the result as promising but unstable; first add "
            "stronger controls or more recordings before unfreezing more."
        )
    if track_accuracy <= baseline:
        return (
            "Unfreezing the final layer did not beat the 36% frozen baseline. "
            "The current data and representation are the main bottlenecks; "
            "expand the number of original recordings before deeper tuning."
        )
    return (
        "The result improved slightly but stayed below 40%. It is "
        "inconclusive because one recording changes accuracy by four points; "
        "expand the dataset before making the model more trainable."
    )


def write_report(
    path: Path,
    args: argparse.Namespace,
    selected: dict[str, list[dict[str, str]]],
    overall: dict[str, Any],
    fold_frame: pd.DataFrame,
    cluster: dict[str, float],
) -> None:
    baseline_gap = (
        args.baseline_train_accuracy - args.baseline_val_accuracy
    )
    track_accuracy = overall["track"]["accuracy"]
    mean_gap = float(fold_frame["train_val_track_accuracy_gap"].mean())
    decision = decision_text(
        track_accuracy,
        args.baseline_track_accuracy,
        mean_gap,
    )
    lines = [
        (
            "# MERT Final-Layer Fine-Tuning Report"
            if args.mode == "last_layer"
            else "# MERT Matched Frozen-Control Report"
        ),
        "",
        "## Question",
        "",
        (
            "Does allowing only MERT's final transformer layer to learn "
            "improve raga classification over the frozen 36% baseline, or "
            "does it mainly increase overfitting?"
            if args.mode == "last_layer"
            else
            "What accuracy does the identical raw-audio training loop produce "
            "when the entire MERT backbone is frozen?"
        ),
        "",
        "## Dataset And Split",
        "",
        f"- Ragas: {', '.join(selected)}",
        "- Five original recordings per raga",
        "- Four 30-second, 24 kHz clips per recording",
        "- Five folds: 3 training, 1 validation and 1 test recording per raga",
        "- Every recording is used as unseen test data exactly once",
        "- No recording crosses train, validation and test",
        "",
        "## Trainable And Frozen Parts",
        "",
        (
            "- Frozen: MERT CNN feature encoder and transformer layers 1-11"
            if args.mode == "last_layer"
            else "- Frozen: the complete MERT backbone"
        ),
        (
            "- Trainable: transformer layer 12 and the external classifier"
            if args.mode == "last_layer"
            else "- Trainable: only the external classifier"
        ),
        f"- Classifier: 768 -> {args.hidden_dim} -> 5, GELU, "
        f"dropout {args.dropout}",
        "",
        "## Hyperparameters",
        "",
        "| Variable | Value |",
        "|---|---:|",
        f"| MERT final-layer learning rate | "
        f"{args.backbone_lr if args.mode == 'last_layer' else 0.0} |",
        f"| Classifier learning rate | {args.classifier_lr} |",
        f"| Weight decay | {args.weight_decay} |",
        f"| Label smoothing | {args.label_smoothing} |",
        f"| Batch size | {args.batch_size} |",
        f"| Gradient accumulation | {args.gradient_accumulation} |",
        f"| Maximum epochs | {args.epochs} |",
        f"| Early-stopping patience | {args.patience} |",
        f"| Random seed | {RANDOM_SEED} |",
        "",
        "## Main Results",
        "",
        "| Experiment | Track accuracy | Track macro F1 | "
        "Train accuracy | Validation accuracy | Train-validation gap |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Frozen MERT + regularized classifier | "
        f"{args.baseline_track_accuracy:.3f} | "
        f"{args.baseline_track_f1:.3f} | "
        f"{args.baseline_train_accuracy:.3f} | "
        f"{args.baseline_val_accuracy:.3f} | {baseline_gap:.3f} |",
        f"| {'Last-layer fine-tuned MERT' if args.mode == 'last_layer' else 'Matched frozen control'} | "
        f"{track_accuracy:.3f} | "
        f"{overall['track']['macro_f1']:.3f} | "
        f"{fold_frame['train_track_accuracy'].mean():.3f} | "
        f"{fold_frame['val_track_accuracy'].mean():.3f} | {mean_gap:.3f} |",
        "",
        f"- Fine-tuned clip accuracy: {overall['clip']['accuracy']:.3f}",
        f"- Fine-tuned track top-3 accuracy: "
        f"{overall['track']['top_3_accuracy']:.3f}",
        f"- Correct unseen recordings: "
        f"{round(track_accuracy * 25)}/25",
        f"- Change from frozen baseline: "
        f"{track_accuracy - args.baseline_track_accuracy:+.3f}",
        "",
        "## Fold Results",
        "",
        "| Fold | Best epoch | Train track acc | Val track acc | "
        "Test track acc | Test macro F1 | Weight drift | Embedding cosine |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in fold_frame.iterrows():
        lines.append(
            f"| {int(row['fold']) + 1} | {int(row['best_epoch'])} | "
            f"{row['train_track_accuracy']:.3f} | "
            f"{row['val_track_accuracy']:.3f} | "
            f"{row['test_track_accuracy']:.3f} | "
            f"{row['test_track_macro_f1']:.3f} | "
            f"{row['relative_last_layer_weight_drift']:.4f} | "
            f"{row['test_embedding_cosine_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Representation Change And Possible Drawbacks",
            "",
            f"- Mean relative change in final-layer weights: "
            f"{fold_frame['relative_last_layer_weight_drift'].mean():.4f}",
            f"- Mean cosine similarity between pretrained and fine-tuned "
            f"test embeddings: "
            f"{fold_frame['test_embedding_cosine_mean'].mean():.4f}",
            f"- Pretrained out-of-fold silhouette: "
            f"{cluster['pretrained_silhouette']:.3f}",
            f"- Post-training out-of-fold silhouette: "
            f"{cluster['after_silhouette']:.3f}",
            "",
            "A lower embedding cosine or larger weight change means the "
            "representation moved further from the pretrained model. This "
            "measures drift, not proven catastrophic forgetting. Proving "
            "forgetting would require evaluating unrelated music tasks before "
            "and after fine-tuning.",
            "",
            "Other drawbacks are higher GPU cost, slower experiments, greater "
            "sensitivity to the validation recordings and a larger risk of "
            "memorizing the small training set.",
            "",
            "## Decision",
            "",
            (
                decision
                if args.mode == "last_layer"
                else
                "This result is the matched frozen control. Compare it with "
                "the last-layer run produced by the same code. The earlier "
                "36% frozen benchmark used validation-selected embedding "
                "layers and regularization settings, so it answers a broader "
                "best-frozen-system question rather than isolating the effect "
                "of unfreezing layer 12."
            ),
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def package_results(output_dir: Path) -> Path:
    archive_path = output_dir / "mert_training_mode_report.zip"
    archive_path.unlink(missing_ok=True)
    with zipfile.ZipFile(
        archive_path,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for item in (
            output_dir / "REPORT.md",
            output_dir / "metrics",
            output_dir / "figures",
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
    return archive_path


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle P100 or T4 GPU and rerun.")
    if args.tracks_per_raga != 5:
        raise RuntimeError("This diagnostic expects five tracks/folds.")
    if args.segment_seconds != 30:
        raise RuntimeError("Keep 30-second clips for a fair comparison.")
    if args.batch_size != 1:
        print("WARNING: batch size 1 is safest for 30-second MERT training.")

    seed_everything(RANDOM_SEED)
    sns.set_theme(style="whitegrid")
    output_dir = Path(args.output_dir).resolve()
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    base_args = argparse.Namespace(
        dataset="saraga_carnatic",
        kaggle_data_root=args.kaggle_data_root,
        num_ragas=args.num_ragas,
        tracks_per_raga=args.tracks_per_raga,
        exclude_raga=args.exclude_raga,
        segments_per_track=args.segments_per_track,
        segment_seconds=args.segment_seconds,
    )
    cache_id = CV.fingerprint(CV.configuration(base_args))
    clips_dir = (
        Path(args.base_output_dir).resolve()
        / "cache"
        / cache_id
        / "clips"
    )
    rows = BASE.collect_kaggle_tracks(
        Path(args.kaggle_data_root).resolve()
    )
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
    CV.validate_clips(manifest, selected, base_args)
    json_dump(metrics_dir / "selected_tracks.json", selected)
    json_dump(metrics_dir / "folds.json", folds)
    print(f"Using 30-second clip cache: {clips_dir}")
    print(f"Clips: {len(manifest)}")

    string_labels = np.asarray(
        [item["raga"] for item in manifest],
        dtype=object,
    )
    track_ids = np.asarray(
        [item["track_id"] for item in manifest],
        dtype=object,
    )
    label_encoder = LabelEncoder().fit(string_labels)
    encoded_labels = label_encoder.transform(string_labels)
    num_classes = len(label_encoder.classes_)

    fold_outputs = []
    for fold_index, fold in enumerate(folds):
        fold_outputs.append(
            train_fold(
                fold_index,
                fold,
                manifest,
                encoded_labels,
                track_ids,
                num_classes,
                args,
                output_dir,
            )
        )

    oof_probabilities = np.full(
        (len(manifest), num_classes),
        np.nan,
        dtype=np.float32,
    )
    before_embeddings = np.full(
        (len(manifest), 768),
        np.nan,
        dtype=np.float32,
    )
    after_embeddings = np.full_like(before_embeddings, np.nan)
    fold_rows = []
    for output in fold_outputs:
        arrays = output["arrays"]
        indices = arrays["indices"].astype(int)
        oof_probabilities[indices] = arrays["probabilities"]
        before_embeddings[indices] = arrays["before_embeddings"]
        after_embeddings[indices] = arrays["after_embeddings"]
        summary = output["summary"]
        fold_rows.append(
            {
                "fold": summary["fold"],
                "best_epoch": summary["best_epoch"],
                "train_loss": summary["train_loss"],
                "val_loss": summary["val_loss"],
                "test_loss": summary["test_loss"],
                "train_clip_accuracy": summary["train_clip"]["accuracy"],
                "train_track_accuracy": summary["train_track"]["accuracy"],
                "val_clip_accuracy": summary["val_clip"]["accuracy"],
                "val_track_accuracy": summary["val_track"]["accuracy"],
                "test_clip_accuracy": summary["test_clip"]["accuracy"],
                "test_track_accuracy": summary["test_track"]["accuracy"],
                "test_track_macro_f1": summary["test_track"]["macro_f1"],
                "train_val_track_accuracy_gap": summary[
                    "train_val_track_accuracy_gap"
                ],
                "relative_last_layer_weight_drift": summary[
                    "relative_last_layer_weight_drift"
                ],
                "test_embedding_cosine_mean": summary[
                    "test_embedding_cosine_mean"
                ],
                "test_embedding_cosine_std": summary[
                    "test_embedding_cosine_std"
                ],
                "elapsed_minutes": summary["elapsed_minutes"],
                "trainable_parameters": summary["trainable_parameters"],
                "total_parameters": summary["total_parameters"],
            }
        )
    if (
        np.isnan(oof_probabilities).any()
        or np.isnan(before_embeddings).any()
        or np.isnan(after_embeddings).any()
    ):
        raise RuntimeError("One or more folds did not produce predictions.")

    clip_metrics = metrics(
        encoded_labels,
        oof_probabilities,
        num_classes,
    )
    track_probabilities, track_labels, _ = aggregate_tracks(
        oof_probabilities,
        encoded_labels,
        track_ids,
    )
    track_metrics = metrics(
        track_labels,
        track_probabilities,
        num_classes,
    )
    overall = {"clip": clip_metrics, "track": track_metrics}
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(metrics_dir / "fold_results.csv", index=False)
    json_dump(metrics_dir / "overall_results.json", overall)
    json_dump(
        metrics_dir / "experiment_variables.json",
        {
            "model": MODEL_NAME,
            "mode": args.mode,
            "num_ragas": args.num_ragas,
            "tracks_per_raga": args.tracks_per_raga,
            "segments_per_track": args.segments_per_track,
            "segment_seconds": args.segment_seconds,
            "sample_rate": SAMPLE_RATE,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "epochs": args.epochs,
            "patience": args.patience,
            "backbone_lr": args.backbone_lr,
            "classifier_lr": args.classifier_lr,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "random_seed": RANDOM_SEED,
            "frozen_baseline": {
                "track_accuracy": args.baseline_track_accuracy,
                "track_macro_f1": args.baseline_track_f1,
                "train_accuracy": args.baseline_train_accuracy,
                "val_accuracy": args.baseline_val_accuracy,
            },
        },
    )

    plot_curves(output_dir)
    plot_confusion(
        encoded_labels,
        oof_probabilities,
        label_encoder.classes_,
        figures_dir / "finetuned_confusion_matrix.png",
    )
    cluster = plot_embedding_comparison(
        before_embeddings,
        after_embeddings,
        string_labels,
        track_ids,
        label_encoder.classes_,
        figures_dir / "embedding_before_after.png",
        args.mode,
    )
    json_dump(metrics_dir / "embedding_drift.json", cluster)
    np.savez_compressed(
        metrics_dir / "oof_predictions.npz",
        labels=encoded_labels,
        tracks=track_ids,
        probabilities=oof_probabilities,
        before_embeddings=before_embeddings,
        after_embeddings=after_embeddings,
        classes=label_encoder.classes_,
    )
    write_report(
        output_dir / "REPORT.md",
        args,
        selected,
        overall,
        fold_frame,
        cluster,
    )
    archive = package_results(output_dir)
    print("\nFinal-layer fine-tuning completed.")
    print(json.dumps(overall, indent=2))
    print(f"Report: {output_dir / 'REPORT.md'}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
