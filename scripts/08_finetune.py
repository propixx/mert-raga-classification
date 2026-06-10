#!/usr/bin/env python3
"""Fine-tune MERT/CultureMERT with a small supervised raga classifier."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class MERTClassifier(nn.Module):
    """Backbone plus a deliberately simple classification head."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        freeze_mode: str,
        unfreeze_last_n: int = 0,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_dim = int(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(256, num_classes),
        )
        self.apply_freeze_mode(freeze_mode, unfreeze_last_n)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M ({100 * trainable / total:.1f}%)")

    def apply_freeze_mode(self, freeze_mode: str, unfreeze_last_n: int) -> None:
        if freeze_mode == "frozen":
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_mode == "partial":
            for param in self.backbone.parameters():
                param.requires_grad = False
            layers = self.backbone.encoder.layers
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif freeze_mode == "full":
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze_mode: {freeze_mode}")

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_values=input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


class SaragaAudioDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        feature_extractor,
        label_encoder: LabelEncoder,
        max_duration: int = 10,
    ) -> None:
        with open(manifest_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        self.sample_rate = int(feature_extractor.sampling_rate)
        self.max_samples = int(max_duration * self.sample_rate)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.data[idx]
        audio, _ = librosa.load(item["path"], sr=self.sample_rate, mono=True)
        audio = audio.astype(np.float32)

        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)), mode="constant")

        inputs = self.feature_extractor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        input_values = inputs["input_values"].squeeze(0)
        label = int(self.label_encoder.transform([item["raga"]])[0])
        return input_values, label


def find_experiment(configs: dict[str, Any], name: str) -> dict[str, Any]:
    for experiment in configs["experiments"]:
        if experiment["name"] == name:
            return experiment
    raise KeyError(f"Experiment {name!r} not found in config.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for inputs, labels in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_items += int(labels.size(0))

    return total_loss / total_items, total_correct / total_items


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for inputs, labels in tqdm(loader, desc="eval", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_items += int(labels.size(0))

    return total_loss / total_items, total_correct / total_items


def plot_history(history: list[dict], experiment_name: str) -> None:
    figures_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, [item["train_loss"] for item in history], label="train")
    ax1.plot(epochs, [item["val_loss"] for item in history], label="val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{experiment_name}: loss")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.plot(epochs, [item["train_acc"] for item in history], label="train")
    ax2.plot(epochs, [item["val_acc"] for item in history], label="val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{experiment_name}: accuracy")
    ax2.grid(alpha=0.25)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{experiment_name}_training_curves.png"), dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    exp = find_experiment(configs, args.experiment)

    print(f"\nRunning experiment: {exp['name']}")
    print(json.dumps(exp, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_labels_path = os.path.join(PROJECT_ROOT, "embeddings", exp["embedding_source"], "train_labels.npy")
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(
            f"Missing {train_labels_path}. Extract embeddings first so the label set is fixed."
        )
    train_labels = np.load(train_labels_path, allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_labels)
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(exp["hf_model"], trust_remote_code=True)
    model = MERTClassifier(
        model_name=exp["hf_model"],
        num_classes=len(label_encoder.classes_),
        freeze_mode=exp["freeze_mode"],
        unfreeze_last_n=int(exp.get("unfreeze_last_n", 0)),
        classifier_dropout=float(exp.get("classifier_dropout", 0.2)),
    ).to(device)

    train_ds = SaragaAudioDataset(
        os.path.join(PROJECT_ROOT, "data", "splits", "train.json"),
        feature_extractor,
        label_encoder,
    )
    val_ds = SaragaAudioDataset(
        os.path.join(PROJECT_ROOT, "data", "splits", "val.json"),
        feature_extractor,
        label_encoder,
    )
    test_ds = SaragaAudioDataset(
        os.path.join(PROJECT_ROOT, "data", "splits", "test.json"),
        feature_extractor,
        label_encoder,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(exp["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(exp["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(exp["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(exp["lr"]),
        weight_decay=float(exp.get("weight_decay", 0.01)),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(exp["epochs"]))
    criterion = nn.CrossEntropyLoss()

    save_dir = os.path.join(PROJECT_ROOT, "models", "finetuned", exp["name"])
    os.makedirs(save_dir, exist_ok=True)
    metrics_dir = os.path.join(PROJECT_ROOT, "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    best_val_acc = -1.0
    history = []
    for epoch in range(1, int(exp["epochs"]) + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{exp['epochs']}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print(f"Saved best model so far: val_acc={best_val_acc:.4f}")

    best_path = os.path.join(save_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTEST: loss={test_loss:.4f}, accuracy={test_acc:.4f}")

    results = {
        "experiment": exp["name"],
        "config": exp,
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "history": history,
        "classes": list(label_encoder.classes_),
    }
    out_path = os.path.join(metrics_dir, f"{exp['name']}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_history(history, exp["name"])
    print(f"Done. Results saved to: {out_path}")


if __name__ == "__main__":
    main()
