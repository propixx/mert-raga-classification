#!/usr/bin/env python3
"""Train layer-wise logistic-regression probes on frozen embeddings."""

from __future__ import annotations

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, "embeddings")
PROBES_DIR = os.path.join(PROJECT_ROOT, "models", "probes")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")
MODELS = ["mert_95m", "culturemert_95m"]
NUM_LAYERS = {"mert_95m": 13, "culturemert_95m": 13, "mert_330m": 25}


def load_layer(model_name: str, split: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(os.path.join(EMBEDDINGS_DIR, model_name, f"{split}_layer{layer:02d}.npy"))
    labels = np.load(os.path.join(EMBEDDINGS_DIR, model_name, f"{split}_labels.npy"), allow_pickle=True)
    return embeddings, labels


def filter_known_labels(
    embeddings: np.ndarray,
    labels: np.ndarray,
    known_labels: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.array([label in known_labels for label in labels])
    return embeddings[mask], labels[mask]


def train_probe(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[Pipeline, LabelEncoder, float, float, np.ndarray, np.ndarray]:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_val = label_encoder.transform(val_labels)

    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    solver="lbfgs",
                    multi_class="auto",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clf.fit(train_embeddings, y_train)
    y_pred = clf.predict(val_embeddings)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    return clf, label_encoder, acc, f1, y_pred, y_val


def plot_layer_scores(model_name: str, layer_results: list[dict], best_layer: int) -> None:
    layers = [item["layer"] for item in layer_results]
    accs = [item["accuracy"] for item in layer_results]
    f1s = [item["f1_macro"] for item in layer_results]

    plt.figure(figsize=(10, 5))
    plt.plot(layers, accs, "o-", label="Validation accuracy")
    plt.plot(layers, f1s, "s-", label="Validation macro F1")
    plt.axvline(best_layer, color="green", linestyle="--", alpha=0.6, label=f"Best layer {best_layer}")
    plt.xlabel("Layer index")
    plt.ylabel("Score")
    plt.title(f"{model_name}: linear probe by layer")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"{model_name}_layer_probe_accuracy.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    model_name: str,
    label_encoder: LabelEncoder,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_layer: int,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"{model_name}: confusion matrix, layer {best_layer}")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(PROBES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    all_results: dict[str, dict] = {}

    for model_name in MODELS:
        print(f"\n{'=' * 70}\nTraining probes for {model_name}\n{'=' * 70}")
        n_layers = NUM_LAYERS[model_name]
        train_labels = np.load(
            os.path.join(EMBEDDINGS_DIR, model_name, "train_labels.npy"),
            allow_pickle=True,
        )
        known_labels = set(train_labels)

        best_acc = -1.0
        best_layer = -1
        best_clf = None
        best_label_encoder = None
        best_y_pred = None
        best_y_val = None
        layer_results = []

        for layer_idx in range(n_layers):
            train_embeddings, train_labels_layer = load_layer(model_name, "train", layer_idx)
            val_embeddings, val_labels = load_layer(model_name, "val", layer_idx)
            val_embeddings, val_labels = filter_known_labels(val_embeddings, val_labels, known_labels)

            clf, label_encoder, acc, f1, y_pred, y_val = train_probe(
                train_embeddings,
                train_labels_layer,
                val_embeddings,
                val_labels,
            )
            layer_results.append({"layer": layer_idx, "accuracy": float(acc), "f1_macro": float(f1)})
            print(f"Layer {layer_idx:02d}: val_acc={acc:.4f}, val_macro_f1={f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_layer = layer_idx
                best_clf = clf
                best_label_encoder = label_encoder
                best_y_pred = y_pred
                best_y_val = y_val

        assert best_clf is not None
        assert best_label_encoder is not None
        assert best_y_pred is not None
        assert best_y_val is not None

        joblib.dump(best_clf, os.path.join(PROBES_DIR, f"{model_name}_best_probe.pkl"))
        joblib.dump(best_label_encoder, os.path.join(PROBES_DIR, f"{model_name}_label_encoder.pkl"))
        plot_layer_scores(model_name, layer_results, best_layer)
        plot_confusion_matrix(model_name, best_label_encoder, best_y_val, best_y_pred, best_layer)

        test_embeddings, test_labels = load_layer(model_name, "test", best_layer)
        test_embeddings, test_labels = filter_known_labels(test_embeddings, test_labels, set(best_label_encoder.classes_))
        y_test = best_label_encoder.transform(test_labels)
        y_test_pred = best_clf.predict(test_embeddings)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

        print(f"\nBest layer: {best_layer}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test macro F1: {test_f1:.4f}")
        print(
            classification_report(
                y_test,
                y_test_pred,
                labels=np.arange(len(best_label_encoder.classes_)),
                target_names=best_label_encoder.classes_,
                zero_division=0,
            )
        )

        all_results[model_name] = {
            "best_layer": best_layer,
            "val_accuracy": float(best_acc),
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1),
            "layer_results": layer_results,
        }

    out_path = os.path.join(METRICS_DIR, "probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved probe metrics to: {out_path}")
    print("\nComparison")
    print(f"{'Model':<22} {'Best layer':>10} {'Val acc':>10} {'Test acc':>10} {'Test F1':>10}")
    for model_name, result in all_results.items():
        print(
            f"{model_name:<22} {result['best_layer']:>10} "
            f"{result['val_accuracy']:>10.4f} {result['test_accuracy']:>10.4f} "
            f"{result['test_f1_macro']:>10.4f}"
        )


if __name__ == "__main__":
    main()
