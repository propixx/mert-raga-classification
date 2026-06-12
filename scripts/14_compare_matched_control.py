#!/usr/bin/env python3
"""Compare matched frozen and last-layer MERT training runs."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--finetuned-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/mert_matched_comparison",
    )
    parser.add_argument(
        "--optimized-frozen-reference",
        type=float,
        default=0.36,
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_run(path: Path) -> dict[str, Any]:
    metrics_dir = path / "metrics"
    required = (
        metrics_dir / "overall_results.json",
        metrics_dir / "fold_results.csv",
        metrics_dir / "experiment_variables.json",
        metrics_dir / "oof_predictions.npz",
    )
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise RuntimeError(
            "Run output is incomplete. Missing: " + ", ".join(missing)
        )
    with np.load(
        metrics_dir / "oof_predictions.npz",
        allow_pickle=True,
    ) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    return {
        "path": path,
        "overall": load_json(metrics_dir / "overall_results.json"),
        "folds": pd.read_csv(metrics_dir / "fold_results.csv"),
        "variables": load_json(
            metrics_dir / "experiment_variables.json"
        ),
        "arrays": arrays,
        "drift": (
            load_json(metrics_dir / "embedding_drift.json")
            if (metrics_dir / "embedding_drift.json").exists()
            else {}
        ),
    }


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


def exact_mcnemar(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    smaller = min(b, c)
    tail = sum(
        math.comb(discordant, index)
        for index in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def validate_pair(
    frozen: dict[str, Any],
    finetuned: dict[str, Any],
) -> None:
    frozen_arrays = frozen["arrays"]
    tuned_arrays = finetuned["arrays"]
    for key in ("labels", "tracks", "classes"):
        if not np.array_equal(frozen_arrays[key], tuned_arrays[key]):
            raise RuntimeError(
                f"Matched runs do not share the same {key}."
            )
    keys = (
        "num_ragas",
        "tracks_per_raga",
        "segments_per_track",
        "segment_seconds",
        "batch_size",
        "gradient_accumulation",
        "epochs",
        "patience",
        "classifier_lr",
        "hidden_dim",
        "dropout",
        "weight_decay",
        "label_smoothing",
        "random_seed",
    )
    unequal = [
        key
        for key in keys
        if frozen["variables"].get(key)
        != finetuned["variables"].get(key)
    ]
    if unequal:
        raise RuntimeError(
            "The runs are not matched for: " + ", ".join(unequal)
        )
    if frozen["variables"].get("mode") != "frozen":
        raise RuntimeError("The frozen directory is not a frozen-mode run.")
    if finetuned["variables"].get("mode") != "last_layer":
        raise RuntimeError(
            "The fine-tuned directory is not a last-layer run."
        )


def comparison_decision(
    frozen_accuracy: float,
    tuned_accuracy: float,
    frozen_gap: float,
    tuned_gap: float,
) -> str:
    delta = tuned_accuracy - frozen_accuracy
    if delta >= 0.08 and tuned_gap <= frozen_gap + 0.05:
        return (
            "Unfreezing layer 12 helped under the matched training procedure. "
            "Expand the dataset and repeat partial fine-tuning before "
            "unfreezing additional layers."
        )
    if delta <= -0.08:
        return (
            "Unfreezing layer 12 harmed generalization under the matched "
            "procedure. Keep MERT frozen and expand the number of original "
            "recordings."
        )
    return (
        "The matched difference is too small or unstable to support deeper "
        "fine-tuning. Expand the dataset and retain the frozen model as the "
        "safer baseline."
    )


def plot_summary(
    summary: pd.DataFrame,
    optimized_reference: float,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#4C78A8", "#D65F5F"]
    axes[0].bar(
        summary["experiment"],
        summary["track_accuracy"],
        color=colors,
    )
    axes[0].axhline(
        optimized_reference,
        color="black",
        linestyle="--",
        label=f"Earlier optimized frozen result ({optimized_reference:.0%})",
    )
    axes[0].axhline(
        0.20,
        color="gray",
        linestyle=":",
        label="Chance (20%)",
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Track-level accuracy")
    axes[0].set_title("Matched overall comparison")
    axes[0].legend(fontsize=8)

    axes[1].bar(
        summary["experiment"],
        summary["train_val_gap"],
        color=colors,
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Train-validation track accuracy gap")
    axes[1].set_title("Overfitting gap")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_fold_comparison(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(frame))
    width = 0.36
    ax.bar(
        positions - width / 2,
        frame["frozen_test_track_accuracy"],
        width,
        label="Frozen",
        color="#4C78A8",
    )
    ax.bar(
        positions + width / 2,
        frame["last_layer_test_track_accuracy"],
        width,
        label="Last layer trainable",
        color="#D65F5F",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"Fold {int(value) + 1}" for value in frame["fold"]]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test track accuracy")
    ax.set_title("Matched test accuracy in each fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_confusions(
    labels: np.ndarray,
    frozen_predictions: np.ndarray,
    tuned_predictions: np.ndarray,
    classes: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, predictions, title in (
        (axes[0], frozen_predictions, "Matched frozen control"),
        (axes[1], tuned_predictions, "Last layer trainable"),
    ):
        matrix = confusion_matrix(
            labels,
            predictions,
            labels=np.arange(len(classes)),
        )
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
    fig.suptitle("Track-level out-of-fold confusion matrices")
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def write_report(
    path: Path,
    summary: pd.DataFrame,
    fold_frame: pd.DataFrame,
    pair_stats: dict[str, Any],
    optimized_reference: float,
    decision: str,
) -> None:
    frozen = summary.iloc[0]
    tuned = summary.iloc[1]
    lines = [
        "# Matched Frozen vs Last-Layer MERT Comparison",
        "",
        "## Why This Control Was Needed",
        "",
        "The earlier optimized frozen system reached 36%, while the first "
        "last-layer fine-tuning run reached 16%. Those systems did not use "
        "exactly the same classifier-selection procedure. This control runs "
        "the same raw-audio code twice with identical data, folds, classifier, "
        "initialization, batch order, optimizer settings and early stopping. "
        "The only intended difference is whether MERT transformer layer 12 is "
        "frozen or trainable.",
        "",
        "## Overall Results",
        "",
        "| Experiment | Track accuracy | Track macro F1 | Clip accuracy | "
        "Mean train accuracy | Mean validation accuracy | Train-validation gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['experiment']} | {row['track_accuracy']:.3f} | "
            f"{row['track_macro_f1']:.3f} | "
            f"{row['clip_accuracy']:.3f} | "
            f"{row['train_accuracy']:.3f} | "
            f"{row['val_accuracy']:.3f} | "
            f"{row['train_val_gap']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Earlier optimized frozen reference: "
            f"{optimized_reference:.3f}",
            f"- Matched last-layer minus frozen accuracy: "
            f"{tuned['track_accuracy'] - frozen['track_accuracy']:+.3f}",
            f"- Frozen correct / fine-tuned wrong: {pair_stats['b']}",
            f"- Frozen wrong / fine-tuned correct: {pair_stats['c']}",
            f"- Exact paired McNemar p-value: {pair_stats['p_value']:.3f}",
            "",
            "With only 25 test recordings, one recording changes overall "
            "accuracy by four percentage points. The paired p-value is "
            "descriptive here; this experiment is too small for a strong "
            "statistical conclusion.",
            "",
            "## Fold Results",
            "",
            "| Fold | Frozen test acc | Last-layer test acc | Difference | "
            "Frozen gap | Last-layer gap |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in fold_frame.iterrows():
        lines.append(
            f"| {int(row['fold']) + 1} | "
            f"{row['frozen_test_track_accuracy']:.3f} | "
            f"{row['last_layer_test_track_accuracy']:.3f} | "
            f"{row['test_accuracy_difference']:+.3f} | "
            f"{row['frozen_train_val_gap']:.3f} | "
            f"{row['last_layer_train_val_gap']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            decision,
            "",
            "The earlier 36% result remains useful as the best frozen system "
            "tested so far because it selected informative hidden layers and "
            "regularization settings. The matched control answers a narrower "
            "question: whether updating layer 12 helps when everything else "
            "is held fixed.",
            "",
            "## Next Step",
            "",
            "Do not unfreeze additional MERT layers unless the matched "
            "last-layer run clearly and consistently beats the matched frozen "
            "control. Otherwise, increase the number of original recordings "
            "per raga and repeat the frozen benchmark first.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def package(output_dir: Path) -> Path:
    path = output_dir / "mert_matched_control_report.zip"
    path.unlink(missing_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
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
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frozen = load_run(Path(args.frozen_dir).resolve())
    tuned = load_run(Path(args.finetuned_dir).resolve())
    validate_pair(frozen, tuned)

    rows = []
    for name, run in (
        ("Matched frozen control", frozen),
        ("Last layer trainable", tuned),
    ):
        folds = run["folds"]
        rows.append(
            {
                "experiment": name,
                "track_accuracy": run["overall"]["track"]["accuracy"],
                "track_macro_f1": run["overall"]["track"]["macro_f1"],
                "clip_accuracy": run["overall"]["clip"]["accuracy"],
                "clip_macro_f1": run["overall"]["clip"]["macro_f1"],
                "train_accuracy": folds[
                    "train_track_accuracy"
                ].mean(),
                "val_accuracy": folds["val_track_accuracy"].mean(),
                "train_val_gap": folds[
                    "train_val_track_accuracy_gap"
                ].mean(),
                "mean_best_epoch": folds["best_epoch"].mean(),
                "mean_elapsed_minutes": folds[
                    "elapsed_minutes"
                ].mean(),
            }
        )
    summary = pd.DataFrame(rows)

    frozen_folds = frozen["folds"].add_prefix("frozen_")
    tuned_folds = tuned["folds"].add_prefix("last_layer_")
    fold_frame = frozen_folds.merge(
        tuned_folds,
        left_on="frozen_fold",
        right_on="last_layer_fold",
    )
    fold_frame["fold"] = fold_frame["frozen_fold"]
    fold_frame["test_accuracy_difference"] = (
        fold_frame["last_layer_test_track_accuracy"]
        - fold_frame["frozen_test_track_accuracy"]
    )
    fold_frame = fold_frame[
        [
            "fold",
            "frozen_test_track_accuracy",
            "last_layer_test_track_accuracy",
            "test_accuracy_difference",
            "frozen_train_val_track_accuracy_gap",
            "last_layer_train_val_track_accuracy_gap",
        ]
    ].rename(
        columns={
            "frozen_train_val_track_accuracy_gap": "frozen_train_val_gap",
            "last_layer_train_val_track_accuracy_gap": "last_layer_train_val_gap",
        }
    )

    frozen_arrays = frozen["arrays"]
    tuned_arrays = tuned["arrays"]
    frozen_track_prob, track_labels, track_ids = aggregate_tracks(
        frozen_arrays["probabilities"],
        frozen_arrays["labels"],
        frozen_arrays["tracks"],
    )
    tuned_track_prob, tuned_track_labels, tuned_track_ids = aggregate_tracks(
        tuned_arrays["probabilities"],
        tuned_arrays["labels"],
        tuned_arrays["tracks"],
    )
    if (
        not np.array_equal(track_labels, tuned_track_labels)
        or not np.array_equal(track_ids, tuned_track_ids)
    ):
        raise RuntimeError("Track aggregation lost matched ordering.")
    frozen_prediction = frozen_track_prob.argmax(axis=1)
    tuned_prediction = tuned_track_prob.argmax(axis=1)
    frozen_correct = frozen_prediction == track_labels
    tuned_correct = tuned_prediction == track_labels
    b = int(np.sum(frozen_correct & ~tuned_correct))
    c = int(np.sum(~frozen_correct & tuned_correct))
    pair_stats = {
        "b": b,
        "c": c,
        "discordant": b + c,
        "p_value": exact_mcnemar(b, c),
    }
    per_track = pd.DataFrame(
        {
            "track_id": track_ids,
            "true_label": track_labels,
            "frozen_prediction": frozen_prediction,
            "last_layer_prediction": tuned_prediction,
            "frozen_correct": frozen_correct,
            "last_layer_correct": tuned_correct,
        }
    )

    frozen_gap = float(summary.iloc[0]["train_val_gap"])
    tuned_gap = float(summary.iloc[1]["train_val_gap"])
    decision = comparison_decision(
        float(summary.iloc[0]["track_accuracy"]),
        float(summary.iloc[1]["track_accuracy"]),
        frozen_gap,
        tuned_gap,
    )

    summary.to_csv(metrics_dir / "matched_summary.csv", index=False)
    fold_frame.to_csv(
        metrics_dir / "matched_fold_comparison.csv",
        index=False,
    )
    per_track.to_csv(
        metrics_dir / "matched_track_predictions.csv",
        index=False,
    )
    json_dump(metrics_dir / "paired_statistics.json", pair_stats)
    json_dump(
        metrics_dir / "decision.json",
        {"decision": decision},
    )
    plot_summary(
        summary,
        args.optimized_frozen_reference,
        figures_dir / "matched_overall_comparison.png",
    )
    plot_fold_comparison(
        fold_frame,
        figures_dir / "matched_fold_comparison.png",
    )
    plot_confusions(
        track_labels,
        frozen_prediction,
        tuned_prediction,
        frozen_arrays["classes"],
        figures_dir / "matched_track_confusions.png",
    )
    write_report(
        output_dir / "REPORT.md",
        summary,
        fold_frame,
        pair_stats,
        args.optimized_frozen_reference,
        decision,
    )
    archive = package(output_dir)
    print(summary.to_string(index=False))
    print(f"\nDecision: {decision}")
    print(f"Report: {output_dir / 'REPORT.md'}")
    print(f"Archive: {archive}")


if __name__ == "__main__":
    main()
