#!/usr/bin/env python3
"""Compile Task 1 metrics into summary tables."""

from __future__ import annotations

import glob
import json
import os

import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")


def load_finetune_results() -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(METRICS_DIR, "*_results.json")):
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        config = result.get("config", {})
        rows.append(
            {
                "type": "finetune",
                "experiment": result.get("experiment"),
                "model": config.get("hf_model"),
                "freeze_mode": config.get("freeze_mode"),
                "lr": config.get("lr"),
                "best_val_acc": result.get("best_val_acc"),
                "test_acc": result.get("test_acc"),
                "test_f1_macro": None,
            }
        )
    return pd.DataFrame(rows)


def load_probe_results() -> pd.DataFrame:
    path = os.path.join(METRICS_DIR, "probe_results.json")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    rows = []
    for model_name, result in results.items():
        rows.append(
            {
                "type": "linear_probe",
                "experiment": f"{model_name}_best_layer_{result['best_layer']}",
                "model": model_name,
                "freeze_mode": "frozen_embeddings",
                "lr": None,
                "best_val_acc": result.get("val_accuracy"),
                "test_acc": result.get("test_accuracy"),
                "test_f1_macro": result.get("test_f1_macro"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(METRICS_DIR, exist_ok=True)
    frames = [load_probe_results(), load_finetune_results()]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        print("No metrics found yet. Run probes or fine-tuning first.")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("test_acc", ascending=False, na_position="last")
    out_csv = os.path.join(METRICS_DIR, "all_experiments_summary.csv")
    df.to_csv(out_csv, index=False)

    print("\nExperiment summary, sorted by test accuracy:\n")
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))
    print(f"\nSaved summary CSV to: {out_csv}")


if __name__ == "__main__":
    main()
