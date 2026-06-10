#!/usr/bin/env python3
"""Create track-level train/validation/test splits from preprocessed clips."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split


TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")


def can_stratify(labels: list[str]) -> bool:
    counts = Counter(labels)
    return len(counts) > 1 and min(counts.values()) >= 2


def split_ids(ids: list[str], labels: list[str], test_size: float) -> tuple[list[str], list[str]]:
    stratify = labels if can_stratify(labels) else None
    try:
        return train_test_split(
            ids,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=stratify,
        )
    except ValueError as exc:
        print(f"WARNING: stratified split failed ({exc}). Falling back to random split.")
        return train_test_split(ids, test_size=test_size, random_state=RANDOM_SEED)


def main() -> None:
    os.makedirs(SPLITS_DIR, exist_ok=True)

    manifest_path = os.path.join(PROCESSED_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing {manifest_path}. Run scripts/03_preprocess.py first.")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    track_to_raga: dict[str, str] = {}
    track_to_segments: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest:
        track_id = str(entry["track_id"])
        track_to_raga[track_id] = entry["raga"]
        track_to_segments[track_id].append(entry)

    track_ids = sorted(track_to_raga)
    track_ragas = [track_to_raga[track_id] for track_id in track_ids]

    print(f"Total tracks: {len(track_ids)}")
    print(f"Total segments: {len(manifest)}")
    print(f"Track-level raga distribution: {Counter(track_ragas).most_common()}")

    trainval_ids, test_ids = split_ids(track_ids, track_ragas, TEST_SIZE)
    trainval_labels = [track_to_raga[track_id] for track_id in trainval_ids]
    val_fraction_of_trainval = VAL_SIZE / (1.0 - TEST_SIZE)
    train_ids, val_ids = split_ids(trainval_ids, trainval_labels, val_fraction_of_trainval)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert train_set.isdisjoint(val_set), "LEAK: train and val share tracks"
    assert train_set.isdisjoint(test_set), "LEAK: train and test share tracks"
    assert val_set.isdisjoint(test_set), "LEAK: val and test share tracks"

    split_data = {
        "train": [entry for entry in manifest if str(entry["track_id"]) in train_set],
        "val": [entry for entry in manifest if str(entry["track_id"]) in val_set],
        "test": [entry for entry in manifest if str(entry["track_id"]) in test_set],
    }

    for split_name, entries in split_data.items():
        out_path = os.path.join(SPLITS_DIR, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"\nSplits saved to: {SPLITS_DIR}")
    print(f"Train: {len(split_data['train'])} segments from {len(train_set)} tracks")
    print(f"Val:   {len(split_data['val'])} segments from {len(val_set)} tracks")
    print(f"Test:  {len(split_data['test'])} segments from {len(test_set)} tracks")
    print("\nNo data leakage detected. Splits are clean at track level.")

    for split_name, entries in split_data.items():
        print(f"\n{split_name.title()} raga distribution:")
        for raga, count in Counter(entry["raga"] for entry in entries).most_common():
            print(f"  {raga}: {count} segments")


if __name__ == "__main__":
    main()
