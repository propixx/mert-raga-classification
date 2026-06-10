#!/usr/bin/env python3
"""Explore Saraga Carnatic metadata and raga distribution."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import mirdata


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_HOME = os.path.join(PROJECT_ROOT, "data", "raw", "saraga_carnatic")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")


def read_metadata_value(track: Any, keys: list[str]) -> str | None:
    metadata = getattr(track, "metadata", None)
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if value:
                return str(value)
    for key in keys:
        value = getattr(track, key, None)
        if value:
            return str(value)
    return None


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    dataset = mirdata.initialize("saraga_carnatic", data_home=DATA_HOME)
    tracks = dataset.load_tracks()

    track_info = []
    for track_id, track in tracks.items():
        raga = read_metadata_value(track, ["raaga", "raga", "raag"])
        audio_path = getattr(track, "audio_path", None)
        if raga and audio_path:
            track_info.append(
                {
                    "track_id": str(track_id),
                    "raga": raga,
                    "audio_path": os.path.abspath(audio_path),
                }
            )

    raga_counts = Counter(item["raga"] for item in track_info)

    print(f"Total tracks in dataset: {len(tracks)}")
    print(f"Tracks with valid raga + audio path: {len(track_info)}")
    print(f"Unique ragas: {len(raga_counts)}")
    print("\nRagas sorted by track count:")
    for raga, count in raga_counts.most_common():
        print(f"  {raga}: {count}")

    top_ragas = raga_counts.most_common(15)
    if top_ragas:
        plt.figure(figsize=(12, 6))
        plt.barh([name for name, _ in reversed(top_ragas)], [count for _, count in reversed(top_ragas)])
        plt.xlabel("Number of tracks")
        plt.title("Saraga Carnatic: top ragas by track count")
        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, "raga_distribution.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"\nSaved raga distribution plot to: {out_path}")

    metadata_path = os.path.join(DATA_HOME, "track_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(track_info, f, indent=2, ensure_ascii=False)
    print(f"Saved usable track metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
