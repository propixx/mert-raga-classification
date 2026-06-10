#!/usr/bin/env python3
"""Preprocess Saraga audio into 24 kHz mono 10-second WAV clips for MERT."""

from __future__ import annotations

import json
import os
from collections import Counter

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


TARGET_SR = 24000
SEGMENT_DURATION = 10
MIN_RAGA_TRACKS = 3
TOP_N_RAGAS = 5
MAX_SEGMENTS_PER_TRACK = 8
MAX_SEGMENTS_PER_RAGA = 40
MAX_SEGMENTS_TOTAL = 200
SILENCE_THRESHOLD = 0.01

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_HOME = os.path.join(PROJECT_ROOT, "data", "raw", "saraga_carnatic")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata_path = os.path.join(DATA_HOME, "track_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run scripts/02_explore_dataset.py before preprocessing."
        )

    with open(metadata_path, encoding="utf-8") as f:
        all_tracks = json.load(f)

    raga_counts = Counter(item["raga"] for item in all_tracks)
    eligible_ragas = [raga for raga, count in raga_counts.most_common() if count >= MIN_RAGA_TRACKS]
    if TOP_N_RAGAS is not None:
        eligible_ragas = eligible_ragas[:TOP_N_RAGAS]

    tracks_to_process = [item for item in all_tracks if item["raga"] in eligible_ragas]
    print(f"Selected {len(eligible_ragas)} ragas: {eligible_ragas}")
    print(f"Tracks to process: {len(tracks_to_process)}")

    manifest = []
    segment_counts = Counter()
    segment_samples = TARGET_SR * SEGMENT_DURATION

    for track in tqdm(tracks_to_process, desc="Preprocessing"):
        if len(manifest) >= MAX_SEGMENTS_TOTAL:
            break

        track_id = str(track["track_id"])
        raga = track["raga"]
        audio_path = track["audio_path"]

        if segment_counts[raga] >= MAX_SEGMENTS_PER_RAGA:
            continue

        if not os.path.exists(audio_path):
            print(f"WARNING: missing audio file, skipping: {audio_path}")
            continue

        try:
            audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        except Exception as exc:
            print(f"WARNING: failed to load {audio_path}: {exc}")
            continue

        num_segments = len(audio) // segment_samples
        saved_for_track = 0
        for segment_idx in range(num_segments):
            if len(manifest) >= MAX_SEGMENTS_TOTAL:
                break
            if segment_counts[raga] >= MAX_SEGMENTS_PER_RAGA:
                break
            if saved_for_track >= MAX_SEGMENTS_PER_TRACK:
                break

            start = segment_idx * segment_samples
            segment = audio[start : start + segment_samples]

            if float(np.max(np.abs(segment))) < SILENCE_THRESHOLD:
                continue

            filename = f"{safe_filename(track_id)}_seg{segment_idx:04d}.wav"
            out_path = os.path.join(OUTPUT_DIR, filename)
            sf.write(out_path, segment, TARGET_SR)

            manifest.append(
                {
                    "path": os.path.abspath(out_path),
                    "raga": raga,
                    "track_id": track_id,
                    "segment_idx": segment_idx,
                    "duration": SEGMENT_DURATION,
                    "sample_rate": TARGET_SR,
                }
            )
            segment_counts[raga] += 1
            saved_for_track += 1

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nPreprocessing complete.")
    print(f"Total saved segments: {len(manifest)}")
    print("Segments per raga:")
    for raga, count in segment_counts.most_common():
        print(f"  {raga}: {count}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
