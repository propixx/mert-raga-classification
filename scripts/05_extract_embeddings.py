#!/usr/bin/env python3
"""Extract mean-pooled hidden-state embeddings from MERT/CultureMERT."""

from __future__ import annotations

import argparse
import json
import os

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


MODEL_REGISTRY = {
    "mert_95m": {
        "hf_name": "m-a-p/MERT-v1-95M",
        "num_layers": 13,
        "hidden_dim": 768,
    },
    "culturemert_95m": {
        "hf_name": "ntua-slp/CultureMERT-95M",
        "num_layers": 13,
        "hidden_dim": 768,
    },
    "mert_330m": {
        "hf_name": "m-a-p/MERT-v1-330M",
        "num_layers": 25,
        "hidden_dim": 1024,
    },
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def extract_all_layers(
    audio_path: str,
    model: torch.nn.Module,
    feature_extractor,
    device: torch.device,
) -> list[np.ndarray]:
    audio, sample_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate, mono=True)
    inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return [
        hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        for hidden_state in outputs.hidden_states
    ]


def save_split_embeddings(
    split_name: str,
    manifest: list[dict],
    output_dir: str,
    model: torch.nn.Module,
    feature_extractor,
    device: torch.device,
    expected_layers: int,
    expected_dim: int,
) -> None:
    layer_rows: list[list[np.ndarray]] = [[] for _ in range(expected_layers)]
    labels: list[str] = []
    metadata: list[dict] = []
    failed: list[dict] = []

    for item in tqdm(manifest, desc=f"{split_name}"):
        try:
            embeddings = extract_all_layers(item["path"], model, feature_extractor, device)
            if len(embeddings) != expected_layers:
                raise RuntimeError(f"expected {expected_layers} layers, got {len(embeddings)}")
            for layer_idx, embedding in enumerate(embeddings):
                if embedding.shape[0] != expected_dim:
                    raise RuntimeError(
                        f"layer {layer_idx} expected dim {expected_dim}, got {embedding.shape[0]}"
                    )
                layer_rows[layer_idx].append(embedding)
            labels.append(item["raga"])
            metadata.append(item)
        except Exception as exc:
            failed.append({"path": item.get("path"), "error": str(exc), "raga": item.get("raga")})
            print(f"FAILED: {item.get('path')}: {exc}")

    if not labels:
        raise RuntimeError(f"No embeddings were extracted for split {split_name}.")

    for layer_idx, rows in enumerate(layer_rows):
        arr = np.stack(rows, axis=0).astype(np.float32)
        np.save(os.path.join(output_dir, f"{split_name}_layer{layer_idx:02d}.npy"), arr)

    np.save(os.path.join(output_dir, f"{split_name}_labels.npy"), np.array(labels, dtype=object))
    with open(os.path.join(output_dir, f"{split_name}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, f"{split_name}_failed.json"), "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)

    print(
        f"Saved {len(labels)} examples for {split_name}; "
        f"failed={len(failed)}; output={output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--splits-dir", default=os.path.join(PROJECT_ROOT, "data", "splits"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing layer files.")
    args = parser.parse_args()

    model_info = MODEL_REGISTRY[args.model]
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, "embeddings", args.model)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_info['hf_name']} on {device}")
    model = AutoModel.from_pretrained(model_info["hf_name"], trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_info["hf_name"], trust_remote_code=True)
    model.eval().to(device)

    for split_name in ["train", "val", "test"]:
        final_layer_path = os.path.join(
            output_dir, f"{split_name}_layer{model_info['num_layers'] - 1:02d}.npy"
        )
        if os.path.exists(final_layer_path) and not args.force:
            print(f"Skipping {split_name}; embeddings already exist. Use --force to overwrite.")
            continue

        split_path = os.path.join(args.splits_dir, f"{split_name}.json")
        if not os.path.exists(split_path):
            print(f"WARNING: missing split file, skipping: {split_path}")
            continue

        with open(split_path, encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"\nProcessing {split_name}: {len(manifest)} segments")
        save_split_embeddings(
            split_name=split_name,
            manifest=manifest,
            output_dir=output_dir,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            expected_layers=model_info["num_layers"],
            expected_dim=model_info["hidden_dim"],
        )

    print("\nEmbedding extraction complete.")


if __name__ == "__main__":
    main()
