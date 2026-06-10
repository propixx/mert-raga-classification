#!/usr/bin/env python3
"""Small environment check before running the heavier experiments."""

from __future__ import annotations

import importlib


def print_version(package_name: str) -> None:
    module = importlib.import_module(package_name)
    version = getattr(module, "__version__", "unknown")
    print(f"{package_name}: {version}")


def main() -> None:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected. CPU works for debugging, but model runs will be slow.")

    for package in ["transformers", "librosa", "mirdata", "numpy", "sklearn"]:
        print_version(package)

    print("Environment OK.")


if __name__ == "__main__":
    main()
