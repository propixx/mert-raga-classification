#!/usr/bin/env python3
"""Download the Saraga Carnatic dataset using mirdata."""

from __future__ import annotations

import os

import mirdata


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_HOME = os.path.join(PROJECT_ROOT, "data", "raw", "saraga_carnatic")


def main() -> None:
    os.makedirs(DATA_HOME, exist_ok=True)
    print(f"Downloading Saraga Carnatic to: {DATA_HOME}")
    dataset = mirdata.initialize("saraga_carnatic", data_home=DATA_HOME)
    dataset.download()
    dataset.validate()
    print("Download complete.")


if __name__ == "__main__":
    main()
