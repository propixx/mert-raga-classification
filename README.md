# Task 1: Foundation Models for Raga Classification

[![Open quick benchmark in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/propixx/mert-raga-classification/blob/main/notebooks/02_balanced_raga_benchmark_colab.ipynb)

This project compares MERT-v1-95M and CultureMERT-95M on Indian classical raga classification using the Saraga Carnatic dataset.

The core idea is to keep the experiment simple and honest:

1. cut Saraga recordings into fixed-length clips,
2. split by original track so there is no leakage,
3. extract embeddings from every layer of both models,
4. visualize the embedding space,
5. train a lightweight linear probe,
6. then try supervised fine-tuning if the GPU allows it.

## Recommended Notebook

For the current project iteration, use:

[`notebooks/03_raga_benchmark_kaggle_30s.ipynb`](notebooks/03_raga_benchmark_kaggle_30s.ipynb)

It is a reproducible 30-second benchmark designed for a Kaggle T4 or P100.
Compared with the first quick run, it improves the experimental setup by:

- guaranteeing all selected ragas appear in train, validation and test;
- splitting by original recording rather than randomly splitting chunks;
- sampling four 30-second clips from different positions in each recording;
- selecting the best hidden layer using validation macro F1;
- reporting clip-level and track-level top-1/top-3 accuracy;
- adding silhouette and Davies-Bouldin embedding metrics;
- comparing a linear probe with a `768 -> 256 -> 6` external neural head;
- caching each completed model/split embedding file for interrupted runs.

The notebook is intentionally limited to six well-supported ragas, six tracks
per raga and approximately 144 clips. The original MERT and CultureMERT
backbones remain frozen. Only the lightweight classifiers are trained.

The earlier eight-second Colab notebook is kept as a quick pipeline check:
[`notebooks/02_balanced_raga_benchmark_colab.ipynb`](notebooks/02_balanced_raga_benchmark_colab.ipynb).

### Related Work Reviewed

I also reviewed [`ritgit24/MERT`](https://github.com/ritgit24/MERT), which uses
the original MERT implementation on Saraga Hindustani and reports top-k and
embedding-cluster metrics. This project keeps those useful evaluation ideas but
uses MERT's required 24 kHz input and enforces track-level leakage protection.

## Models

Primary models:

- `m-a-p/MERT-v1-95M`
- `ntua-slp/CultureMERT-95M`

Optional extension:

- `m-a-p/MERT-v1-330M`

Both primary models use raw mono 24 kHz audio and produce 13 hidden-state outputs: one frontend representation plus 12 transformer layers.

## Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Then check the environment:

```bash
python scripts/00_check_env.py
```

## Reproduction Order

Run these commands from the `task1_raga_classification` directory.

```bash
python scripts/00_check_env.py

python scripts/01_download_saraga.py
python scripts/02_explore_dataset.py
python scripts/03_preprocess.py
python scripts/04_create_splits.py

python scripts/05_extract_embeddings.py --model mert_95m
python scripts/05_extract_embeddings.py --model culturemert_95m

python scripts/06_visualize_embeddings.py
python scripts/07_train_probe.py

python scripts/08_finetune.py --config configs/experiment_configs.yaml --experiment mert_95m_frozen
python scripts/08_finetune.py --config configs/experiment_configs.yaml --experiment culturemert_95m_frozen

python scripts/08_finetune.py --config configs/experiment_configs.yaml --experiment mert_95m_partial_unfreeze
python scripts/08_finetune.py --config configs/experiment_configs.yaml --experiment mert_95m_full_finetune
python scripts/08_finetune.py --config configs/experiment_configs.yaml --experiment culturemert_95m_partial_unfreeze
python scripts/08_finetune.py --config configs/experiment_configs.yaml --experiment culturemert_95m_full_finetune

python scripts/09_evaluate.py
```

If GPU memory is limited, run only the frozen fine-tuning experiments after the probe. The embedding/probe comparison is the most important baseline.

## Output Locations

- raw Saraga data: `data/raw/saraga_carnatic`
- processed 24 kHz clips: `data/processed`
- train/val/test manifests: `data/splits`
- embeddings: `embeddings/mert_95m` and `embeddings/culturemert_95m`
- probe models: `models/probes`
- fine-tuned checkpoints: `models/finetuned`
- figures: `results/figures`
- metrics: `results/metrics`
- study notes: `notes/model_study.md`
- final analysis draft: `notes/final_analysis.md`

## Important Method Choices

### Track-Level Splitting

All splits are by `track_id`, not by segment. This matters because two clips from the same recording can share performer, room, microphone, tonic, and local melodic material. Segment-level splitting would make the task look easier than it really is.

### Layer-Wise Probing

The probe script tests every layer instead of assuming the final layer is best. This is useful because raga identity may depend on pitch contour information that could be stronger in middle layers.

### No Fake Results

The final analysis file is currently a draft. It should be filled only after the scripts produce real metrics and figures.

## Current Status

The 30-second Kaggle notebook and benchmark code are ready. Actual scores must
come from a completed Kaggle GPU run; no result is filled in beforehand.

## Known Risks

- Saraga download can be slow or fail depending on mirror availability.
- MERT embedding extraction is compute-heavy without a GPU.
- Full fine-tuning may run out of VRAM.
- If only CPU is available, prioritize the linear-probe pipeline and document the hardware limitation.

## References

- MERT paper: https://arxiv.org/abs/2306.00107
- MERT-v1-95M model card: https://huggingface.co/m-a-p/MERT-v1-95M
- CultureMERT paper: https://arxiv.org/abs/2506.17818
- CultureMERT-95M model card: https://huggingface.co/ntua-slp/CultureMERT-95M
