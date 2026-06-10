# Colab Runner Notes

Use Colab for the heavy parts of Task 1.

Why Colab:

- the local machine currently has CPU-only PyTorch
- the Kaggle CLI token on this machine returns `401 Unauthorized`
- Colab usually gives a T4 GPU on the free tier, which is enough for MERT-95M embeddings and frozen/partial fine-tuning

## What to Upload

Upload this zip to Colab:

```text
task1_raga_classification_colab.zip
```

Then open:

```text
notebooks/task1_colab_runner.ipynb
```

## Colab Runtime

In Colab, select:

```text
Runtime -> Change runtime type -> T4 GPU
```

Then run the notebook cells from top to bottom.

## Suggested Order in Colab

Start with the reliable core:

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
python scripts/09_evaluate.py
```

Then try fine-tuning:

```bash
python scripts/08_finetune.py --config configs/experiment_configs.yaml --experiment mert_95m_frozen
python scripts/08_finetune.py --config configs/experiment_configs.yaml --experiment culturemert_95m_frozen
```

Only after those work, try partial and full fine-tuning.

## Honest Reporting Rule

Do not fill the final analysis with guessed numbers. Copy the real metrics from:

```text
results/metrics/probe_results.json
results/metrics/all_experiments_summary.csv
```

and the plots from:

```text
results/figures/
```
