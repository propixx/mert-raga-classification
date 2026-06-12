# MERT vs CultureMERT for Raga Classification

I compared frozen MERT-95M and CultureMERT-95M embeddings on a small balanced
subset of Saraga Carnatic.

## What I Did

- Selected 5 ragas with 5 recordings each.
- Excluded `Rāgamālika` because it contains multiple ragas.
- Took four 30-second clips from every recording.
- Used five track-wise folds, so every recording was tested once.
- Tested all 13 embedding layers.
- Trained a logistic-regression classifier and a small neural classifier.
- Tried different regularization, learning-rate and dropout values.

The MERT and CultureMERT models were kept frozen. Only the classifiers were
trained.

## Results

| Model | Classifier | Clip accuracy | Track accuracy | Track macro F1 |
|---|---|---:|---:|---:|
| MERT | Linear | 22% | 24% | 25.8% |
| CultureMERT | Linear | 24% | 20% | 18.2% |
| MERT | Neural head | 19% | 16% | 12.3% |
| CultureMERT | Neural head | 19% | 24% | 23.8% |

Random guessing for five ragas is 20%.

The result is weak, but the experiment did not fail. MERT and CultureMERT both
produced valid embeddings and the classifiers trained normally. The scores
changed across folds, and the neural classifiers performed much better on
training than validation. This suggests overfitting because there are only
five independent recordings per raga.

CultureMERT did not clearly outperform MERT in this experiment. The best
track accuracy was 24%, reached by MERT with the linear classifier and
CultureMERT with the neural classifier.

More details are in [RESULTS.md](RESULTS.md).

## Main Files

- `notebooks/04_raga_crossval_kaggle_30s.ipynb`: complete Kaggle notebook
- `scripts/11_crossval_benchmark.py`: five-fold experiment
- `notes/model_study.md`: notes on MERT and CultureMERT
- `results/crossval/`: result tables and plots

## Run It

Upload `notebooks/04_raga_crossval_kaggle_30s.ipynb` to Kaggle, enable a GPU
and Internet, then run all cells.

The notebook uses Kaggle's Saraga Carnatic audio copy, checkpoints embedding
extraction, and creates the final report automatically.

## Main Limitation

Five recordings per raga are not enough for a strong conclusion. My next step
would be to add more independent recordings. After that, I can compare the
frozen experiment with partial fine-tuning of the last MERT layers.
