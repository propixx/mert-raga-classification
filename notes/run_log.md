# Run Log

This is a plain log of what has actually been run locally. I am keeping this separate from the final analysis so that the report does not accidentally sound like experiments were completed before they were.

## 2026-05-29

Created the Task 1 project structure and added:

- environment check
- Saraga download/exploration/preprocessing scripts
- track-level split script
- all-layer embedding extraction script
- t-SNE/UMAP visualization script
- layer-wise linear probe script
- fine-tuning script
- result collation script
- experiment config YAML
- model study notes
- final analysis draft
- README

Ran:

```bash
python scripts/00_check_env.py
```

Observed:

```text
PyTorch version: 2.3.0+cpu
CUDA available: False
WARNING: No GPU detected. CPU works for debugging, but model runs will be slow.
transformers: 4.41.0
librosa: 0.11.0
mirdata: 1.0.0
numpy: 1.26.2
sklearn: 1.8.0
Environment OK.
```

Conclusion:

- The local Python environment can run the scripts.
- This machine does not currently expose a CUDA GPU.
- Full embedding extraction and fine-tuning should be run on a GPU machine if possible.
- Local CPU can still be used for script debugging and small smoke tests.
