# Model Study: MERT and CultureMERT for Raga Classification

These notes are my starting map before running the experiments. The main question I care about is simple: if two models have almost the same architecture, but one has been adapted with more culturally diverse music, does that show up when the task is Indian classical raga classification?

## Sources I Checked

- MERT paper: https://arxiv.org/abs/2306.00107
- MERT-v1-95M model card: https://huggingface.co/m-a-p/MERT-v1-95M
- MERT-v1-95M config: https://huggingface.co/m-a-p/MERT-v1-95M/blob/main/config.json
- CultureMERT paper: https://arxiv.org/abs/2506.17818
- CultureMERT-95M model card: https://huggingface.co/ntua-slp/CultureMERT-95M

## Architecture Sketch

```text
24 kHz mono waveform
    |
    v
7-layer 1D CNN frontend
    |
    v
Transformer encoder block 1
    |
    v
...
    |
    v
Transformer encoder block 12
    |
    v
hidden states from every depth
    |
    v
mean over time
    |
    v
fixed-size embedding for visualization / probe / classifier
```

The important practical point is that these models do not take spectrogram images as input. They take raw mono audio, resampled to 24 kHz. The hidden states are time sequences, so for a simple raga classifier I reduce each layer by mean-pooling over time.

## MERT-v1-95M

MERT stands for Music undERstanding model with large-scale self-supervised Training. It follows the general style of speech SSL models like HuBERT, but it is trained for music audio instead of speech.

The MERT paper describes a masked prediction setup where the model learns from teacher targets. The paper specifically mentions two kinds of musical/acoustic supervision: an acoustic teacher based on residual vector quantization and a musical teacher based on CQT-style information. I read this as a useful design choice for music, because raga identity depends heavily on pitch movement and tonal behavior, not just general audio texture.

From the MERT-v1-95M config/model card:

- Hugging Face ID: `m-a-p/MERT-v1-95M`
- model type: custom `mert_model`
- frontend: 7 convolutional feature extraction layers
- transformer encoder layers: 12
- hidden size: 768
- attention heads: 12
- output hidden-state count used here: 13, meaning CNN/features plus 12 transformer layers
- sample rate: 24 kHz
- feature rate: 75 Hz

One detail I want to keep honest: the assignment text says MERT was trained on 160K hours. The model card table lists `MERT-v1-95M` as 20K hours and `MERT-v1-330M` as 160K hours, while the page also says the MERT-v1 family trained with more data, up to 160K hours. So in my report I will avoid claiming that the 95M checkpoint itself definitely used 160K hours.

## CultureMERT-95M

CultureMERT-95M is the more interesting comparison for this project. Its model card describes it as a 95M-parameter model based on MERT-v1-95M, adapted through continual pre-training for cross-cultural music representation learning.

From the CultureMERT paper/model card:

- Hugging Face ID: `ntua-slp/CultureMERT-95M`
- base model: `m-a-p/MERT-v1-95M`
- architecture: 12-layer transformer encoder, 768 hidden dimension, 7-layer CNN frontend
- input: raw mono 24 kHz audio
- output: 13 hidden-state layers
- continual pre-training data mix: Greek, Turkish/Ottoman, Hindustani, and Carnatic music
- the model card lists 200 hours Hindustani and 200 hours Carnatic audio in the adaptation mix
- the paper reports improvements on non-Western music tagging benchmarks with limited forgetting on Western benchmarks

This makes the comparison cleaner than comparing two unrelated model families. If CultureMERT does better on Saraga, the likely explanation is not bigger parameter count or different architecture. It would more likely be the cultural adaptation data and continued training.

## Why Layer-Wise Evaluation Matters

I do not want to assume that the last layer is automatically best. In audio SSL models, different layers often specialize differently:

- early layers: local signal shape, onset/timbre, energy, basic spectral information
- middle layers: more stable pitch and melodic contour information
- later layers: more abstract summaries, sometimes better for broad tags but not always best for fine-grained music labels

Raga classification is not the same as genre classification. A model has to preserve pitch relations and characteristic phrases. My guess before running the experiments is that a middle-to-late layer may beat the final layer, but I will let the linear probe results decide.

## Why CultureMERT Might Help Here

Raga is a culturally specific musical category. It is not just a label for instrument, mood, or surface texture. If the pre-training data mostly contains Western popular music, then a model can still learn useful audio features, but it may not naturally organize Indian classical pitch behavior in a way that separates ragas.

CultureMERT is adapted on Indian classical material, including both Hindustani and Carnatic audio according to its model card. That should make it more sensitive to things like:

- long melodic contours
- ornamentation/gamaka-like pitch movement
- drone-centered tonality
- non-Western scale usage
- performance textures common in Indian classical recordings

The experiment will test whether this intuition is visible in:

1. t-SNE/UMAP clustering
2. best-layer linear probe accuracy
3. confusion matrices
4. fine-tuning results

## Experimental Hypotheses

Before seeing results, my hypotheses are:

1. CultureMERT-95M will outperform MERT-v1-95M on the linear probe.
2. The best probe layer may not be the last layer.
3. Fine-tuning should improve over a frozen linear probe, but full fine-tuning may overfit if the dataset is small.
4. Some ragas will be confused because recordings can share tonic-centered textures, similar phrases, or similar instrumentation.
5. Splitting by segment would inflate performance, so all splits must be by original track ID.

## Notes for Implementation

The scripts use this path through the project:

```text
Saraga tracks
  -> metadata exploration
  -> top raga selection
  -> 24 kHz / mono / 10 second clips
  -> track-level train/val/test splits
  -> all-layer embeddings
  -> t-SNE and UMAP
  -> layer-wise logistic regression probe
  -> supervised fine-tuning
  -> final comparison table
```

The track-level split is not optional. If segments from one concert or track leak into both train and test, the classifier might learn recording identity instead of raga structure.
