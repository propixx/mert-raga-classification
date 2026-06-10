# Initial Small Run

Before preparing the balanced benchmark, I ran a fast five-raga experiment with
at most 200 clips. The split was by original track.

Observed linear-probe results:

| Model | Best layer | Validation accuracy | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| MERT-v1-95M | 3 | 0.567 | 0.083 | 0.078 |
| CultureMERT-95M | 3 | 0.467 | 0.125 | 0.078 |

CultureMERT had slightly higher test accuracy, but the result was unstable and
far below validation performance. The likely issue was that the very small
track split did not provide a representative test set. This motivated the new
balanced benchmark with guaranteed per-raga tracks in every split and
track-level probability aggregation.
