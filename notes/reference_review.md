# Notes After Reviewing ritgit24/MERT

I reviewed the public repository at <https://github.com/ritgit24/MERT> because
it tackles a closely related Saraga Hindustani classification problem.

Useful ideas I carried into the new benchmark:

- report top-k accuracy, not only top-1;
- compare embedding spaces quantitatively with silhouette and
  Davies-Bouldin scores;
- save embeddings and figures so the result can be inspected later;
- keep the supervised classifier simple enough to interpret.

I did not copy its experimental setup directly. Two details matter:

1. Its preprocessing and standalone embedding script use 16 kHz audio, while
   the MERT-v1 checkpoints used here expect 24 kHz.
2. Its manifest code shuffles and splits chunks within each label. If several
   chunks came from one original recording, they can enter different splits.
   This may let recording-specific information leak into test evaluation.

The balanced notebook therefore splits original tracks first and creates clips
afterwards. It also reports both clip-level and track-level metrics, since
several clips from one test track are related observations rather than fully
independent examples.
