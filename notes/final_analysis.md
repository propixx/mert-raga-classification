# Final Analysis

I compared frozen MERT-95M and CultureMERT-95M on five Saraga Carnatic ragas.
I used five track-wise folds so that every original recording was tested once.

The best track accuracy was 24%. MERT reached it with logistic regression and
CultureMERT reached it with the neural classifier. Random accuracy is 20%, so
the result is only slightly above chance.

CultureMERT did not clearly beat MERT. The selected embedding layer also
changed between folds, which shows that the result is unstable with this small
dataset.

The neural classifiers learned the training data much better than the
validation data. This is overfitting. It explains why adding a larger
classifier did not improve the final test result.

My conclusion is not that MERT or CultureMERT is broken. The experiment shows
that five recordings per raga are not enough for these frozen embeddings to
generalize reliably to unseen performances.

The next step should be to collect more independent recordings. Partial
fine-tuning of the last few transformer layers can then be tested against this
frozen baseline.
