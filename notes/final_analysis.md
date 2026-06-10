# Final Analysis Draft

This file is intentionally a draft until the scripts have actually been run. I am not filling in fake numbers.

## Questions to Answer After Running

1. Which MERT-v1-95M layer gives the best validation accuracy for raga classification?
2. Which CultureMERT-95M layer gives the best validation accuracy?
3. Does CultureMERT improve over MERT on test accuracy and macro F1?
4. Do the visualizations show cleaner raga clusters for one model?
5. Which ragas are confused most often?
6. Does frozen probing hold up against fine-tuning?
7. Does partial unfreezing help, or does it overfit?
8. Which learning rate is safest for this dataset size?

## Results Table to Fill

| Experiment | Best layer | Validation accuracy | Test accuracy | Test macro F1 | Notes |
|---|---:|---:|---:|---:|---|
| MERT-95M linear probe | TODO | TODO | TODO | TODO | TODO |
| CultureMERT-95M linear probe | TODO | TODO | TODO | TODO | TODO |
| MERT-95M frozen fine-tune | n/a | TODO | TODO | TODO | TODO |
| CultureMERT-95M frozen fine-tune | n/a | TODO | TODO | TODO | TODO |
| MERT-95M partial/full fine-tune | n/a | TODO | TODO | TODO | if GPU allows |
| CultureMERT-95M partial/full fine-tune | n/a | TODO | TODO | TODO | if GPU allows |

## Early Interpretation Template

After the experiments finish, I will compare the numbers against the starting hypothesis:

- If CultureMERT wins clearly, I will treat that as evidence that culturally adapted pre-training helps this raga task.
- If MERT and CultureMERT are close, I will look at confusion matrices and layer plots before concluding there is no difference.
- If full fine-tuning beats everything but validation is unstable, I will discuss overfitting risk.
- If frozen probes are already strong, that suggests these models already encode useful raga-relevant information before supervised training.

## Things Not to Overclaim

- A high score does not prove the model understands raga in the way a musician does.
- Saraga Carnatic is not the whole of Indian classical music.
- Segment-level labels are inherited from track-level metadata, so there can be noisy segments.
- The top-raga subset makes the problem manageable but narrows the conclusion.
