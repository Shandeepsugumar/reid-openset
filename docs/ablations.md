# Ablation Experiments Summary

Aggregated validation hold-out and test-set open-set evaluation across all ablation studies.

| Experiment | Val Rank-1 | Val mAP | Val FAR@TAR=80 | Val DIR@FAR=1% | Test Rank-1 | Test mAP | Test FAR@TAR=80 | Test FAR@TAR=90 | Test DIR@FAR=1% | Test DIR@FAR=5% | Test max TAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_resnet18 | not run | not run | not run | not run | 90.23% | 76.89% | 13.81% ± 2.33% | 39.34% ± 4.01% | 37.78% ± 3.93% | 66.44% ± 2.58% | 93.12% ± 0.42% |
| e0_baseline_val | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| e1_cosface | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| e2_cosface_hardid | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| e3_cosface_hardid_absneg | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| e1b_arcface | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |

> [!NOTE]
> Validation metrics are extracted from the held-out validation identities at the final training epoch. Test metrics represent evaluation on the official Market-1501 query/gallery test split using Evaluator v2 (mean ± std over 5 deterministic seeds) evaluated on `checkpoint_epoch_060.pth`. Experiments marked 'not run' await GPU execution.
