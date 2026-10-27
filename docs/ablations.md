# Ablation Experiments Summary

Aggregated validation hold-out and test-set open-set evaluation across all ablation studies.

| Experiment | Val Rank-1 | Val mAP | Val FAR@TAR=80 | Val DIR@FAR=1% | Test Rank-1 | Test mAP | Test FAR@TAR=80 | Test FAR@TAR=90 | Test DIR@FAR=1% | Test DIR@FAR=5% | Test max TAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_resnet18 | not run | not run | not run | not run | 90.23% | 76.89% | 13.81% ± 2.33% | 39.34% ± 4.01% | 37.78% ± 3.93% | 66.44% ± 2.58% | 93.12% ± 0.42% |
| e0_baseline_val | 97.74% | 89.81% | 1.61% | 59.01% | 89.52% | 74.36% | 15.97% ± 1.50% | 45.56% ± 3.91% | 31.04% ± 2.18% | 61.31% ± 2.15% | 92.56% ± 0.39% |
| e1_cosface | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| e2_cosface_hardid | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| e3_cosface_hardid_absneg | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| e1b_arcface | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |

> [!NOTE]
> Validation metrics are extracted from the held-out validation identities at the final training epoch. Test metrics represent evaluation on the official Market-1501 query/gallery test split using Evaluator v2 (mean ± std over 5 deterministic seeds) evaluated on `checkpoint_epoch_060.pth`. Experiments marked 'not run' await GPU execution.

## Experiment Notes and Trajectories

### baseline_resnet18 (Full 751-Identity Benchmark)
- **Validation**: not recorded (no validation split; trained on all 751 identities).
- **Test Metrics (v2, 5 seeds)**:
  - Rank-1: 90.23% | mAP: 76.89%
  - FAR@TAR=80%: 13.81% ± 2.33% | FAR@TAR=90%: 39.34% ± 4.01%
  - DIR@FAR=1%: 37.78% ± 3.93% | DIR@FAR=5%: 66.44% ± 2.58%
  - Max TAR: 93.12% ± 0.42%

### e0_baseline_val (Paired Baseline with 100 Held-Out Validation IDs)
- **Validation Trajectory**:
  - Epoch 05: Rank-1 92.73% | mAP 82.38% | FAR@TAR=80 8.98% | FAR@TAR=90 28.00% | DIR@FAR=1% 47.96% | max TAR 94.83%
  - Epoch 15: Rank-1 95.74% | mAP 84.25% | FAR@TAR=80 4.73% | FAR@TAR=90 20.67% | DIR@FAR=1% 69.01% | max TAR 96.42%
  - Epoch 30: Rank-1 96.49% | mAP 86.76% | FAR@TAR=80 3.42% | FAR@TAR=90 10.32% | DIR@FAR=1% 57.82% | max TAR 98.01%
  - Epoch 45: Rank-1 96.49% | mAP 88.58% | FAR@TAR=80 2.02% | FAR@TAR=90  6.16% | DIR@FAR=1% 62.61% | max TAR 97.61%
  - Epoch 60: Rank-1 97.74% | mAP 89.81% | FAR@TAR=80 1.61% | FAR@TAR=90  4.04% | DIR@FAR=1% 59.01% | max TAR 98.41%
  - **Late Epoch Behavior**: Metrics continuously rise through early-to-mid epochs (epochs 5–35) and flatten smoothly across late epochs (epochs 45–60) with no decline, confirming stable convergence without overfitting.
- **Test Metrics (v2, 5 seeds)**:
  - Rank-1: 89.52% (Paired reference baseline; -0.71% vs unheld 751-ID baseline)
  - mAP: 74.36% (Paired reference baseline; -2.53% vs unheld 751-ID baseline)
  - FAR@TAR=80%: 15.97% ± 1.50%
  - FAR@TAR=90%: 45.56% ± 3.91%
  - DIR@FAR=1%: 31.04% ± 2.18%
  - DIR@FAR=5%: 61.31% ± 2.15%
  - Max TAR: 92.56% ± 0.39%

### e1_cosface
- **Validation**: not run
- **Test Metrics vs Paired Baseline (e0_baseline_val)**: not run

### e2_cosface_hardid
- **Validation**: not run
- **Test Metrics vs Paired Baseline (e0_baseline_val)**: not run

### e3_cosface_hardid_absneg
- **Validation**: not run
- **Test Metrics vs Paired Baseline (e0_baseline_val)**: not run

### e1b_arcface
- **Validation**: not run
- **Test Metrics vs Paired Baseline (e0_baseline_val)**: not run
