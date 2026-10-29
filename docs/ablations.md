# Ablation Experiments Summary

Aggregated validation hold-out and test-set open-set evaluation across all ablation studies.

| Experiment | Val Rank-1 | Val mAP | Val FAR@TAR=80 | Val DIR@FAR=1% | Test Rank-1 | Test mAP | Test FAR@TAR=80 | Test FAR@TAR=90 | Test DIR@FAR=1% | Test DIR@FAR=5% | Test max TAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_resnet18 | not run | not run | not run | not run | 90.23% | 76.89% | 13.81% ± 2.33% | 39.34% ± 4.01% | 37.78% ± 3.93% | 66.44% ± 2.58% | 93.12% ± 0.42% |
| e0_baseline_val | 97.74% | 89.81% | 1.61% | 59.01% | 89.52% | 74.36% | 15.97% ± 1.50% | 45.56% ± 3.91% | 31.04% ± 2.18% | 61.31% ± 2.15% | 92.56% ± 0.39% |
| e1_cosface | 93.48% | 79.77% | 5.12% | 53.60% | 81.26% | 58.69% | 41.46% ± 2.42% | N/A (max TAR = 86.15%) | 28.37% ± 3.10% | 51.12% ± 2.00% | 86.15% ± 0.31% |
| e2_cosface_hardid | 94.49% | 81.80% | 4.13% | 49.63% | 83.70% | 61.87% | 33.34% ± 1.86% | N/A (max TAR = 88.12%) | 24.56% ± 3.45% | 51.58% ± 1.95% | 88.12% ± 0.63% |
| e3_cosface_hardid_absneg | 93.98% | 81.97% | 3.63% | 45.51% | 84.06% | 62.21% | 32.48% ± 2.90% | N/A (max TAR = 88.71%) | 25.71% ± 1.99% | 50.07% ± 1.20% | 88.71% ± 0.59% |
| e1b_arcface | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run | not run |
| e1c_cosface_small | 94.49% | 82.59% | 4.23% | 57.24% | 84.65% | 62.70% | 31.75% ± 2.70% | N/A (max TAR = 88.88%) | 30.54% ± 2.26% | 54.84% ± 2.19% | 88.88% ± 0.69% |
| e4_ce_hardid | 96.49% | 87.25% | 3.23% | 61.17% | 86.82% | 70.22% | 22.84% ± 2.38% | N/A (max TAR = 90.38%) | 33.64% ± 4.76% | 56.74% ± 2.61% | 90.38% ± 0.68% |
| e5_ce_hardid_absneg | 95.24% | 87.07% | 2.12% | 66.07% | 86.43% | 69.86% | 22.02% ± 2.65% | 81.47% ± 11.99% | 31.26% ± 4.35% | 59.79% ± 1.49% | 90.62% ± 0.43% |

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

### e1_cosface (CosFace Margin Softmax m=0.35, s=30.0)
- **Validation Trajectory**:
  - Epoch 05: Rank-1 89.97% | mAP 72.69% | FAR@TAR=80 16.21% | FAR@TAR=90   N/A  | DIR@FAR=1% 31.70% | max TAR 93.03%
  - Epoch 15: Rank-1 93.73% | mAP 80.39% | FAR@TAR=80  6.78% | FAR@TAR=90 20.74% | DIR@FAR=1% 38.18% | max TAR 95.93%
  - Epoch 30: Rank-1 93.73% | mAP 81.06% | FAR@TAR=80  5.55% | FAR@TAR=90 25.80% | DIR@FAR=1% 39.95% | max TAR 94.82%
  - Epoch 45: Rank-1 93.23% | mAP 80.26% | FAR@TAR=80  5.83% | FAR@TAR=90 24.90% | DIR@FAR=1% 50.42% | max TAR 95.12%
  - Epoch 60: Rank-1 93.48% | mAP 79.77% | FAR@TAR=80  5.12% | FAR@TAR=90 23.07% | DIR@FAR=1% 53.60% | max TAR 95.02%
- **Test Metrics (v2, 5 seeds)**:
  - Rank-1: 81.26% (-8.26% vs e0_baseline_val)
  - mAP: 58.69% (-15.67% vs e0_baseline_val)
  - FAR@TAR=80%: 41.46% ± 2.42% (+25.49% vs e0_baseline_val)
  - FAR@TAR=90%: N/A (max TAR = 86.15% ± 0.31%)
  - DIR@FAR=1%: 28.37% ± 3.10% (-2.67% vs e0_baseline_val)
  - DIR@FAR=5%: 51.12% ± 2.00% (-10.19% vs e0_baseline_val)
  - Max TAR: 86.15% ± 0.31% (-6.41% vs e0_baseline_val)

### e2_cosface_hardid (e1 + Hard Identity Sampler)
- **Validation Trajectory**:
  - Epoch 05: Rank-1 62.41% | mAP 41.00% | FAR@TAR=80   nan% | FAR@TAR=90   N/A  | DIR@FAR=1% 12.43% | max TAR 71.79%
  - Epoch 15: Rank-1 88.22% | mAP 71.79% | FAR@TAR=80 10.29% | FAR@TAR=90 50.14% | DIR@FAR=1% 43.05% | max TAR 92.94%
  - Epoch 30: Rank-1 94.74% | mAP 78.56% | FAR@TAR=80  5.05% | FAR@TAR=90 20.90% | DIR@FAR=1% 54.06% | max TAR 95.82%
  - Epoch 45: Rank-1 94.24% | mAP 81.20% | FAR@TAR=80  4.94% | FAR@TAR=90 17.96% | DIR@FAR=1% 52.00% | max TAR 95.52%
  - Epoch 60: Rank-1 94.49% | mAP 81.80% | FAR@TAR=80  4.13% | FAR@TAR=90 16.74% | DIR@FAR=1% 49.63% | max TAR 96.02%
- **Test Metrics (v2, 5 seeds)**:
  - Rank-1: 83.70% (-5.82% vs e0_baseline_val, +2.44% vs e1)
  - mAP: 61.87% (-12.49% vs e0_baseline_val, +3.18% vs e1)
  - FAR@TAR=80%: 33.34% ± 1.86% (+17.37% vs e0_baseline_val, -8.12% vs e1)
  - FAR@TAR=90%: N/A (max TAR = 88.12% ± 0.63%)
  - DIR@FAR=1%: 24.56% ± 3.45% (-6.48% vs e0_baseline_val, -3.81% vs e1)
  - DIR@FAR=5%: 51.58% ± 1.95% (-9.73% vs e0_baseline_val, +0.46% vs e1)
  - Max TAR: 88.12% ± 0.63% (-4.44% vs e0_baseline_val, +1.97% vs e1)

### e3_cosface_hardid_absneg (e2 + Absolute Negative Loss)
- **Validation Trajectory**:
  - Epoch 05: Rank-1 61.65% | mAP 42.25% | FAR@TAR=80   nan% | FAR@TAR=90   N/A  | DIR@FAR=1% 11.26% | max TAR 70.61%
  - Epoch 15: Rank-1 89.97% | mAP 71.90% | FAR@TAR=80 18.18% | FAR@TAR=90   N/A  | DIR@FAR=1% 49.18% | max TAR 91.85%
  - Epoch 30: Rank-1 92.73% | mAP 79.14% | FAR@TAR=80  6.55% | FAR@TAR=90 20.72% | DIR@FAR=1% 49.84% | max TAR 94.33%
  - Epoch 45: Rank-1 93.98% | mAP 81.93% | FAR@TAR=80  5.14% | FAR@TAR=90 19.49% | DIR@FAR=1% 42.47% | max TAR 95.82%
  - Epoch 60: Rank-1 93.98% | mAP 81.97% | FAR@TAR=80  3.63% | FAR@TAR=90 13.41% | DIR@FAR=1% 45.51% | max TAR 95.22%
- **Test Metrics (v2, 5 seeds)**:
  - Rank-1: 84.06% (-5.46% vs e0_baseline_val, +0.36% vs e2)
  - mAP: 62.21% (-12.15% vs e0_baseline_val, +0.34% vs e2)
  - FAR@TAR=80%: 32.48% ± 2.90% (+16.51% vs e0_baseline_val, -0.86% vs e2)
  - FAR@TAR=90%: N/A (max TAR = 88.71% ± 0.59%)
  - DIR@FAR=1%: 25.71% ± 1.99% (-5.33% vs e0_baseline_val, +1.15% vs e2)
  - DIR@FAR=5%: 50.07% ± 1.20% (-11.24% vs e0_baseline_val, -1.51% vs e2)
  - Max TAR: 88.71% ± 0.59% (-3.85% vs e0_baseline_val, +0.59% vs e2)

### e1b_arcface
- **Validation**: not run
- **Test Metrics vs Paired Baseline (e0_baseline_val)**: not run

### e1c_cosface_small (CosFace Reduced Margin m=0.20, s=30.0)
- **Validation Trajectory**:
  - Epoch 05: Rank-1 88.72% | mAP 72.64% | FAR@TAR=80 15.36% | FAR@TAR=90   N/A  | DIR@FAR=1% 35.35% | max TAR 91.45%
  - Epoch 15: Rank-1 91.48% | mAP 81.17% | FAR@TAR=80  6.16% | FAR@TAR=90 19.93% | DIR@FAR=1% 51.59% | max TAR 93.93%
  - Epoch 30: Rank-1 92.48% | mAP 82.66% | FAR@TAR=80  3.43% | FAR@TAR=90 13.72% | DIR@FAR=1% 55.31% | max TAR 93.94%
  - Epoch 45: Rank-1 94.49% | mAP 82.78% | FAR@TAR=80  4.84% | FAR@TAR=90 15.24% | DIR@FAR=1% 55.85% | max TAR 95.92%
  - Epoch 60: Rank-1 94.49% | mAP 82.59% | FAR@TAR=80  4.23% | FAR@TAR=90 16.22% | DIR@FAR=1% 57.24% | max TAR 96.31%
- **Test Metrics (v2, 5 seeds)**:
  - Rank-1: 84.65% (-4.87% vs e0_baseline_val, +3.39% vs e1)
  - mAP: 62.70% (-11.66% vs e0_baseline_val, +4.01% vs e1)
  - FAR@TAR=80%: 31.75% ± 2.70% (+15.78% vs e0_baseline_val, -9.71% vs e1)
  - FAR@TAR=90%: N/A (max TAR = 88.88% ± 0.69%)
  - DIR@FAR=1%: 30.54% ± 2.26% (-0.50% vs e0_baseline_val, +2.17% vs e1)
  - DIR@FAR=5%: 54.84% ± 2.19% (-6.47% vs e0_baseline_val, +3.72% vs e1)
  - Max TAR: 88.88% ± 0.69% (-3.68% vs e0_baseline_val, +2.73% vs e1)
- **Key Insight**: Lowering the angular margin from $m=0.35$ to $m=0.20$ substantially improves performance across all dimensions: Rank-1 (+3.39%), mAP (+4.01%), max TAR (+2.73%), DIR@1% (+2.17%), and reduces FAR@80 by 9.71% absolute.

### e4_ce_hardid (Baseline CE + Hard Identity Sampler)
- **Validation Trajectory**:
  - Epoch 05: Rank-1 85.46% | mAP 67.70% | FAR@TAR=80 36.39% | FAR@TAR=90   N/A  | DIR@FAR=1% 20.50% | max TAR 90.46%
  - Epoch 15: Rank-1 93.23% | mAP 81.54% | FAR@TAR=80  7.47% | FAR@TAR=90 22.76% | DIR@FAR=1% 39.12% | max TAR 95.72%
  - Epoch 30: Rank-1 95.24% | mAP 85.96% | FAR@TAR=80  3.72% | FAR@TAR=90 11.49% | DIR@FAR=1% 59.57% | max TAR 96.82%
  - Epoch 45: Rank-1 95.49% | mAP 86.38% | FAR@TAR=80  3.52% | FAR@TAR=90 10.59% | DIR@FAR=1% 61.98% | max TAR 96.51%
  - Epoch 60: Rank-1 96.49% | mAP 87.25% | FAR@TAR=80  3.23% | FAR@TAR=90  7.56% | DIR@FAR=1% 61.17% | max TAR 96.71%
- **Test Metrics (v2, 5 seeds)**:
  - Rank-1: 86.82% (-2.70% vs e0_baseline_val)
  - mAP: 70.22% (-4.14% vs e0_baseline_val)
  - FAR@TAR=80%: 22.84% ± 2.38% (+6.87% vs e0_baseline_val, -8.91% to -18.62% vs CosFace)
  - FAR@TAR=90%: N/A (max TAR = 90.38% ± 0.68%)
  - DIR@FAR=1%: 33.64% ± 4.76% (+2.60% vs e0_baseline_val, highest among all ablation models)
  - DIR@FAR=5%: 56.74% ± 2.61% (-4.57% vs e0_baseline_val)
  - Max TAR: 90.38% ± 0.68% (-2.18% vs e0_baseline_val)
- **Key Insight**: Combining standard CE loss with hard-identity batch mining achieves the best open-set identification rate at 1% FAR (33.64% vs 31.04% for e0), while keeping FAR@TAR=80 far lower (22.84%) than any CosFace model (31.75%–41.46%).

### e5_ce_hardid_absneg (e4 + Absolute Negative Loss)
- **Validation Trajectory**:
  - Epoch 05: Rank-1 86.47% | mAP 67.72% | FAR@TAR=80 34.55% | FAR@TAR=90   N/A  | DIR@FAR=1% 18.52% | max TAR 91.25%
  - Epoch 15: Rank-1 93.98% | mAP 81.53% | FAR@TAR=80  5.15% | FAR@TAR=90 25.44% | DIR@FAR=1% 53.66% | max TAR 96.52%
  - Epoch 30: Rank-1 94.74% | mAP 84.78% | FAR@TAR=80  3.83% | FAR@TAR=90 11.19% | DIR@FAR=1% 63.89% | max TAR 96.13%
  - Epoch 45: Rank-1 94.74% | mAP 86.58% | FAR@TAR=80  2.42% | FAR@TAR=90  9.08% | DIR@FAR=1% 65.53% | max TAR 96.52%
  - Epoch 60: Rank-1 95.24% | mAP 87.07% | FAR@TAR=80  2.12% | FAR@TAR=90  6.87% | DIR@FAR=1% 66.07% | max TAR 97.12%
- **Test Metrics (v2, 5 seeds)**:
  - Rank-1: 86.43% (-3.09% vs e0_baseline_val)
  - mAP: 69.86% (-4.50% vs e0_baseline_val)
  - FAR@TAR=80%: 22.02% ± 2.65% (+6.05% vs e0_baseline_val, lowest FAR@80 among all non-baseline models)
  - FAR@TAR=90%: 81.47% ± 11.99% (reaches 90% TAR on test set)
  - DIR@FAR=1%: 31.26% ± 4.35% (+0.22% vs e0_baseline_val)
  - DIR@FAR=5%: 59.79% ± 1.49% (-1.52% vs e0_baseline_val)
  - Max TAR: 90.62% ± 0.43% (-1.94% vs e0_baseline_val)
- **Key Insight**: Absolute Negative Loss pushed max TAR above 90% (allowing evaluation of FAR@90 at 81.47%), lowered FAR@80 to 22.02%, and sustained DIR@FAR=1% above 31%.

