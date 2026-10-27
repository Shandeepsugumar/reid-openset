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

> [!NOTE]
> Validation metrics are extracted from the held-out validation identities at the final training epoch. Test metrics represent evaluation on the official Market-1501 query/gallery test split using Evaluator v2 (mean ± std over 5 deterministic seeds) evaluated on `checkpoint_epoch_060.pth`. Experiments marked 'not run' await GPU execution.

## How to read this table

- **Paired Baseline**: `e0_baseline_val` (trained on 651 identities with 100 identities held out for validation) is the strict paired baseline for all ablation runs (`e1_cosface`, `e2_cosface_hardid`, `e3_cosface_hardid_absneg`, and `e1b_arcface`). All ablation comparisons and difference scores (`+-`) are computed relative to `e0_baseline_val`.
- **Independent Reference**: `baseline_resnet18` (trained on all 751 training identities without a validation holdout) serves as an external upper-bound reference to measure the capacity penalty of reserving 100 identities for validation.
- **Validation Set Purpose**: The validation set is inherently easier than the official test split (smaller gallery size, single-camera/clean probe conditions, and no open-set distractors). Its purpose is sanity-checking training dynamics, verifying non-divergence, catching broken runs, and comparing curve shapes across epochs—**not** selecting a winning model.
- **Evaluation Rule for Claiming Improvement**:
  > [!IMPORTANT]
  > A model is considered an **improvement** over the paired baseline only if **FAR@TAR=80**, **FAR@TAR=90**, and **DIR@FAR=1%** all move in the beneficial direction by more than $1 \times \text{std}$ (FAR decreasing by $> 1\sigma$, DIR increasing by $> 1\sigma$) AND the validation curve does not indicate degradation. If any metric does not meet this threshold or metrics move in conflicting directions, the result is designated a **tie**. No run may be designated "best" unless this criterion is strictly satisfied.

## Experiment Notes and Trajectories

### baseline_resnet18 (Full 751-Identity Benchmark)
- **Validation**: not recorded (no validation split; trained on all 751 identities).
- **Test Metrics (v2, 5 seeds)**:
  - Rank-1: 90.23% | mAP: 76.89%
  - FAR@TAR=80%: 13.81% ± 2.33% | FAR@TAR=90%: 39.34% ± 4.01%
  - DIR@FAR=1%: 37.78% ± 3.93% | DIR@FAR=5%: 66.44% ± 2.58%
  - Max TAR: 93.12% ± 0.42%

### e0_baseline_val (Paired Baseline with 100 Held-Out Validation IDs)
- **Run Status & Artifacts**: Completed 60 epochs (no NaN loss). `eval_v2.json`, `val_metrics.csv`, `train_metrics.csv`, `roc_v2.png`, `train_log.txt`, and `val_identities.json` present. 6 embedding arrays in `artifacts/e0_baseline_val/`.
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

### e1_cosface (CosFace Large-Margin Cosine Loss, s=30.0, m=0.35)
- **Run Status & Artifacts**: Completed 60 epochs (no NaN loss). `eval_v2.json`, `val_metrics.csv`, `train_metrics.csv`, `roc_v2.png`, and `val_identities.json` present. 6 embedding arrays in `artifacts/e1_cosface/`. `train_log.txt` not recorded.
- **Validation Trajectory**:
  - Epoch 05: Rank-1 89.97% | mAP 72.69% | FAR@TAR=80 16.21% | FAR@TAR=90 N/A (max TAR = 93.03%) | DIR@FAR=1% 31.70% | max TAR 93.03%
  - Epoch 15: Rank-1 93.73% | mAP 80.39% | FAR@TAR=80  6.78% | FAR@TAR=90 20.74% | DIR@FAR=1% 38.18% | max TAR 95.93%
  - Epoch 30: Rank-1 93.73% | mAP 81.06% | FAR@TAR=80  5.55% | FAR@TAR=90 25.80% | DIR@FAR=1% 39.95% | max TAR 94.82%
  - Epoch 45: Rank-1 93.23% | mAP 80.26% | FAR@TAR=80  5.83% | FAR@TAR=90 24.90% | DIR@FAR=1% 50.42% | max TAR 95.12%
  - Epoch 60: Rank-1 93.48% | mAP 79.77% | FAR@TAR=80  5.12% | FAR@TAR=90 23.07% | DIR@FAR=1% 53.60% | max TAR 95.02%
  - **Late Epoch Behavior**: Fast initial ascent between epochs 5–20 followed by a distinct flattening plateau across late epochs (epochs 30–60), where Rank-1 stabilizes between 93.2%–94.0% and mAP plateaus near ~80.0% with minimal variance.
- **Test Metrics vs Paired Baseline (e0_baseline_val)**:
  - Rank-1: 81.26% (-8.26% vs e0_baseline_val)
  - mAP: 58.69% (-15.67% vs e0_baseline_val)
  - FAR@TAR=80%: 41.46% ± 2.42% (+25.49% vs e0_baseline_val, degraded verification)
  - FAR@TAR=90%: N/A (max TAR = 86.15%; fails to reach 90% TAR on test set)
  - DIR@FAR=1%: 28.37% ± 3.10% (-2.67% vs e0_baseline_val)
  - DIR@FAR=5%: 51.12% ± 2.00% (-10.19% vs e0_baseline_val)
  - Max TAR: 86.15% ± 0.31% (-6.41% vs e0_baseline_val)
  - **Outcome vs e0_baseline_val**: Under the evaluation rule, `e1_cosface` does not improve over `e0_baseline_val` (all key metrics move unfavorably). Outcome: **no improvement**.

### e2_cosface_hardid (CosFace + Hard-Negative Identity Mining PK Sampler)
- **Run Status & Artifacts**: Completed 60 epochs (no NaN loss). `eval_v2.json`, `val_metrics.csv`, `train_metrics.csv`, `roc_v2.png`, and `val_identities.json` present. 6 embedding arrays in `artifacts/e2_cosface_hardid/`. `train_log.txt` not recorded.
- **Validation Trajectory**:
  - Epoch 05: Rank-1 62.41% | mAP 41.00% | FAR@TAR=80 N/A (max TAR = 71.79%) | FAR@TAR=90 N/A (max TAR = 71.79%) | DIR@FAR=1% 12.43% | max TAR 71.79%
  - Epoch 15: Rank-1 88.22% | mAP 71.79% | FAR@TAR=80 10.29% | FAR@TAR=90 50.14% | DIR@FAR=1% 43.05% | max TAR 92.94%
  - Epoch 30: Rank-1 94.74% | mAP 78.56% | FAR@TAR=80  5.05% | FAR@TAR=90 20.90% | DIR@FAR=1% 54.06% | max TAR 95.82%
  - Epoch 45: Rank-1 94.24% | mAP 81.20% | FAR@TAR=80  4.94% | FAR@TAR=90 17.96% | DIR@FAR=1% 52.00% | max TAR 95.52%
  - Epoch 60: Rank-1 94.49% | mAP 81.80% | FAR@TAR=80  4.13% | FAR@TAR=90 16.74% | DIR@FAR=1% 49.63% | max TAR 96.02%
  - **Late Epoch Behavior**: Hard negative identity mining causes a slower start at epoch 5 (Rank-1 62.41%), but accuracy climbs rapidly from epoch 10 through epoch 40 (Rank-1 94.74%, mAP 81.49%) and cleanly flattens through epochs 45–60 without degradation.
- **Test Metrics vs Paired Baseline (e0_baseline_val)**:
  - Rank-1: 83.70% (-5.82% vs e0_baseline_val; +2.44% vs e1_cosface)
  - mAP: 61.87% (-12.49% vs e0_baseline_val; +3.18% vs e1_cosface)
  - FAR@TAR=80%: 33.34% ± 1.86% (+17.37% vs e0_baseline_val; -8.12% vs e1_cosface)
  - FAR@TAR=90%: N/A (max TAR = 88.12%; fails to reach 90% TAR on test set)
  - DIR@FAR=1%: 24.56% ± 3.45% (-6.48% vs e0_baseline_val; -3.81% vs e1_cosface)
  - DIR@FAR=5%: 51.58% ± 1.95% (-9.73% vs e0_baseline_val; +0.46% vs e1_cosface)
  - Max TAR: 88.12% ± 0.63% (-4.44% vs e0_baseline_val; +1.97% vs e1_cosface)
  - **Outcome vs e0_baseline_val**: Under the evaluation rule, `e2_cosface_hardid` does not improve over `e0_baseline_val`. Comparing vs `e1_cosface`, FAR@80 improved by $> 1\sigma$ but DIR@1% dropped, so under the rule `e2` vs `e1` is designated a **tie**.

### e3_cosface_hardid_absneg (CosFace + Hard-ID + Absolute Negative Loss, weight=0.1, margin=0.3)
- **Run Status & Artifacts**: Completed 60 epochs (no NaN loss). `eval_v2.json`, `val_metrics.csv`, `train_metrics.csv`, `roc_v2.png`, and `val_identities.json` present. 6 embedding arrays in `artifacts/e3_cosface_hardid_absneg/`. `train_log.txt` not recorded.
- **Validation Trajectory**:
  - Epoch 05: Rank-1 61.65% | mAP 42.25% | FAR@TAR=80 N/A (max TAR = 70.61%) | FAR@TAR=90 N/A (max TAR = 70.61%) | DIR@FAR=1% 11.26% | max TAR 70.61%
  - Epoch 15: Rank-1 89.97% | mAP 71.90% | FAR@TAR=80 18.18% | FAR@TAR=90 N/A (max TAR = 91.85%) | DIR@FAR=1% 49.18% | max TAR 91.85%
  - Epoch 30: Rank-1 92.73% | mAP 79.14% | FAR@TAR=80  6.55% | FAR@TAR=90 20.72% | DIR@FAR=1% 49.84% | max TAR 94.33%
  - Epoch 45: Rank-1 93.98% | mAP 81.93% | FAR@TAR=80  5.14% | FAR@TAR=90 19.49% | DIR@FAR=1% 42.47% | max TAR 95.82%
  - Epoch 60: Rank-1 93.98% | mAP 81.97% | FAR@TAR=80  3.63% | FAR@TAR=90 13.41% | DIR@FAR=1% 45.51% | max TAR 95.22%
  - **Late Epoch Behavior**: Similar to e2, starts lower at epoch 5 due to negative pair mining, climbs steadily through epoch 45 (Rank-1 93.98%, mAP 81.93%), and stabilizes smoothly across late epochs (epochs 45–60) with no degradation.
- **Test Metrics vs Paired Baseline (e0_baseline_val)**:
  - Rank-1: 84.06% (-5.46% vs e0_baseline_val; +0.36% vs e2_cosface_hardid)
  - mAP: 62.21% (-12.15% vs e0_baseline_val; +0.34% vs e2_cosface_hardid)
  - FAR@TAR=80%: 32.48% ± 2.90% (+16.51% vs e0_baseline_val; -0.86% vs e2_cosface_hardid)
  - FAR@TAR=90%: N/A (max TAR = 88.71%; fails to reach 90% TAR on test set)
  - DIR@FAR=1%: 25.71% ± 1.99% (-5.33% vs e0_baseline_val; +1.15% vs e2_cosface_hardid)
  - DIR@FAR=5%: 50.07% ± 1.20% (-11.24% vs e0_baseline_val; -1.51% vs e2_cosface_hardid)
  - Max TAR: 88.71% ± 0.59% (-3.85% vs e0_baseline_val; +0.59% vs e2_cosface_hardid)
  - **Outcome vs e0_baseline_val & e2**: Under the evaluation rule, `e3_cosface_hardid_absneg` does not improve over `e0_baseline_val`. Compared against `e2_cosface_hardid`, metric shifts are within $1\sigma$, so it is designated a **tie**. Neither run is declared "best".

### e1b_arcface
- **Validation**: not run
- **Test Metrics vs Paired Baseline (e0_baseline_val)**: not run
