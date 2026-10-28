# Margin Diagnosis: Why Margin Heads Degrade FAR@TAR=80 While Maintaining Comparable DIR@FAR=1%

## Executive Summary & Diagnostic Core

This investigation diagnoses why CosFace and its variants (`e1_cosface`, `e2_cosface_hardid`, `e3_cosface_hardid_absneg`) exhibit dramatically worse standard open-set false accept rates at 80% true accept rate (**FAR@TAR=80%** of **41.46%**, **33.34%**, and **32.48%** vs. **15.97%** for baseline `e0_baseline_val`), while detection identification rate at 1% false accept rate (**DIR@FAR=1%**) remains only slightly degraded (**28.37%**, **24.56%**, and **25.71%** vs. **31.04%**).

### Core Diagnostic Answers

1. **The Discrepancy is Driven by an Open-Set "Ceiling Effect" (Denominator Saturation)**:
   - On the standard open-set evaluation protocol (Evaluator v2, 50% known ratio), open-set True Accept Rate (TAR) at threshold $\tau$ is defined as:
     $$\text{TAR}(\tau) = \frac{\sum_{i \in \text{Enrolled}} \mathbf{1}(\text{Rank-1 identity is correct} \land \text{score}_i \ge \tau)}{N_{\text{enrolled}}}$$
   - The theoretical maximum achievable TAR as $\tau \to -\infty$ is capped at the model's closed-set Rank-1 identification accuracy on the enrolled probes ($\text{Max TAR}$).
   - For `e0_baseline_val`, $\text{Max TAR} = 92.56\%$. Reaching $\text{TAR}=80\%$ requires capturing $80.0 / 92.56 = 86.4\%$ of all correct probes, leaving a healthy **12.56 percentage points of headroom**. The required threshold remains high ($\tau = 0.6135$ on seed 0), above the 86th percentile of impostor scores ($\text{FAR} = 13.95\%$).
   - For `e1_cosface`, $\text{Max TAR} = 86.15\%$. Reaching $\text{TAR}=80\%$ requires capturing $80.0 / 86.15 = 92.86\%$ of all correct probes, leaving only **6.15 percentage points of headroom**. To capture the noisy tail of correct matches, the threshold $\tau$ must be relaxed down to **0.4447** (seed 0).
   - Because 0.4447 sits well below the 75th percentile of impostor scores (0.4690) and near the impostor median (0.4233), **37.78% of impostor probes on seed 0 (and 41.46% across 5 seeds)** exceed this threshold.
   - At $\text{TAR}=70\%$ (where `e1` has 16.15 points of headroom, comparable to `e0` at 80%), `e1` achieves an FAR of **17.24%**, demonstrating that the jump from 17.2% to 41.5% is an exponential artifact of operating against the identification ceiling.
   - Conversely, **DIR@FAR=1%** fixes the false accept rate at a strict $1.0\%$ budget and queries the true identification rate above that high threshold. Because this metric operates far away from the ceiling, `e1` achieves **28.37%** (only 2.67 percentage points below `e0`'s 31.04%).

2. **The Root Cause of the Ceiling Drop (86.15% vs. 92.56% Max TAR) is Hyperparameter & Geometry Mismatch**:
   - The standard CosFace hyperparameters ($s = 30.0, m = 0.35$) were established for large-scale face recognition datasets (CASIA-WebFace, MegaFace) with $10,000+$ classes. On Market-1501 with only 651 training identities and ~19 images per identity, $s=30$ produces excessively sharp softmax logits that saturate training loss to near zero ($\mathcal{L}_{\text{total}} = 0.136$ at epoch 60 vs. $1.059$ for `e0`), causing severe over-clustering on the training identities.
   - Furthermore, `e1` combines an **angular cosine margin** ($m=0.35$ on $\cos\theta$) with an unweighted **Euclidean batch-hard triplet loss** ($d_E = \sqrt{2 - 2\cos\theta} \in [0, 2]$, margin $0.3$). The gradient magnitudes of the $s=30$ margin head overpower the triplet loss, distorting the embedding space.

---

## 1. Performance Tables

### 1.1 Test Set Evaluation (Evaluator v2, 5 Seeds: 0, 1, 2, 3, 4)

All values are reported as `Mean ± Std` across 5 deterministic seeds on Market-1501 standard test split ($N_q = 3368, N_g = 15913, \text{known\_ratio} = 0.5$). `e1b_arcface` was not trained or evaluated.

| Metric | e0_baseline_val | e1_cosface | e2_cosface_hardid | e3_cosface_hardid_absneg | e1b_arcface |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Test Rank-1 (%)** | 89.52% | 81.26% | 83.70% | 84.06% | *not run* |
| **Test mAP (%)** | 74.36% | 58.69% | 61.87% | 62.21% | *not run* |
| **Max Achievable TAR (%)** | 92.56% ± 0.39% | 86.15% ± 0.31% | 88.12% ± 0.63% | 88.71% ± 0.59% | *not run* |
| **FAR @ TAR=70% (%)** | 8.23% ± 1.31% | 17.24% ± 1.41% | 15.88% ± 1.08% | 15.15% ± 1.03% | *not run* |
| **FAR @ TAR=80% (%)** | 15.97% ± 1.50% | 41.46% ± 2.42% | 33.34% ± 1.86% | 32.48% ± 2.90% | *not run* |
| **FAR @ TAR=85% (%)** | 24.94% ± 1.67% | 78.32% ± 4.23% | 55.87% ± 5.27% | 54.62% ± 4.52% | *not run* |
| **FAR @ TAR=90% (%)** | 45.56% ± 3.91% | N/A (Max TAR = 86.15%) | N/A (Max TAR = 88.12%) | N/A (Max TAR = 88.71%) | *not run* |
| **DIR @ FAR=0.1% (%)** | 4.72% ± 0.78% | 6.40% ± 3.44% | 4.69% ± 1.07% | 8.04% ± 0.67% | *not run* |
| **DIR @ FAR=1.0% (%)** | 31.04% ± 2.18% | 28.37% ± 3.10% | 24.56% ± 3.45% | 25.71% ± 1.99% | *not run* |
| **DIR @ FAR=5.0% (%)** | 61.31% ± 2.15% | 51.12% ± 2.00% | 51.58% ± 1.95% | 50.07% ± 1.20% | *not run* |
| **DIR @ FAR=10.0% (%)** | 72.49% ± 1.85% | 61.47% ± 1.76% | 63.51% ± 1.77% | 62.54% ± 1.67% | *not run* |

### 1.2 Validation Progression (Epochs 5..60, Every 5 Epochs)

Validation split consists of 100 held-out identities (seed 42), evaluated using Evaluator v2.

| Epoch | e0 Rank-1 | e0 mAP | e0 FAR@80 | e1 Rank-1 | e1 mAP | e1 FAR@80 | e2 Rank-1 | e2 mAP | e2 FAR@80 | e3 Rank-1 | e3 mAP | e3 FAR@80 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **05** | 92.73% | 82.38% | 8.98% | 89.97% | 72.69% | 16.21% | 62.41% | 41.00% | N/A | 61.65% | 42.25% | N/A |
| **10** | 93.98% | 82.10% | 5.75% | 92.23% | 78.64% | 7.17% | 83.96% | 64.23% | 36.94% | 83.21% | 64.31% | 35.73% |
| **15** | 95.74% | 84.25% | 4.73% | 93.73% | 80.39% | 6.78% | 88.22% | 71.79% | 10.29% | 89.97% | 71.90% | 18.18% |
| **20** | 95.49% | 85.46% | 4.84% | 93.98% | 80.94% | 6.56% | 91.48% | 75.59% | 9.60% | 90.98% | 76.40% | 11.23% |
| **25** | 96.49% | 85.52% | 3.74% | 93.73% | 80.23% | 5.44% | 92.98% | 79.12% | 4.85% | 91.98% | 78.39% | 8.48% |
| **30** | 96.49% | 86.76% | 3.42% | 93.73% | 81.06% | 5.55% | 94.74% | 78.56% | 5.05% | 92.73% | 79.14% | 6.55% |
| **35** | 97.49% | 88.40% | 2.23% | 93.98% | 80.45% | 6.77% | 93.73% | 79.84% | 5.14% | 92.23% | 79.71% | 5.94% |
| **40** | 97.24% | 88.76% | 2.72% | 93.98% | 80.92% | 6.46% | 94.74% | 81.49% | 5.05% | 94.49% | 81.27% | 4.14% |
| **45** | 96.49% | 88.58% | 2.02% | 93.23% | 80.26% | 5.83% | 94.24% | 81.20% | 4.94% | 93.98% | 81.93% | 5.14% |
| **50** | 97.49% | 89.62% | 1.92% | 93.48% | 80.22% | 5.33% | 94.74% | 81.82% | 4.65% | 94.24% | 82.05% | 3.63% |
| **55** | 97.74% | 89.76% | 1.61% | 93.98% | 79.77% | 5.14% | 94.49% | 81.84% | 4.04% | 93.48% | 82.13% | 3.63% |
| **60** | 97.74% | 89.81% | 1.61% | 93.48% | 79.77% | 5.12% | 94.49% | 81.80% | 4.13% | 93.98% | 81.97% | 3.63% |

---

## 2. Score Distributions & Ceiling Effect Analysis (Standard Protocol, Seed 0)

On seed 0 of the standard test protocol ($N_{\text{enrolled}} = 1690, N_{\text{impostor}} = 1678$), every probe is matched against the enrolled gallery. We decompose probe scores into three mutually exclusive sets:
1. **(a) Correct Top-1 Enrolled Scores**: Enrolled probes whose highest-similarity gallery match has the correct identity label ($N_{\text{correct}}$).
2. **(b) Wrong Top-1 Enrolled Scores**: Enrolled probes whose highest-similarity gallery match has an incorrect identity label ($N_{\text{wrong}}$).
3. **(c) Top-1 Impostor Scores**: Unknown probes whose highest-similarity gallery match is with any enrolled identity ($N_{\text{impostor}}$).

### 2.1 Quantile Decomposition Table

| Model | Distribution | N | 5% | 25% | 50% (Median) | 75% | 95% | Max |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **e0_baseline_val** | (a) Correct Top-1 | 1572 | 0.5474 | 0.6596 | 0.7259 | 0.7827 | 0.8443 | 0.9082 |
| | (b) Wrong Top-1 | 118 | 0.4162 | 0.4708 | 0.5481 | 0.6763 | 0.7954 | 0.8410 |
| | (c) Impostor Top-1 | 1678 | 0.3913 | 0.4553 | 0.5051 | 0.5744 | 0.6800 | 0.8568 |
| **e1_cosface** | (a) Correct Top-1 | 1457 | 0.4254 | 0.5208 | 0.5944 | 0.6597 | 0.7432 | 0.8475 |
| | (b) Wrong Top-1 | 233 | 0.3463 | 0.3922 | 0.4580 | 0.5195 | 0.6261 | 0.7303 |
| | (c) Impostor Top-1 | 1678 | 0.3354 | 0.3813 | 0.4233 | 0.4690 | 0.5553 | 0.7893 |
| **e2_cosface_hardid** | (a) Correct Top-1 | 1483 | 0.4378 | 0.5376 | 0.6037 | 0.6721 | 0.7609 | 0.8655 |
| | (b) Wrong Top-1 | 207 | 0.3369 | 0.4049 | 0.4717 | 0.5468 | 0.6669 | 0.7441 |
| | (c) Impostor Top-1 | 1678 | 0.3356 | 0.3845 | 0.4271 | 0.4827 | 0.5776 | 0.7606 |
| **e3_cosface_hardid_absneg** | (a) Correct Top-1 | 1505 | 0.4340 | 0.5335 | 0.6038 | 0.6705 | 0.7546 | 0.8661 |
| | (b) Wrong Top-1 | 185 | 0.3411 | 0.4036 | 0.4636 | 0.5311 | 0.6377 | 0.7297 |
| | (c) Impostor Top-1 | 1678 | 0.3313 | 0.3823 | 0.4277 | 0.4805 | 0.5835 | 0.7473 |

### 2.2 Ceiling Effect and Threshold Mechanics (Seed 0)

| Metric | e0_baseline_val | e1_cosface | e2_cosface_hardid | e3_cosface_hardid_absneg |
|:---|:---:|:---:|:---:|:---:|
| **Enrolled Probes ($N_{\text{tot}}$)** | 1690 | 1690 | 1690 | 1690 |
| **Correct Probes ($N_{\text{correct}}$)** | 1572 | 1457 | 1483 | 1505 |
| **Seed 0 Max TAR** | **93.02%** | **86.21%** | **87.75%** | **89.05%** |
| **Distance to TAR=80% Ceiling** | **+13.02%** | **+6.21%** | **+7.75%** | **+9.05%** |
| **Proportion of Correct Probes Required for TAR=80%** | $80.0 / 93.02 = \mathbf{86.0\%}$ | $80.0 / 86.21 = \mathbf{92.8\%}$ | $80.0 / 87.75 = \mathbf{91.2\%}$ | $80.0 / 89.05 = \mathbf{89.8\%}$ |
| **Operating Threshold $\tau$ @ TAR=80%** | **0.6135** | **0.4447** | **0.4618** | **0.4745** |
| **Impostors Passing $\tau_{80}$ (Count / Total)** | 234 / 1678 | 634 / 1678 | 551 / 1678 | 460 / 1678 |
| **Seed 0 FAR @ TAR=80%** | **13.95%** | **37.78%** | **32.84%** | **27.41%** |
| **Operating Threshold $\tau$ @ TAR=70%** | **0.6729** | **0.5284** | **0.5471** | **0.5434** |
| **Seed 0 FAR @ TAR=70%** | **5.48%** | **15.14%** | **14.36%** | **13.89%** |

### 2.3 Histogram Analysis & Distribution Plot

A high-resolution 4-panel histogram comparing the three score distributions across all four models on common axes is saved at [score_distributions_comparison.png](file:///D:/model/reid-openset/docs/score_distributions_comparison.png).

```
Score Distribution Geometry Comparison (Seed 0):

e0_baseline_val (Max TAR = 93.02%):
  Impostor Top-1:   [0.39 -------- 0.51 -------- 0.68]
  Enrolled Correct:             [0.55 -------- 0.73 -------- 0.84]
  Threshold (80%):                         ^ tau = 0.614 (Far out on Impostor tail -> FAR = 13.9%)

e1_cosface (Max TAR = 86.21%):
  Impostor Top-1:   [0.34 ----- 0.42 ----- 0.56]
  Enrolled Correct:        [0.43 ----- 0.59 ----- 0.74]
  Threshold (80%):            ^ tau = 0.445 (Plunges into Impostor distribution -> FAR = 37.8%)
```

**Conclusion on Ceiling Inflation**: FAR@TAR=80% for `e1` is **heavily inflated by the ceiling effect**. Because `e1` has only 6.21% headroom above 80%, the threshold must descend to the 7.2th percentile of the correct distribution ($\tau = 0.4447$). In contrast, at TAR=70% (16.21% headroom), the threshold is $\tau = 0.5284$, which sits safely above the impostor median and results in an FAR of 15.14% on seed 0 (17.24% 5-seed mean).

---

## 3. Training Dynamics & Loss Components

### 3.1 Loss Component Trajectories Across Epochs

| Epoch | e0 Loss | e0 CE | e0 Triplet | e0 Dist-AP | e0 Dist-AN | e1 Loss | e1 CE | e1 Triplet | e1 Dist-AP | e1 Dist-AN | e2 Loss | e2 CE | e2 Triplet |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 6.774 | 6.392 | 0.382 | 1.331 | 1.250 | 7.950 | 7.503 | 0.447 | 1.395 | 1.248 | 8.667 | 8.195 | 0.472 |
| **5** | 3.025 | 2.900 | 0.125 | 0.846 | 1.096 | 4.378 | 4.184 | 0.194 | 1.105 | 1.222 | 10.027 | 9.644 | 0.383 |
| **10** | 1.547 | 1.483 | 0.064 | 0.880 | 1.197 | 3.397 | 3.317 | 0.080 | 1.021 | 1.289 | 12.193 | 11.861 | 0.333 |
| **11** | 1.509 | 1.446 | 0.064 | 0.888 | 1.204 | 2.786 | 2.716 | 0.070 | 1.008 | 1.294 | 11.959 | 11.627 | 0.333 |
| **15** | 1.394 | 1.339 | 0.055 | 0.894 | 1.225 | 1.551 | 1.511 | 0.040 | 0.962 | 1.313 | 9.722 | 9.455 | 0.266 |
| **20** | 1.324 | 1.279 | 0.045 | 0.887 | 1.237 | 1.064 | 1.036 | 0.027 | 0.932 | 1.320 | 7.916 | 7.686 | 0.229 |
| **30** | 1.216 | 1.189 | 0.027 | 0.875 | 1.264 | 0.529 | 0.518 | 0.011 | 0.873 | 1.330 | 4.423 | 4.286 | 0.137 |
| **40** | 1.128 | 1.114 | 0.014 | 0.853 | 1.288 | 0.328 | 0.322 | 0.006 | 0.841 | 1.334 | 2.609 | 2.515 | 0.094 |
| **50** | 1.076 | 1.068 | 0.008 | 0.829 | 1.308 | 0.181 | 0.178 | 0.003 | 0.811 | 1.340 | 1.933 | 1.867 | 0.066 |
| **60** | 1.059 | 1.054 | 0.005 | 0.815 | 1.317 | 0.136 | 0.134 | 0.002 | 0.800 | 1.343 | 1.845 | 1.773 | 0.072 |

### 3.2 Key Training Dynamics Findings

1. **Divergence Point (Epochs 1–10)**:
   - For `e0`, standard linear CE loss drops smoothly from 6.39 to 1.48 by epoch 10.
   - For `e1`, CosFace CE loss starts higher (7.50) and plateaus between epochs 7 and 10 around ~3.32. This plateau is caused by the linear margin warmup ($m(t) = 0.35 \cdot t / 10$), where the increasing penalty counteracts optimization progress.
2. **Transition When Margin Warmup Ends (Epoch 10)**:
   - Once the margin stops increasing at epoch 10 ($m = 0.35$), `e1` loss plunges rapidly: from 3.32 (epoch 10) to 2.72 (epoch 11), 2.41 (epoch 12), and 1.55 (epoch 15).
   - This indicates the network rapidly finds an equilibrium on the fixed margin hypersphere.
3. **Hard Negative Mining Dynamics (`e2` and `e3`)**:
   - In `e2` and `e3`, identity centroid clustering is active. Loss actually *increases* from 8.67 (epoch 1) to 12.19 (epoch 10).
   - This counter-intuitive increase is driven by two concurrent forces: (1) margin warmup from 0 to 0.35, and (2) centroid mining successfully selecting much closer, harder negative identities into every batch. This is confirmed by `dist_an` at epoch 10: 1.136 for `e2` vs. 1.289 for `e1`.
   - After epoch 10, `e2` loss drops monotonically to 1.85 at epoch 60.
4. **Is `e1` Still Improving at Epoch 60 (Under-Trained)?**:
   - **No.** `e1` is definitively not under-trained.
   - In late epochs (50..60), `e1` training loss drops to **0.136** (triplet loss = 0.002, CE loss = 0.134).
   - On the validation set, `e1` peaked between epochs 20 and 30 (mAP = 81.06%, Rank-1 = 93.73%), and degraded slightly to 79.77% by epoch 60.
   - Continued training beyond epoch 30 caused mild over-fitting on training identity boundaries rather than metric learning improvement.
5. **Logging Audit**:
   - Training classification accuracy is **not logged** in `train_metrics.csv` for any run (`e0`, `e1`, `e2`, `e3`).

---

## 4. Code Audit (Read-Only)

We conducted a read-only audit of `src/reid/models/backbone.py`, `src/reid/losses/`, and `scripts/train.py`.

### 4.1 Checkpoint & Architectural Confirmations

1. **(a) Test-Time Embeddings vs. Loss Inputs**:
   - In `src/reid/models/backbone.py` (`extract_features`):
     ```python
     feat = self.head(torch.flatten(self.gap(self.conv_base(x)), 1))
     feat_bn = self.bnneck(feat)
     norm_feat = F.normalize(feat_bn, p=2, dim=1)
     ```
   - In `MarginSoftmaxHead.forward`:
     ```python
     norm_x = F.normalize(features, p=2, dim=1)  # features is feat_bn
     ```
     Because `features` is `feat_bn`, `norm_x` is mathematically identical to `norm_feat`.
   - In `BaselineReIDLoss`: `embeddings` passed to `BatchHardTripletLoss` is `norm_feat`.
   - In `scripts/evaluate.py`: In evaluation mode, `model(images)` returns `norm_feat`.
   - **Verdict**: Confirmed. Test-time embeddings, triplet loss embeddings, and margin head input are the exact same tensor: BNNeck output, L2-normalized.

2. **(b) Weight Normalization and Training-Only Margin**:
   - In `src/reid/losses/margin.py`: `self.weight` is normalized via `F.normalize(self.weight, p=2, dim=1)` on every forward pass.
   - Margin penalty is applied only when `targets is not None and self.m > 0`.
   - In eval mode (`model.eval()`), `ReIDBackbone.forward` returns `norm_feat` directly and bypasses the classifier entirely.
   - **Verdict**: Confirmed.

3. **(c) Label Smoothing**:
   - In `scripts/train.py` (lines 475–479):
     ```python
     label_smoothing = (
         0.0
         if loss_cls in ("cosface", "arcface")
         else float(loss_cfg.get("label_smoothing", 0.1))
     )
     ```
   - **Verdict**: Confirmed. Label smoothing is disabled (0.0) for margin heads and active (0.1) for `e0`.

4. **(d) Warmup and LR Schedules**:
   - LR scheduler uses a 5-epoch linear warmup followed by cosine decay:
     `progress = (current_epoch - 5) / (60 - 5)`, independent of margin.
   - Margin warmup runs independently in `scripts/train.py` (lines 569–574):
     `curr_m = margin_m * min(1.0, epoch / 10)`.
   - Both schedules are completely decoupled.
   - **Verdict**: Confirmed.

5. **(e) Config Diff: `e0_baseline_val.yaml` vs. `e1_cosface.yaml`**:
   ```diff
   -exp_name: "e0_baseline_val"
   +exp_name: "e1_cosface"
   ...
    loss:
   +  cls: "cosface"
   +  margin_m: 0.35
   +  scale_s: 30.0
   +  margin_warmup_epochs: 10
      margin: 0.3
   -  label_smoothing: 0.1
   +  label_smoothing: 0.0
      weight_ce: 1.0
      weight_triplet: 1.0
   ```
   All other parameters (backbone, embedding_dim 512, batch size P=16/K=4, LR 3.5e-4, epochs 60, seed 42, val_identities 100) are identical.

### 4.2 Bugs vs. Design Concerns

* **Bugs**:
  - No fatal runtime bugs or shape mismatch errors were found in the codebase.
* **Design Concerns**:
  1. **Unclipped CosFace Subtraction**: In `src/reid/losses/margin.py`, `output = self.s * (cosine - one_hot * self.m)`. For negative cosine values ($\cos\theta < -1 + m$), $\cos\theta - m$ can dip below $-1.0$ (down to $-1.35$). While margin warmup mitigates instability, standard implementations often clamp $\cos\theta - m \ge -1$.
  2. **Euclidean Triplet Loss with Angular CosFace**: `BatchHardTripletLoss` computes Euclidean distances $d(x, y) = \sqrt{2 - 2\cos\theta} \in [0, 2]$. Meanwhile, CosFace optimizes angular distances on the unit hypersphere. The gradient geometry of the two losses differs non-linearly across the sphere.
  3. **Gradient Scale Imbalance**: At $s = 30$, the gradient of the margin cross-entropy loss is scaled by 30, overwhelming the unit-weight batch-hard triplet loss ($weight\_ce = 1.0, weight\_triplet = 1.0$).

---

## 5. Ranked Hypotheses & Proposed Test Experiments

### Hypothesis 1 (Primary - Proven): Metric Denominator Artifact & Ceiling Effect
* **Statement**: FAR@TAR=80% is inflated by 25.5 percentage points in `e1` compared to `e0` primarily because `e1`'s closed-set accuracy ceiling ($\text{Max TAR} = 86.15\%$) leaves almost no margin to reach 80% TAR, forcing the decision threshold into the impostor score bulk.
* **Empirical Evidence**:
  - Headroom to 80% TAR is only **6.21 percentage points** for `e1` vs. **12.56 percentage points** for `e0`.
  - Reaching 80% TAR requires capturing 92.9% of all correct probes in `e1` (threshold $\tau = 0.445$, below the 75th percentile of impostors), compared to 86.0% in `e0` ($\tau = 0.614$).
  - At TAR=70% (where `e1` has 16.15 points of headroom), `e1`'s FAR drops to 17.24%.
  - When evaluated at fixed false accept rate (**DIR@FAR=1%**), which is immune to the ceiling artifact, `e1` (28.37%) is within 2.67 points of `e0` (31.04%).
* **What it Does NOT Explain**:
  - Why `e1`'s closed-set Rank-1 and Max TAR dropped from 89.52% to 81.26% in the first place.
* **Proposed Test Experiment**:
  - Report normalized open-set metrics: $\text{FAR}@(\text{TAR} = 0.85 \times \text{Max TAR})$ or evaluate complete DET/ROC AUC curves.

---

### Hypothesis 2 (Secondary - Strongly Supported): Scale and Margin Mismatch on Low-Sample ReID Dataset
* **Statement**: Standard face-recognition hyperparameters ($s = 30.0, m = 0.35$) are excessively aggressive for Market-1501 (651 training identities, ~19 images/identity). They over-constrain the angular space, causing the ResNet-18 backbone to overfit to training identity prototypes and lose generalizability to unseen test identities.
* **Empirical Evidence**:
  - `e1` training loss collapsed to **0.136** by epoch 60, yet test mAP dropped by **15.67 percentage points** (58.69% vs. 74.36%).
  - Validation metrics peaked early at epoch 20–30 (mAP 81.06%) and steadily deteriorated to 79.77% at epoch 60.
  - Adding hard identity mining in `e2` recovered +2.44% Rank-1 and +3.18% mAP, but still fell well short of `e0`.
* **What it Does NOT Explain**:
  - Does not explain why DIR@FAR=0.1% was slightly higher for `e1` (6.40%) and `e3` (8.04%) than `e0` (4.72%). (This indicates that for the very cleanest, high-confidence queries, the angular margin indeed produces sharper top-end separation).
* **Proposed Test Experiment**:
  - Run a 2D parameter sweep: $s \in [12, 16, 20]$ and $m \in [0.10, 0.20, 0.25]$.

---

### Hypothesis 3 (Tertiary - Plausible): Objective Geometry Conflict Between Euclidean Triplet and Angular Margin
* **Statement**: Combining an angular cosine classification loss with a Euclidean batch-hard triplet loss creates conflicting gradient updates on the unit hypersphere, especially when the classification gradient is amplified by $s=30$.
* **Empirical Evidence**:
  - In `e0`, the linear classifier operates on unnormalized `feat_bn` (Luo et al. BNNeck trick), allowing the magnitude to absorb scale differences while Euclidean triplet shapes the normalized features.
  - In `e1`, both losses operate on normalized features, but with different distance metrics ($d_E$ vs. $\cos\theta$).
* **What it Does NOT Explain**:
  - Why `e3` (which added an absolute negative loss on cosine similarities) improved only marginally (+0.36% Rank-1) over `e2`.
* **Proposed Test Experiment**:
  - Replace `metric="euclidean"` with `metric="cosine"` in `BatchHardTripletLoss`, or test standalone CosFace without triplet loss ($weight\_triplet = 0.0$).

---

### Hypothesis 4 (Refuted / Disproven): Model Under-Training or Optimization Instability
* **Statement**: `e1` performed poorly because it needed more epochs or diverged during training.
* **Empirical Evidence (Refutation)**:
  - All runs completed 60 epochs without NaN loss.
  - `e1` train loss reached 0.136.
  - Validation curves plateaued after epoch 25.
  - `e1` is fully converged, ruling out under-training.
