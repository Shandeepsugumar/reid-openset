# Experiment 0: ResNet-18 Baseline

This document records the official benchmark results for the baseline model trained on a Google Colab Tesla T4 GPU. All numbers below are extracted verbatim from the recorded run logs (`results/baseline_resnet18/train_log.txt`), metrics CSV (`results/baseline_resnet18/train_metrics.csv`), and evaluation JSON (`results/baseline_resnet18/eval_results.json`).

---

## 1. Experimental Setup

- **Experiment Name**: `baseline_resnet18`
- **Hardware Accelerator**: NVIDIA Tesla T4 GPU
- **VRAM Utilization**: Peak allocated 0.72 GB
- **Operating Environment**: Google Colab (Linux, Python 3.13 / PyTorch with CUDA)
- **Dataset**: Market-1501 (`Market-1501-v15.09.15`)
  - **Train split**: 12,936 images | 751 identities | 6 cameras
  - **Query split**: 3,368 images | 750 identities | 6 cameras
  - **Gallery split**: 19,732 images | 752 identities | 6 cameras
- **Backbone**: ResNet-18 (ImageNet pretrained weights)
- **Embedding Head**: 512-dim linear projection with BatchNorm neck (BNNeck)
- **Output Representation**: L2-normalized embedding after BNNeck

---

## 2. Hyperparameters & Training Dynamics

| Hyperparameter | Value |
| :--- | :--- |
| **Image Resolution** | $256 \times 128$ (height $\times$ width) |
| **Batch Sampling ($P \times K$)** | $P = 16$ identities, $K = 4$ instances per identity (Batch Size = 64) |
| **DataLoader Workers** | 2 (`pin_memory=True`, `persistent_workers=True`) |
| **Total Epochs** | 60 |
| **Warmup Epochs** | 5 (linear warmup from $7.0 \times 10^{-5}$ to $3.5 \times 10^{-4}$) |
| **Base Learning Rate** | $3.5 \times 10^{-4}$ (cosine decay to $1.0 \times 10^{-6}$) |
| **Weight Decay** | $5.0 \times 10^{-4}$ |
| **Optimizer** | Adam |
| **Loss Function** | Cross-Entropy ID Loss (label smoothing 0.1) + Batch-Hard Triplet Loss (margin 0.3) |
| **Loss Weights** | $w_{\text{ce}} = 1.0, w_{\text{triplet}} = 1.0$ |
| **Precision** | Mixed Precision (AMP enabled) |
| **Random Seed** | 42 |

### Timing & Convergence
- **Time per Epoch**: ~29.5 seconds (range: 27.3s to 30.4s)
- **Total Training Time**: ~1,770 seconds (~29.5 minutes)
- **Initial Loss (Epoch 1)**: 6.8891 (CE: 6.5164, Triplet: 0.3727) | $d(ap)$: 1.308, $d(an)$: 1.237
- **Final Loss (Epoch 60)**: 1.0786 (CE: 1.0720, Triplet: 0.0066) | $d(ap)$: 0.830, $d(an)$: 1.316

---

## 3. Evaluation Results

### Closed-Set Protocol (Market-1501 Standard)
Evaluated with query identity matching against gallery (excluding same identity & same camera pairs and junk index $-1$):

| Metric | Score (%) | Exact Float Value |
| :--- | :---: | :---: |
| **Rank-1 Accuracy** | **90.23%** | 0.9023159144893111 |
| **Rank-5 Accuracy** | **96.29%** | 0.9628859857482185 |
| **Rank-10 Accuracy** | **97.98%** | 0.9798099762470309 |
| **Mean Average Precision (mAP)** | **76.89%** | 0.7689482869467333 |

### Open-Set Protocol (Enrolled vs. Impostor Detection)
Evaluated with a 50% query identity split (fixed seed 42):
- **Enrolled Probes (Targets)**: 1,690 queries
- **Impostor Probes (Non-targets)**: 1,678 queries

| Operating Point | False Alarm Rate (FAR) | Exact Float Value |
| :--- | :---: | :---: |
| **FAR @ TAR = 80%** | **14.42%** | 0.14421930870083433 |
| **FAR @ TAR = 90%** | **37.37%** | 0.3736591179976162 |
| **FAR @ TAR = 95%** | **100.00%** | 1.0 |

---

## 4. Note on the TAR = 95% Operating Point

In the open-set evaluation protocol, a genuine target query probe counts as a **True Accept** only when:
1. Its maximum gallery similarity score satisfies $s^* \ge \tau$, **and**
2. The top-1 retrieved identity matches the probe's genuine identity ($\hat{y} == y$).

Because the model's closed-set Rank-1 identification accuracy is **90.23%**, approximately 9.77% of genuine probes are incorrectly identified at Rank-1 regardless of the similarity score. Consequently, the true accept rate (TAR) is fundamentally bounded from above by Rank-1 accuracy ($\text{TAR} \le 90.23\%$).

When queried for $\text{TAR} = 95\%$, no threshold $\tau \in [-0.2, 1.0]$ satisfies $\text{TAR}(\tau) \ge 0.95$. In this scenario, the metric lookup function returns `1.0` (100.00%) as a fallback. This 100% figure is therefore an artifact of requesting an unreachable operating point and does not reflect a calibration failure.

---

## 5. Stored Artifacts

- **Model Checkpoints**: `checkpoints/checkpoint_epoch_060.pth` (final epoch snapshot evaluated for baseline) and `checkpoints/latest_checkpoint.pth` (recorded in `results/baseline_resnet18/eval_results.json`)
- **Metric Logs**: `results/baseline_resnet18/train_metrics.csv`

> [!NOTE]
> Training saves `latest_checkpoint.pth` every epoch for auto-resumption and saves epoch snapshots `checkpoint_epoch_0<epoch>.pth` (at epochs 10, 20, 30, 40, 50, and 60). Baseline evaluation is performed strictly on the final epoch snapshot (`checkpoint_epoch_060.pth`), never on a "best" checkpoint chosen by test-set metrics, strictly preventing test-set data leakage.
- **ROC Curve Plot**: `results/baseline_resnet18/roc_curve.png`
- **Summary JSON**: `results/baseline_resnet18/eval_results.json`
- **Query Embeddings**: `artifacts/baseline_resnet18/query_embeddings.npy` ($3,368 \times 512$)
- **Query PIDs / CamIDs**: `artifacts/baseline_resnet18/query_pids.npy`, `query_camids.npy`
- **Gallery Embeddings**: `artifacts/baseline_resnet18/gallery_embeddings.npy` ($19,732 \times 512$)
- **Gallery PIDs / CamIDs**: `artifacts/baseline_resnet18/gallery_pids.npy`, `gallery_camids.npy`
