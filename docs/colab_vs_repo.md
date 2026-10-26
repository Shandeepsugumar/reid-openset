# Colab vs Repository Comparison Report

This document records the exact differences between the Google Colab baseline reference implementation (`colab/reid_colab_baseline.py`) and the initial repository code (`src/reid/`, `scripts/train.py`, `configs/default.yaml`).

Per project requirements, where behavior differences exist, the **Colab implementation represents the ground-truth reference baseline**.

---

## 1. Component Comparison

| Component | Colab Reference (`colab/`) | Initial Repo (`src/`, `scripts/`) | Resolution in Feature Port |
| :--- | :--- | :--- | :--- |
| **Dataset & Filename Parsing** | Regex `([-\d]+)_c(\d+)`. Drops `pid <= 0` in train. Contiguous ID relabeling. Supports `smoke_test` ID subsetting. | Regex `([-\d]+)_c(\d+)`. Drops `pid <= 0` in train. Contiguous ID relabeling. No smoke test slicing in dataset. | Adopt Colab's `smoke_test` ID filtering (`smoke_test_identities=50`). |
| **Data Transforms** | Resize (256, 128), RandomFlip(0.5), Pad(10)+Crop(256, 128), Normalize(ImageNet), RandomErasing(0.5). | Identical transform pipeline and constants. | Identical. |
| **PK Batch Sampler** | `PKSampler(data_source, p=16, k=4)`. Fixed `P * K = 64`. Samples $P$ identities, $K$ instances each (with replacement if $< K$). | Identical logic with additional flexible initialization options. | Keep compatible with Colab default ($P=16, K=4$). |
| **Model Architecture** | `ReIDBackbone`: ResNet-18, GAP, 512-dim linear projection, `BNNeck` with `bias.requires_grad_(False)`, classifier `Linear(512, num_classes, bias=False)`. L2 normalization of `feat_bn` ("after"). | Identical architecture. | Kept identical. |
| **Loss Functions** | Combined loss: CrossEntropy (label smoothing 0.1) + Batch-Hard Triplet Loss (margin 0.3, Euclidean distance). Equal weights (1.0, 1.0). | Identical loss formulations. | Kept identical. |
| **Optimizer & Scheduler** | Adam (`lr=3.5e-4`, `weight_decay=5.0e-4`). 5-epoch linear warmup followed by cosine annealing down to `min_lr=1.0e-6`. | Adam optimizer without the Warmup + Cosine scheduler. | Port Colab's Warmup + Cosine scheduler into `scripts/train.py`. |
| **Mixed Precision (AMP)** | `torch.cuda.amp.autocast()` and `torch.cuda.amp.GradScaler()` (deprecated in PyTorch 2.x). | Legacy `torch.cuda.amp` syntax. | Upgrade to modern non-deprecated `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")` with automatic CPU fallback. |
| **DataLoader Settings** | `num_workers=2`, `pin_memory=True`, `persistent_workers=True`, `drop_last=True`. | `num_workers=0`, `persistent_workers=False`. | Make configurable via config/CLI with Colab defaults (`num_workers=2`). |
| **Evaluation Protocols** | Standalone inside notebook: Closed-set CMC (Rank-1, 5, 10) & mAP (excluding same camera & junk). Open-set TAR vs. FAR sweep (500 thresholds). Saves `.npy` arrays. | Modular `src/reid/eval/` was an empty package placeholder. | Port full closed-set and open-set evaluation into `src/reid/eval/`, `scripts/evaluate.py`, and `scripts/evaluate_embeddings.py`. |
| **Artifacts & Logs** | Saves `train_metrics.csv`, `latest_checkpoint.pth`, periodic checkpoints, 6 `.npy` files, `roc_curve.png`, and `eval_results.json`. | Saves only `train_metrics.csv` and checkpoint. | Align output organization under experiment directories `<output-root>/<exp_name>/`. |

---

## 2. Suspected Bugs & Nuances in the Colab Code

### 1. Unreachable TAR Target in Open-Set Protocol (`FAR @ TAR=95% = 100.00%`)
- **Observation**: The Colab evaluation outputs:
  ```text
  FAR @ TAR=80% : 14.42%
  FAR @ TAR=90% : 37.37%
  FAR @ TAR=95% : 100.00%
  ```
- **Explanation**: In this protocol, a query probe is counted as a **True Accept** only if:
  1. Its maximum gallery similarity score $s^* \ge \tau$, **AND**
  2. The top-1 retrieved gallery identity matches the genuine probe identity ($\hat{y} == y$).
- Because the baseline model's closed-set Rank-1 identification accuracy is **90.23%**, at least ~9.77% of genuine probes are incorrectly matched at Rank-1 regardless of the threshold $\tau$.
- Consequently, the maximum possible true accept rate (TAR) achievable by this model is $\le 90.23\%$. The condition `tar_arr >= 0.95` is never satisfied.
- In `compute_openset_tar_far()`, the fallback handler executes:
  ```python
  def get_far_at_tar(desired_tar: float) -> float:
      valid_indices = np.where(tar_arr >= desired_tar)[0]
      if len(valid_indices) == 0:
          return 1.0  # Returns 100% when desired TAR is unreachable
      return float(np.min(far_arr[valid_indices]))
  ```
- **Conclusion**: The 100% figure is a mathematical artifact of the threshold retrieval function when queried for an unreachable TAR target, not a training failure. As instructed, baseline logic is preserved exactly.

### 2. PyTorch 2.x AMP Deprecation Warnings
- **Observation**: Colab logged `FutureWarning: torch.cuda.amp.GradScaler(args...) is deprecated` and `FutureWarning: torch.cuda.amp.autocast(args...) is deprecated`.
- **Resolution**: Ported code uses the official replacement API:
  `torch.amp.GradScaler("cuda", enabled=use_amp)` and `torch.amp.autocast("cuda", enabled=use_amp)`.

### 3. MultiProcessing DataLoader Child Process Teardown Warning
- **Observation**: Python 3.13 / Colab logged `AssertionError: can only test a child process` during garbage collection of `_MultiProcessingDataLoaderIter.__del__`.
- **Resolution**: Harmless warning during interpreter exit; mitigated by graceful dataloader iteration and `persistent_workers` scoping.

### 4. Open-Set Evaluation Nuances & Matching Protocol Details
- **Normal Split Variance**: The single-seed (seed 42) result of $\text{FAR@TAR=90\%} = 37.37\%$ versus the 5-seed mean of $39.34\% \pm 4.01\%$ is within expected normal partition noise ($|37.37 - 39.34| = 1.97\% < 1\,\sigma$).
- **Single Model Split Variation**: The $\pm \text{std}$ across seeds reflects only the variation over random enrolled/unknown query identity partitions of the *single* trained baseline model, not training-seed variation.
- **Gallery Junk vs. Distractor Treatment**: During matching time (in both closed-set and open-set protocols):
  - Junk images (`pid == -1`) and same-identity same-camera pairs are discarded from the candidate gallery pool (`g_pids != -1`).
  - Distractor images (`pid == 0`, false-alarm DPM bounding boxes) are **retained** in the gallery pool and act as negative distractors during retrieval.
- **Score-Only AUC and EER**: Open-set ROC AUC and EER are score-only metrics evaluated on the binary discrimination between enrolled probe max-similarity scores and impostor probe max-similarity scores; they do not check whether the retrieved gallery identity matches the genuine probe identity.
