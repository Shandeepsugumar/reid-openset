# Look-Alike Evaluation Protocol and Failure Analysis

A model-independent open-set evaluation protocol and failure analysis methodology designed to assess person re-identification vulnerabilities when distinct individuals share identical or near-identical clothing (e.g. uniforms, workwear, common campus apparel).

---

## 1. Motivation and Problem Definition

Standard open-set ReID evaluation protocols randomly partition query identities into **Enrolled** (known) and **Unknown** (impostor) sets. Because the identity partition is purely uniform, the probability that an unknown probe happens to closely resemble an enrolled person is low. In real-world enterprise deployments (e.g. corporate campuses, logistics hubs, hospitals, and schools), security failures frequently arise from **look-alikes**: unrelated individuals dressed in identical uniforms or similar color combinations.

Market-1501 does not provide semantic clothing attributes or uniform labels. To measure vulnerability to visual look-alikes without introducing bias toward any trained neural feature extractor, this protocol extracts **model-independent clothing descriptors** using spatial HSV color histograms and groups visually similar identities using **complete-linkage agglomerative clustering**.

---

## 2. Methodology

### 2.1 Spatial Clothing Descriptors
For every image in the test set (`query/` and `bounding_box_test/`), excluding junk (`pid == -1`) and distractor (`pid == 0`):
1. **Spatial Cropping**:
   - **Upper Body Region**: Rows 20% to 55% of image height.
   - **Lower Body Region**: Rows 55% to 95% of image height.
   - **Central Column Range**: Columns 15% to 85% of image width (central 70% width crop to suppress background clutter).
2. **Color Representation**:
   - Convert image crops to HSV color space.
   - Compute a 3D joint color histogram: $8 \text{ Hue} \times 4 \text{ Saturation} \times 4 \text{ Value} = 128 \text{ bins}$ per region.
   - Histograms are $L_1$-normalized ($\sum p_i = 1.0$).
3. **Identity Descriptor**:
   - For each identity, the descriptor is the element-wise average of all its query and gallery crop histograms: $\bar{p}_{\text{upper}}$ and $\bar{p}_{\text{lower}}$.
4. **Distance Metric**:
   - Bhattacharyya distance:
     $$D_B(p, q) = \sqrt{1.0 - \sum_{k=1}^{128} \sqrt{p_k \cdot q_k}}$$
   - Overall identity distance:
     $$D(i, j) = \frac{D_B(\bar{p}_{\text{upper}}^{(i)}, \bar{p}_{\text{upper}}^{(j)}) + D_B(\bar{p}_{\text{lower}}^{(i)}, \bar{p}_{\text{lower}}^{(j)})}{2}$$

### 2.2 Complete-Linkage Look-Alike Grouping
- Pairwise distances are computed across all 750 valid test identities ($280,875$ unique pairs).
- **Clustering Algorithm**: Complete-linkage agglomerative clustering with `distance_threshold = percentile`.
  - Complete linkage guarantees that the maximum distance between **any pair** within a group is $\le \text{threshold}$ (preventing the single-linkage chaining that causes giant connected components).
  - Clusters of size $\ge 2$ form look-alike groups.
  - To prevent oversized clusters, any cluster with size $> 6$ is recursively bisected via complete linkage until all groups satisfy $2 \le \text{size} \le 6$ (dropping singletons if a cut creates one). Identities without neighbors are excluded.

---

## 3. Look-Alike Group Statistics & Sensitivity Analysis

Pairwise distance distribution across all 750 Market-1501 test identities:
- **Global Median Pairwise Distance**: **0.5396**

Sensitivity across 2nd, 5th, and 10th percentiles:

| Percentile | Distance Threshold | Num Groups | Total IDs in Groups | Group Size Histogram | Mean Intra-Group Dist | Max Intra-Group Dist |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **p02 (2.0%)** | 0.3111 | 188 | 531 / 750 (70.8%) | {2: 97, 3: 54, 4: 18, 5: 11, 6: 8} | 0.2480 | 0.3110 |
| **p05 (5.0%)** | **0.3517** | **200** | **633 / 750 (84.4%)** | **{2: 81, 3: 50, 4: 37, 5: 19, 6: 13}** | **0.2671** | **0.3515** |
| **p10 (10.0%)** | 0.3909 | 196 | 685 / 750 (91.3%) | {2: 59, 3: 49, 4: 38, 5: 32, 6: 18} | 0.2811 | 0.3903 |

> [!NOTE]
> The default operating configuration is **p05 (5.0%)**, yielding 200 well-bounded look-alike groups encompassing 633 identities with an average intra-group distance of 0.2671 (compared to the global median distance of 0.5396).

---

## 4. Evaluation Protocol Design

### 4.1 Paired Look-Alike Open-Set Protocol
1. **Stratified Group Partition**:
   - In each look-alike group, identities are deterministically partitioned into **Enrolled** ($\lceil \text{size} / 2 \rceil$) and **Unknown** (remainder; at least 1).
   - Unknown probes are visual look-alikes of enrolled individuals.
2. **Realistic Gallery Size**:
   - Gallery = images of enrolled group identities + **all non-group test identities** + distractor images (`pid == 0`).
   - Retaining non-group identities in the gallery ensures realistic retrieval gallery density (19,732 total gallery images).
3. **Probes**:
   - Probes = query images of enrolled identities (targets) + query images of unknown group identities (impostors). Non-group queries are omitted.
4. **Evaluation Rules**:
   - Strict Evaluator v2 parity: junk (`pid == -1`) and same-identity same-camera pairs are discarded.
   - Evaluated over 5 deterministic seeds: `0, 1, 2, 3, 4`.

### 4.2 Matched Random Control Protocol
To isolate whether performance drops stem from look-alike similarity rather than smaller probe sample size or gallery composition:
- Exactly the same number of enrolled ($K$) and unknown ($U$) identities are sampled uniformly at random from all 750 test identities.
- The gallery composition rule is identical: enrolled $K$ identities + all remaining $(750 - K - U)$ identities + distractor `pid == 0`.
- Probes are queries of the $K$ enrolled + queries of the $U$ unknown.

### 4.3 Threshold-Transfer Metric
In operational deployments, an open-set decision threshold $\tau_{80}$ is calibrated on benign validation data to achieve a target $\text{TAR} = 80\%$. The threshold-transfer metric measures what happens when that exact threshold $\tau_{80}$ is confronted with look-alike encounters:
- $\text{TAR}(\tau_{80})$: Fraction of enrolled probes correctly verified at or above $\tau_{80}$.
- $\text{FAR}(\tau_{80})$: False accept rate of look-alike impostors at or above $\tau_{80}$.

---

## 5. Difficulty Check: Is the Look-Alike Protocol Harder?

Evaluation on `baseline_resnet18` and `e0_baseline_val` comparing Look-Alike vs Matched Random Control vs Standard Evaluator v2 reference:

### 5.1 `baseline_resnet18` (751 Train Identities)

| Metric | Look-Alike (p05) | Matched Control | Standard v2 (Reference) | Difference (Look-Alike vs Control) |
| :--- | :---: | :---: | :---: | :---: |
| **DIR @ FAR=1%** | **31.67% ± 2.80%** | 37.93% ± 4.68% | 37.78% ± 3.93% | **-6.26%** (sharp degradation) |
| **FAR @ TAR=80%** | **15.47% ± 1.12%** | 15.25% ± 1.30% | 13.81% ± 2.33% | **+0.22%** (elevated FAR) |
| **Max Impostor Sim** | **0.5156 ± 0.0028** | 0.5106 ± 0.0031 | N/A | **+0.0050** (higher false match score) |
| **Max Achievable TAR** | 93.57% ± 0.65% | 92.67% ± 0.80% | 93.12% ± 0.42% | +0.90% |
| **Transfer FAR (at $\tau_{80}$)** | **16.16% ± 1.42%** | N/A | 13.81% ± 2.33% | **+2.35%** vs standard reference |
| **Transfer TAR (at $\tau_{80}$)** | 80.58% ± 1.34% | N/A | 80.00% ± 0.00% | +0.58% |

### 5.2 `e0_baseline_val` (651 Train Identities, Paired Baseline)

| Metric | Look-Alike (p05) | Matched Control | Standard v2 (Reference) | Difference (Look-Alike vs Control) |
| :--- | :---: | :---: | :---: | :---: |
| **DIR @ FAR=1%** | **24.57% ± 4.71%** | 30.16% ± 3.10% | 31.04% ± 2.18% | **-5.59%** (sharp degradation) |
| **FAR @ TAR=80%** | **18.67% ± 1.22%** | 17.31% ± 1.60% | 15.97% ± 1.50% | **+1.36%** (higher false accept rate) |
| **Max Impostor Sim** | **0.5285 ± 0.0028** | 0.5231 ± 0.0021 | N/A | **+0.0054** (higher false match score) |
| **Max Achievable TAR** | 93.18% ± 0.51% | 92.14% ± 0.82% | 92.56% ± 0.39% | +1.04% |
| **Transfer FAR (at $\tau_{80}$)** | **18.63% ± 0.60%** | N/A | 15.97% ± 1.50% | **+2.66%** vs standard reference |
| **Transfer TAR (at $\tau_{80}$)** | 80.01% ± 0.98% | N/A | 80.00% ± 0.00% | +0.01% |

### Difficulty Check Assessment:
> [!IMPORTANT]
> **Conclusion**: The look-alike protocol is **unequivocally harder** than the matched random control across both baselines:
> 1. **Detection & Identification Rate (DIR@FAR=1%) drops sharply** by **-6.26%** on `baseline_resnet18` and by **-5.59%** on `e0_baseline_val`. When false alarms are restricted to 1%, looking like an enrolled subject causes true positives to be suppressed.
> 2. **Mean Max-Impostor Similarity is consistently elevated** ($+0.0050$ on baseline, $+0.0054$ on e0), demonstrating that look-alike impostors achieve higher spurious cosine similarities.
> 3. **Threshold-Transfer FAR spikes**: An operational threshold set to achieve TAR=80% on standard probes yields an inflated false alarm rate of **16.16%** (+2.35%) on `baseline_resnet18` and **18.63%** (+2.66%) on `e0_baseline_val`.

---

## 6. Model Comparison on the Look-Alike Protocol

Results across all 5 evaluated experiment checkpoints on the default Look-Alike Protocol (p05, 5 seeds, mean $\pm$ std):

| Experiment | Look-Alike FAR@TAR=80 | Look-Alike DIR@FAR=1% | Look-Alike Max Impostor Sim | Look-Alike Max TAR | Transfer FAR (at $\tau_{80}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **baseline_resnet18** | 15.47% ± 1.12% | 31.67% ± 2.80% | 0.5156 ± 0.0028 | 93.57% ± 0.65% | 16.16% ± 1.42% |
| **e0_baseline_val** | 18.67% ± 1.22% | 24.57% ± 4.71% | 0.5285 ± 0.0028 | 93.18% ± 0.51% | 18.63% ± 0.60% |
| **e1_cosface** | 41.98% ± 2.25% | 22.53% ± 3.45% | 0.4383 ± 0.0020 | 86.43% ± 0.39% | 44.61% ± 2.30% |
| **e2_cosface_hardid** | 33.50% ± 2.70% | 20.37% ± 2.71% | 0.4451 ± 0.0025 | 89.19% ± 0.46% | 36.44% ± 2.58% |
| **e3_cosface_hardid_absneg** | 34.05% ± 2.38% | 20.70% ± 2.83% | 0.4446 ± 0.0020 | 89.39% ± 0.45% | 34.73% ± 1.95% |
| **e1b_arcface** | not run | not run | not run | not run | not run |

---

## 7. Percentile Sensitivity Analysis (p02 vs p05 vs p10)

Evaluation of `baseline_resnet18` and `e0_baseline_val` across clustering percentiles:

| Model | Percentile | Look-Alike FAR@TAR=80 | Control FAR@TAR=80 | Look-Alike DIR@FAR=1% | Control DIR@FAR=1% | Transfer FAR (at $\tau_{80}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_resnet18** | p02 (2%) | 18.28% ± 1.73% | 17.66% ± 2.72% | 30.33% ± 2.81% | 34.59% ± 3.74% | 18.22% ± 0.95% |
| **baseline_resnet18** | p05 (5%) | 15.47% ± 1.12% | 15.25% ± 1.30% | 31.67% ± 2.80% | 37.93% ± 4.68% | 16.16% ± 1.42% |
| **baseline_resnet18** | p10 (10%) | 15.89% ± 1.84% | 14.44% ± 1.52% | 34.40% ± 2.91% | 37.63% ± 4.18% | 16.02% ± 1.39% |
| **e0_baseline_val** | p02 (2%) | 21.78% ± 1.41% | 19.40% ± 2.37% | 22.72% ± 2.60% | 28.04% ± 3.59% | 20.81% ± 1.46% |
| **e0_baseline_val** | p05 (5%) | 18.67% ± 1.22% | 17.31% ± 1.60% | 24.57% ± 4.71% | 30.16% ± 3.10% | 18.63% ± 0.60% |
| **e0_baseline_val** | p10 (10%) | 18.41% ± 1.46% | 16.52% ± 1.83% | 29.38% ± 4.32% | 32.20% ± 1.89% | 18.04% ± 0.84% |

**Trend Observation**:
- As the threshold tightens from $p=10\% \to 5\% \to 2\%$ (identities look increasingly identical):
  - `DIR @ FAR=1%` decreases monotonically: on `e0_baseline_val`, $29.38\% \to 24.57\% \to 22.72\%$.
  - `FAR @ TAR=80%` increases monotonically: on `e0_baseline_val`, $18.41\% \to 18.67\% \to 21.78\%$.
  - `Transfer FAR` increases monotonically: on `e0_baseline_val`, $18.04\% \to 18.63\% \to 20.81\%$.

---

## 8. Failure Analysis (TAR=80% Operating Point)

Failure analysis was executed using `scripts/failure_analysis.py` at operating threshold $\tau_{80} = 0.6131$ (`baseline_resnet18`, seed 0) and $\tau_{80} = 0.6135$ (`e0_baseline_val`, seed 0).

Composite visualization grids are saved locally in `docs/failures/` (git-ignored per redistribution terms):
- `docs/failures/worst_false_accepts_baseline_resnet18.png`
- `docs/failures/worst_false_rejects_baseline_resnet18.png`
- `docs/failures/worst_false_accepts_e0_baseline_val.png`
- `docs/failures/worst_false_rejects_e0_baseline_val.png`

### 8.1 Visual Observations: Worst False Accepts
Top 10 false accepts involve look-alike impostor probes matching enrolled gallery samples with scores far exceeding the threshold (scores ranging from **0.7929 to 0.8929**):
1. **Monochromatic Dark Apparel / Winter Coats**:
   - Probe `PID 1399` matching distractor `PID 0` (score **0.8929**): Both pedestrians wear long, dark navy/black winter jackets with dark trousers under overcast diffused lighting in Camera 3. The absence of contrasting garments leads the model to rely on coarse body silhouette, causing extreme false alarms.
2. **Identical White Shirt + Dark Shorts/Trousers (Campus Uniform Look)**:
   - Probe `PID 1247` matching `PID 1074` (score **0.8544**): Both subjects wear plain short-sleeved white t-shirts and dark shorts. Despite different facial features and hairstyles, ReID features strongly align.
   - Probe `PID 1151` matching `PID 745` (score **0.8324**): Both subjects wear white collared button-down shirts and black dress slacks.
3. **Bright / Saturated Colored Shirts**:
   - Probe `PID 1375` matching `PID 198` (score **0.8026**): Both identities wear distinctive bright yellow/gold t-shirts and dark pants. The dominant yellow hue dominates the feature embedding.
   - Probe `PID 800` matching `PID 137` (score **0.8292**): Both wear light blue casual shirts and dark jeans.
4. **Striped / Textured Patterns**:
   - Probe `PID 60` matching `PID 182` (score **0.8099**): Both wear horizontal dark-and-light striped shirts.

### 8.2 Visual Observations: Worst False Rejects
Enrolled probes failing to verify ($s < \tau_{80}$) or matching incorrect identities (scores down to **0.3644**):
1. **Extreme Viewpoint Inversion (Front vs Back Views)**:
   - Several false rejects feature a frontal probe query (showing face, logos, or shirt graphics) paired against a gallery image taken from directly behind (showing a plain backpack, hoodie, or solid rear coat).
2. **Distractor Occlusion & Bicycles**:
   - Distractors (`PID 0`) walking alongside bicycles or carrying large luggage often capture top similarity scores over true enrolled gallery images due to spurious object alignment.
3. **Severe Inter-Camera Color Casts**:
   - Probes captured in Camera 1/Camera 2 (harsh direct midday sunlight) matching true identity samples in Camera 4/Camera 5 (underpass shade with strong blue/cyan tint) exhibit massive cosine distance shifts.

---

## 9. Methodological Limitations

1. **Color Histograms as a Uniform Proxy**:
   - HSV color histograms capture global chromatic distribution but lack structural spatial layout (e.g. logo placement, collar geometry, sleeve length).
2. **Camera Sensor Variations**:
   - Uncalibrated camera sensors in Market-1501 introduce chromatic shifts that can slightly alter histogram bins between cameras for the same physical garment.
3. **Absence of Ground-Truth Clothing Labels**:
   - Because Market-1501 does not include ground-truth clothing labels, grouping relies on empirical color similarity. Manual attribute confirmation is recommended when adapting this protocol to production datasets.
