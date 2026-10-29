# Look-Alike Evaluation Protocol and Failure Analysis

A model-independent open-set evaluation protocol and failure analysis methodology designed to evaluate person re-identification (ReID) vulnerabilities when distinct individuals share identical or similar clothing (e.g., uniforms, workwear, common campus apparel).

---

## 1. Motivation and Problem Definition

Standard open-set ReID evaluation protocols uniformly partition test identities into **Enrolled** (known) and **Unknown** (impostor) sets. Under uniform random splitting, the chance that an unknown probe happens to visually resemble an enrolled gallery subject is low. However, in security-critical deployments (e.g. corporate offices, industrial sites, airports, schools), impostor attempts and false accepts frequently involve **look-alikes**: distinct people wearing identical uniforms or matching color schemes.

Market-1501 does not include ground-truth clothing labels or semantic attribute annotations. To evaluate look-alike vulnerability without introducing bias toward any neural feature extractor, this protocol extracts **model-independent clothing descriptors** using spatial HSV color histograms with dedicated achromatic gray-level bins and groups visually similar identities using **complete-linkage agglomerative clustering**.

---

## 2. Methodology

### 2.1 Spatial Clothing Descriptors with Achromatic Bins
For every image in the test set (`query/` and `bounding_box_test/`), ignoring junk (`pid == -1`) and distractor (`pid == 0`):
1. **Spatial Cropping**:
   - **Upper Body Region**: Rows 20% to 55% of image height.
   - **Lower Body Region**: Rows 55% to 95% of image height.
   - **Central Column Range**: Columns 15% to 85% of image width (central 70% width crop to suppress background clutter).
2. **Achromatic Pixel Handling and Circular Hue**:
   - In standard HSV color space, pixels with very low saturation (white, gray, neutral) or very low value (black, dark shadows) have ill-defined or unstable circular hue angles.
   - Configurable thresholds (`sat_threshold = 0.15`, `val_threshold = 0.15`, `num_gray_bins = 4`):
     - Pixels with $S \le 0.15 \times 255$ or $V \le 0.15 \times 255$ are classified as **achromatic** and assigned to 4 dedicated gray-level bins partitioned by Value: $\lfloor V \times 4 / 256 \rfloor$.
     - Remaining **chromatic** pixels are partitioned into an $8 \text{ Hue} \times 4 \text{ Saturation} \times 4 \text{ Value} = 128 \text{ bins}$ histogram, where Hue circularly spans $[0, 256)$.
   - Total descriptor length per crop: $4 \text{ (gray)} + 128 \text{ (chromatic)} = 132 \text{ bins}$, $L_1$-normalized ($\sum p_i = 1.0$).
3. **Identity Descriptor**:
   - For each identity $i$, the mean descriptors $\bar{p}_{\text{upper}}^{(i)}$ and $\bar{p}_{\text{lower}}^{(i)}$ are averaged across all query and gallery images of identity $i$.
4. **Pairwise Bhattacharyya Distance**:
   $$D_B(p, q) = \sqrt{1.0 - \sum_{k=1}^{132} \sqrt{p_k \cdot q_k}}$$
   $$D(i, j) = 0.5 \left( D_B(\bar{p}_{\text{upper}}^{(i)}, \bar{p}_{\text{upper}}^{(j)}) + D_B(\bar{p}_{\text{lower}}^{(i)}, \bar{p}_{\text{lower}}^{(j)}) \right)$$

### 2.2 Complete-Linkage Look-Alike Grouping
- Pairwise distances are computed across all 750 valid test identities ($280,875$ unique pairs).
- **Clustering Algorithm**: Agglomerative clustering with complete linkage and `distance_threshold = percentile`.
  - Complete linkage guarantees that the maximum distance between **any pair** within a group is $\le \text{threshold}$.
  - Clusters of size $\ge 2$ form look-alike groups.
  - To prevent oversized clusters, any cluster with size $> 6$ is recursively split via complete linkage until all groups satisfy $2 \le \text{size} \le 6$. Identities without neighbors below the threshold remain ungrouped.

---

## 3. Look-Alike Group Statistics & Sensitivity Analysis

Pairwise distance distribution across all 750 Market-1501 test identities:
- **Global Median Pairwise Distance**: **`0.5022`**

| Percentile | Distance Threshold ($\tau_{cluster}$) | Num Groups | Total IDs in Groups | Group Size Histogram | Mean Intra-Group Dist | Max Intra-Group Dist |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`p01 (1.0%)`** | 0.2345 | 153 | 409 / 750 (54.5%) | `{2: 94, 3: 33, 4: 12, 5: 10, 6: 4}` | 0.1850 | 0.2341 |
| **`p02 (2.0%)`** | 0.2594 | 180 | 509 / 750 (67.9%) | `{2: 102, 3: 36, 4: 19, 5: 17, 6: 6}` | 0.1979 | 0.2589 |
| **`p05 (5.0%)`** | **0.3004** | **191** | **613 / 750 (81.7%)** | **`{2: 78, 3: 49, 4: 24, 5: 26, 6: 14}`** | **0.2172** | **0.3004** |
| **`p10 (10.0%)`** | 0.3413 | 192 | 670 / 750 (89.3%) | `{2: 60, 3: 49, 4: 31, 5: 33, 6: 19}` | 0.2304 | 0.3405 |

### 3.1 Primary Percentile Selection Rule
The primary operating percentile was determined strictly from group statistics prior to model evaluation:
- **Decision Rule**: Choose `p05` as primary if it puts at least $\approx 80$ identities into groups; otherwise choose `p10`.
- **Outcome**: `p05` assigns **613 identities** into 191 groups (81.7% of all test identities), satisfying the rule. Therefore, **`p05` is selected as the primary protocol**.

### 3.2 Sanity Check: Most and Least Similar Groups
- **5 Most Similar Groups (Tightest Clothing Look-Alikes)**:
  1. Mean Dist = **0.1252** (Max: 0.1428) | Size 3 | PIDs: `[92, 137, 153]` (Identical black coats / dark pants)
  2. Mean Dist = **0.1368** (Max: 0.1368) | Size 2 | PIDs: `[230, 275]` (Matching dark blue jacket, beige trousers)
  3. Mean Dist = **0.1440** (Max: 0.1683) | Size 4 | PIDs: `[1141, 1185, 1298, 1340]` (Bright white short-sleeve shirts, dark shorts)
  4. Mean Dist = **0.1466** (Max: 0.1466) | Size 2 | PIDs: `[1482, 1493]` (Red graphic t-shirts, jeans)
  5. Mean Dist = **0.1493** (Max: 0.1857) | Size 5 | PIDs: `[183, 1120, 1144, 1211, 1444]` (Light gray hoodies, dark bottoms)
- **5 Least Similar Groups (Near Cutoff $\tau_{cluster} = 0.3004$)**:
  1. Mean Dist = **0.2969** (Max: 0.2969) | Size 2 | PIDs: `[18, 304]`
  2. Mean Dist = **0.2982** (Max: 0.2982) | Size 2 | PIDs: `[664, 794]`
  3. Mean Dist = **0.2987** (Max: 0.2987) | Size 2 | PIDs: `[858, 1043]`
  4. Mean Dist = **0.3001** (Max: 0.3001) | Size 2 | PIDs: `[388, 1054]`
  5. Mean Dist = **0.3004** (Max: 0.3004) | Size 2 | PIDs: `[295, 1188]`

*(High-resolution visual grids for 12 random groups and 12 random contrast pairs are saved to git-ignored `docs/failures/group_samples/`)*

---

## 4. Evaluation Protocol Design

### 4.1 Look-Alike Open-Set Protocol
1. **Stratified Group Partition**:
   - Within each look-alike group $G$, identities are randomly split into **Enrolled** ($\max(1, \lfloor |G|/2 \rfloor)$) and **Unknown** (remainder; at least 1).
   - Unknown probes are visual look-alikes of enrolled individuals.
2. **Realistic Gallery Size**:
   - Gallery = images of enrolled group identities + **all non-group test identities** (extra enrolled) + distractor images (`pid == 0`).
   - Retaining non-group identities in the gallery ensures realistic search gallery density (19,732 gallery images).
3. **Probes**:
   - Probes = query images of enrolled identities (targets) + query images of unknown group identities (impostors). Non-group queries are omitted.
4. **Evaluation Rules**:
   - Evaluated over 5 deterministic seeds: `0, 1, 2, 3, 4`.
   - Discards junk (`pid == -1`) and same-identity same-camera pairs.

### 4.2 Matched Random Control Protocol
To isolate whether performance differences stem from look-alike visual similarity rather than reduced probe counts or gallery density:
- Exactly the same number of enrolled ($K$) and unknown ($U$) identities are sampled uniformly at random from all 750 test identities.
- Gallery composition rule is identical: enrolled $K$ identities + all remaining $(750 - K - U)$ identities + distractor `pid == 0`.
- Probes are queries of the $K$ enrolled + queries of the $U$ unknown.

### 4.3 Probe Counts and Statistical Precision
The number of probes evaluated varies across protocols:
- **Standard Protocol v2**: 375 enrolled IDs / 375 unknown IDs $\implies 1685.4 \pm 12.4$ target probes, $1682.6 \pm 12.4$ impostor probes (3,368 total).
- **`p05` Look-Alike & Matched Control**: 269 enrolled IDs / 344 unknown IDs $\implies 1215.0 \pm 7.6$ target probes, $1571.0 \pm 7.6$ impostor probes (2,786 total; 582 non-group queries omitted).
- **`p01` Look-Alike & Matched Control**: 184 enrolled IDs / 225 unknown IDs $\implies 843.4 \pm 7.5$ target probes, $1027.0 \pm 7.5$ impostor probes (1,870 total; 1,498 non-group queries omitted).

> [!NOTE]
> Uncertainty is wider when probe sets are small. The tighter protocols (`p01`, `p02`) evaluate fewer probes, resulting in larger standard deviations across seeds.

---

## 5. Experimental Results

### 5.1 Complete Model Performance Benchmark

All metrics evaluated on official Market-1501 test embeddings across 5 deterministic seeds (mean ± std):

| Model | Closed-Set Rank-1 | Closed-Set mAP | Max Achievable TAR | Standard FAR@TAR=80 | Standard DIR@FAR=1% | Look-Alike (p05) FAR@TAR=80 | Matched Control FAR@TAR=80 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_resnet18`** | 90.23% | 76.89% | 93.12% ± 0.42% | 13.81% ± 2.33% | 37.78% ± 3.93% | **17.26% ± 1.46%** | 15.52% ± 0.84% |
| **`e0_baseline_val`** | 89.52% | 74.36% | 92.56% ± 0.39% | 15.97% ± 1.50% | 31.04% ± 2.18% | **19.29% ± 2.05%** | 17.63% ± 1.40% |
| **`e1_cosface`** | 81.26% | 58.69% | 86.15% ± 0.31% | 41.46% ± 2.42% | 28.37% ± 3.10% | **49.41% ± 3.18%** | 47.90% ± 6.56% |
| **`e2_cosface_hardid`** | 83.70% | 61.87% | 88.12% ± 0.63% | 33.34% ± 1.86% | 24.56% ± 3.45% | **40.08% ± 0.64%** | 36.68% ± 4.06% |
| **`e3_cosface_hardid_absneg`** | 84.06% | 62.21% | 88.71% ± 0.59% | 32.48% ± 2.90% | 25.71% ± 1.99% | **37.81% ± 2.45%** | 35.52% ± 2.96% |

> [!IMPORTANT]
> **Comparison of `e1`–`e3` against `e0_baseline_val` on Standard Protocol**:
> On the standard open-set evaluation protocol, `e1_cosface` (FAR@80: **41.46% ± 2.42%**), `e2_cosface_hardid` (FAR@80: **33.34% ± 1.86%**), and `e3_cosface_hardid_absneg` (FAR@80: **32.48% ± 2.90%**) exhibit substantially worse false alarm rates than `e0_baseline_val` (FAR@80: **15.97% ± 1.50%**). The cause of this degradation is analyzed in a separate task.

---

### 5.2 Paired Differences: Look-Alike vs. Matched Control

To rigorously quantify "how much harder" the look-alike protocol is compared to an identically sized random control, we compute the **per-seed paired difference** ($\Delta = \text{Look-Alike} - \text{Matched Control}$) for each seed.

#### Primary Protocol (`p05`) Paired Statistics

| Model | $\Delta \text{FAR@TAR=80}$ (Mean ± Std) | Seeds $> 0$ | $\Delta \text{DIR@FAR=1\%}$ (Mean ± Std) | $\Delta \text{Max Impostor Sim}$ (Mean ± Std) | Transfer FAR at 5-seed Mean $\tau_{std}$ | Standard Reference FAR (5-seed Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_resnet18`** | **+1.75% ± 1.93%** | 3 / 5 | -3.66% ± 4.45% | +0.0047 ± 0.0050 | **16.21% ± 1.50%** | 13.81% ± 2.33% |
| **`e0_baseline_val`** | **+1.66% ± 3.31%** | 2 / 5 | -0.57% ± 5.67% | +0.0045 ± 0.0035 | **18.34% ± 1.18%** | 15.97% ± 1.50% |
| **`e1_cosface`** | **+1.52% ± 8.10%** | 4 / 5 | -3.66% ± 5.05% | +0.0034 ± 0.0026 | **45.42% ± 2.51%** | 41.46% ± 2.42% |
| **`e2_cosface_hardid`** | **+3.40% ± 4.24%** | 4 / 5 | -3.34% ± 3.98% | +0.0032 ± 0.0021 | **36.75% ± 1.60%** | 33.34% ± 1.86% |
| **`e3_cosface_hardid_absneg`** | **+2.29% ± 3.68%** | 4 / 5 | -1.74% ± 1.18% | +0.0016 ± 0.0017 | **34.74% ± 2.42%** | 32.48% ± 2.90% |

#### Paired Differences across All Percentiles (`Look-Alike - Matched Control`)

- **`p01 (1.0% Threshold = 0.2345)`**:
  - `baseline_resnet18`: $\Delta \text{FAR@80} = +1.24\% \pm 3.43\%$ (3/5 $> 0$) | $\Delta \text{DIR@1\%} = +1.49\% \pm 6.07\%$ | $\Delta \text{Sim} = +0.0014 \pm 0.0036$
  - `e0_baseline_val`: $\Delta \text{FAR@80} = +0.51\% \pm 3.68\%$ (1/5 $> 0$) | $\Delta \text{DIR@1\%} = -2.34\% \pm 4.73\%$ | $\Delta \text{Sim} = +0.0015 \pm 0.0035$
  - `e1_cosface`: $\Delta \text{FAR@80} = -6.03\% \pm 10.42\%$ (1/5 $> 0$) | $\Delta \text{DIR@1\%} = +1.37\% \pm 7.44\%$ | $\Delta \text{Sim} = -0.0013 \pm 0.0025$
  - `e2_cosface_hardid`: $\Delta \text{FAR@80} = +4.98\% \pm 5.65\%$ (4/5 $> 0$) | $\Delta \text{DIR@1\%} = +0.69\% \pm 2.76\%$ | $\Delta \text{Sim} = -0.0036 \pm 0.0019$
  - `e3_cosface_hardid_absneg`: $\Delta \text{FAR@80} = -1.04\% \pm 6.24\%$ (2/5 $> 0$) | $\Delta \text{DIR@1\%} = -0.47\% \pm 3.63\%$ | $\Delta \text{Sim} = -0.0030 \pm 0.0022$
- **`p02 (2.0% Threshold = 0.2594)`**:
  - `baseline_resnet18`: $\Delta \text{FAR@80} = +1.89\% \pm 3.33\%$ (4/5 $> 0$) | $\Delta \text{DIR@1\%} = -1.77\% \pm 8.53\%$ | $\Delta \text{Sim} = +0.0043 \pm 0.0072$
  - `e0_baseline_val`: $\Delta \text{FAR@80} = +3.06\% \pm 3.26\%$ (4/5 $> 0$) | $\Delta \text{DIR@1\%} = -4.52\% \pm 4.00\%$ | $\Delta \text{Sim} = +0.0036 \pm 0.0069$
  - `e1_cosface`: $\Delta \text{FAR@80} = -3.22\% \pm 10.43\%$ (3/5 $> 0$) | $\Delta \text{DIR@1\%} = -3.70\% \pm 5.08\%$ | $\Delta \text{Sim} = +0.0021 \pm 0.0031$
  - `e2_cosface_hardid`: $\Delta \text{FAR@80} = +5.75\% \pm 7.22\%$ (4/5 $> 0$) | $\Delta \text{DIR@1\%} = -5.17\% \pm 4.58\%$ | $\Delta \text{Sim} = +0.0014 \pm 0.0045$
  - `e3_cosface_hardid_absneg`: $\Delta \text{FAR@80} = +3.06\% \pm 5.20\%$ (3/5 $> 0$) | $\Delta \text{DIR@1\%} = -4.00\% \pm 4.57\%$ | $\Delta \text{Sim} = +0.0008 \pm 0.0034$
- **`p10 (10.0% Threshold = 0.3413)`**:
  - `baseline_resnet18`: $\Delta \text{FAR@80} = +0.88\% \pm 1.41\%$ (4/5 $> 0$) | $\Delta \text{DIR@1\%} = -2.54\% \pm 4.67\%$ | $\Delta \text{Sim} = +0.0029 \pm 0.0018$
  - `e0_baseline_val`: $\Delta \text{FAR@80} = +2.59\% \pm 1.88\%$ (5/5 $> 0$) | $\Delta \text{DIR@1\%} = -3.97\% \pm 5.39\%$ | $\Delta \text{Sim} = +0.0049 \pm 0.0012$
  - `e1_cosface`: $\Delta \text{FAR@80} = +3.04\% \pm 6.39\%$ (4/5 $> 0$) | $\Delta \text{DIR@1\%} = -4.54\% \pm 3.91\%$ | $\Delta \text{Sim} = +0.0032 \pm 0.0011$
  - `e2_cosface_hardid`: $\Delta \text{FAR@80} = +2.92\% \pm 5.29\%$ (3/5 $> 0$) | $\Delta \text{DIR@1\%} = -2.99\% \pm 1.56\%$ | $\Delta \text{Sim} = +0.0021 \pm 0.0025$
  - `e3_cosface_hardid_absneg`: $\Delta \text{FAR@80} = +1.65\% \pm 3.58\%$ (4/5 $> 0$) | $\Delta \text{DIR@1\%} = -3.62\% \pm 2.28\%$ | $\Delta \text{Sim} = +0.0023 \pm 0.0011$

### 5.3 Difficulty Assessment & Protocol Interpretation
When comparing against the matched control:
- In the primary `p05` protocol, the paired difference in FAR@TAR=80 averages **$+1.52\%$ to $+3.40\%$** across models.
- However, the cross-seed standard deviation of the paired difference ($\pm 1.9\%$ to $\pm 8.1\%$) exceeds the mean difference in every model except `baseline_resnet18`, and the paired difference is not positive on all 5 seeds (ranging from 2/5 to 4/5 seeds positive).
- Consequently, the look-alike protocol cannot be characterized as "systematically" harder than a matched random partition; rather, it introduces a moderate, variable increase in false alarm rate that reflects real-world look-alike pairing noise.
- When an operational threshold $\tau_{std}$ calibrated on benign data is transferred to look-alike scenarios, the 5-seed mean transfer FAR is elevated by **+2.26% to +3.96%** above the standard operating baseline across models.

---

## 6. Tightness Analysis: Quantile Binning by Color Distance

To evaluate whether clothing color distance alone predicts false accepts, unknown probes in `baseline_resnet18` were binned into 4 quantile intervals based on the color distance $d_{color}$ between the probe identity and its nearest enrolled group-mate:

$$\text{Quantile Edges (5-seed mean)}: [0.1014, 0.1799, 0.2149, 0.2498, 0.3004]$$

The false accept rate at the standard-protocol operating threshold ($\tau_{std}$, calibrated for $\text{TAR} = 80\%$ on the same seed) was evaluated per bin:

| Bin Interval ($d_{color}$) | Look-Alike Probe Count | Look-Alike FAR@$\tau_{std}$ | Matched Control Probe Count | Matched Control FAR@$\tau_{std}$ |
| :---: | :---: | :---: | :---: | :---: |
| **Bin 1 $[0.1014, 0.1799)$** (Closest) | $390.4 \pm 2.9$ | **20.69% ± 1.65%** | $311.6 \pm 37.5$ | 18.25% ± 2.45% |
| **Bin 2 $[0.1799, 0.2149)$** | $392.8 \pm 2.4$ | **13.89% ± 1.15%** | $319.6 \pm 33.9$ | 14.10% ± 3.57% |
| **Bin 3 $[0.2149, 0.2498)$** | $392.4 \pm 1.4$ | **17.37% ± 2.04%** | $326.8 \pm 22.0$ | 14.80% ± 2.16% |
| **Bin 4 $[0.2498, 0.3004]$** (Loosest) | $395.4 \pm 3.4$ | **12.93% ± 2.53%** | $330.2 \pm 21.8$ | 13.96% ± 2.57% |
| **Beyond Bin 4 $(> 0.3004)$** | 0.0 (bounded by $\tau$) | N/A | $253.8 \pm 19.8$ | 11.22% ± 5.41% |

### Key Findings:
1. **Elevated FAR at Close Distances**: In both the look-alike and matched control protocols, probes with the closest color distance ($d_{color} < 0.18$, Bin 1) exhibit the highest false accept rate (**~18%–21%**), confirming that close clothing color similarity increases false alarm vulnerability.
2. **Non-Monotonicity within Clusters**: Within the clustered range ($0.18 \le d_{color} \le 0.30$), FAR does not decrease monotonically with color distance (Bin 3 shows higher FAR than Bin 2 in both protocols). Fine-grained spatial cues, body build, accessories, and camera viewpoint variations heavily influence whether a false accept occurs.
3. **Parity with Matched Control within Same Distance Range**: When matched control impostor probes happen to have a color neighbor within the same distance range, their false accept rate matches that of the look-alike protocol. The drop in control FAR occurs beyond $0.3004$ (falling to 11.22%), a range excluded from look-alike groups by complete-linkage clustering.

---

## 7. Failure Analysis & Base Rate Analysis

### 7.1 Group-Mate Attribution vs. Random Base Rate

When an unknown impostor probe produces a false accept, we measure whether its top-1 gallery match is an individual from its **own look-alike group**:
- **Random Base Rate**: If top-1 matches were distributed uniformly among enrolled identities, the probability that an unknown probe matches one of its own group-mates is:
  $$\mathbb{E}[\text{Base Rate}] = \frac{|G_{\text{enrolled}}|}{N_{\text{enrolled}}} \approx \frac{1.6}{269} = \mathbf{0.60\%}$$
- **Observed Rate (`baseline_resnet18`)**: Across all unknown probes, the top-1 match is an enrolled group-mate in **`21.87% ± 0.77%`** of cases.
- **Enrichment Factor**: The observed group-mate retrieval rate is **`36.5x` higher than random chance** ($20.95\% \pm 0.83\%$ on `e0_baseline_val`, a **`35.0x`** enrichment).

### 7.2 Worst False Accepts (Seed 0, `baseline_resnet18`, $\tau_{std} = 0.6131$)

Among the 10 worst false accepts (highest similarity scores $\ge \tau_{std}$), **`6 / 10` are matched directly to their own look-alike group-mate**:

| Rank | Score | Status | Probe Identity & Camera | Top Match Identity & Camera | Visual Pattern |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **#01** | **0.8406** | `[OWN GROUP MATE]` | Probe PID 182 (cam 4, `0182_c4s1_034626_00.jpg`) | Match PID 60 (cam 4, `0060_c4s1_007651_01.jpg`) | Identical black winter jacket, dark trousers, similar build |
| **#02** | 0.8385 | `[OTHER ID]` | Probe PID 1247 (cam 3, `1247_c3s3_020678_00.jpg`) | Match PID 1074 (cam 1, `1074_c1s5_007861_06.jpg`) | Similar light gray t-shirt and dark backpack straps |
| **#03** | **0.8265** | `[OWN GROUP MATE]` | Probe PID 137 (cam 5, `0137_c5s1_024301_00.jpg`) | Match PID 153 (cam 5, `0153_c5s1_025776_01.jpg`) | Identical navy blue polo shirt, khaki trousers |
| **#04** | 0.8246 | `[OTHER ID]` | Probe PID 913 (cam 3, `0913_c3s2_113553_00.jpg`) | Match PID 189 (cam 3, `0189_c3s1_041001_03.jpg`) | White t-shirt, dark shorts, comparable hair length |
| **#05** | 0.8231 | `[OTHER ID]` | Probe PID 1151 (cam 2, `1151_c2s3_003407_00.jpg`) | Match PID 745 (cam 2, `0745_c2s2_101807_03.jpg`) | Dark blazer, white collared shirt |
| **#06** | **0.8228** | `[OWN GROUP MATE]` | Probe PID 360 (cam 4, `0360_c4s2_009842_00.jpg`) | Match PID 725 (cam 3, `0725_c3s2_061753_02.jpg`) | Gray long-sleeve hoodie, black pants |
| **#07** | **0.8026** | `[OWN GROUP MATE]` | Probe PID 1062 (cam 3, `1062_c3s2_141794_00.jpg`) | Match PID 502 (cam 3, `0502_c3s1_133683_03.jpg`) | Bright red t-shirt, beige shorts |
| **#08** | **0.7933** | `[OWN GROUP MATE]` | Probe PID 182 (cam 2, `0182_c2s1_033676_00.jpg`) | Match PID 60 (cam 1, `0060_c1s1_008526_01.jpg`) | Same pair as #01 under cross-camera illumination |
| **#09** | **0.7864** | `[OWN GROUP MATE]` | Probe PID 1361 (cam 3, `1361_c3s3_042203_00.jpg`) | Match PID 786 (cam 3, `0786_c3s2_088903_02.jpg`) | Yellow t-shirt, blue denim jeans |
| **#10** | 0.7768 | `[OTHER ID]` | Probe PID 61 (cam 4, `0061_c4s1_009376_00.jpg`) | Match PID 80 (cam 4, `0080_c4s1_012951_01.jpg`) | Black t-shirt, dark shorts |

### 7.3 Standard Protocol Worst False Accepts (Reference)
In the standard random split at the same threshold ($\tau_{std} = 0.6131$), look-alike pairs (such as `PID 182` and `PID 60`, or `PID 137` and `PID 153`) also appear at the top of the false accept list:
- `#01`: Score 0.8606 | Probe PID 189 $\rightarrow$ Match PID 913
- `#02`: Score 0.8406 | Probe PID 182 $\rightarrow$ Match PID 60
- `#03`: Score 0.8265 | Probe PID 137 $\rightarrow$ Match PID 153
- `#04`: Score 0.8231 | Probe PID 1151 $\rightarrow$ Match PID 745
- `#06`: Score 0.7933 | Probe PID 182 $\rightarrow$ Match PID 60
- `#08`: Score 0.7779 | Probe PID 189 $\rightarrow$ Match PID 913
- `#10`: Score 0.7642 | Probe PID 182 $\rightarrow$ Match PID 60

In the standard protocol, these pairs appear only when random chance happens to place them on opposite sides of the split. The look-alike protocol systematically tests this vulnerability across all groups.

---

## 8. Limitations

1. **Color Histograms as a Weak Clothing Proxy**:
   - Spatial HSV color histograms capture global color distribution across upper and lower body crops, but ignore fine texture, logos, fabric patterns, garment cuts, and accessory details.
   - Distinct people with similar overall colors (e.g. black coats or white t-shirts) are grouped together even if their physical appearance and facial structures differ markedly.
2. **High Coverage at `p05`**:
   - At the 5th percentile (`p05`), 613 of 750 test identities (81.7%) are assigned to look-alike groups. This indicates that Market-1501 pedestrian clothing contains substantial color redundancy (many students wearing dark jackets, white shirts, or jeans).
   - While `p01` provides a tighter clustering cutoff (threshold 0.2345), it includes fewer identities (409), reducing probe sample size.
3. **Statistical Uncertainty on Subsets**:
   - Because non-group identities are excluded from the probe pool (though preserved in the gallery), look-alike protocols evaluate fewer probes (e.g. ~1,215 targets and ~1,571 impostors in `p05`, vs ~1,685 of each in standard protocol).
   - This smaller probe sample size yields wider standard deviations across random partition seeds.
