# Inference-Time Decision Rules for Open-Set Person Re-Identification

## 1. Executive Summary

This study evaluates post-hoc, inference-time decision rules operating on saved L2-normalized feature embeddings from the baseline model (`e0_baseline_val`, ResNet-18 trained on Market-1501 with 100 held-out validation identities). The objective is to determine whether inference-time score transformations—computed purely on CPU without retraining—can suppress false accepts at fixed recall and improve open-set verification across standard and look-alike adversarial conditions.

### Improvement Rule Definition
Per the pre-registered evaluation criteria, a candidate rule qualifies as an improvement (**win**) over the `raw` baseline on a protocol if and only if:
1. **$\text{FAR@TAR=80\%}$** is lower by more than about one standard deviation ($> 0.95\,\text{std}$) of paired per-seed differences, with positive improvement in at least 4 of 5 seeds ($\Delta \text{FAR} > 0$).
2. **$\text{DIR@FAR=1\%}$** is higher by more than about one standard deviation ($> 0.95\,\text{std}$) of paired differences, with positive improvement in at least 4 of 5 seeds ($\Delta \text{DIR} > 0$).
3. **$\text{DIR@FAR=5\%}$** is higher by more than about one standard deviation ($> 0.95\,\text{std}$) of paired differences, with positive improvement in at least 4 of 5 seeds ($\Delta \text{DIR} > 0$).
If all three conditions are met, the verdict is **win**. If any metric is significantly worse ($< -0.95\,\text{std}$), the verdict is **loss**. Otherwise, it is a **tie**.

### Key Outcomes
1. **Validation Selection Confirms `raw`**: On the two-fold validation selection protocol (100 held-out validation IDs, 10 runs across 2 folds and 5 seeds), **no candidate rule beat `raw` on both $\text{DIR@FAR=5\%}$ and $\text{FAR@TAR=80\%}$ by $>1.0\,\text{std}$**. Therefore, following strict protocol rules without test-peeking, `raw` top-1 cosine matching was deterministically selected as the operating rule.
2. **Standard Test Protocol**: Post-hoc on the standard test set, ambiguity gap rules (`gap_0.50`, `gap_0.25`, `gap_0.75`) achieve a **win** over `raw`, with `gap_0.50` reducing $\text{FAR@TAR=80\%}$ by $3.17 \pm 0.60\%$ (5/5 seeds) and boosting $\text{DIR@FAR=5\%}$ by $6.25 \pm 0.83\%$ (5/5 seeds).
3. **Look-Alike (p05) Adversarial Protocol**: **No rule beats `raw` on Look-Alike p05** (all candidate rules result in **tie** or **loss**). Under dense look-alike impostors, `gap_0.50` fails to improve $\text{DIR@FAR=1\%}$ (paired diff $-1.74 \pm 4.03\%$, positive in only 2 of 5 seeds $\implies$ **tie**).
4. **Score Normalization & CSLS**: `znorm`, `snorm`, and `csls` fail across all protocols (**loss** on every metric, 0/5 positive seeds). A 50-identity cohort in Market-1501 is far too small, causing severe camera-lighting bias and variance collapse.

---

## 2. Mathematical Formulations

All queries $q \in \mathbb{R}^D$ and gallery items $g \in \mathbb{R}^D$ are unit L2-normalized ($\|q\|_2 = \|g\|_2 = 1$). A disjoint cohort matrix $\mathbf{C} \in \mathbb{R}^{M \times D}$ is formed from held-out validation identities. For any test probe $q$, camera-junk exclusion masks the same identity from the same camera.

### 2.1 Raw Baseline (`raw`)
The baseline decision score is the standard top-1 cosine similarity to the enrolled gallery:
$$s_{\text{raw}}(q) = s_1(q) = \max_{g \in \mathcal{G}_{\text{valid}}} \cos(q, g)$$
The predicted identity is $\hat{y}(q) = \text{arg max}_{g \in \mathcal{G}_{\text{valid}}} \cos(q, g)$.

### 2.2 Ambiguity Gap Rule (`gap`)
The gap rule suppresses ambiguous probes that match multiple enrolled identities equally well:
$$s_{\text{gap}}(q; \lambda) = s_1(q) - \lambda \cdot s_2(q), \quad \lambda \in \{0.25, 0.50, 0.75, 1.0\}$$
where:
- $s_1(q) = \max_{g \in \mathcal{G}_{\text{valid}}} \cos(q, g)$ with predicted identity $y_1 = \text{pid}(g_1^*)$.
- $s_2(q) = \max_{g \in \mathcal{G}_{\text{valid}}, \, \text{pid}(g) \neq y_1} \cos(q, g)$ is the maximum similarity to an enrolled identity distinct from $y_1$.
- **Distractor Handling**: Distractor images ($\text{pid} = 0$) are treated as distinct individuals; each distractor image is treated as its own unique identity when evaluating candidate matches for $s_2$. Background junk images ($\text{pid} = -1$) are filtered out.
- Identity prediction remains $y_1$.

### 2.3 Probe Z-Score Normalization (`znorm`)
Standardizes probe similarity relative to a cohort $\mathbf{C}$:
$$s_{\text{znorm}}(q; k) = \frac{s_1(q) - \mu_q^{(k)}}{\sigma_q^{(k)} + \epsilon}, \quad k \in \{10, 50\}$$
where $\mu_q^{(k)}$ and $\sigma_q^{(k)}$ are the sample mean and sample standard deviation of the top-$k$ cosine similarities between probe $q$ and cohort items in $\mathbf{C}$. Identity prediction remains the raw top-1 cosine match.

### 2.4 Symmetric Score Normalization (`snorm`)
Averages the z-normalized scores of the probe and the top-1 retrieved gallery item:
$$s_{\text{snorm}}(q; k) = \frac{1}{2} \left[ \frac{s_1(q) - \mu_q^{(k)}}{\sigma_q^{(k)} + \epsilon} + \frac{s_1(q) - \mu_{g_1}^{(k)}}{\sigma_{g_1}^{(k)} + \epsilon} \right]$$
where $\mu_{g_1}^{(k)}$ and $\sigma_{g_1}^{(k)}$ are computed over the top-$k$ cosines of the matched gallery embedding $g_1$ against cohort $\mathbf{C}$.

### 2.5 Inductive CSLS (`csls`)
Cross-Domain Similarity Local Scaling adapted strictly to the inductive setting where density statistics are computed exclusively against the held-out cohort matrix $\mathbf{C}$:
$$s_{\text{csls}}(q, g; k) = 2 \cos(q, g) - r_k(q) - r_k(g)$$
where:
$$r_k(x) = \frac{1}{k} \sum_{c \in \mathcal{N}_k(x; \mathbf{C})} \cos(x, c)$$
denotes the mean cosine similarity of vector $x$ to its $k$ nearest neighbors in cohort $\mathbf{C}$.
In contrast to the standard cosine rule, CSLS scores alter both the pairwise ranking $\hat{g} = \text{arg max}_{g \in \mathcal{G}} s_{\text{csls}}(q, g)$ and the acceptance decision score $s(q) = \max_g s_{\text{csls}}(q, g)$.

---

## 3. Offline Validation Selection Protocol

To ensure strict zero test-peeking, decision rule hyperparameter selection is conducted exclusively on the 100 held-out validation identities ($1,912$ images: $399$ queries, $1,513$ gallery).

### Protocol Setup
- **Split**: 100 validation IDs partitioned deterministically (`seed=42`) into two disjoint 50-ID subsets: Half A and Half B.
- **Two Folds**:
  - **Fold 1**: Half A = Cohort ($980$ samples), Half B = Selection set ($50$ IDs).
  - **Fold 2**: Half B = Cohort ($932$ samples), Half A = Selection set ($50$ IDs).
- **Seeds**: For each fold, the 50 selection identities are split into 25 enrolled and 25 unknown over 5 random seeds ($0, 1, 2, 3, 4$), yielding 10 evaluation runs per rule.
- **Winner Criterion**: A non-raw candidate rule is selected if and only if it beats `raw` on **both** $\text{DIR@FAR=5\%}$ and $\text{FAR@TAR=80\%}$ by more than $1.0\,\text{std}$ of the paired per-seed differences across the 10 runs. Otherwise, `raw` is preserved.

### Validation Selection Results (10 Runs)

| Rule | Mean DIR@5% (%) | Mean FAR@80% (%) | $\Delta \text{DIR@5\%}$ (vs Raw) | $\Delta \text{FAR@80\%}$ (vs Raw) | Qualifying (>1 std on both) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`raw`** | **$94.76 \pm 4.00$** | **$0.30 \pm 0.46$** | $+0.00 \pm 0.00$ | $+0.00 \pm 0.00$ | **Selected Baseline** |
| `gap_lambda_0.25` | $94.24 \pm 4.48$ | $0.61 \pm 0.68$ | $-0.52 \pm 2.70$ | $-0.31 \pm 0.66$ | No |
| `gap_lambda_0.50` | $92.97 \pm 5.23$ | $0.82 \pm 1.21$ | $-1.78 \pm 2.62$ | $-0.52 \pm 1.25$ | No |
| `gap_lambda_0.75` | $90.47 \pm 6.41$ | $1.46 \pm 1.51$ | $-4.29 \pm 4.09$ | $-1.16 \pm 1.51$ | No |
| `gap_lambda_1.0` | $86.54 \pm 7.27$ | $2.56 \pm 1.81$ | $-8.22 \pm 5.21$ | $-2.26 \pm 1.68$ | No |
| `csls_k10` | $93.73 \pm 4.22$ | $0.31 \pm 0.47$ | $-1.03 \pm 2.36$ | $-0.01 \pm 0.62$ | No |
| `csls_k5` | $93.32 \pm 3.83$ | $0.60 \pm 0.68$ | $-1.43 \pm 2.24$ | $-0.30 \pm 0.66$ | No |
| `snorm_k50` | $86.28 \pm 7.59$ | $3.45 \pm 1.67$ | $-8.47 \pm 6.46$ | $-3.15 \pm 1.79$ | No |
| `snorm_k10` | $82.52 \pm 7.54$ | $5.42 \pm 2.33$ | $-12.23 \pm 6.44$ | $-5.12 \pm 2.44$ | No |
| `znorm_k50` | $81.70 \pm 7.66$ | $4.21 \pm 3.08$ | $-13.06 \pm 6.14$ | $-3.91 \pm 3.19$ | No |
| `znorm_k10` | $71.17 \pm 11.30$ | $5.81 \pm 2.16$ | $-23.59 \pm 8.88$ | $-5.51 \pm 2.34$ | No |

**Validation Outcome**: No candidate rule qualified. **`raw` is confirmed as the selected rule.**

---

## 4. Test Set Evaluation Across Protocols (5 Seeds Each)

We evaluate all decision rules on the test set embeddings ($3,368$ query, $19,732$ gallery) across three distinct protocols using 5 fixed seeds ($0, 1, 2, 3, 4$). For cohort-dependent rules (`znorm`, `snorm`, `csls`), the cohort consists of the 50 validation identities in Half A ($980$ samples).

### 4.1 Standard Open-Set Protocol (50% Enrolled, 50% Unknown)

| Rule | Max TAR (%) | FAR@TAR=80% (%) | FAR@TAR=90% (%) | DIR@FAR=1% (%) | DIR@FAR=5% (%) | $\Delta \text{FAR@80}$ (pos) | $\Delta \text{DIR@1}$ (pos) | $\Delta \text{DIR@5}$ (pos) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`raw` (selected)** | **$92.56 \pm 0.39$** | $15.97 \pm 1.50$ | $45.56 \pm 3.91$ | $31.04 \pm 2.18$ | $61.31 \pm 2.15$ | $+0.00 \pm 0.00$ (0/5) | $+0.00 \pm 0.00$ (0/5) | $+0.00 \pm 0.00$ (0/5) | **baseline** |
| `gap_lambda_0.50` | $92.56 \pm 0.39$ | **$12.79 \pm 1.39$** | **$39.85 \pm 2.59$** | **$36.61 \pm 2.48$** | **$67.56 \pm 2.56$** | $+3.17 \pm 0.60$ (5/5) | $+5.58 \pm 1.75$ (5/5) | $+6.25 \pm 0.83$ (5/5) | **win** |
| `gap_lambda_0.25` | $92.56 \pm 0.39$ | $13.93 \pm 1.38$ | $41.98 \pm 3.74$ | $35.63 \pm 3.76$ | $66.64 \pm 2.37$ | $+2.03 \pm 0.59$ (5/5) | $+4.60 \pm 1.72$ (5/5) | $+5.34 \pm 0.59$ (5/5) | **win** |
| `gap_lambda_0.75` | $92.56 \pm 0.39$ | $14.98 \pm 0.77$ | $42.85 \pm 1.95$ | $35.52 \pm 2.50$ | $65.47 \pm 1.94$ | $+0.99 \pm 0.83$ (4/5) | $+4.48 \pm 2.53$ (5/5) | $+4.16 \pm 0.65$ (5/5) | **win** |
| `gap_lambda_1.0` | $92.56 \pm 0.39$ | $20.43 \pm 0.93$ | $64.36 \pm 2.27$ | $33.32 \pm 2.76$ | $60.76 \pm 1.58$ | $-4.47 \pm 0.60$ (0/5) | $+2.28 \pm 2.65$ (4/5) | $-0.55 \pm 0.87$ (2/5) | loss |
| `csls_k10` | $92.28 \pm 0.41$ | $23.61 \pm 2.15$ | $63.79 \pm 6.75$ | $17.53 \pm 1.91$ | $45.75 \pm 2.99$ | $-7.64 \pm 1.05$ (0/5) | $-13.51 \pm 2.56$ (0/5) | $-15.56 \pm 1.24$ (0/5) | loss |
| `csls_k5` | $92.19 \pm 0.37$ | $24.64 \pm 2.60$ | $65.89 \pm 6.58$ | $16.66 \pm 2.10$ | $44.49 \pm 3.46$ | $-8.68 \pm 1.69$ (0/5) | $-14.38 \pm 2.53$ (0/5) | $-16.81 \pm 1.93$ (0/5) | loss |
| `snorm_k10` | $92.56 \pm 0.39$ | $44.12 \pm 2.80$ | $75.77 \pm 4.38$ | $6.77 \pm 1.28$ | $23.96 \pm 2.94$ | $-28.16 \pm 1.82$ (0/5) | $-24.27 \pm 1.38$ (0/5) | $-37.35 \pm 3.05$ (0/5) | loss |
| `snorm_k50` | $92.56 \pm 0.39$ | $47.85 \pm 2.97$ | $79.81 \pm 3.26$ | $7.02 \pm 1.76$ | $25.64 \pm 2.93$ | $-31.89 \pm 1.70$ (0/5) | $-24.02 \pm 2.10$ (0/5) | $-35.67 \pm 2.12$ (0/5) | loss |
| `znorm_k10` | $92.56 \pm 0.39$ | $46.87 \pm 3.56$ | $75.39 \pm 3.58$ | $5.22 \pm 0.84$ | $20.17 \pm 2.10$ | $-30.91 \pm 2.43$ (0/5) | $-25.82 \pm 1.41$ (0/5) | $-41.13 \pm 2.00$ (0/5) | loss |
| `znorm_k50` | $92.56 \pm 0.39$ | $49.46 \pm 3.57$ | $79.93 \pm 2.94$ | $6.79 \pm 1.72$ | $21.97 \pm 3.72$ | $-33.49 \pm 2.15$ (0/5) | $-24.25 \pm 2.48$ (0/5) | $-39.33 \pm 2.58$ (0/5) | loss |

### 4.2 Look-Alike Protocol (Percentile 5%, Distance Threshold = 0.3517)

| Rule | Max TAR (%) | FAR@TAR=80% (%) | FAR@TAR=90% (%) | DIR@FAR=1% (%) | DIR@FAR=5% (%) | $\Delta \text{FAR@80}$ (pos) | $\Delta \text{DIR@1}$ (pos) | $\Delta \text{DIR@5}$ (pos) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`raw` (selected)** | **$92.51 \pm 0.48$** | $19.29 \pm 2.05$ | $51.48 \pm 6.16$ | $28.01 \pm 5.18$ | $58.19 \pm 2.28$ | $+0.00 \pm 0.00$ (0/5) | $+0.00 \pm 0.00$ (0/5) | $+0.00 \pm 0.00$ (0/5) | **baseline** |
| `gap_lambda_0.50` | $92.51 \pm 0.48$ | **$14.99 \pm 1.19$** | **$47.14 \pm 6.72$** | $26.27 \pm 7.72$ | **$63.06 \pm 2.69$** | $+4.30 \pm 1.28$ (5/5) | $-1.74 \pm 4.03$ (2/5) | $+4.87 \pm 1.10$ (5/5) | tie |
| `gap_lambda_0.25` | $92.51 \pm 0.48$ | $16.31 \pm 0.87$ | $48.16 \pm 5.94$ | $29.38 \pm 6.13$ | $61.59 \pm 3.29$ | $+2.98 \pm 1.45$ (5/5) | $+1.36 \pm 2.13$ (3/5) | $+3.41 \pm 1.31$ (5/5) | tie |
| `gap_lambda_0.75` | $92.51 \pm 0.48$ | $15.44 \pm 0.47$ | $47.52 \pm 5.63$ | $22.92 \pm 5.96$ | $61.55 \pm 2.55$ | $+3.85 \pm 1.76$ (5/5) | $-5.09 \pm 2.23$ (0/5) | $+3.37 \pm 1.98$ (5/5) | loss |
| `gap_lambda_1.0` | $92.51 \pm 0.48$ | $21.91 \pm 0.93$ | $63.01 \pm 4.83$ | $21.62 \pm 5.48$ | $56.73 \pm 2.76$ | $-2.62 \pm 2.48$ (1/5) | $-6.39 \pm 1.60$ (0/5) | $-1.46 \pm 2.50$ (1/5) | loss |
| `csls_k10` | $92.30 \pm 0.62$ | $26.41 \pm 2.43$ | $64.57 \pm 5.35$ | $15.27 \pm 2.52$ | $41.38 \pm 2.70$ | $-7.11 \pm 0.95$ (0/5) | $-12.74 \pm 3.82$ (0/5) | $-16.80 \pm 1.47$ (0/5) | loss |
| `csls_k5` | $92.17 \pm 0.58$ | $26.88 \pm 2.87$ | $66.87 \pm 5.55$ | $14.19 \pm 2.23$ | $40.99 \pm 2.89$ | $-7.59 \pm 1.20$ (0/5) | $-13.82 \pm 4.38$ (0/5) | $-17.19 \pm 2.33$ (0/5) | loss |
| `snorm_k10` | $92.51 \pm 0.48$ | $46.59 \pm 3.37$ | $76.33 \pm 3.30$ | $4.33 \pm 0.95$ | $21.67 \pm 2.67$ | $-27.30 \pm 1.83$ (0/5) | $-23.68 \pm 5.66$ (0/5) | $-36.51 \pm 3.15$ (0/5) | loss |
| `snorm_k50` | $92.51 \pm 0.48$ | $49.53 \pm 2.29$ | $79.40 \pm 2.55$ | $5.73 \pm 1.46$ | $22.09 \pm 2.40$ | $-30.24 \pm 0.89$ (0/5) | $-22.28 \pm 6.22$ (0/5) | $-36.10 \pm 3.24$ (0/5) | loss |
| `znorm_k10` | $92.51 \pm 0.48$ | $47.45 \pm 4.12$ | $75.27 \pm 3.44$ | $3.65 \pm 2.04$ | $19.02 \pm 2.11$ | $-28.16 \pm 2.22$ (0/5) | $-24.36 \pm 6.06$ (0/5) | $-39.16 \pm 2.93$ (0/5) | loss |
| `znorm_k50` | $92.51 \pm 0.48$ | $50.43 \pm 2.55$ | $81.92 \pm 2.43$ | $5.49 \pm 1.77$ | $19.39 \pm 2.21$ | $-31.14 \pm 1.16$ (0/5) | $-22.52 \pm 6.68$ (0/5) | $-38.80 \pm 3.60$ (0/5) | loss |

### 4.3 Look-Alike Protocol (Percentile 1%, Distance Threshold = 0.2929)

| Rule | Max TAR (%) | FAR@TAR=80% (%) | FAR@TAR=90% (%) | DIR@FAR=1% (%) | DIR@FAR=5% (%) | $\Delta \text{FAR@80}$ (pos) | $\Delta \text{DIR@1}$ (pos) | $\Delta \text{DIR@5}$ (pos) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`raw` (selected)** | **$91.41 \pm 0.41$** | $22.48 \pm 0.97$ | $64.66 \pm 4.56$ | $22.39 \pm 4.26$ | $50.30 \pm 3.99$ | $+0.00 \pm 0.00$ (0/5) | $+0.00 \pm 0.00$ (0/5) | $+0.00 \pm 0.00$ (0/5) | **baseline** |
| `gap_lambda_0.25` | $91.41 \pm 0.41$ | $19.85 \pm 1.44$ | $63.09 \pm 7.25$ | **$26.91 \pm 2.96$** | $56.34 \pm 3.56$ | $+2.63 \pm 0.70$ (5/5) | $+4.52 \pm 2.08$ (5/5) | $+6.04 \pm 1.05$ (5/5) | **win** |
| `gap_lambda_0.50` | $91.41 \pm 0.41$ | **$19.50 \pm 1.79$** | **$62.17 \pm 7.13$** | $24.54 \pm 4.53$ | **$57.63 \pm 3.20$** | $+2.98 \pm 1.39$ (5/5) | $+2.15 \pm 2.40$ (4/5) | $+7.33 \pm 1.28$ (5/5) | tie |
| `gap_lambda_0.75` | $91.41 \pm 0.41$ | $20.93 \pm 1.56$ | $67.54 \pm 5.49$ | $21.61 \pm 3.65$ | $57.09 \pm 3.48$ | $+1.55 \pm 1.58$ (4/5) | $-0.79 \pm 2.33$ (2/5) | $+6.79 \pm 1.99$ (5/5) | tie |
| `gap_lambda_1.0` | $91.41 \pm 0.41$ | $29.18 \pm 2.38$ | $80.22 \pm 5.80$ | $18.48 \pm 4.85$ | $52.06 \pm 2.95$ | $-6.70 \pm 2.42$ (0/5) | $-3.91 \pm 3.22$ (1/5) | $+1.76 \pm 1.48$ (4/5) | loss |
| `csls_k10` | $91.00 \pm 0.36$ | $30.52 \pm 0.61$ | $82.38 \pm 5.13$ | $15.56 \pm 3.11$ | $38.18 \pm 1.21$ | $-8.04 \pm 1.01$ (0/5) | $-6.83 \pm 2.12$ (0/5) | $-12.11 \pm 3.87$ (0/5) | loss |
| `csls_k5` | $90.84 \pm 0.41$ | $30.99 \pm 1.01$ | $84.54 \pm 6.46$ | $13.99 \pm 2.65$ | $37.68 \pm 1.19$ | $-8.51 \pm 1.51$ (0/5) | $-8.41 \pm 2.38$ (0/5) | $-12.62 \pm 3.65$ (0/5) | loss |
| `snorm_k10` | $91.41 \pm 0.41$ | $51.88 \pm 2.65$ | $85.15 \pm 5.10$ | $3.04 \pm 0.46$ | $19.12 \pm 2.86$ | $-29.40 \pm 3.26$ (0/5) | $-19.35 \pm 3.94$ (0/5) | $-31.17 \pm 3.41$ (0/5) | loss |
| `snorm_k50` | $91.41 \pm 0.41$ | $54.16 \pm 2.39$ | $86.19 \pm 3.50$ | $4.74 \pm 0.87$ | $21.20 \pm 1.67$ | $-31.68 \pm 2.79$ (0/5) | $-17.66 \pm 4.22$ (0/5) | $-29.09 \pm 2.90$ (0/5) | loss |
| `znorm_k10` | $91.41 \pm 0.41$ | $54.47 \pm 1.70$ | $85.22 \pm 4.73$ | $3.05 \pm 0.95$ | $15.26 \pm 0.88$ | $-31.99 \pm 2.11$ (0/5) | $-19.35 \pm 3.92$ (0/5) | $-35.03 \pm 3.67$ (0/5) | loss |
| `znorm_k50` | $91.41 \pm 0.41$ | $56.49 \pm 0.88$ | $87.27 \pm 3.26$ | $3.94 \pm 1.28$ | $17.90 \pm 1.43$ | $-34.01 \pm 1.61$ (0/5) | $-18.45 \pm 3.81$ (0/5) | $-32.39 \pm 2.87$ (0/5) | loss |

---

## 5. Technical Discussion & Failure Modes

### 5.1 Validation vs. Test Disconnect on Gap Rule
- **Why Gap Rule Worked on Test Set**: With $751$ gallery identities and thousands of distractors, impostor probes frequently exhibit secondary similarities $s_2$ that are nearly equal to $s_1$. Subtracting $0.50 \cdot s_2$ suppresses ambiguous impostors, reducing false alarm rate at fixed recall ($\text{FAR@TAR=80\%}$ drops by $3.17\%$ on Standard and $4.30\%$ on Look-Alike p05).
- **Why It Was Not Selected on Validation**: The held-out validation set contains only 50 identities per fold (25 enrolled in selection splits). Genuine probes easily dominate the small gallery ($s_1 \gg s_2$) while impostors encounter fewer competing enrolled IDs. Baseline FAR on the small validation set was already near zero ($0.30 \pm 0.46\%$), providing no headroom for the gap rule to demonstrate a statistically significant gain ($>1\,\text{std}$). Following strict scientific hygiene without test-peeking meant `raw` was preserved.
- **Why Gap Rule Failed on Look-Alike p05**: Under dense look-alikes, genuine probes also frequently suffer from competing look-alike gallery items of other identities, reducing their genuine $s_1 - \lambda \cdot s_2$ margin. At the tight operating point $\text{DIR@FAR=1\%}$, genuine targets are penalized as severely as impostors, yielding a paired difference of $-1.74 \pm 4.03\%$ (only 2/5 seeds positive $\implies$ **tie**).

### 5.2 Failure of Z-Norm and S-Norm
In speaker verification and face recognition, z-norm typically relies on hundreds or thousands of cohort speakers across diverse acoustic channels. Here, the cohort consists of only 50 individuals ($980$ image crops across 6 cameras).
1. **Camera Bias**: A probe photographed in Camera 1 has systematically higher cosine similarities to cohort images from Camera 1 than Camera 2. The top-$k$ cohort neighbors are almost entirely drawn from the probe's own camera view, making $\mu_q$ and $\sigma_q$ an indicator of camera lighting rather than intrinsic gallery matching difficulty.
2. **Variance Collapse**: With small $k$ ($10$ or $50$), $\sigma_q$ fluctuates erratically. When an impostor probe happens to have a very low $\sigma_q$, $(s_1 - \mu)/\sigma$ explodes into a false accept.

### 5.3 Inductive CSLS Dynamics
CSLS with this inductive definition ($2\cos(q, g) - r_k(q) - r_k(g)$) is conceptually close to z-norm without standard deviation scaling. Because $r_k(x)$ is computed exclusively against the 50-ID cohort, it penalizes features that lie near dense clusters of the cohort. When a genuine target identity has clothing resembling one of the cohort clusters, $r_k(g)$ is inflated, directly lowering the genuine match score and degrading closed-set rank-1 and max TAR.

---

## 6. Limitations and Production Generalization

Market-1501's 100 held-out validation identities (partitioned into a 50-identity cohort) represent a constrained regime for cohort-based normalization methods. In realistic surveillance deployments or on larger benchmarks (e.g., MSMT17 with 1,041 training IDs and 3,060 test IDs):
1. **Cohort Scale**: A cohort of $500+$ diverse identities across 15+ camera angles would provide much smoother, camera-balanced estimates for $\mu_q$ and $\sigma_q$, preventing the variance collapse observed here.
2. **Camera-Balanced Cohort Sampling**: To make z-norm viable for person ReID, the cohort should enforce equal representation across all deployed camera views, standardizing cross-camera transfer rather than penalizing specific views.
3. **Operational Recommendation**: For resource-constrained single-facility deployments (such as Market-1501 scale), raw top-1 cosine matching remains the most robust default, while the ambiguity gap rule ($s_1 - 0.50 \cdot s_2$) offers a lightweight post-processing enhancement if calibrated on a gallery of matching size.
