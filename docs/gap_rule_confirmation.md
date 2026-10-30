# Pre-Registration: Confirmation of the Ambiguity Gap Decision Rule

**Date**: 2026-10-30  
**Status**: Pre-registered protocol (Commit order establishes pre-registration before test execution).  
**Context**: In exploratory evaluation on `e0_baseline_val`, the ambiguity gap rule $s_{\text{gap}}(q; \lambda) = s_1(q) - \lambda \cdot s_2(q)$ demonstrated improved open-set verification performance on the standard test protocol. However, because this rule was identified post-hoc after inspecting test-set metrics (the 100-identity validation set was too easy to discriminate rules), it must be treated strictly as an **exploratory candidate**. This document pre-registers the confirmation protocol and success criteria with zero post-hoc modifications.

---

## 1. Pre-Registered Confirmation Criteria

### Criterion 1 (C1): Replication Across Independent Models
- **Protocol**: Standard open-set protocol (50% enrolled, 50% unknown query identities) across 5 fixed random split seeds ($0, 1, 2, 3, 4$).
- **Parameters**: Primary $\lambda = 0.50$; sensitivity analysis at $\lambda \in \{0.25, 0.75\}$.
- **Metric Requirement**: For a model to replicate, the gap rule must beat the `raw` baseline on **all three metrics**:
  1. $\Delta \text{FAR@TAR=80\%} > 1.0 \times \text{std}(\Delta \text{FAR@80})$ and positive in at least 4 of 5 seeds ($\Delta > 0$).
  2. $\Delta \text{DIR@FAR=1\%} > 1.0 \times \text{std}(\Delta \text{DIR@1})$ and positive in at least 4 of 5 seeds ($\Delta > 0$).
  3. $\Delta \text{DIR@FAR=5\%} > 1.0 \times \text{std}(\Delta \text{DIR@5})$ and positive in at least 4 of 5 seeds ($\Delta > 0$).
  *(Sign convention: positive delta $\Delta$ is always better for both DIR and FAR).*
- **Confirmation Threshold**: C1 passes if and only if the primary rule ($\lambda = 0.50$) achieves a **win** on at least **6 of the 8 available models** with saved test embeddings:
  - `baseline_resnet18`
  - `e0_baseline_val`
  - `e1_cosface`
  - `e1c_cosface_small`
  - `e2_cosface_hardid`
  - `e3_cosface_hardid_absneg`
  - `e4_ce_hardid`
  - `e5_ce_hardid_absneg`
  *(Models without artifacts, e.g. e6/e7, are listed as "not run" and not counted).*

### Criterion 2 (C2): Cross-Fitting on Test Identities
- **Protocol**: 2-fold cross-fitting on the test query identities ($751$ identities):
  - Partition the $751$ test query identities deterministically into two disjoint halves (Half 1 and Half 2) using fixed seed $42$.
  - **Fold 1**: Select $\lambda^* \in \{0.25, 0.50, 0.75, 1.0\}$ on Half 1 (criterion: highest mean paired improvement of $\text{FAR@TAR=80\%}$ and $\text{DIR@FAR=5\%}$ across 5 random splits of Half 1). Evaluate the chosen $\lambda^*$ on the held-out Half 2 identities across 5 random splits.
  - **Fold 2**: Select $\lambda^*$ on Half 2; evaluate on held-out Half 1.
- **Confirmation Threshold**: C2 passes if and only if the held-out paired improvement is strictly positive on **both folds** ($\Delta \text{FAR@80} > 0$ and $\Delta \text{DIR@5} > 0$ on both held-out sets). Both chosen $\lambda^*$ values must be documented.

### Criterion 3 (C3): No Harm on Look-Alike Adversarial Sets
- **Protocol**: Look-Alike p05 (distance threshold $\le 0.3517$) and Look-Alike p01 (distance threshold $\le 0.2929$) protocols evaluated on `e0_baseline_val` and `baseline_resnet18` across 5 fixed random seeds.
- **Parameters**: Primary $\lambda = 0.50$; Secondary $\lambda = 0.25$.
- **Confirmation Threshold**: C3 passes if on both look-alike protocols (p05 and p01) and for both models:
  - $\Delta \text{FAR@TAR=80\%} \ge -1.0 \times \text{std}(\Delta \text{FAR@80})$ (not worse by more than one std).
  - $\Delta \text{DIR@FAR=5\%} \ge -1.0 \times \text{std}(\Delta \text{DIR@5})$ (not worse by more than one std).
  - $\text{DIR@FAR=1\%}$ is reported but explicitly marked as noisy due to high variance at tight operating thresholds.

---

## 2. Evaluation Conventions

1. **Sign Convention**:
   - $\Delta \text{FAR@80} = \text{FAR@80}_{\text{raw}} - \text{FAR@80}_{\text{rule}}$ (positive = lower false alarm rate, better).
   - $\Delta \text{DIR@1} = \text{DIR@1}_{\text{rule}} - \text{DIR@1}_{\text{raw}}$ (positive = higher identification rate, better).
   - $\Delta \text{DIR@5} = \text{DIR@5}_{\text{rule}} - \text{DIR@5}_{\text{raw}}$ (positive = higher identification rate, better).
2. **Seeds**:
   - "5/5 seeds" refers to 5 deterministic random enrolled/unknown splits of ONE trained model checkpoint, not 5 separate training runs.
3. **Verdict Categories**:
   - **win**: All three metrics better by $> 0.95\,\text{std}$ with positive sign in $\ge 4/5$ seeds.
   - **loss**: Any metric worse by $> 0.95\,\text{std}$ ($< -0.95\,\text{std}$).
   - **tie**: Differences fall within one standard deviation or have mixed signs.

---

## 3. Mechanism Check Specification (Hypothesis Testing)

- **Hypothesis Under Test**: *"The open-set verification gain of the ambiguity gap rule depends systematically on the number of enrolled identities in the gallery."*
- **Experimental Sweep**: On the standard protocol for `e0_baseline_val` and `baseline_resnet18`, keep all impostor probes fixed while subsampling the enrolled gallery identities to $N_{\text{enrolled}} \in \{25, 50, 100, 200, \text{full}\}$ across the same 5 seeds.
- **Reporting**: Report $\text{FAR@TAR=80\%}$, $\text{DIR@FAR=1\%}$, and $\text{DIR@FAR=5\%}$ for `raw` and `gap` ($\lambda \in \{0.25, 0.50, 0.75\}$). Plot improvement $\Delta$ vs. $N_{\text{enrolled}}$.
- **Transparency**: Honestly report if the empirical results do not support the hypothesis.

---

## 4. Deployment Behavior & Safeguards

1. **Fallback Logic**: If the gallery contains fewer than 2 enrolled identities (i.e. $s_2$ is undefined), the gap rule must gracefully fall back to returning the raw top-1 score $s_1$.
2. **Minimum Gallery Parameter**: Introduce `min_gallery_identities` (integer $\ge 1$). If the enrolled gallery size is strictly less than `min_gallery_identities`, return $s_1$. Default value is selected based on the smallest gallery size in the sweep where the gap rule is not worse than raw by more than 1 std (or 2 if no dependence is observed).
3. **Scope**: Changes will be restricted to `src/reid/decision/rules.py` with accompanying unit tests; production service code will remain untouched.

---

## 5. Final Confirmation Verdict Rule

- **Confirmed**: C1, C2, and C3 all pass.
- **Partially Confirmed**: At least 2 of {C1, C2, C3} pass.
- **Not Confirmed**: Fewer than 2 pass, or C1 fails completely.
