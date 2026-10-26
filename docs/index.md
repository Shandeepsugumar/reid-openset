# Documentation: Open-Set Person Re-Identification

## Overview
Open-set person re-identification differs from classical closed-set ReID:
1. **Gallery**: Known registered identities.
2. **Probe/Query**: Can be either a registered person (target) or an unseen impostor (non-target).
3. **Objective**: Accurately match known identities while detecting and rejecting unknown identities with high confidence.

## Key Modules
- **`models/`**: Feature extractor backbones (e.g. ResNet, OSNet, ViT).
- **`losses/`**: Objectives optimizing intra-class compactness and inter-class margin.
- **`calibration/`**: Estimating open-set thresholds via Extreme Value Theory (EVT) / Weibull fitting or probability calibration.
- **`eval/`**: DIR (Detection and Identification Rate) vs. FAR (False Alarm Rate) curves and open-set CMC metrics.
