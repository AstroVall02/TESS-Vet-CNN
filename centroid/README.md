# Centroid sub-study

This sub-study was performed in order to verify whether the transit signal for a given TIC ID was indeed due to the target star or was the result of **background blends** (the signal originates from a background star eclipsing binary, transit or other astrophysical phenomenon) by checking whether the in-transit light centroid
moves, following the precedent of Ansdell et al. (2018).

## Setup

This part has extra dependencies and is not needed for the main CNN architecture:

```bash
uv pip install -r ../requirements-centroid.txt
```

`centroid_checks.py` imports the third-party [transit-diffImage](https://github.com/stevepur/transit-diffImage)
package (difference images and PRF centroid fits). Install it from its repository and make sure it is importable.
Downloading TESS FFI cutouts needs access to MAST.

## Contents

| File | Purpose |
|---|---|
| `centroid_checks.py` | Helpers: aberration-corrected target reference, difference image, split-half and control-epoch fits, pre-registered classification rules |
| `results/centroid_validation_results.csv` | 12 stars: 3 clean, 9 known background false positives |
| `check_spoc_coverage.py` | Counts SPOC 2-minute sectors per TIC |
| `extract_centroid_features.py` | Per-star centroid-shift features from SPOC light-curve centroid columns |
| `results/centroid_pilot.csv` | Partial pilot output (~48 stars) |

## Findings

* Clean-star baseline offset: 0.1 to 0.3 px.
* An early 1 px baseline was an artifact of using an un-aberrated reference position. The target reference must include
  stellar aberration (~1 TESS pixel).
* One known background blend was detected (1.75 px, stable across halves, controls did not reproduce it); two more
  landed on the named host but control epochs reproduced the fit, so they were not counted; five gave no usable measurement.
* Not tested: clean targets with bright neighbours.
* The scalable SPOC-centroid feature was piloted but dropped since the speed was a crawling ~5 stars/hour from MAST and the 48-star pilot
  was too small to be conclusive.
