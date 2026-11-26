# ECE 549 final project: classical change detection

Shared code for the team: the synthetic benchmark, dataset loaders, evaluation
harness and plots. Registration and change-detection methods plug into it.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest                      # ~10 s, should be all green
```

Keep `data/` and `results/` out of git and out of OneDrive/iCloud folders. They are large
and can be regenerated.

## Conventions (read before writing code)

* **Images** are RGB `uint8`, shape `H x W x 3`. OpenCV loads BGR, so convert.
* **Pixel coordinates** follow OpenCV: `x` = column, `y` = row, 0-based.
* **Every homography maps TARGET pixels to REFERENCE pixels.** This is what
  `cv2.findHomography(tgt_pts, ref_pts, ...)` returns. Align with
  `changedet.geometry.warp_to_ref(tgt, H, ref.shape)`.
  * OpenCV's `findTransformECC(template=ref, input=tgt)` uses the opposite direction:
    its warp matrix is `inv(H)`.
* Pixels outside the warped target's footprint are zero and must not be scored.

## Plugging in your method

| Who | Where | Function signature |
|---|---|---|
| Registration (Yuting, Shaun) | `changedet/registration/<name>.py` | `register(ref, tgt) -> (H or None, info: dict)` |
| Change detection (Yvonne) | `changedet/change/<name>.py` | `change_map(ref, aligned, valid) -> float H x W` |

The full contracts are in the docstrings of `changedet/registration/__init__.py` and
`changedet/change/__init__.py`. The reference implementations
(`registration/reference.py`: identity, SIFT+RANSAC; `change/reference.py`: absolute
difference) exist only to test the harness.

## Synthetic benchmark

```bash
python -m changedet.synth --out data/synth/v0 --n 10            # ~10 s, ~210 MB
python -m changedet.synth --out data/synth/tune --n 20 --seed 1 --src-dir <other photos>
```

Each pair gets an object **added**, **removed**, or **no change** (10% of pairs). It is
then rendered in every *set*. Sweeps vary one factor from the base setting
(viewpoint 16 px, noise σ 2, everything else off) and keep the scene and every random
direction fixed, so only the magnitude changes along a sweep.

| Factor | Levels | Meaning |
|---|---|---|
| `viewpoint` | 0, 8, 16, 32, 64, 96 | max displacement of each image corner (px) |
| `noise` | 0, 5, 10, 20, 30 | Gaussian σ in 8-bit units |
| `blur` | 0, 1, 2, 3, 5 | Gaussian σ (px), target only |
| `illum` | 0 – 1 | gain ±40%, gamma 2^±0.6, shading ramp ±50% at 1.0 |
| `shadow` | 0, 0.2, 0.4, 0.6 | fraction of light removed in a soft cast shadow (not a change) |
| `occlusion` | 0, 0.05, 0.1, 0.2 | occluder area as a fraction of the image |
| `mixed` | random | all factors drawn at random |

Files per pair, in `data/synth/v0/<set>/<pair id>/`:

| File | Contents |
|---|---|
| `ref.png`, `tgt.png` | the image pair |
| `H_gt.npy` | ground-truth homography (target → reference) |
| `mask_gt.png` | changed pixels, in the reference frame |
| `valid.png` | reference pixels visible in the target |
| `meta.json` | factor levels, change type, sources, per-object contrast |

**Team decisions baked in (change them if you disagree):**

* **Occluders:** pixels hidden by an occluder are excluded from scoring by default. Use
  `--occluder change` to label them as change instead.
* **Shadows and lighting:** these are never labeled as change. That is the point:
  they produce false positives.
* **Source images:** the default photos are scikit-image samples, which are fine for
  testing. For real results, use a larger freely licensed photo set via `--src-dir`.
  Don't use HPatches images, since thresholds tuned on synthetic data would leak into
  the HPatches test. Tune on a benchmark generated with a different `--seed`/`--src-dir`.

## Evaluation

```bash
# end to end with your methods
python -m changedet.evaluate --data data/synth/v0 \
    --register changedet.registration.<module>:<function> \
    --change changedet.change.<module>:<function> --out results/<run name>

# oracle alignment: change detection with the true homography
python -m changedet.evaluate --data data/synth/v0 --register oracle --change absdiff --out results/oracle_absdiff

# controlled misalignment: true homography pushed off by exactly 0/1/2/4/8 px
python -m changedet.evaluate --data data/synth/v0 --sets viewpoint_16 --perturb 0,1,2,4,8 \
    --change absdiff --out results/misalign_absdiff

# registration only on HPatches (expects data/hpatches-sequences-release; --sets i or v)
python -m changedet.evaluate --dataset hpatches --data data/hpatches-sequences-release --register sift --out results/hpatches_sift

# compare runs in one set of figures
python -m changedet.plots results/oracle_absdiff results/sift_absdiff --labels "Oracle,SIFT" --out results/compare
```

Each run writes these files:

| File | Contents |
|---|---|
| `per_pair.csv` | one row per pair: corner error, timings, your `info` fields, per-pair P/R/F1/IoU |
| `summary.json` | per set: registration results, change-detection results |
| `pr_curves.json` | precision-recall curves |
| `plots/` | figures |

`summary.json` reports registration as success rate at 1/3/5/10 px, the median corner
error and the number of failures. It reports change detection as AP, best F1,
P/R/F1/IoU at the threshold rule, the mean runtime and the overlap coverage.

**Scoring rules:**

* Change detection is scored on the ground-truth valid region.
* Pixels the estimated warp doesn't cover count as "not detected", and a failed
  registration (`None`) detects nothing.
* `--threshold otsu` (the default) applies a fixed rule to each image. A number gives a
  threshold tuned on separate data.
* `best_f1` picks its threshold on the test data, so it is optimistic. Report it only
  next to the fixed-rule F1.

## Datasets

* **HPatches** (registration): get `hpatches-sequences-release` from the HPatches project
  (github.com/hpatches/hpatches-dataset) and unpack it into `data/`.
* **VL-CMU-CD** (real change masks): the loader isn't written yet.

## Status

* Done:
  * benchmark generator v0
  * HPatches loader
  * metrics
  * evaluation harness with oracle and controlled-misalignment modes
  * plots
  * tests
* To do:
  * VL-CMU-CD loader
  * a larger, freely licensed source-photo set
  * object-level metrics
  * the full pipeline script once the methods land
