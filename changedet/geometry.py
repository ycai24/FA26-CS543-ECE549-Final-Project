"""Homography helpers shared by the benchmark generator, dataset loaders and evaluator.

Conventions used everywhere in this repo
----------------------------------------
* Pixel coordinates follow OpenCV: x = column, y = row, 0-based, (0, 0) is the
  centre of the top-left pixel. (The course notes, Ch4 §4.2.1, use a 1-based
  matrix-style convention; don't mix the two.)
* Every homography ``H`` maps TARGET pixel coordinates to REFERENCE pixel
  coordinates:  [x_ref, y_ref, 1]^T ~ H @ [x_tgt, y_tgt, 1]^T.
  ``cv2.findHomography(tgt_pts, ref_pts, ...)`` returns an H in this direction.
* ``cv2.warpPerspective(tgt, H, (W_ref, H_ref))`` resamples the target onto the
  reference grid (no ``WARP_INVERSE_MAP`` flag). ``warp_to_ref`` below does this.
* OpenCV's ``findTransformECC(template, input, warp)`` uses the OPPOSITE direction
  (template -> input, applied with ``WARP_INVERSE_MAP``). With template = ref and
  input = tgt, its warp matrix is ``inv(H)``.
"""
from __future__ import annotations

import cv2
import numpy as np


def normalize_h(H: np.ndarray) -> np.ndarray:
    """Return H as float64 scaled so that H[2, 2] == 1 (unchanged if that is impossible)."""
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got shape {H.shape}")
    if np.all(np.isfinite(H)) and abs(H[2, 2]) > 1e-12:
        return H / H[2, 2]
    return H


def image_corners(w: int, h: int) -> np.ndarray:
    """The four corner pixel centres of a w x h image, clockwise from top-left (4x2)."""
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography H to an (N, 2) array of points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, dtype=np.float64).T
    return ph[:, :2] / ph[:, 2:3]


def corner_error(H_est, H_gt: np.ndarray, w: int, h: int) -> float:
    """Mean distance in pixels (reference frame) between the target image's four
    corners mapped by ``H_est`` and by ``H_gt``.

    ``w, h`` are the TARGET image's width and height. A missing (None) or
    non-finite ``H_est`` counts as a failure and returns ``inf``.
    """
    if H_est is None:
        return float("inf")
    H_est = np.asarray(H_est, dtype=np.float64)
    if H_est.shape != (3, 3) or not np.all(np.isfinite(H_est)):
        return float("inf")
    c = image_corners(w, h)
    with np.errstate(all="ignore"):
        d = np.linalg.norm(apply_h(H_est, c) - apply_h(H_gt, c), axis=1)
    return float(d.mean()) if np.all(np.isfinite(d)) else float("inf")


def warp_to_ref(tgt: np.ndarray, H: np.ndarray, ref_shape, interp: int = cv2.INTER_LINEAR):
    """Resample ``tgt`` onto the reference grid with H (tgt -> ref).

    Returns ``(aligned, footprint)``. ``footprint`` is a bool mask of reference
    pixels that received real target data. It is eroded by one pixel so the
    interpolated (half-black) border pixels are excluded. Pixels outside it are
    zero in ``aligned`` and must not be scored (Ch4 §4.2.4).
    """
    h, w = ref_shape[:2]
    H = np.asarray(H, dtype=np.float64)
    aligned = cv2.warpPerspective(tgt, H, (w, h), flags=interp,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    footprint = warp_mask_to_ref(np.ones(tgt.shape[:2], dtype=bool), H, ref_shape)
    footprint = cv2.erode(footprint.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return aligned, footprint


def warp_mask_to_ref(mask: np.ndarray, H: np.ndarray, ref_shape) -> np.ndarray:
    """Nearest-neighbour warp of a bool mask from the target frame to the reference frame."""
    h, w = ref_shape[:2]
    out = cv2.warpPerspective(mask.astype(np.uint8) * 255, np.asarray(H, dtype=np.float64), (w, h),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out > 127


def perturb_homography(H: np.ndarray, w: int, h: int, err_px: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Return H' whose corner error relative to H is exactly ``err_px``.

    Each of the target's four corners, mapped by H, is pushed ``err_px`` pixels in
    an independent random direction, and H' is re-solved from the moved corners.
    Use this to inject controlled misregistration.
    """
    src = image_corners(w, h)
    dst = apply_h(H, src)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=4)
    dst = dst + err_px * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    Hp = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    return normalize_h(Hp)
