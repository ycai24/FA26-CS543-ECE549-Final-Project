"""Minimal reference registrations, used to sanity-check the benchmark and the
evaluation harness. The project's real registration code goes in its own modules."""
from __future__ import annotations

import cv2
import numpy as np

from ..geometry import normalize_h


def identity(ref: np.ndarray, tgt: np.ndarray):
    return np.eye(3), {"method": "identity"}


def sift_ransac(ref: np.ndarray, tgt: np.ndarray, ratio: float = 0.75, ransac_px: float = 3.0):
    """SIFT, Lowe ratio test, then a RANSAC homography (tgt -> ref)."""
    g_ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    g_tgt = cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    k_ref, d_ref = sift.detectAndCompute(g_ref, None)
    k_tgt, d_tgt = sift.detectAndCompute(g_tgt, None)
    info = {"method": "sift_ransac", "kp_ref": len(k_ref), "kp_tgt": len(k_tgt), "matches": 0, "inliers": 0}
    if d_ref is None or d_tgt is None or len(k_ref) < 4 or len(k_tgt) < 4:
        return None, info
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d_tgt, d_ref, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    info["matches"] = len(good)
    if len(good) < 4:
        return None, info
    src = np.float32([k_tgt[m.queryIdx].pt for m in good])
    dst = np.float32([k_ref[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    if H is None:
        return None, info
    info["inliers"] = int(inliers.sum())
    return normalize_h(H), info
