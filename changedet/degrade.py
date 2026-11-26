"""Shape helpers and photometric degradations for the synthetic benchmark.

Images are float32 RGB in [0, 1]. Randomness that fixes a *direction* (shading
direction, polygon shape, noise field) is drawn once per pair and passed in, so
the same pair can be rendered at several severity levels with only the
magnitude changing.
"""
from __future__ import annotations

import cv2
import numpy as np


# ----------------------------------------------------------------------------- shapes
def random_polygon(rng: np.random.Generator, center, radius: float,
                   n_min: int = 5, n_max: int = 10) -> np.ndarray:
    """Random star-convex polygon (N x 2) around ``center`` with vertices at 55-100% of ``radius``."""
    n = int(rng.integers(n_min, n_max + 1))
    ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
    r = radius * rng.uniform(0.55, 1.0, n)
    return np.stack([center[0] + r * np.cos(ang), center[1] + r * np.sin(ang)], axis=1)


def polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_mask(shape, poly: np.ndarray) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def feathered_mask(shape, poly: np.ndarray, sigma: float) -> np.ndarray:
    """Polygon alpha matte in [0, 1] with Gaussian-softened edges.

    Hard pasting leaves visible seams; blending with weights that fall off at the
    boundary hides them (course notes Ch4 §4.2.2).
    """
    a = polygon_mask(shape, poly).astype(np.float32)
    if sigma > 0:
        a = cv2.GaussianBlur(a, (0, 0), sigma)
    return a


def composite(base: np.ndarray, patch: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return a * patch + (1.0 - a) * base


# ------------------------------------------------------------------------ degradations
def illumination(img: np.ndarray, level: float, gain_sign: float, gamma_sign: float,
                 shade_angle: float) -> np.ndarray:
    """Global gain, gamma, and a smooth linear shading ramp, all scaled by ``level`` in [0, 1].

    At level 1: gain 1 +/- 0.4, gamma 2^(+/-0.6) (about 0.66 or 1.5), and shading
    varying +/-50% across the image. The ramp models slowly varying illumination
    (course notes §27.4.1).
    """
    if level <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (np.cos(shade_angle) * (xx - w / 2) + np.sin(shade_angle) * (yy - h / 2)) / (0.5 * np.hypot(w, h))
    shading = 1.0 + 0.5 * level * ramp
    gain = 1.0 + 0.4 * level * gain_sign
    gamma = 2.0 ** (0.6 * level * gamma_sign)
    out = np.clip(img, 0.0, 1.0) ** gamma * gain * shading[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def cast_shadow(img: np.ndarray, level: float, poly: np.ndarray, softness_px: float = 8.0) -> np.ndarray:
    """Darken a soft-edged polygon region by the fraction ``level`` (0 = no shadow).

    Outdoor shadows are blurry and seldom fully dark (course notes §27.3.1). A
    shadow is an illumination change, not a scene change, so it is never part of
    the change mask.
    """
    if level <= 0:
        return img
    m = feathered_mask(img.shape, poly, softness_px)
    return (img * (1.0 - level * m)[..., None]).astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def add_noise(img: np.ndarray, sigma_255: float, field: np.ndarray) -> np.ndarray:
    """Additive Gaussian noise; ``sigma_255`` is in 8-bit units, ``field`` is a pre-drawn N(0, 1) array."""
    if sigma_255 <= 0:
        return img
    return np.clip(img + (sigma_255 / 255.0) * field, 0.0, 1.0).astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
