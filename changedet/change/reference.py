"""Smoke-test change map for exercising the harness. The project's real
change-detection methods go in their own modules."""
from __future__ import annotations

import cv2
import numpy as np


def absdiff(ref: np.ndarray, aligned: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Absolute grayscale difference in [0, 1]."""
    a = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    b = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.abs(a - b)
