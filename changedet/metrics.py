"""Registration and change-detection metrics."""
from __future__ import annotations

import math

import numpy as np

SUCCESS_THRESHOLDS = (1, 3, 5, 10)


# --------------------------------------------------------------------- registration
def registration_summary(errors, times=None, thresholds=SUCCESS_THRESHOLDS) -> dict:
    """Success rate at each corner-error threshold, plus the median.

    Report success rates and the median rather than the mean: one failed
    homography (error ~inf) would swamp the average.
    """
    e = np.asarray([x for x in errors if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    out = {"n": int(e.size)}
    if e.size == 0:
        return out
    for t in thresholds:
        out[f"success@{t}px"] = float(np.mean(e <= t))
    out["median_corner_err"] = float(np.median(e))
    out["failures"] = int(np.sum(~np.isfinite(e)))
    if times is not None and len(times):
        out["mean_time_s"] = float(np.mean(times))
    return out


# ---------------------------------------------------------------- change detection
def confusion(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) counted only inside ``region``."""
    p, g = pred[region], gt[region]
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    tn = int(np.sum(~p & ~g))
    return tp, fp, fn, tn


def prf_iou(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall, F1 and IoU. A ratio with a zero denominator is NaN."""
    def div(a, b):
        return a / b if b > 0 else float("nan")
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    return {"precision": p, "recall": r, "f1": f1, "iou": div(tp, tp + fp + fn)}


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu threshold of the finite values (a fixed rule, not tuned on the test set)."""
    v = values[np.isfinite(values)]
    if v.size == 0 or v.min() == v.max():
        return float(v.max()) if v.size else float("inf")
    from skimage.filters import threshold_otsu
    return float(threshold_otsu(v))


def check_score(score: np.ndarray, shape) -> np.ndarray:
    score = np.asarray(score)
    if score.shape != tuple(shape[:2]):
        raise ValueError(f"change map has shape {score.shape}, expected {tuple(shape[:2])}")
    return score.astype(np.float32, copy=False)


class PRAccumulator:
    """Collects (score, label) pairs over a dataset for precision-recall curves.

    Each pixel of the evaluation region is kept with probability ``sample_rate``.
    Sampling at a fixed rate keeps every image weighted by its area, as an exact
    pixel-level curve would. Scores of -inf mean "never detected" (e.g. the
    registration failed). NaN or +inf inside the region raises, so a broken
    change map is caught early.
    """

    def __init__(self, sample_rate: float = 0.05, seed: int = 0):
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)
        self._scores: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []

    def add(self, score: np.ndarray, gt: np.ndarray, region: np.ndarray) -> None:
        s, g = score[region], gt[region]
        if s.size == 0:
            return
        bad = np.isnan(s) | np.isposinf(s)
        if bad.any():
            raise ValueError(f"change map has {int(bad.sum())} NaN/+inf values inside the evaluation region")
        keep = self.rng.random(s.size) < self.sample_rate
        self._scores.append(s[keep].astype(np.float32))
        self._labels.append(g[keep].astype(bool))

    @property
    def n_positive(self) -> int:
        return int(sum(l.sum() for l in self._labels))

    def curve(self):
        """(precision, recall, thresholds), predicting change where score >= threshold.
        None if there are no positive samples."""
        if not self._scores:
            return None
        s = np.concatenate(self._scores)
        y = np.concatenate(self._labels)
        P = int(y.sum())
        if P == 0:
            return None
        order = np.argsort(-s, kind="stable")
        s, y = s[order], y[order]
        tp = np.cumsum(y)
        fp = np.cumsum(~y)
        with np.errstate(invalid="ignore"):
            last_of_group = np.r_[np.nonzero(np.diff(s) != 0)[0], s.size - 1]
        tp, fp, thr = tp[last_of_group], fp[last_of_group], s[last_of_group]
        finite = np.isfinite(thr)
        tp, fp, thr = tp[finite], fp[finite], thr[finite]
        return tp / np.maximum(tp + fp, 1), tp / P, thr

    def average_precision(self) -> float:
        c = self.curve()
        if c is None:
            return float("nan")
        p, r, _ = c
        return float(np.sum((r - np.r_[0.0, r[:-1]]) * p))

    def best_f1(self) -> tuple[float, float]:
        """Best F1 over thresholds, and its threshold. Optimistic: the threshold is
        chosen on the evaluation data itself. Report it next to a fixed-rule F1."""
        c = self.curve()
        if c is None:
            return float("nan"), float("nan")
        p, r, thr = c
        f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
        i = int(np.argmax(f1))
        return float(f1[i]), float(thr[i])

    def curve_points(self, max_points: int = 200):
        """The PR curve downsampled to at most ``max_points`` points (for saving and plotting)."""
        c = self.curve()
        if c is None:
            return None
        p, r, thr = c
        idx = np.unique(np.linspace(0, len(p) - 1, min(max_points, len(p))).round().astype(int))
        return {"precision": p[idx].tolist(), "recall": r[idx].tolist(), "threshold": thr[idx].tolist()}
