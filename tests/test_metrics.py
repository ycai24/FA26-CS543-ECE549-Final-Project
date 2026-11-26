import numpy as np
import pytest

from changedet.metrics import PRAccumulator, confusion, prf_iou, registration_summary


def test_confusion_counts_only_inside_region():
    pred = np.array([[1, 1, 0, 0]], bool)
    gt = np.array([[1, 0, 1, 0]], bool)
    region = np.array([[1, 1, 1, 0]], bool)
    assert confusion(pred, gt, region) == (1, 1, 1, 0)
    m = prf_iou(1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["iou"] == pytest.approx(1 / 3)


def test_prf_undefined_is_nan():
    assert np.isnan(prf_iou(0, 0, 0)["f1"])


def test_pr_perfect_separation():
    gt = np.zeros((50, 50), bool)
    gt[10:20, 10:20] = True
    score = gt.astype(np.float32) + 0.01
    acc = PRAccumulator(sample_rate=1.0)
    acc.add(score, gt, np.ones_like(gt))
    assert acc.average_precision() == pytest.approx(1.0)
    f1, thr = acc.best_f1()
    assert f1 == pytest.approx(1.0) and thr == pytest.approx(1.01)


def test_pr_random_scores_give_prevalence():
    rng = np.random.default_rng(0)
    gt = rng.random((400, 400)) < 0.1
    acc = PRAccumulator(sample_rate=1.0)
    acc.add(rng.random((400, 400)).astype(np.float32), gt, np.ones_like(gt))
    assert acc.average_precision() == pytest.approx(0.1, abs=0.02)


def test_never_detected_pixels_lower_recall():
    gt = np.ones((10, 10), bool)
    score = np.full((10, 10), -np.inf, np.float32)
    score[:5] = 1.0                                     # half the positives are detectable
    acc = PRAccumulator(sample_rate=1.0)
    acc.add(score, gt, np.ones_like(gt))
    p, r, _ = acc.curve()
    assert r.max() == pytest.approx(0.5)


def test_nan_scores_are_rejected():
    acc = PRAccumulator()
    with pytest.raises(ValueError):
        acc.add(np.full((4, 4), np.nan, np.float32), np.ones((4, 4), bool), np.ones((4, 4), bool))


def test_registration_summary_counts_failures():
    s = registration_summary([0.5, 2.0, 4.0, float("inf")])
    assert s["success@1px"] == 0.25 and s["success@5px"] == 0.75 and s["failures"] == 1
