import cv2
import numpy as np
import pytest

from changedet.geometry import (apply_h, corner_error, image_corners, normalize_h,
                                perturb_homography, warp_to_ref)


def random_h(rng, w=320, h=240, rho=20):
    src = image_corners(w, h)
    dst = src + rng.uniform(-rho, rho, size=(4, 2))
    return normalize_h(cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)))


def test_apply_h_identity():
    pts = np.array([[0, 0], [10.5, 3.25]])
    assert np.allclose(apply_h(np.eye(3), pts), pts)


def test_corner_error_zero_and_failures():
    rng = np.random.default_rng(0)
    H = random_h(rng)
    assert corner_error(H, H, 320, 240) == pytest.approx(0, abs=1e-9)
    assert corner_error(None, H, 320, 240) == float("inf")
    assert corner_error(np.full((3, 3), np.nan), H, 320, 240) == float("inf")


@pytest.mark.parametrize("err", [0.5, 2.0, 8.0])
def test_perturbation_gives_exact_corner_error(err):
    rng = np.random.default_rng(1)
    H = random_h(rng)
    Hp = perturb_homography(H, 320, 240, err, rng)
    assert corner_error(Hp, H, 320, 240) == pytest.approx(err, abs=1e-2)


def test_warp_to_ref_undoes_the_target_render():
    rng = np.random.default_rng(2)
    ref = cv2.GaussianBlur(rng.uniform(0, 255, (240, 320, 3)).astype(np.float32), (0, 0), 4)
    H = random_h(rng)                                  # tgt -> ref
    tgt = cv2.warpPerspective(ref, H, (320, 240), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    aligned, foot = warp_to_ref(tgt, H, ref.shape)
    inner = cv2.erode(foot.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert inner.mean() > 0.5
    assert np.abs(aligned - ref)[inner].mean() < 1.0   # double interpolation only
    wrong, _ = warp_to_ref(tgt, np.linalg.inv(H), ref.shape)
    assert np.abs(wrong - ref)[inner].mean() > 5.0     # the other direction is clearly wrong
