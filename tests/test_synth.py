"""Checks that the benchmark's ground truth is self-consistent."""
import json

import cv2
import numpy as np
import pytest

from changedet import synth
from changedet.datasets import SynthDataset
from changedet.evaluate import evaluate
from changedet.geometry import corner_error, warp_to_ref
from changedet.registration.reference import sift_ransac


@pytest.fixture(scope="module")
def sources():
    cfg = synth.GenConfig(ref_noise=0.0)
    names, imgs = synth.load_sources()
    cw, ch = cfg.width + 2 * cfg.margin, cfg.height + 2 * cfg.margin
    canvases = [synth.cover_resize(im, cw, ch).astype(np.float32) / 255 for im in imgs]
    return cfg, names, canvases


def clean_levels(**kw):
    lv = {f: 0.0 for f in synth.FACTORS}
    lv.update(kw)
    return lv


def render(sources, index, **levels):
    cfg, names, canvases = sources
    scene = synth.make_scene(index, canvases, names, cfg)
    draws = synth.make_draws(index, len(canvases), scene.source_index, cfg)
    ref = synth.degrade.to_uint8(scene.ref)
    tgt, H, mask, valid, _ = synth.render_target(scene, clean_levels(**levels), draws, canvases, cfg)
    return scene, ref, tgt, H, mask, valid


@pytest.mark.parametrize("index", range(6))
def test_h_gt_aligns_tgt_with_ref_outside_the_change(sources, index):
    scene, ref, tgt, H, mask, valid = render(sources, index, viewpoint=48)
    aligned, foot = warp_to_ref(tgt, H, ref.shape)
    near_change = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    region = valid & foot & ~near_change
    region = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    diff = np.abs(aligned.astype(float) - ref.astype(float)).mean(axis=2)
    assert diff[region].mean() < 4.0, "H_gt does not align tgt onto ref"
    wrong, _ = warp_to_ref(tgt, np.linalg.inv(H), ref.shape)
    assert np.abs(wrong.astype(float) - ref.astype(float)).mean(axis=2)[region].mean() > 2 * diff[region].mean()
    if scene.change_type != "none":
        assert diff[mask & valid].mean() > 2 * diff[region].mean(), "changed pixels barely differ"


def test_no_change_pairs_have_empty_masks(sources):
    cfg, names, canvases = sources
    types = [synth.make_scene(i, canvases, names, cfg).change_type for i in range(40)]
    assert {"add", "remove", "none"} <= set(types)
    for i, t in enumerate(types):
        if t == "none":
            _, _, _, _, mask, _ = render(sources, i)
            assert not mask.any()


def test_objects_are_visible(sources):
    cfg, names, canvases = sources
    c = [x for i in range(30) for x in synth.make_scene(i, canvases, names, cfg).object_contrast]
    assert np.mean(np.array(c) >= cfg.min_object_contrast) > 0.9


def test_sift_recovers_h_gt(sources):
    _, ref, tgt, H, _, _ = render(sources, 0, viewpoint=32)
    H_est, info = sift_ransac(ref, tgt)
    assert corner_error(H_est, H, tgt.shape[1], tgt.shape[0]) < 2.0, info


def test_occluder_is_excluded_from_valid(sources):
    _, _, _, _, _, valid_clean = render(sources, 1)
    _, _, _, _, _, valid_occ = render(sources, 1, occlusion=0.1)
    assert valid_occ.sum() < valid_clean.sum() - 0.05 * valid_clean.size


def test_generate_and_evaluate_end_to_end(tmp_path):
    root = tmp_path / "bench"
    synth.generate(root, n_pairs=2, which="viewpoint,noise", verbose=False)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["n_pairs"] == 2 and "viewpoint_16" in manifest["sets"]
    ds = SynthDataset(root, sets=["viewpoint_16", "noise_*"])
    assert len(ds) == 2 * (1 + len(synth.FACTORS["noise"]))
    rows, summary, curves = evaluate(ds, register="oracle", change="absdiff", verbose=False)
    assert summary["viewpoint_16"]["registration"]["success@1px"] == 1.0
    _, summary_p, _ = evaluate(SynthDataset(root, sets=["viewpoint_16"]), change="absdiff",
                               perturb=[0, 4], verbose=False)
    assert summary_p["viewpoint_16@err4"]["registration"]["median_corner_err"] == pytest.approx(4, abs=0.05)
