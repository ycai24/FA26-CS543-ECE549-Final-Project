"""Synthetic change-detection benchmark generator.

    python -m changedet.synth --out data/synth/v0 --n 10

How one pair is built (all randomness is seeded by (seed, pair index)):

1. A source photo is resized to cover a canvas of (W + 2m) x (H + 2m) pixels.
2. 1-3 objects (random polygons cut from a *different* source photo) are blended
   into a copy of the canvas with feathered edges.
3. "add": ref scene = canvas, tgt scene = canvas + objects. "remove": the same
   two scenes, swapped. "none": no objects (measures false positives).
4. ref = the central W x H crop of its scene (no resampling).
5. tgt = the tgt scene seen through a homography. The ref rectangle's corners
   are each moved by up to rho px (viewpoint severity), and that quad is warped
   to W x H.
6. Degradations are applied to tgt after the warp, in this order:
   occluder -> illumination -> cast shadow -> blur -> noise -> 8-bit.

Sweeps vary one factor at a time away from BASE. They reuse the same scene and
the same random *directions* (corner directions, shading direction,
shadow/occluder shape, noise field) at every level, so only the magnitude
changes along a sweep.

Output: <out>/<set>/<pair id>/ holding
    ref.png, tgt.png  RGB images
    H_gt.npy          3x3, maps tgt pixel coords -> ref pixel coords (see geometry.py)
    mask_gt.png       255 = changed pixel, in the ref frame
    valid.png         255 = ref pixel visible in tgt (and not hidden by an occluder)
    meta.json         everything needed to reproduce / slice the results
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from . import degrade
from .geometry import image_corners, normalize_h, warp_mask_to_ref, warp_to_ref

GENERATOR_VERSION = "0.1"

# Severity levels per factor. Units: viewpoint = max corner displacement (px);
# noise = Gaussian sigma in 8-bit units; blur = Gaussian sigma (px);
# illum = strength in [0, 1]; shadow = fraction of light removed;
# occlusion = occluder area as a fraction of the image.
FACTORS: dict[str, list[float]] = {
    "viewpoint": [0, 8, 16, 32, 64, 96],
    "noise": [0, 5, 10, 20, 30],
    "blur": [0, 1, 2, 3, 5],
    "illum": [0, 0.25, 0.5, 0.75, 1.0],
    "shadow": [0, 0.2, 0.4, 0.6],
    "occlusion": [0, 0.05, 0.10, 0.20],
}
# Every factor sits at its BASE value except the one being swept.
BASE: dict[str, float] = dict(viewpoint=16, noise=2, blur=0, illum=0, shadow=0, occlusion=0)
# "mixed" set: every factor drawn at random from these ranges.
MIXED_RANGES: dict[str, tuple[float, float]] = dict(
    viewpoint=(0, 64), noise=(0, 15), blur=(0, 2), illum=(0, 0.75), shadow=(0, 0.5), occlusion=(0, 0.10))
MIXED_P_SHADOW = 0.5
MIXED_P_OCCLUSION = 0.3

# Images bundled with scikit-image, used when no --src-dir is given. Fine for a
# smoke test; use a larger, freely licensed photo set for real experiments.
# Don't use HPatches images: thresholds tuned on synthetic data would leak into
# the HPatches test.
SKIMAGE_SAMPLES = ["astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry", "camera",
                   "brick", "grass", "gravel", "moon", "hubble_deep_field", "coins"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class GenConfig:
    width: int = 512
    height: int = 384
    margin: int = 100                 # canvas border; must be >= the largest viewpoint level
    ref_noise: float = 2.0            # sensor noise on ref (8-bit sigma), fixed per pair
    max_objects: int = 3
    obj_radius_min: float = 0.06      # object radius as a fraction of min(W, H)
    obj_radius_max: float = 0.14
    feather_sigma: float = 1.5        # edge softening of pasted objects / occluders (px)
    min_object_contrast: float = 0.12 # mean |object - covered background| in [0, 1]; else retry
    object_tries: int = 10
    p_nochange: float = 0.1           # fraction of pairs with no change at all
    shadow_radius: float = 0.35       # shadow polygon radius as a fraction of min(W, H)
    shadow_softness: float = 8.0      # px
    occluder_mode: str = "ignore"     # "ignore": occluded pixels are not scored; "change": they count as change
    seed: int = 0


# ------------------------------------------------------------------------- sources
def _as_rgb_u8(im: np.ndarray) -> np.ndarray | None:
    if im is None or im.dtype == bool:
        return None
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)
    if im.ndim == 2:
        return np.stack([im] * 3, axis=-1)
    if im.ndim == 3 and im.shape[2] == 4:
        return im[..., :3].copy()
    if im.ndim == 3 and im.shape[2] == 3:
        return im
    return None


def load_sources(src_dir: str | Path | None = None) -> tuple[list[str], list[np.ndarray]]:
    names, imgs = [], []
    if src_dir:
        for p in sorted(Path(src_dir).rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                im = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if im is not None:
                    names.append(p.name)
                    imgs.append(im)
    else:
        from skimage import data
        for n in SKIMAGE_SAMPLES:
            try:
                im = _as_rgb_u8(getattr(data, n)())
            except Exception:  # sample not bundled / needs a download
                im = None
            if im is not None:
                names.append(f"skimage:{n}")
                imgs.append(im)
    if len(imgs) < 2:
        raise ValueError("need at least two source images (one scene, one for object patches)")
    return names, imgs


def cover_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize (keeping aspect ratio) so the image covers w x h, then centre-crop."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0:y0 + h, x0:x0 + w]


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


# --------------------------------------------------------------------------- scene
@dataclass
class Scene:
    ref: np.ndarray        # float32 H x W x 3, before ref noise
    tgt_scene: np.ndarray  # float32 canvas the target view is rendered from
    mask_gt: np.ndarray    # bool H x W, changed pixels in the ref frame
    change_type: str       # "add" | "remove" | "none"
    source: str
    source_index: int
    patch_sources: list
    object_contrast: list  # mean |object - covered background| per object, in [0, 1]


def make_scene(index: int, canvases: list[np.ndarray], names: list[str], cfg: GenConfig) -> Scene:
    rng = np.random.default_rng([cfg.seed, index, 0])
    W, H, m = cfg.width, cfg.height, cfg.margin
    n = len(canvases)
    si = int(rng.integers(n))
    canvas = canvases[si]

    if rng.random() < cfg.p_nochange:
        change_type, n_obj = "none", 0
    else:
        change_type = "add" if rng.random() < 0.5 else "remove"
        n_obj = int(rng.integers(1, cfg.max_objects + 1))

    with_obj = canvas.copy()
    alpha_union = np.zeros(canvas.shape[:2], np.float32)
    patch_names, contrasts = [], []
    for _ in range(n_obj):
        r = rng.uniform(cfg.obj_radius_min, cfg.obj_radius_max) * min(W, H)
        center = (m + rng.uniform(0.2, 0.8) * W, m + rng.uniform(0.2, 0.8) * H)
        poly = degrade.random_polygon(rng, center, r)
        alpha = degrade.feathered_mask(canvas.shape, poly, cfg.feather_sigma)
        inside = alpha > 0.5
        # Pick patch content until the object visibly differs from what it covers;
        # otherwise the mask would label pixels that did not really change.
        best = None
        for _try in range(cfg.object_tries):
            pj = int(rng.integers(n - 1))
            pj += pj >= si                   # any source except the scene's own
            dx, dy = rng.uniform(-m, m, size=2)
            patch = _shift(canvases[pj], dx, dy)
            if inside.any():
                # Pull the patch halfway towards the local brightness so objects are not
                # trivially brighter/darker than their surroundings (cf. Ch4 Fig. 4.3).
                bg, pm = float(with_obj[inside].mean()), float(patch[inside].mean())
                patch = np.clip(patch * (0.5 + 0.5 * bg / max(pm, 1e-3)), 0.0, 1.0)
            contrast = float(np.abs(patch[inside] - with_obj[inside]).mean()) if inside.any() else 0.0
            if best is None or contrast > best[0]:
                best = (contrast, pj, patch)
            if contrast >= cfg.min_object_contrast:
                break
        contrast, pj, patch = best
        with_obj = degrade.composite(with_obj, patch, alpha)
        alpha_union = np.maximum(alpha_union, alpha)
        patch_names.append(names[pj])
        contrasts.append(round(contrast, 4))

    ref_scene, tgt_scene = (with_obj, canvas) if change_type == "remove" else (canvas, with_obj)
    crop = (slice(m, m + H), slice(m, m + W))
    return Scene(ref=ref_scene[crop].copy(), tgt_scene=tgt_scene.astype(np.float32),
                 mask_gt=alpha_union[crop] >= 0.5, change_type=change_type,
                 source=names[si], source_index=si, patch_sources=patch_names,
                 object_contrast=contrasts)


@dataclass
class Draws:
    """Per-pair random directions, reused at every severity level."""
    corner_dirs: np.ndarray      # 4 x 2, uniform in [-1, 1]
    gain_sign: float
    gamma_sign: float
    shade_angle: float
    shadow_poly: np.ndarray      # tgt-frame polygon
    occ_poly_unit: np.ndarray    # polygon around (0, 0), radius ~1
    occ_center: np.ndarray
    occ_source: int
    occ_shift: np.ndarray
    noise_field: np.ndarray      # H x W x 3, N(0, 1), for tgt
    ref_noise_field: np.ndarray  # H x W x 3, N(0, 1), for ref


def make_draws(index: int, n_sources: int, scene_source: int, cfg: GenConfig) -> Draws:
    rng = np.random.default_rng([cfg.seed, index, 1])
    W, H = cfg.width, cfg.height
    occ_src = int(rng.integers(n_sources - 1))
    occ_src += occ_src >= scene_source
    return Draws(
        corner_dirs=rng.uniform(-1.0, 1.0, size=(4, 2)),
        gain_sign=float(rng.choice([-1.0, 1.0])),
        gamma_sign=float(rng.choice([-1.0, 1.0])),
        shade_angle=float(rng.uniform(0, 2 * np.pi)),
        shadow_poly=degrade.random_polygon(
            rng, (rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H), cfg.shadow_radius * min(W, H)),
        occ_poly_unit=degrade.random_polygon(rng, (0.0, 0.0), 1.0),
        occ_center=np.array([rng.uniform(0.25, 0.75) * W, rng.uniform(0.25, 0.75) * H]),
        occ_source=occ_src,
        occ_shift=rng.uniform(-cfg.margin, cfg.margin, size=2),
        noise_field=rng.standard_normal((H, W, 3)).astype(np.float32),
        ref_noise_field=rng.standard_normal((H, W, 3)).astype(np.float32),
    )


def mixed_levels(index: int, cfg: GenConfig) -> dict[str, float]:
    rng = np.random.default_rng([cfg.seed, index, 2])
    lv = {f: float(rng.uniform(*MIXED_RANGES[f])) for f in FACTORS}
    if rng.random() >= MIXED_P_SHADOW:
        lv["shadow"] = 0.0
    if rng.random() >= MIXED_P_OCCLUSION:
        lv["occlusion"] = 0.0
    return lv


def render_target(scene: Scene, levels: dict[str, float], draws: Draws,
                  canvases: list[np.ndarray], cfg: GenConfig):
    """Render the target view. Returns (tgt_u8, H_gt, mask_gt, valid, occluded_ref)."""
    W, H, m = cfg.width, cfg.height, cfg.margin
    rect = image_corners(W, H)
    quad = rect + m + levels["viewpoint"] * draws.corner_dirs       # canvas coords
    H_tc = cv2.getPerspectiveTransform(rect.astype(np.float32), quad.astype(np.float32)).astype(np.float64)
    # tgt(x) = scene(H_tc x)
    tgt = cv2.warpPerspective(scene.tgt_scene, H_tc, (W, H),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT101)
    to_ref = np.array([[1.0, 0.0, -m], [0.0, 1.0, -m], [0.0, 0.0, 1.0]])
    H_gt = normalize_h(to_ref @ H_tc)                                 # tgt -> ref

    occluded = np.zeros((H, W), bool)
    if levels["occlusion"] > 0:
        r = np.sqrt(levels["occlusion"] * W * H / degrade.polygon_area(draws.occ_poly_unit))
        poly = draws.occ_center + r * draws.occ_poly_unit
        alpha = degrade.feathered_mask((H, W), poly, cfg.feather_sigma)
        patch = _shift(canvases[draws.occ_source], *draws.occ_shift)[m:m + H, m:m + W]
        tgt = degrade.composite(tgt, patch, alpha)
        occluded = alpha >= 0.5

    tgt = degrade.illumination(tgt, levels["illum"], draws.gain_sign, draws.gamma_sign, draws.shade_angle)
    tgt = degrade.cast_shadow(tgt, levels["shadow"], draws.shadow_poly, cfg.shadow_softness)
    tgt = degrade.gaussian_blur(tgt, levels["blur"])
    tgt = degrade.add_noise(tgt, levels["noise"], draws.noise_field)
    tgt_u8 = degrade.to_uint8(tgt)

    _, valid = warp_to_ref(np.zeros((H, W), np.uint8), H_gt, (H, W))
    occluded_ref = warp_mask_to_ref(occluded, H_gt, (H, W)) & valid
    mask = scene.mask_gt.copy()
    if cfg.occluder_mode == "ignore":
        valid &= ~occluded_ref
    elif cfg.occluder_mode == "change":
        mask |= occluded_ref
    else:
        raise ValueError(f"unknown occluder_mode {cfg.occluder_mode!r}")
    return tgt_u8, H_gt, mask, valid, occluded_ref


# --------------------------------------------------------------------------- sets
def build_sets(which: str) -> list[tuple[str, dict | None, str]]:
    """Parse --sets: 'all', 'sweeps', 'mixed', or a comma list of factor names / 'mixed'."""
    tokens = [t.strip() for t in which.split(",") if t.strip()]
    if tokens == ["all"]:
        tokens = list(FACTORS) + ["mixed"]
    elif tokens == ["sweeps"]:
        tokens = list(FACTORS)
    sets = []
    for t in tokens:
        if t == "mixed":
            sets.append(("mixed", None, "mixed"))
        elif t in FACTORS:
            for lv in FACTORS[t]:
                levels = dict(BASE)
                levels[t] = lv
                sets.append((f"{t}_{lv:g}", levels, t))
        else:
            raise ValueError(f"unknown set {t!r}; choose from {list(FACTORS) + ['mixed', 'sweeps', 'all']}")
    return sets


def save_pair(d: Path, ref_u8, tgt_u8, H_gt, mask, valid, meta) -> None:
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "ref.png"), cv2.cvtColor(ref_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(d / "tgt.png"), cv2.cvtColor(tgt_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(d / "mask_gt.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(d / "valid.png"), valid.astype(np.uint8) * 255)
    np.save(d / "H_gt.npy", H_gt)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def generate(out_dir: str | Path, n_pairs: int, which: str = "all", cfg: GenConfig | None = None,
             src_dir: str | Path | None = None, overwrite: bool = False, verbose: bool = True) -> Path:
    cfg = cfg or GenConfig()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"{out} is not empty; choose a new --out or pass --overwrite")
    max_rho = max(FACTORS["viewpoint"] + [MIXED_RANGES["viewpoint"][1]])
    if cfg.margin < max_rho:
        raise ValueError(f"margin ({cfg.margin}) must be >= the largest viewpoint level ({max_rho})")

    names, imgs = load_sources(src_dir)
    cw, ch = cfg.width + 2 * cfg.margin, cfg.height + 2 * cfg.margin
    canvases = [cover_resize(im, cw, ch).astype(np.float32) / 255.0 for im in imgs]
    sets = build_sets(which)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for i in range(n_pairs):
        scene = make_scene(i, canvases, names, cfg)
        draws = make_draws(i, len(canvases), scene.source_index, cfg)
        ref_u8 = degrade.to_uint8(degrade.add_noise(scene.ref, cfg.ref_noise, draws.ref_noise_field))
        for set_name, levels, factor in sets:
            lv = mixed_levels(i, cfg) if levels is None else levels
            tgt_u8, H_gt, mask, valid, occ = render_target(scene, lv, draws, canvases, cfg)
            meta = {
                "pid": f"{i:04d}", "set": set_name, "factor": factor, "levels": lv,
                "change_type": scene.change_type, "n_objects": len(scene.patch_sources),
                "changed_fraction": float(mask[valid].mean()) if valid.any() else 0.0,
                "valid_fraction": float(valid.mean()), "occluded_fraction": float(occ.mean()),
                "source": scene.source, "patch_sources": scene.patch_sources,
                "object_contrast": scene.object_contrast,
                "seed": cfg.seed, "occluder_mode": cfg.occluder_mode,
                "H_convention": "H_gt maps tgt pixel (x, y, 1) to ref pixel coords (OpenCV, 0-based)",
            }
            save_pair(out / set_name / f"{i:04d}", ref_u8, tgt_u8, H_gt, mask, valid, meta)
        if verbose:
            print(f"pair {i + 1}/{n_pairs} done ({time.perf_counter() - t0:.1f}s)", flush=True)

    manifest = {
        "generator_version": GENERATOR_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": n_pairs, "sets": [s[0] for s in sets], "config": asdict(cfg),
        "factors": FACTORS, "base": BASE, "mixed_ranges": MIXED_RANGES,
        "sources": names, "src_dir": str(src_dir) if src_dir else None,
        "python": platform.python_version(), "opencv": cv2.__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate the synthetic change-detection benchmark.")
    p.add_argument("--out", default="data/synth/v0")
    p.add_argument("--n", type=int, default=10, help="number of scene pairs (each is rendered in every set)")
    p.add_argument("--sets", default="all", help="'all', 'sweeps', 'mixed', or comma list of factors")
    p.add_argument("--src-dir", default=None, help="folder of source photos (default: scikit-image samples)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", default="512x384", help="WxH of the output images")
    p.add_argument("--occluder", choices=["ignore", "change"], default="ignore")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    w, h = (int(v) for v in a.size.lower().split("x"))
    cfg = GenConfig(width=w, height=h, seed=a.seed, occluder_mode=a.occluder)
    try:
        out = generate(a.out, a.n, a.sets, cfg, a.src_dir, a.overwrite)
    except (FileExistsError, ValueError) as e:
        sys.exit(f"error: {e}")
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
