"""Dataset loaders. Every loader yields ``Sample`` objects with the same fields.

Images are RGB uint8 (H x W x 3). Masks are bool. ``H_gt`` maps tgt pixel
coordinates to ref pixel coordinates (see geometry.py). Fields a dataset lacks
are None: HPatches has no change mask, VL-CMU-CD has no homography.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geometry import normalize_h, warp_to_ref


@dataclass
class Sample:
    pid: str
    set: str
    ref: np.ndarray
    tgt: np.ndarray
    H_gt: np.ndarray | None = None
    mask_gt: np.ndarray | None = None
    valid: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


def load_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"could not read mask {path}")
    return m > 127


class SynthDataset:
    """Synthetic benchmark written by ``changedet.synth``.

    ``sets`` is a list of glob patterns over set names, e.g. ["viewpoint_*", "mixed"].
    ``limit`` caps the number of pairs per set (handy for quick runs).
    """

    def __init__(self, root: str | Path, sets: list[str] | None = None, limit: int | None = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"no benchmark at {self.root}")
        all_sets = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        if sets:
            all_sets = [s for s in all_sets if any(fnmatch.fnmatch(s, pat) for pat in sets)]
        self.items: list[Path] = []
        for s in all_sets:
            pairs = sorted(p for p in (self.root / s).iterdir() if (p / "meta.json").exists())
            self.items.extend(pairs[:limit] if limit else pairs)
        if not self.items:
            raise ValueError(f"no pairs found under {self.root} for sets={sets}")

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        for d in self.items:
            meta = json.loads((d / "meta.json").read_text())
            yield Sample(pid=meta["pid"], set=meta["set"], ref=load_rgb(d / "ref.png"),
                         tgt=load_rgb(d / "tgt.png"), H_gt=np.load(d / "H_gt.npy"),
                         mask_gt=load_mask(d / "mask_gt.png"), valid=load_mask(d / "valid.png"), meta=meta)


class HPatchesDataset:
    """HPatches sequences (hpatches-sequences-release).

    Each sequence folder holds 1.ppm ... 6.ppm and text files H_1_2 ... H_1_6, where
    H_1_k maps image-1 coordinates to image-k coordinates. Pairs are (ref = 1, tgt = k),
    so ``H_gt = inv(H_1_k)``. Folder prefixes: ``i_`` = illumination, ``v_`` = viewpoint.
    """

    def __init__(self, root: str | Path, subset: str = "all", limit: int | None = None):
        self.root = Path(root)
        if subset not in ("all", "i", "v"):
            raise ValueError("subset must be 'all', 'i' or 'v'")
        seqs = sorted(d for d in self.root.iterdir() if d.is_dir() and d.name[:2] in ("i_", "v_"))
        if subset != "all":
            seqs = [d for d in seqs if d.name.startswith(subset + "_")]
        if not seqs:
            raise ValueError(f"no HPatches sequences found under {self.root}")
        self.seqs = seqs[:limit] if limit else seqs

    def __len__(self) -> int:
        return 5 * len(self.seqs)

    def __iter__(self):
        for seq in self.seqs:
            ref = load_rgb(seq / "1.ppm")
            for k in range(2, 7):
                tgt = load_rgb(seq / f"{k}.ppm")
                H_gt = normalize_h(np.linalg.inv(np.loadtxt(seq / f"H_1_{k}")))
                _, valid = warp_to_ref(np.zeros(tgt.shape[:2], np.uint8), H_gt, ref.shape)
                yield Sample(pid=f"{seq.name}_1_{k}", set=f"hpatches_{seq.name[0]}", ref=ref, tgt=tgt,
                             H_gt=H_gt, valid=valid, meta={"sequence": seq.name, "k": k})


def open_dataset(kind: str, root: str | Path, sets: list[str] | None = None, limit: int | None = None):
    if kind == "synth":
        return SynthDataset(root, sets, limit)
    if kind == "hpatches":
        subset = sets[0] if sets else "all"
        return HPatchesDataset(root, subset, limit)
    raise ValueError(f"unknown dataset kind {kind!r} (VL-CMU-CD loader not written yet)")
