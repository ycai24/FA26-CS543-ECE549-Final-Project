"""Evaluation harness: run a registration method and/or a change-detection method
over a dataset and write per-pair results, a summary, PR curves and plots.

Examples (run from the repo root):

  # Oracle alignment: change detection given the TRUE homography
  python -m changedet.evaluate --data data/synth/v0 --register oracle --change absdiff --out results/oracle_absdiff

  # Controlled misalignment: true homography pushed off by exactly 0/1/2/4/8 px
  python -m changedet.evaluate --data data/synth/v0 --sets viewpoint_16 --perturb 0,1,2,4,8 \
      --change absdiff --out results/misalign_absdiff

  # A teammate's methods end to end
  python -m changedet.evaluate --data data/synth/v0 --register changedet.registration.cascade:register \
      --change changedet.change.ssim:change_map --out results/cascade_ssim

  # Registration only, on HPatches (--sets i or v picks the subset)
  python -m changedet.evaluate --dataset hpatches --data data/hpatches-sequences-release \
      --register sift --out results/hpatches_sift

Scoring rules
* Registration: corner error of the target's corners (geometry.corner_error).
  Summaries give success rates at 1/3/5/10 px plus the median.
* Change detection is scored on the ground-truth valid region. Pixels the
  estimated warp doesn't cover count as "not detected", so a bad registration
  can't hide changed pixels by shrinking the overlap. A failed registration
  (H is None) detects nothing.
* Thresholds: precision-recall curves and AP need none. Fixed-threshold metrics
  use --threshold: 'otsu' (per image, a fixed rule) or a number tuned on
  separate data. best_f1 is also reported, but it picks the threshold on the
  test data, so it is optimistic.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from . import datasets
from .change import reference as change_reference
from .geometry import corner_error, perturb_homography, warp_to_ref
from .metrics import (PRAccumulator, check_score, confusion, otsu_threshold, prf_iou,
                      registration_summary)
from .registration import reference as reg_reference

BUILTIN_REGISTER = {"identity": reg_reference.identity, "sift": reg_reference.sift_ransac}
BUILTIN_CHANGE = {"absdiff": change_reference.absdiff}


def resolve(spec: str, builtins: dict):
    """A builtin name, or 'package.module:function'."""
    if spec in builtins:
        return builtins[spec]
    if ":" not in spec:
        raise ValueError(f"{spec!r} is not a builtin ({sorted(builtins)}) or 'module:function'")
    mod, fn = spec.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


def _scalars(d: dict, prefix: str) -> dict:
    ok = (int, float, str, bool, np.integer, np.floating, np.bool_)
    return {prefix + k: (v.item() if hasattr(v, "item") else v) for k, v in (d or {}).items() if isinstance(v, ok)}


def evaluate(dataset, register: str = "oracle", change: str | None = None, perturb=None,
             threshold: str = "otsu", sample_rate: float = 0.05, seed: int = 0, verbose: bool = True):
    """Returns (rows, summary, pr_curves)."""
    reg_fn = None if (register == "oracle" or perturb) else resolve(register, BUILTIN_REGISTER)
    cd_fn = resolve(change, BUILTIN_CHANGE) if change else None
    variants = list(perturb) if perturb else [None]
    rows, groups = [], {}
    t_start = time.perf_counter()

    for i, s in enumerate(dataset):
        th, tw = s.tgt.shape[:2]
        for e in variants:
            gname = s.set if e is None else f"{s.set}@err{e:g}"
            g = groups.setdefault(gname, dict(set=s.set, perturb_px=e, n=0, acc=PRAccumulator(sample_rate, seed),
                                              tp=0, fp=0, fn=0, errs=[], reg_times=[], cd_times=[], coverage=[]))
            g["n"] += 1
            row = {"set": s.set, "pid": s.pid, "group": gname, "perturb_px": "" if e is None else e}

            # ---- registration
            if e is not None or reg_fn is None:
                if s.H_gt is None:
                    raise ValueError(f"{s.set}/{s.pid}: oracle/perturbed registration needs a ground-truth H")
                if e is None:
                    H, info = s.H_gt, {"method": "oracle"}
                else:
                    rng = np.random.default_rng([seed, i, int(round(e * 1000))])
                    H, info = perturb_homography(s.H_gt, tw, th, e, rng), {"method": "perturbed_gt"}
                t_reg = 0.0
            else:
                t0 = time.perf_counter()
                try:
                    H, info = reg_fn(s.ref, s.tgt)
                except Exception as exc:
                    raise RuntimeError(f"registration {register!r} crashed on {s.set}/{s.pid}") from exc
                t_reg = time.perf_counter() - t0
            row.update(_scalars(info, "reg_"))
            row["reg_time_s"] = t_reg
            row["reg_failed"] = H is None
            g["reg_times"].append(t_reg)
            if s.H_gt is not None:
                err = corner_error(H, s.H_gt, tw, th)
                row["corner_err"] = err
                g["errs"].append(err)

            # ---- change detection
            if cd_fn is not None and s.mask_gt is not None:
                shape = s.ref.shape[:2]
                base = s.valid if s.valid is not None else np.ones(shape, bool)
                score_full = np.full(shape, -np.inf, np.float32)
                if H is None:
                    region, thr, t_cd = np.zeros(shape, bool), float("inf"), 0.0
                else:
                    aligned, foot = warp_to_ref(s.tgt, H, s.ref.shape)
                    region = base & foot
                    t0 = time.perf_counter()
                    try:
                        score = check_score(cd_fn(s.ref, aligned, region.copy()), shape)
                    except Exception as exc:
                        raise RuntimeError(f"change map {change!r} failed on {s.set}/{s.pid}") from exc
                    t_cd = time.perf_counter() - t0
                    score_full[region] = score[region]
                    thr = otsu_threshold(score[region]) if threshold == "otsu" else float(threshold)
                g["acc"].add(score_full, s.mask_gt, base)
                tp, fp, fn, _ = confusion(score_full >= thr, s.mask_gt, base)
                g["tp"] += tp
                g["fp"] += fp
                g["fn"] += fn
                g["cd_times"].append(t_cd)
                cov = float(region.sum() / max(base.sum(), 1))
                g["coverage"].append(cov)
                row.update(cd_time_s=t_cd, threshold=thr, tp=tp, fp=fp, fn=fn, coverage=cov,
                           **prf_iou(tp, fp, fn))
            rows.append(row)
        if verbose and ((i + 1) % 25 == 0 or i + 1 == len(dataset)):
            print(f"{i + 1}/{len(dataset)} samples ({time.perf_counter() - t_start:.1f}s)", flush=True)

    summary, curves = {}, {}
    for name, g in sorted(groups.items()):
        entry = {"set": g["set"], "perturb_px": g["perturb_px"], "n": g["n"]}
        if g["errs"]:
            entry["registration"] = registration_summary(g["errs"], g["reg_times"])
        if g["cd_times"]:
            acc = g["acc"]
            best_f1, best_thr = acc.best_f1()
            entry["change"] = {
                "AP": acc.average_precision(), "best_f1": best_f1, "best_f1_threshold": best_thr,
                "threshold_rule": threshold, **prf_iou(g["tp"], g["fp"], g["fn"]),
                "mean_time_s": float(np.mean(g["cd_times"])), "mean_coverage": float(np.mean(g["coverage"])),
                "positive_samples": acc.n_positive,
            }
            curves[name] = acc.curve_points()
        summary[name] = entry
    return rows, summary, curves


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def write_results(out: Path, rows, summary, curves, config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(out / "per_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(_json_safe({"config": config, "groups": summary}), indent=2))
    (out / "pr_curves.json").write_text(json.dumps(_json_safe(curves)))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", choices=["synth", "hpatches"], default="synth")
    p.add_argument("--data", required=True, help="dataset root folder")
    p.add_argument("--sets", default=None, help="comma list of set-name globs (synth) or i / v (hpatches)")
    p.add_argument("--limit", type=int, default=None, help="max pairs per set (synth) / sequences (hpatches)")
    p.add_argument("--register", default="oracle", help="oracle | identity | sift | module:function")
    p.add_argument("--change", default=None, help="absdiff | module:function (omit for registration only)")
    p.add_argument("--perturb", default=None, help="comma list of injected corner errors in px, e.g. 0,1,2,4,8")
    p.add_argument("--threshold", default="otsu", help="'otsu' or a fixed number")
    p.add_argument("--sample-rate", type=float, default=0.05, help="pixel sampling rate for PR curves")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)

    out = Path(a.out)
    if (out / "summary.json").exists() and not a.overwrite:
        sys.exit(f"error: {out} already has results; choose a new --out or pass --overwrite")
    if a.threshold != "otsu":
        float(a.threshold)  # fail early on a typo
    sets = [s.strip() for s in a.sets.split(",")] if a.sets else None
    perturb = [float(v) for v in a.perturb.split(",")] if a.perturb else None
    ds = datasets.open_dataset(a.dataset, a.data, sets, a.limit)

    rows, summary, curves = evaluate(ds, a.register, a.change, perturb, a.threshold, a.sample_rate, a.seed)
    config = {k: v for k, v in vars(a).items() if k not in ("overwrite", "no_plots")}
    config["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_results(out, rows, summary, curves, config)
    print(f"wrote {out.resolve()}")
    if not a.no_plots:
        from .plots import make_run_plots
        for path in make_run_plots(out):
            print(f"  plot: {path}")


if __name__ == "__main__":
    main()
