"""Plots for evaluation runs.

A single run's plots are written automatically by ``changedet.evaluate``. To
compare methods, pass several run folders (colors follow the order given):

  python -m changedet.plots results/oracle_absdiff results/sift_absdiff --labels "Oracle,SIFT" --out results/compare
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

# Categorical palette (validated colorblind-safe for adjacent series, light mode).
# Colors are assigned in this fixed order, by run, never cycled. Each series also
# gets its own marker shape so identity never depends on color alone (print, CVD).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

FACTOR_LABELS = {
    "viewpoint": "Viewpoint: max corner shift (px)",
    "noise": "Noise σ (8-bit units)",
    "blur": "Blur σ (px)",
    "illum": "Illumination change (strength)",
    "shadow": "Shadow: fraction of light removed",
    "occlusion": "Occluder area (fraction of image)",
}
GROUP_RE = re.compile(r"^(?P<factor>[a-z]+)_(?P<level>[0-9.]+)(@err(?P<err>[0-9.]+))?$")


def _plt():
    import logging

    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # fallback-font chatter
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
    })
    return plt


def _num(v):
    if v is None or v in ("inf", "-inf"):
        return math.nan
    return float(v)


def load_run(run_dir: str | Path, label: str | None = None) -> dict:
    d = Path(run_dir)
    summary = json.loads((d / "summary.json").read_text())
    pr_path = d / "pr_curves.json"
    curves = json.loads(pr_path.read_text()) if pr_path.exists() else {}
    return {"label": label or d.name, "config": summary["config"], "groups": summary["groups"], "pr": curves}


def _get(entry: dict, metric: str):
    section, key = metric.split(".", 1)
    return _num(entry.get(section, {}).get(key))


def _sweep_points(run: dict, metric: str) -> dict:
    """factor -> sorted [(level, value)], one-factor sweeps only (no perturbation groups)."""
    out: dict[str, list] = {}
    for name, entry in run["groups"].items():
        m = GROUP_RE.match(name)
        if not m or m.group("err") is not None:
            continue
        v = _get(entry, metric)
        if not math.isnan(v):
            out.setdefault(m.group("factor"), []).append((float(m.group("level")), v))
    return {f: sorted(pts) for f, pts in out.items() if len(pts) >= 2}


def _finish(fig, runs, path, title, plt, ncol_legend=4):
    if len(runs) >= 2:
        handles, labels = [], []
        for ax in fig.axes:                       # the first panel may not have every run
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        fig.suptitle(title, fontsize=11, fontweight="semibold", color=INK)
        fig.legend(handles, labels, loc="outside lower center", ncol=min(ncol_legend, len(labels)),
                   handlelength=2.2)
    else:
        fig.suptitle(f"{title} — {runs[0]['label']}", fontsize=11, fontweight="semibold", color=INK)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _direct_labels(ax, items, min_gap_pt: float = 11.0, dx_pt: float = 6.0) -> None:
    """Label points (x, y, text) to their right, nudged apart vertically so they never collide.
    Text stays in ink colors; the line beside it carries the series color."""
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_pt = 72.0 / fig.dpi
    ys = [ax.transData.transform((x, y))[1] * to_pt for x, y, _ in items]
    placed, last = {}, -1e9
    for i in sorted(range(len(items)), key=lambda j: ys[j]):
        placed[i] = max(ys[i], last + min_gap_pt)
        last = placed[i]
    for i, (x, y, text) in enumerate(items):
        ax.annotate(text, (x, y), xytext=(dx_pt, placed[i] - ys[i]), textcoords="offset points",
                    va="center", color=INK2, fontsize=8, annotation_clip=False)


def sweep_figure(runs: list[dict], metric: str, ylabel: str, title: str, path: Path):
    """Small multiples: one panel per degradation factor, one line per run."""
    plt = _plt()
    per_run = [_sweep_points(r, metric) for r in runs]
    factors = [f for f in FACTOR_LABELS if any(f in pr for pr in per_run)]
    if not factors:
        return None
    ncols = min(3, len(factors))
    nrows = math.ceil(len(factors) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows), squeeze=False,
                             sharey=True, constrained_layout=True)
    for ax, f in zip(axes.flat, factors):
        for k, (run, pts) in enumerate(zip(runs, per_run)):
            if f in pts:
                xs, ys = zip(*pts[f])
                ax.plot(xs, ys, color=SERIES[k], marker=MARKERS[k], label=run["label"],
                        markeredgecolor="white", markeredgewidth=1.0)
        ax.set_title(FACTOR_LABELS[f], loc="left")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes.flat[len(factors):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    return _finish(fig, runs, path, title, plt)


def pr_figure(runs: list[dict], group: str, path: Path):
    """Precision-recall curves for one set; the dot marks each run's best-F1 point."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.8), constrained_layout=True)
    drawn, labels = 0, []
    for k, run in enumerate(runs):
        c = run["pr"].get(group)
        if not c:
            continue
        ax.plot(c["recall"], c["precision"], color=SERIES[k], label=run["label"])
        bf = run["groups"][group].get("change", {})
        f1 = _num(bf.get("best_f1"))
        if not math.isnan(f1):
            p, r = c["precision"], c["recall"]
            i = max(range(len(p)), key=lambda j: 2 * p[j] * r[j] / (p[j] + r[j]) if p[j] + r[j] > 0 else 0)
            ax.plot([r[i]], [p[i]], marker=MARKERS[k], color=SERIES[k], markersize=8,
                    markeredgecolor="white", markeredgewidth=1.2, linestyle="none")
            labels.append((r[i], p[i], f"{run['label'] + ': ' if len(runs) > 1 else ''}best F1 {f1:.2f}"))
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    if len(labels) <= 4:
        # Label to the left of each best-F1 dot: PR curves leave that area empty.
        _direct_labels(ax, labels, dx_pt=-8)
        for t in ax.texts:
            t.set_horizontalalignment("right")
    return _finish(fig, runs, path, f"Precision–recall, {group}", plt, ncol_legend=2)


def misregistration_figure(runs: list[dict], metric: str, ylabel: str, path: Path):
    """Change-detection quality vs. injected corner error (from --perturb runs)."""
    plt = _plt()
    series = []
    for run in runs:
        by_set: dict[str, list] = {}
        for name, entry in run["groups"].items():
            if entry.get("perturb_px") is None:
                continue
            v = _get(entry, metric)
            if not math.isnan(v):
                by_set.setdefault(entry["set"], []).append((float(entry["perturb_px"]), v))
        for s, pts in sorted(by_set.items()):
            if len(runs) == 1:
                label = s
            else:
                label = run["label"] if len(by_set) == 1 else f"{run['label']} · {s}"
            series.append((label, sorted(pts)))
    if not series:
        return None
    series = series[:len(SERIES)]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ends = []
    for k, (label, pts) in enumerate(series):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=SERIES[k], marker=MARKERS[k], label=label,
                markeredgecolor="white", markeredgewidth=1.0)
        ends.append((xs[-1], ys[-1], label))
    ax.set_xlabel("Injected corner error (px)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    if 1 < len(series) <= 4:
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + 0.35 * (x1 - x0))   # room for the end labels inside the axes
        _direct_labels(ax, ends)
    title = "Change detection vs. misregistration"
    if len(runs) == 1:
        title += f" — {runs[0]['label']}"
    pseudo_runs = [{"label": lbl} for lbl, _ in series]
    if len(pseudo_runs) == 1:
        pseudo_runs[0]["label"] = runs[0]["label"]
        title = "Change detection vs. misregistration"
    return _finish(fig, pseudo_runs, path, title, plt, ncol_legend=2)


def _default_pr_group(runs: list[dict]) -> str | None:
    names = [n for n, c in runs[0]["pr"].items() if c]
    for pref in ("viewpoint_16", "viewpoint_16@err0", "mixed", "mixed@err0"):
        if pref in names:
            return pref
    return names[0] if names else None


def make_plots(runs: list[dict], out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    real_reg = any(r["config"].get("register") not in ("oracle", None) and not r["config"].get("perturb")
                   for r in runs)
    if real_reg:
        made.append(sweep_figure(runs, "registration.success@3px", "Success rate (≤ 3 px)",
                                 "Registration success vs. degradation", out / "registration_success.png"))
    made.append(sweep_figure(runs, "change.AP", "Average precision",
                             "Change detection AP vs. degradation", out / "change_ap.png"))
    made.append(sweep_figure(runs, "change.f1", "F1 (fixed threshold rule)",
                             "Change detection F1 vs. degradation", out / "change_f1.png"))
    group = _default_pr_group(runs)
    if group:
        made.append(pr_figure(runs, group, out / f"pr_{group}.png"))
    made.append(misregistration_figure(runs, "change.AP", "Average precision", out / "misregistration_ap.png"))
    return [p for p in made if p is not None]


def make_run_plots(run_dir: str | Path) -> list[Path]:
    return make_plots([load_run(run_dir)], Path(run_dir) / "plots")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Compare evaluation runs.")
    p.add_argument("runs", nargs="+", help="result folders written by changedet.evaluate")
    p.add_argument("--labels", default=None, help="comma list, same order as the runs")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    labels = a.labels.split(",") if a.labels else [None] * len(a.runs)
    if len(labels) != len(a.runs):
        raise SystemExit("error: --labels must have one entry per run")
    if len(a.runs) > len(SERIES):
        raise SystemExit(f"error: at most {len(SERIES)} runs per figure; split into several comparisons")
    runs = [load_run(r, l) for r, l in zip(a.runs, labels)]
    for path in make_plots(runs, a.out):
        print(f"plot: {path}")


if __name__ == "__main__":
    main()
