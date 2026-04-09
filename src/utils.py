"""
Visualisation and analysis utilities for Flow Matching Assignment (COMP8650).

Functions
─────────
plot_comparison       — side-by-side ground-truth vs generated scatter plots
plot_2x2_grid         — compact grid of scatter plots (e.g. for 4 param combos)
plot_grid             — general N-panel grid of scatter plots
plot_loss_curve       — smoothed training loss
plot_step_comparison  — generated samples at multiple ODE step counts
save_fig              — save a figure with auto-directory creation

Metric
──────
nearest_neighbour_dist — simple NND-based quality metric
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend when running headlessly (e.g. in certain Colab modes)
# matplotlib.use("Agg")  # uncomment if plots don't render in your environment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCATTER_KW = dict(s=2, alpha=0.45, rasterized=True)


def _set_ax(ax: plt.Axes, title: str = "", lims: float | None = None) -> None:
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.tick_params(labelsize=7)
    if lims is not None:
        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)


def save_fig(fig: plt.Figure, path: str | Path) -> None:
    """Save figure, creating parent directories as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=120, bbox_inches="tight")
    print(f"[Saved] {p}")


# ---------------------------------------------------------------------------
# Side-by-side GT vs Generated
# ---------------------------------------------------------------------------

def plot_comparison(
    gt_2d: np.ndarray,
    gen_2d: np.ndarray,
    *,
    title: str = "",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 4),
    lims: float | None = None,
) -> plt.Figure:
    """
    Two-panel scatter: ground truth (blue) | generated (red).

    Args:
        gt_2d     : (N, 2) ground-truth samples
        gen_2d    : (N, 2) generated samples
        title     : super-title
        save_path : if given, save figure here
        figsize   : figure size in inches
        lims      : if given, sets axis limits to ±lims

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(gt_2d[:, 0],  gt_2d[:, 1],  c="steelblue", **_SCATTER_KW)
    _set_ax(axes[0], "Ground Truth", lims=lims)

    axes[1].scatter(gen_2d[:, 0], gen_2d[:, 1], c="tomato", **_SCATTER_KW)
    _set_ax(axes[1], "Generated", lims=lims)

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Compact 2×2 grid (4 param combos)
# ---------------------------------------------------------------------------

def plot_2x2_grid(
    panels: list[tuple[np.ndarray, str]],
    *,
    suptitle: str = "",
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 9),
    lims: float | None = None,
) -> plt.Figure:
    """
    2×2 grid of scatter plots.

    Args:
        panels   : list of (samples_2d, subtitle) — must have exactly 4 entries
        suptitle : super-title
    """
    assert len(panels) == 4, "Exactly 4 panels required for 2×2 grid"
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for ax, (samp, sub) in zip(axes, panels):
        ax.scatter(samp[:, 0], samp[:, 1], **_SCATTER_KW)
        _set_ax(ax, sub, lims=lims)

    fig.suptitle(suptitle, fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# General N-panel grid
# ---------------------------------------------------------------------------

def plot_grid(
    samples_list: Sequence[np.ndarray],
    titles: Sequence[str],
    *,
    n_cols: int = 4,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    suptitle: str = "",
    lims: float | None = None,
) -> plt.Figure:
    """
    Flexible N-panel scatter grid.

    Args:
        samples_list : list of (N, 2) arrays
        titles       : list of panel titles (same length)
        n_cols       : columns per row
        figsize      : figure size (auto if None)
        save_path    : optional save path
        suptitle     : super-title
        lims         : optional symmetric axis limits
    """
    n = len(samples_list)
    n_rows = (n + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (4.0 * n_cols, 3.8 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat = axes.flatten()

    for i, (samp, ttl) in enumerate(zip(samples_list, titles)):
        flat[i].scatter(samp[:, 0], samp[:, 1], **_SCATTER_KW)
        _set_ax(flat[i], ttl, lims=lims)

    # Hide unused axes
    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Training loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(
    loss_history: list[float] | np.ndarray,
    *,
    title: str = "Training Loss",
    smooth: int | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (8, 3),
) -> plt.Figure:
    """
    Plot (optionally smoothed) training loss.

    Args:
        loss_history : list of per-step loss values
        title        : plot title
        smooth       : smoothing window size (auto if None)
        save_path    : optional save path
    """
    arr = np.array(loss_history, dtype=float)
    if smooth is None:
        smooth = max(1, len(arr) // 200)
    smoothed = np.convolve(arr, np.ones(smooth) / smooth, mode="valid")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(arr,     alpha=0.25, lw=0.8, color="steelblue", label="raw")
    ax.plot(np.arange(len(smoothed)) + smooth // 2, smoothed,
            lw=1.5, color="tomato", label=f"smoothed (w={smooth})")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Step-count comparison (Part 4.1)
# ---------------------------------------------------------------------------

def plot_step_comparison(
    step_results: dict[int, np.ndarray],
    gt_2d: np.ndarray,
    *,
    title: str = "Sampling Quality vs. NFE",
    save_path: str | Path | None = None,
    lims: float | None = None,
) -> plt.Figure:
    """
    Plot generated samples at various ODE step counts, plus ground truth.

    Args:
        step_results : {n_steps: samples_2d}
        gt_2d        : (N, 2) ground truth samples
        title        : super-title
    """
    keys   = sorted(step_results.keys())
    panels = [(step_results[k], f"{k} step{'s' if k > 1 else ''}") for k in keys]
    panels = [("gt", gt_2d, "Ground Truth")] + [(k, v, t) for (k, v, t) in
              [(k, step_results[k], f"{k} step{'s' if k > 1 else ''}") for k in keys]]

    # Reformat
    all_samps  = [gt_2d]  + [step_results[k] for k in keys]
    all_titles = ["Ground Truth"] + [f"{k} step{'s' if k > 1 else ''}" for k in keys]

    n      = len(all_samps)
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.8 * n_rows),
                             squeeze=False)
    flat = axes.flatten()

    colors = ["steelblue"] + ["tomato"] * len(keys)
    for i, (samp, ttl, col) in enumerate(zip(all_samps, all_titles, colors)):
        flat[i].scatter(samp[:, 0], samp[:, 1], c=col, **_SCATTER_KW)
        _set_ax(flat[i], ttl, lims=lims)

    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        save_fig(fig, save_path)

    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Nearest-neighbour distance metric
# ---------------------------------------------------------------------------

def nearest_neighbour_dist(
    gen: np.ndarray,
    gt: np.ndarray,
    *,
    n_subsample: int = 2000,
    rng_seed: int = 0,
) -> float:
    """
    Average nearest-neighbour distance from generated to ground-truth samples.

    Lower = better (generated samples are closer to the data manifold).

    Args:
        gen        : (N, D) generated samples
        gt         : (M, D) ground-truth samples
        n_subsample: maximum points to use (for speed)
        rng_seed   : random seed for subsampling

    Returns:
        float — mean NND
    """
    rng  = np.random.default_rng(rng_seed)
    n_g  = min(n_subsample, len(gen))
    n_gt = min(n_subsample, len(gt))

    gen_s = gen[rng.choice(len(gen), n_g,  replace=False)]
    gt_s  = gt [rng.choice(len(gt),  n_gt, replace=False)]

    # Pairwise L2 distances — shape (n_g, n_gt)
    diff  = gen_s[:, None, :] - gt_s[None, :, :]   # (n_g, n_gt, D)
    dists = np.sqrt((diff ** 2).sum(-1))            # (n_g, n_gt)
    return float(dists.min(axis=1).mean())


# ---------------------------------------------------------------------------
# Quick summary table printer
# ---------------------------------------------------------------------------

def print_results_table(results: dict, datasets: list[str], dims: list[int]) -> None:
    """
    Pretty-print a table of NND metrics for all dataset/dim combinations.

    Args:
        results  : {(dataset, dim, pred_type, loss_type): nnd_value}
        datasets : list of dataset names
        dims     : list of dimensions
    """
    pred_types = ["v", "x"]
    loss_types = ["v", "x"]

    header = f"{'Dataset':12s} {'D':>4s} {'pred\\loss':>10s}  " + \
             "  ".join(f"loss={lt}" for lt in loss_types)
    print(header)
    print("-" * len(header))

    for ds in datasets:
        for dim in dims:
            for pt in pred_types:
                row = f"{ds:12s} {dim:>4d} pred={pt}       "
                for lt in loss_types:
                    key = (ds, dim, pt, lt)
                    val = results.get(key, float("nan"))
                    row += f"  {val:8.4f}"
                print(row)
        print()
