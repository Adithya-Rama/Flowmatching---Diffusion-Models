"""
Sampling procedures for Flow Matching (Parts 1-4).

Euler ODE sampler (Parts 1-3):
  Start from z ~ N(0, I) at t=1 and integrate backward to t ~= 0.

MeanFlow sampler (Part 4):
  Uses a direct mean-velocity model u_theta(z, t, h). Each step jumps from
  time t to r = t - h with z_r = z_t - h * u_theta(z_t, t, h).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

T_EPS: float = 1e-4


# ---------------------------------------------------------------------------
# Euler ODE sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(
    model: nn.Module,
    n_samples: int,
    dim: int,
    *,
    n_steps: int = 50,
    pred_type: str = "v",
    device: str = "cpu",
    t_eps: float = T_EPS,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate samples via Euler integration of the learned ODE.

    For v-prediction:
        v = model(z, t)
    For x-prediction:
        x_hat = model(z, t)
        v = (z_t - x_hat) / t
    """
    pred_type = pred_type.lower()
    if pred_type not in {"v", "x"}:
        raise ValueError(f"Unknown pred_type: {pred_type!r}")

    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    z = torch.randn(n_samples, dim, device=device)
    ts = torch.linspace(1.0, t_eps, n_steps + 1, device=device)

    for i in range(n_steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_curr

        t_batch = t_curr.expand(n_samples)
        pred = model(z, t_batch)

        if pred_type == "v":
            v = pred
        else:
            v = (z - pred) / max(float(t_curr), t_eps)

        z = z + v * dt

    return z.cpu().numpy()


# ---------------------------------------------------------------------------
# MeanFlow sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def meanflow_sample(
    model: nn.Module,
    n_samples: int,
    dim: int,
    *,
    n_steps: int = 1,
    pred_type: str = "v",
    device: str = "cpu",
    t_eps: float = T_EPS,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate samples using a direct mean-velocity MeanFlow model.

    The model predicts:
        model(z_t, t, h) -> u_theta(z_t, t, h)

    The update is:
        z_{t-h} = z_t - h * u_theta(z_t, t, h)
    """
    pred_type = pred_type.lower()
    if pred_type != "v":
        raise ValueError(
            "MeanFlow sampler expects direct mean-velocity pred_type='v'; "
            f"got {pred_type!r}."
        )

    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    z = torch.randn(n_samples, dim, device=device)
    ts = torch.linspace(1.0, t_eps, n_steps + 1, device=device)

    for i in range(n_steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        h_val = t_curr - t_next

        t_batch = t_curr.expand(n_samples)
        h_batch = h_val.expand(n_samples)

        u = model(z, t_batch, h_batch)
        z = z - h_val * u

    return z.cpu().numpy()


# ---------------------------------------------------------------------------
# Convenience wrapper: sample and project to 2D for visualisation
# ---------------------------------------------------------------------------

def sample_and_project(
    model: nn.Module,
    dataset,
    n_samples: int = 2000,
    *,
    n_steps: int = 50,
    pred_type: str = "v",
    model_type: str = "fm",
    device: str = "cpu",
    t_eps: float = T_EPS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample from model and return (generated_2d, ground_truth_2d).
    """
    dim = dataset.dim

    if model_type == "fm":
        gen = euler_sample(
            model, n_samples, dim,
            n_steps=n_steps, pred_type=pred_type,
            device=device, t_eps=t_eps, seed=seed,
        )
    elif model_type == "meanflow":
        gen = meanflow_sample(
            model, n_samples, dim,
            n_steps=n_steps, pred_type=pred_type,
            device=device, t_eps=t_eps, seed=seed,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    gen_2d = dataset.to_2d(gen)

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    gt = dataset.data.numpy()[idxs]
    gt_2d = dataset.to_2d(gt)

    return gen_2d, gt_2d
