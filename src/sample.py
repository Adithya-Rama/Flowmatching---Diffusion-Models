"""
Sampling procedures for Flow Matching (Parts 1 – 4).

Euler ODE sampler  (Parts 1–3):
  Start from z ~ N(0, I) at t=1, integrate backwards to t≈0
  using the predicted velocity field.

MeanFlow sampler   (Part 4):
  Uses the mean-velocity model u_θ(z, t, h) with variable step counts.

Reference: SiT (Scalable Interpolant Transformers) for Euler ODE details.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

T_EPS: float = 1e-4   # keep t away from exact 0


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

    The ODE is:  dz/dt = v(z, t)   integrated from t=1 to t≈0.
    Stepping direction is backward (Δt < 0).

    For v-prediction:
        v = model(z, t)
    For x-prediction:
        x̂ = model(z, t)
        v = (z_t − x̂) / t        [from z_t = x + t·v  ⟹  v = (z_t−x)/t]

    Args:
        model     : trained FlowMatchingMLP with model(z, t) interface
        n_samples : number of samples to generate
        dim       : ambient data dimension D
        n_steps   : number of Euler integration steps
        pred_type : 'v' or 'x'
        device    : torch device string
        t_eps     : minimum t value (stopping point of integration)
        seed      : optional random seed for reproducibility

    Returns:
        samples : np.ndarray of shape (n_samples, dim)
    """
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    # ── Initialise z ~ N(0, I) at t=1
    z = torch.randn(n_samples, dim, device=device)

    # ── Time grid: t decreases from 1.0 to t_eps in n_steps steps
    ts = torch.linspace(1.0, t_eps, n_steps + 1, device=device)

    for i in range(n_steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        dt     = t_next - t_curr          # negative (backward in time)

        t_batch = t_curr.expand(n_samples)
        pred    = model(z, t_batch)

        if pred_type == "v":
            v = pred
        elif pred_type == "x":
            # v = (z_t − x̂) / t,  guard against small t
            t_val = float(t_curr)
            v = (z - pred) / max(t_val, t_eps)
        else:
            raise ValueError(f"Unknown pred_type: '{pred_type}'")

        # Euler step:  z ← z + v · Δt
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
    device: str = "cpu",
    t_eps: float = T_EPS,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate samples using the MeanFlow model.

    MeanFlow predicts the mean velocity u_θ(z_t, t, h) over the horizon h = t − r.
    For a single step: h = 1 − t_eps ≈ 1, so z_r = z_t − h · u_θ directly jumps
    from near-noise to near-data.

    For multi-step (n_steps > 1), the horizon shrinks proportionally.

    Update rule per step:
        h    = t_curr − t_next         (current horizon)
        u    = model(z, t_curr, h)     (mean velocity over horizon)
        z   ← z − h · u               (jump to t_next = t_curr − h)

    Args:
        model     : trained MeanFlowMLP with model(z, t, h) interface
        n_samples : number of samples to generate
        dim       : ambient data dimension D
        n_steps   : 1 (single-step) or more
        device    : torch device string
        t_eps     : stopping time
        seed      : optional random seed

    Returns:
        samples : np.ndarray of shape (n_samples, dim)
    """
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    # ── Initialise z ~ N(0, I) at t=1
    z = torch.randn(n_samples, dim, device=device)

    # ── Time grid
    ts = torch.linspace(1.0, t_eps, n_steps + 1, device=device)

    for i in range(n_steps):
        t_curr = ts[i]
        t_next = ts[i + 1]
        h_val  = t_curr - t_next          # horizon (positive)

        t_batch = t_curr.expand(n_samples)
        h_batch = h_val.expand(n_samples)

        # u_θ(z_t, t, h) — mean velocity over the horizon [t−h, t]
        u = model(z, t_batch, h_batch)

        # Jump:  z_{t−h} = z_t − h · u
        z = z - h_val * u

    return z.cpu().numpy()


# ---------------------------------------------------------------------------
# Convenience wrapper: sample and project to 2D for visualisation
# ---------------------------------------------------------------------------

def sample_and_project(
    model: nn.Module,
    dataset,                 # ToyDiffusionDataset instance (for to_2d)
    n_samples: int = 2000,
    *,
    n_steps: int = 50,
    pred_type: str = "v",
    model_type: str = "fm",  # 'fm' or 'meanflow'
    device: str = "cpu",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample from model and return (generated_2d, ground_truth_2d).

    Works for both FlowMatchingMLP (model_type='fm') and
    MeanFlowMLP (model_type='meanflow').
    """
    dim = dataset.dim

    if model_type == "fm":
        gen = euler_sample(
            model, n_samples, dim,
            n_steps=n_steps, pred_type=pred_type,
            device=device, seed=seed,
        )
    elif model_type == "meanflow":
        gen = meanflow_sample(
            model, n_samples, dim,
            n_steps=n_steps, device=device, seed=seed,
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")

    gen_2d = dataset.to_2d(gen)

    # Ground truth: random subset of training data projected to 2D
    rng  = np.random.default_rng(0)
    idxs = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    gt   = dataset.data.numpy()[idxs]
    gt_2d = dataset.to_2d(gt)

    return gen_2d, gt_2d
