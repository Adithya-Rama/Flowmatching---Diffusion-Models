"""
MeanFlow training loop (Part 4).

This implementation uses the direct mean-velocity parameterization:
    model(z_t, t, h) -> u_theta(z_t, t, h)

For the linear interpolant z_t = (1 - t) * x + t * eps, the instantaneous
velocity is v_true = eps - x. MeanFlow trains a finite-horizon mean velocity
with the identity:
    u_target = v_true - h * d/dt u_theta(z_t(t), t, h(t))

The total derivative is computed with JVP tangents (dz/dt, dt/dt, dh/dt) =
(v_true, 1, 1). The h tangent is essential because h = t - r and r is held
fixed when differentiating with respect to t.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

T_EPS: float = 1e-4


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def meanflow_loss(
    model: nn.Module,
    x: torch.Tensor,
    *,
    pred_type: str = "v",
    fm_ratio: float = 0.5,
    device: str = "cpu",
    t_eps: float = T_EPS,
) -> torch.Tensor:
    pred_type = pred_type.lower()
    if pred_type != "v":
        raise ValueError(
            "MeanFlow is configured for direct mean-velocity prediction; "
            f"expected pred_type='v', got {pred_type!r}."
        )
    if not 0.0 <= fm_ratio <= 1.0:
        raise ValueError(f"fm_ratio must be in [0, 1], got {fm_ratio}.")
    if not 0.0 < t_eps < 0.5:
        raise ValueError(f"t_eps must be in (0, 0.5), got {t_eps}.")

    x = x.to(device).float()
    B = x.shape[0]

    split = max(1, min(B, int(B * fm_ratio)))
    losses: list[torch.Tensor] = []

    # 1. FM boundary term (h = 0): u_theta(z_t, t, 0) should match v_true.
    x_fm = x[:split]
    B_fm = x_fm.shape[0]
    if B_fm > 0:
        eps_fm = torch.randn_like(x_fm)
        t_fm = torch.rand(B_fm, device=device).clamp(t_eps, 1.0 - t_eps)
        t4_fm = t_fm.view(B_fm, 1)
        z_fm = (1.0 - t4_fm) * x_fm + t4_fm * eps_fm
        h_zero = torch.zeros(B_fm, device=device)

        pred_fm = model(z_fm, t_fm, h_zero)
        target_fm = eps_fm - x_fm
        losses.append(F.mse_loss(pred_fm, target_fm))

    # 2. Mean-velocity consistency term (h > 0): u = v_true - h * d_t u.
    if B - split > 0:
        x_mv = x[split:]
        B_mv = x_mv.shape[0]
        eps_mv = torch.randn_like(x_mv)

        # Sample r <= t in [t_eps, 1 - t_eps] by ordering two uniforms.
        lo, hi = t_eps, 1.0 - t_eps
        u1 = lo + (hi - lo) * torch.rand(B_mv, device=device)
        u2 = lo + (hi - lo) * torch.rand(B_mv, device=device)
        t_mv = torch.maximum(u1, u2)
        r_mv = torch.minimum(u1, u2)
        h_mv = (t_mv - r_mv).clamp(min=t_eps)

        t4 = t_mv.view(B_mv, 1)
        h4 = h_mv.view(B_mv, 1)
        z_mv = (1.0 - t4) * x_mv + t4 * eps_mv
        v_true = (eps_mv - x_mv).detach()

        pred_h = model(z_mv, t_mv, h_mv)

        def _u_fn(z_in: torch.Tensor, t_in: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
            return model(z_in, t_in, h_in)

        ones = torch.ones(B_mv, device=device)
        _, du_dt = torch.func.jvp(
            _u_fn,
            (z_mv.detach(), t_mv.detach(), h_mv.detach()),
            (v_true, ones, ones),
        )

        target = (v_true - h4 * du_dt).detach()
        losses.append(F.mse_loss(pred_h, target))

    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_meanflow(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    n_steps: int = 25_000,
    lr: float = 1e-3,
    pred_type: str = "v",
    fm_ratio: float = 0.5,
    warmup_frac: float = 0.0,
    t_eps: float = T_EPS,
    device: str = "cpu",
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 5_000,
    resume: bool = True,
    run_name: str = "meanflow",
    log_every: int = 500,
) -> tuple[nn.Module, list[float]]:
    """
    Train a direct mean-velocity MeanFlowMLP.

    Args:
        model           : MeanFlowMLP to train
        dataloader      : DataLoader over ToyDiffusionDataset
        n_steps         : total gradient steps
        lr              : Adam learning rate
        pred_type       : must be 'v'
        fm_ratio        : fraction of each batch for standard FM (h=0) after warmup
        warmup_frac     : fraction of steps to train FM-only before MV consistency
        t_eps           : clamp t to [t_eps, 1 - t_eps] for numerical stability
        device          : torch device
        checkpoint_dir  : directory for saving checkpoints
        checkpoint_every: save frequency in steps
        resume          : auto-resume from existing checkpoint
        run_name        : checkpoint sub-directory name
        log_every       : progress-bar update frequency

    Returns:
        (trained model, loss_history)
    """
    pred_type = pred_type.lower()
    if pred_type != "v":
        raise ValueError(
            "train_meanflow only supports direct mean-velocity pred_type='v'; "
            f"got {pred_type!r}."
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_step: int = 0
    loss_history: list[float] = []
    warmup_steps = int(n_steps * warmup_frac)

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / run_name / "latest.pt"
        if resume and ckpt_path.exists():
            print(f"[Resume] Loading MeanFlow checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = ckpt["step"]
            loss_history = ckpt.get("loss_history", [])
            print(f"[Resume] Continuing from step {start_step}/{n_steps}")

    if start_step >= n_steps:
        print(f"[Skip] MeanFlow already trained to {start_step} steps.")
        return model, loss_history

    model.train()
    data_iter = iter(dataloader)

    pbar = tqdm(
        range(start_step, n_steps),
        desc=f"MeanFlow/{run_name}",
        dynamic_ncols=True,
        leave=True,
    )

    for step in pbar:
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x = next(data_iter)

        # Pure FM warmup: first learn the h=0 velocity boundary cleanly.
        current_fm_ratio = 1.0 if step < warmup_steps else fm_ratio

        optimizer.zero_grad()
        loss = meanflow_loss(
            model, x,
            pred_type=pred_type,
            fm_ratio=current_fm_ratio,
            t_eps=t_eps,
            device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        lv = loss.item()
        loss_history.append(lv)

        if step % log_every == 0:
            pbar.set_postfix({"loss": f"{lv:.5f}"})

        if (
            checkpoint_dir is not None
            and (step + 1) % checkpoint_every == 0
        ):
            _save_ckpt(
                model, optimizer, step + 1, loss_history,
                checkpoint_dir, run_name,
            )

    if checkpoint_dir is not None:
        _save_ckpt(
            model, optimizer, n_steps, loss_history,
            checkpoint_dir, run_name,
        )

    return model, loss_history


# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------

def _save_ckpt(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss_history: list[float],
    checkpoint_dir: str | Path,
    run_name: str,
) -> None:
    path = Path(checkpoint_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "loss_history": loss_history,
    }, path / "latest.pt")
    tqdm.write(f"[Checkpoint] MeanFlow saved at step {step} -> {path / 'latest.pt'}")
