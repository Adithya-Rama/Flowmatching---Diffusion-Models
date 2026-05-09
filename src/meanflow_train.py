"""
MeanFlow training loop (Part 4).

MeanFlow (arXiv 2505.13447) trains a model to predict the *mean velocity*
u_θ(z_t, t, h) over a time horizon h = t − r.

The model ALWAYS outputs mean velocity (not x̂). pred_type only controls
how the FM boundary condition target is computed.

Training objective
──────────────────
Two terms are mixed in each mini-batch (split controlled by `fm_ratio`):

1. Standard FM term  (h = 0, 50 % of batch by default)
   Enforces  u_θ(z_t, t, 0) = v(z_t, t):
     - x-pred:  v_target = (z_t − x) / t
     - v-pred:  v_target = ε − x

2. Mean-velocity consistency term  (h > 0, remaining 50 %)
   Self-consistency condition (Eq. 3 of the MeanFlow paper):

       u_θ(z_t, t, h) = sg[ v_θ(z_t, t) + h · D_t u_θ(z_t, t, h) ]

   where:
     - v_θ(z_t, t) = u_θ(z_t, t, 0)   (model's own h=0 prediction)
     - D_t u_θ = total time derivative along the ODE trajectory
               = JVP(u_θ(·, ·, h), (z_t, t), (v_true, 1))
               with tangents (dz/dt, dt/dt) = (v_true, 1)
     - sg(·) = stop-gradient (detach)
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
    pred_type: str = "x",
    fm_ratio: float = 0.5,
    device: str = "cpu",
    t_eps: float = T_EPS,
) -> torch.Tensor:
    x    = x.to(device).float()
    B, D = x.shape

    split  = max(1, int(B * fm_ratio))
    losses = []

    # ══════════════════════════════════════════════════════════════════════
    # 1.  Standard FM term  (h = 0)
    #     Enforces boundary condition: u_θ(z_t, t, 0) = v(z_t, t)
    #     Model output = mean velocity; target = instantaneous velocity.
    # ══════════════════════════════════════════════════════════════════════
    x_fm   = x[:split]
    B_fm   = x_fm.shape[0]
    eps_fm = torch.randn_like(x_fm)
    t_fm   = torch.rand(B_fm, device=device).clamp(t_eps, 1.0 - t_eps)
    t4_fm  = t_fm.view(B_fm, 1)
    z_fm   = (1.0 - t4_fm) * x_fm + t4_fm * eps_fm
    h_zero = torch.zeros(B_fm, device=device)

    pred_fm = model(z_fm, t_fm, h_zero)
    if pred_type == "x":
        v_target_fm = (z_fm - x_fm) / t4_fm   # v = (z_t − x) / t
    else:
        v_target_fm = eps_fm - x_fm            # v = ε − x
    loss_fm = F.mse_loss(pred_fm, v_target_fm)
    losses.append(loss_fm)

    # ══════════════════════════════════════════════════════════════════════
    # 2.  Mean-velocity consistency term  (h > 0)
    #     Self-consistency: u_θ(z_t, t, h) = sg[ v_θ(z_t,t) + h·D_t u_θ ]
    #     D_t u_θ = JVP of u_θ(z_t(t), t, h) w.r.t. (z_t, t)
    #              with tangents (v_true, 1)  — total time derivative
    # ══════════════════════════════════════════════════════════════════════
    if B - split > 0:
        x_mv   = x[split:]
        B_mv   = x_mv.shape[0]
        eps_mv = torch.randn_like(x_mv)
        r_mv   = torch.rand(B_mv, device=device).clamp(t_eps, 1.0 - t_eps)
        delta  = torch.rand(B_mv, device=device) * (1.0 - r_mv - t_eps)
        t_mv   = (r_mv + delta).clamp(t_eps, 1.0 - t_eps)
        h_mv   = (t_mv - r_mv).clamp(min=t_eps)
        t4_mv  = t_mv.view(B_mv, 1)
        z_mv   = (1.0 - t4_mv) * x_mv + t4_mv * eps_mv

        try:
            # Prediction at h>0 — kept in graph for backward
            u_pred = model(z_mv, t_mv, h_mv)

            # Instantaneous velocity at h=0: v_θ(z_t, t) = u_θ(z_t, t, 0)
            # Goes into target → stop-gradient via no_grad
            h_zero_mv = torch.zeros(B_mv, device=device)
            with torch.no_grad():
                v_hat_0 = model(z_mv, t_mv, h_zero_mv)

            # True velocity tangent dz_t/dt = v = ε − x
            if pred_type == "x":
                v_tan = (z_mv - x_mv) / t4_mv
            else:
                v_tan = eps_mv - x_mv

            # JVP: D_t u_θ(z_t(t), t, h) with h fixed
            # Differentiate w.r.t. (z_t, t), tangents = (v_tan, 1)
            h_mv_d  = h_mv.detach()
            z_mv_d  = z_mv.detach()
            t_mv_d  = t_mv.detach()
            v_tan_d = v_tan.detach()
            ones_t  = torch.ones(B_mv, device=device)

            def _fn(z_in, t_in):
                return model(z_in, t_in, h_mv_d)

            _, du_dt = torch.func.jvp(_fn, (z_mv_d, t_mv_d), (v_tan_d, ones_t))

            # Target: sg[ v_θ(z_t, t) + h · D_t u_θ ]
            h4     = h_mv.view(B_mv, 1)
            target = (v_hat_0 + h4 * du_dt.detach()).detach()
            loss_mv = F.mse_loss(u_pred, target)

        except Exception as _e:
            tqdm.write(f"[MeanFlow] JVP fallback: {_e}")
            pred_fb = model(z_mv, t_mv, h_mv)
            if pred_type == "x":
                v_fb = (z_mv - x_mv) / t4_mv
            else:
                v_fb = eps_mv - x_mv
            loss_mv = F.mse_loss(pred_fb, v_fb)

        losses.append(loss_mv)

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
    pred_type: str = "x",
    fm_ratio: float = 0.5,
    device: str = "cpu",
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 5_000,
    resume: bool = True,
    run_name: str = "meanflow",
    log_every: int = 500,
) -> tuple[nn.Module, list[float]]:
    """
    Train a MeanFlowMLP.

    Args:
        model           : MeanFlowMLP to train
        dataloader      : DataLoader over ToyDiffusionDataset
        n_steps         : total gradient steps
        lr              : Adam learning rate
        pred_type       : best prediction type from Part 2 (default 'x')
        fm_ratio        : fraction of each batch for standard FM (h=0)
        device          : torch device
        checkpoint_dir  : directory for saving checkpoints
        checkpoint_every: save frequency in steps
        resume          : auto-resume from existing checkpoint
        run_name        : checkpoint sub-directory name
        log_every       : progress-bar update frequency

    Returns:
        (trained model, loss_history)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_step: int = 0
    loss_history: list[float] = []

    # ── Resume ───────────────────────────────────────────────────────────
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / run_name / "latest.pt"
        if resume and ckpt_path.exists():
            print(f"[Resume] Loading MeanFlow checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step   = ckpt["step"]
            loss_history = ckpt.get("loss_history", [])
            print(f"[Resume] Continuing from step {start_step}/{n_steps}")

    if start_step >= n_steps:
        print(f"[Skip] MeanFlow already trained to {start_step} steps.")
        return model, loss_history

    # ── Training loop ────────────────────────────────────────────────────
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

        optimizer.zero_grad()
        loss = meanflow_loss(
            model, x,
            pred_type=pred_type,
            fm_ratio=fm_ratio,
            device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # add this
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
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step":            step,
        "loss_history":    loss_history,
    }, path / "latest.pt")
    tqdm.write(f"[Checkpoint] MeanFlow saved at step {step}  →  {path / 'latest.pt'}")
