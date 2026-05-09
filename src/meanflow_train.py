"""
MeanFlow training loop (Part 4).

MeanFlow (arXiv 2505.13447) trains a model to predict the mean velocity
u_θ(z_t, t, h) over a time horizon h = t − r, enabling single-step generation.

Design: model uses x-prediction (outputs x̂) for stability at high D.
Mean velocity derives as  u_θ = (z_t − x̂_θ) / t.

Training objective
──────────────────
Two terms per mini-batch (split controlled by `fm_ratio`):

1. FM term  (h = 0):  MSE( model(z_t, t, 0), x )
   Boundary condition enforcing  x̂(h=0) = x  (standard x-prediction FM).

2. MV consistency (h > 0):  MSE( model(z_t, t, h), x̂_target )

   Self-consistency in x-prediction space (derived from Eq. 3 of paper):

       u_θ(h) = sg[ v_θ(0) + h · D_t u_θ(h) ]
   =>  x̂_θ(h) = sg[ x̂_θ(0) − t · h · D_t[(z_t − x̂_θ(h)) / t] ]

   where D_t is the total time derivative along the trajectory dz/dt = v_true:
       D_t[(z − x̂) / t] = JVP( λ(z,t) → (z − model(z,t,h)) / t,
                                (z_t, t), (v_true, 1) )
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
    # 1.  FM term  (h = 0)
    #     model(z_t, t, 0) predicts x̂ = x (x-prediction, stable at high D)
    # ══════════════════════════════════════════════════════════════════════
    x_fm   = x[:split]
    B_fm   = x_fm.shape[0]
    eps_fm = torch.randn_like(x_fm)
    t_fm   = torch.rand(B_fm, device=device).clamp(t_eps, 1.0 - t_eps)
    t4_fm  = t_fm.view(B_fm, 1)
    z_fm   = (1.0 - t4_fm) * x_fm + t4_fm * eps_fm
    h_zero = torch.zeros(B_fm, device=device)

    pred_fm   = model(z_fm, t_fm, h_zero)
    target_fm = x_fm if pred_type == "x" else (eps_fm - x_fm)
    loss_fm   = F.mse_loss(pred_fm, target_fm)
    losses.append(loss_fm)

    # ══════════════════════════════════════════════════════════════════════
    # 2.  MV consistency (h > 0)
    #
    #  For x-prediction:
    #   Mean velocity  u_θ = (z_t − x̂_θ) / t
    #   Self-consistency:
    #     x̂_θ(h) = sg[ x̂_θ(0)  −  t · h · D_t[(z − x̂_θ(h)) / t] ]
    #   where D_t is the JVP along the training trajectory (dz/dt = v_true = ε−x):
    #     D_t[(z − x̂) / t] = JVP( (z,t) → (z − model(z,t,h)) / t,
    #                              (z_t, t), (v_true, 1) )
    #
    #  Sanity check (optimal/straight-line flow):
    #   x̂(z_t, t, h) = x for all h  →  D_t[(z−x)/t] = D_t[v_true] = 0
    #   target = x̂(0) − 0 = x  ✓  (trivially satisfied, no degenerate solution)
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
            # model(z, t, h>0) — prediction at h>0, tracked for backward
            pred_h = model(z_mv, t_mv, h_mv)

            # x̂_0 = model(z, t, 0) — h=0 x-prediction (stop-gradient, into target)
            h_zero_mv = torch.zeros(B_mv, device=device)
            with torch.no_grad():
                xhat_0 = model(z_mv, t_mv, h_zero_mv)

            if pred_type == "x":
                # Trajectory tangent: dz_t/dt = eps - x
                v_true  = (eps_mv - x_mv).detach()
                ones_t  = torch.ones(B_mv, device=device)
                h_mv_d  = h_mv.detach()
                z_mv_d  = z_mv.detach()
                t_mv_d  = t_mv.detach()

                # JVP of mean-velocity function (z,t) → (z − x̂(z,t,h)) / t
                def _u_fn(z_in, t_in):
                    x_hat = model(z_in, t_in, h_mv_d)
                    return (z_in - x_hat) / t_in.unsqueeze(-1)

                _, du_dt = torch.func.jvp(
                    _u_fn, (z_mv_d, t_mv_d), (v_true, ones_t)
                )

                # x̂_target = sg[ x̂_0 − t · h · D_t u_θ ]
                h4      = h_mv.view(B_mv, 1)
                target  = (xhat_0 - t4_mv * h4 * du_dt.detach()).detach()

            else:  # v-prediction: model outputs velocity directly
                v_true  = (eps_mv - x_mv).detach()
                ones_t  = torch.ones(B_mv, device=device)
                h_mv_d  = h_mv.detach()
                z_mv_d  = z_mv.detach()
                t_mv_d  = t_mv.detach()

                def _fn(z_in, t_in):
                    return model(z_in, t_in, h_mv_d)

                _, du_dt = torch.func.jvp(
                    _fn, (z_mv_d, t_mv_d), (v_true, ones_t)
                )

                h4     = h_mv.view(B_mv, 1)
                target = (xhat_0 + h4 * du_dt.detach()).detach()

            loss_mv = F.mse_loss(pred_h, target)

        except Exception as _e:
            tqdm.write(f"[MeanFlow] JVP fallback: {_e}")
            pred_fb   = model(z_mv, t_mv, h_mv)
            target_fb = x_mv if pred_type == "x" else (eps_mv - x_mv)
            loss_mv   = F.mse_loss(pred_fb, target_fb)

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
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step":            step,
        "loss_history":    loss_history,
    }, path / "latest.pt")
    tqdm.write(f"[Checkpoint] MeanFlow saved at step {step}  →  {path / 'latest.pt'}")
