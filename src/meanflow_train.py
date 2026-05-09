"""
MeanFlow training loop (Part 4).

This implementation uses the direct mean-velocity parameterization:
    model(z_t, t, h) -> u_theta(z_t, t, h)

For the linear interpolant z_t = (1 - t) * x + t * eps, the instantaneous
velocity is v_true = eps - x. MeanFlow trains a finite-horizon mean velocity
with the identity:
    u_target = v_true - h * d/dt u_theta(z_t(t), t, h(t))

The total derivative uses JVP tangents (dz/dt, dt/dt, dh/dt) = (v_true, 1, 1).
The h tangent is essential because h = t - r and r is held fixed when
differentiating with respect to t.

Quality improvements (v3):
  - Long-horizon MV sampling (long_h_frac=0.5):
      50% random ordered-uniform intervals (existing coverage),
      50% long intervals with r≈t_eps, t∈[0.5, 1−t_eps].
      This ensures the 1-step / large-jump regime is always in the training set.
  - Phased optimisation:
      FM warmup (warmup_frac of steps): lr=lr, FM-only batches.
      MV phase: fresh Adam reset to mv_lr, cosine-decayed to mv_lr_min.
  - EMA of model weights (ema_decay) starting at the MV phase transition.
      The returned model has EMA weights applied.
"""

from __future__ import annotations

import math
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
    long_h_frac: float = 0.5,
    device: str = "cpu",
    t_eps: float = T_EPS,
) -> torch.Tensor:
    """
    Compute the MeanFlow loss.

    Args:
        model       : MeanFlowMLP
        x           : clean data batch, shape (B, D)
        pred_type   : must be 'v' (direct mean-velocity)
        fm_ratio    : fraction of batch used for FM boundary term (h=0)
        long_h_frac : fraction of the MV sub-batch using long-horizon sampling
                      (r≈t_eps, t∈[0.5, 1−t_eps]); remainder uses random intervals
        device      : torch device string
        t_eps       : time clamp floor

    Returns:
        scalar MSE loss
    """
    pred_type = pred_type.lower()
    if pred_type != "v":
        raise ValueError(
            "MeanFlow requires direct mean-velocity pred_type='v'; "
            f"got {pred_type!r}."
        )

    x = x.to(device).float()
    B = x.shape[0]

    split = max(1, min(B, int(B * fm_ratio)))
    losses: list[torch.Tensor] = []

    # ── 1.  FM boundary term  (h = 0) ──────────────────────────────────────
    x_fm  = x[:split]
    B_fm  = x_fm.shape[0]
    if B_fm > 0:
        eps_fm = torch.randn_like(x_fm)
        t_fm   = torch.rand(B_fm, device=device).clamp(t_eps, 1.0 - t_eps)
        t4_fm  = t_fm.view(B_fm, 1)
        z_fm   = (1.0 - t4_fm) * x_fm + t4_fm * eps_fm
        h_zero = torch.zeros(B_fm, device=device)

        pred_fm = model(z_fm, t_fm, h_zero)
        losses.append(F.mse_loss(pred_fm, eps_fm - x_fm))

    # ── 2.  Mean-velocity consistency term  (h > 0) ─────────────────────────
    B_mv = B - split
    if B_mv > 0:
        x_mv   = x[split:]
        eps_mv = torch.randn_like(x_mv)

        lo, hi = t_eps, 1.0 - t_eps

        # Build (t, h) pairs as a mixture of two sampling strategies.
        B_long = max(0, int(B_mv * long_h_frac))
        B_rand = B_mv - B_long

        t_parts: list[torch.Tensor] = []
        h_parts: list[torch.Tensor] = []

        if B_rand > 0:
            # Random ordered-uniform intervals: covers the full (r,t) grid.
            u1 = lo + (hi - lo) * torch.rand(B_rand, device=device)
            u2 = lo + (hi - lo) * torch.rand(B_rand, device=device)
            t_rand = torch.maximum(u1, u2)
            h_rand = (t_rand - torch.minimum(u1, u2)).clamp(min=t_eps)
            t_parts.append(t_rand)
            h_parts.append(h_rand)

        if B_long > 0:
            # Long-horizon: r≈t_eps (near-zero), t∈[0.5, 1−t_eps].
            # This directly covers the 1-step and large-jump regime.
            t_long = 0.5 + (hi - 0.5) * torch.rand(B_long, device=device)
            h_long = (t_long - t_eps).clamp(min=t_eps)
            t_parts.append(t_long)
            h_parts.append(h_long)

        t_mv = torch.cat(t_parts) if len(t_parts) > 1 else t_parts[0]
        h_mv = torch.cat(h_parts) if len(h_parts) > 1 else h_parts[0]

        t4     = t_mv.view(B_mv, 1)
        h4     = h_mv.view(B_mv, 1)
        z_mv   = (1.0 - t4) * x_mv + t4 * eps_mv
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
# LR schedule helper
# ---------------------------------------------------------------------------

def _cosine_lr(
    step: int,
    warmup_steps: int,
    n_steps: int,
    mv_lr: float,
    mv_lr_min: float,
) -> float:
    """Cosine-annealed LR for the MV phase: mv_lr → mv_lr_min."""
    mv_step  = step - warmup_steps
    mv_total = max(1, n_steps - warmup_steps)
    t = mv_step / mv_total
    return mv_lr_min + 0.5 * (mv_lr - mv_lr_min) * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_meanflow(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    n_steps: int = 25_000,
    lr: float = 1e-3,
    mv_lr: float = 3e-4,
    mv_lr_min: float = 1e-5,
    pred_type: str = "v",
    fm_ratio: float = 0.5,
    long_h_frac: float = 0.5,
    warmup_frac: float = 0.4,
    ema_decay: float = 0.999,
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

    Two-phase schedule
    ──────────────────
    Phase 1 — FM warmup  (steps 0 … warmup_steps−1)
        lr=lr, fm_ratio=1.0 (FM only, no MV consistency).

    Phase 2 — MV consistency  (steps warmup_steps … n_steps−1)
        Adam reset to mv_lr (fresh momentum), cosine-decayed to mv_lr_min.
        Mixed long-horizon + random-interval MV batches (long_h_frac).
        EMA of model weights with decay=ema_decay.

    The returned model has EMA weights applied (ema_decay > 0).

    Args:
        model            : MeanFlowMLP to train
        dataloader       : DataLoader over ToyDiffusionDataset
        n_steps          : total gradient steps
        lr               : FM-warmup Adam learning rate
        mv_lr            : initial Adam LR for MV phase
        mv_lr_min        : cosine floor LR for MV phase
        pred_type        : must be 'v'
        fm_ratio         : FM-term fraction during MV phase
        long_h_frac      : long-horizon fraction of MV batch
        warmup_frac      : fraction of n_steps for FM-only warmup
        ema_decay        : EMA decay factor; 0 disables EMA
        t_eps            : t clamped to [t_eps, 1−t_eps]
        device           : torch device
        checkpoint_dir   : directory for saving checkpoints (None = off)
        checkpoint_every : save every N steps
        resume           : auto-resume from latest.pt
        run_name         : checkpoint sub-directory name
        log_every        : tqdm progress-bar update frequency

    Returns:
        (model_with_ema_weights, loss_history)
    """
    pred_type = pred_type.lower()
    if pred_type != "v":
        raise ValueError(
            "train_meanflow only supports direct mean-velocity pred_type='v'; "
            f"got {pred_type!r}."
        )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    warmup_steps     = int(n_steps * warmup_frac)
    start_step       = 0
    loss_history: list[float] = []
    mv_phase_started = False
    ema_shadow: dict | None = None

    # ── Resume ───────────────────────────────────────────────────────────────
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / run_name / "latest.pt"
        if resume and ckpt_path.exists():
            print(f"[Resume] Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step       = ckpt["step"]
            loss_history     = ckpt.get("loss_history", [])
            mv_phase_started = ckpt.get("mv_phase_started", False)
            ema_raw          = ckpt.get("ema_shadow")
            if ema_raw is not None:
                ema_shadow = {k: v.float().to(device) for k, v in ema_raw.items()}
            print(f"[Resume] Continuing from step {start_step}/{n_steps}")

    if start_step >= n_steps:
        print(f"[Skip] MeanFlow already trained to {start_step} steps.")
        if ema_shadow is not None:
            model.load_state_dict(ema_shadow)
        return model, loss_history

    # If resuming mid-MV-phase, restore the cosine LR for this step.
    if mv_phase_started:
        for pg in optimizer.param_groups:
            pg["lr"] = _cosine_lr(start_step, warmup_steps, n_steps, mv_lr, mv_lr_min)

    # ── Training loop ────────────────────────────────────────────────────────
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

        # ── Phase transition: FM warmup → MV consistency ─────────────────
        if not mv_phase_started and step >= warmup_steps:
            mv_phase_started = True
            tqdm.write(
                f"[MeanFlow] Switching to MV phase at step {step} "
                f"— resetting Adam  lr={mv_lr:.2e} → {mv_lr_min:.2e} (cosine)"
            )
            # Fresh Adam: clear accumulated FM-phase momentum.
            optimizer = torch.optim.Adam(model.parameters(), lr=mv_lr)
            # Initialise EMA from the FM-warmed model weights.
            if ema_decay > 0.0:
                ema_shadow = {
                    k: v.float().clone()
                    for k, v in model.state_dict().items()
                }

        # ── Cosine LR decay in MV phase ──────────────────────────────────
        if mv_phase_started:
            for pg in optimizer.param_groups:
                pg["lr"] = _cosine_lr(step, warmup_steps, n_steps, mv_lr, mv_lr_min)

        # ── Loss ─────────────────────────────────────────────────────────
        current_fm_ratio = 1.0 if not mv_phase_started else fm_ratio
        current_long_h   = long_h_frac if mv_phase_started else 0.0

        optimizer.zero_grad()
        loss = meanflow_loss(
            model, x,
            pred_type=pred_type,
            fm_ratio=current_fm_ratio,
            long_h_frac=current_long_h,
            t_eps=t_eps,
            device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # ── EMA update ───────────────────────────────────────────────────
        if ema_shadow is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_shadow[k].mul_(ema_decay).add_((1.0 - ema_decay) * v.float())

        lv = loss.item()
        loss_history.append(lv)

        if step % log_every == 0:
            pbar.set_postfix({
                "loss": f"{lv:.5f}",
                "phase": "mv" if mv_phase_started else "fm",
            })

        if checkpoint_dir is not None and (step + 1) % checkpoint_every == 0:
            _save_ckpt(
                model, optimizer, step + 1, loss_history,
                checkpoint_dir, run_name,
                mv_phase_started=mv_phase_started,
                ema_shadow=ema_shadow,
            )

    # ── Final checkpoint ─────────────────────────────────────────────────────
    if checkpoint_dir is not None:
        _save_ckpt(
            model, optimizer, n_steps, loss_history,
            checkpoint_dir, run_name,
            mv_phase_started=mv_phase_started,
            ema_shadow=ema_shadow,
        )

    # ── Apply EMA weights to the returned model ──────────────────────────────
    if ema_shadow is not None:
        model.load_state_dict(ema_shadow)
        tqdm.write("[MeanFlow] EMA weights applied to returned model.")

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
    *,
    mv_phase_started: bool = False,
    ema_shadow: dict | None = None,
) -> None:
    path = Path(checkpoint_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "step":             step,
            "loss_history":     loss_history,
            "mv_phase_started": mv_phase_started,
            "ema_shadow":       (
                {k: v.cpu() for k, v in ema_shadow.items()}
                if ema_shadow is not None else None
            ),
        },
        path / "latest.pt",
    )
    tqdm.write(f"[Checkpoint] MeanFlow saved at step {step} -> {path / 'latest.pt'}")
