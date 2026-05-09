"""
MeanFlow training loop (Part 4).

MeanFlow (arXiv 2505.13447) — paper Eq. 11 target:
    u_target = v(z_t, t)  −  h · D_t[u_θ(z_t, t, h)]
where v(z_t, t) is the instantaneous velocity.  During training on the linear
interpolant z_t = (1−t)x + tε, that velocity is exactly  v_true = ε − x  (ground
truth, NOT the model's h=0 output).  Using the model's own h=0 prediction as
the target is a self-referential bootstrap that is unstable.

x-prediction MeanFlow (best param from Part 2):
    Mean velocity  u_θ = (z_t − x̂_θ)/t.  Plugging into Eq. 11 and using the
    identity  z_t − t·v_true = x  (linear interpolant) gives:

        x̂_target = x  +  t·h · D_t[(z_t − x̂_θ(h)) / t]

    where D_t is the total time-derivative along dz/dt = v_true, computed via
    JVP( λ(z,t)→(z − model(z,t,h))/t, (z_t,t), (v_true, 1) ).

Sanity check (optimal straight-line flow x̂_θ ≡ x):
    D_t[(z−x)/t] = D_t[v_true] = 0  ⇒  target = x  ✓

v-prediction MeanFlow (else branch):
    v̂_target = v_true  −  h · D_t v̂_θ(z_t, t, h)

Training objective: half the batch trains the FM boundary (h=0); the other half
trains MV consistency at h>0 (controlled by `fm_ratio`).
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

    # ── 1. FM term (h = 0): boundary condition  x̂(h=0) = x  (or v̂(h=0) = v_true)
    x_fm = x[:split]
    B_fm = x_fm.shape[0]
    if B_fm > 0:
        eps_fm = torch.randn_like(x_fm)
        t_fm   = torch.rand(B_fm, device=device).clamp(t_eps, 1.0 - t_eps)
        t4_fm  = t_fm.view(B_fm, 1)
        z_fm   = (1.0 - t4_fm) * x_fm + t4_fm * eps_fm
        h_zero = torch.zeros(B_fm, device=device)

        pred_fm   = model(z_fm, t_fm, h_zero)
        target_fm = x_fm if pred_type == "x" else (eps_fm - x_fm)
        losses.append(F.mse_loss(pred_fm, target_fm))

    # ── 2. MV consistency (h > 0): paper Eq. 11 with ground-truth v_true
    if B - split > 0:
        x_mv = x[split:]
        B_mv = x_mv.shape[0]
        eps_mv = torch.randn_like(x_mv)

        # Sample r ≤ t in [t_eps, 1−t_eps] by ordering two uniforms
        u1   = torch.rand(B_mv, device=device)
        u2   = torch.rand(B_mv, device=device)
        t_mv = torch.maximum(u1, u2).clamp(t_eps, 1.0 - t_eps)
        r_mv = torch.minimum(u1, u2).clamp(t_eps, 1.0 - t_eps)
        h_mv = (t_mv - r_mv).clamp(min=t_eps)

        t4 = t_mv.view(B_mv, 1)
        h4 = h_mv.view(B_mv, 1)
        z_mv   = (1.0 - t4) * x_mv + t4 * eps_mv
        v_true = (eps_mv - x_mv).detach()

        pred_h = model(z_mv, t_mv, h_mv)  # tracked for backward

        ones_t = torch.ones(B_mv, device=device)
        h_mv_d = h_mv.detach()
        z_mv_d = z_mv.detach()
        t_mv_d = t_mv.detach()

        try:
            if pred_type == "x":
                # u_θ = (z − x̂_θ)/t.  x̂_target = x + t·h·D_t[u_θ]
                def _u_fn(z_in, t_in):
                    x_hat = model(z_in, t_in, h_mv_d)
                    return (z_in - x_hat) / t_in.unsqueeze(-1)

                _, du_dt = torch.func.jvp(
                    _u_fn, (z_mv_d, t_mv_d), (v_true, ones_t)
                )
                target = (x_mv + t4 * h4 * du_dt).detach()

            else:  # v-prediction: v̂_target = v_true − h·D_t v̂_θ
                def _v_fn(z_in, t_in):
                    return model(z_in, t_in, h_mv_d)

                _, dv_dt = torch.func.jvp(
                    _v_fn, (z_mv_d, t_mv_d), (v_true, ones_t)
                )
                target = (v_true - h4 * dv_dt).detach()

            losses.append(F.mse_loss(pred_h, target))

        except Exception as _e:
            tqdm.write(f"[MeanFlow] JVP fallback: {_e}")
            target_fb = x_mv if pred_type == "x" else v_true
            losses.append(F.mse_loss(pred_h, target_fb))

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
    Train a MeanFlowMLP.

    Args:
        model           : MeanFlowMLP to train
        dataloader      : DataLoader over ToyDiffusionDataset
        n_steps         : total gradient steps
        lr              : Adam learning rate
        pred_type       : 'v' or 'x'
        fm_ratio        : fraction of each batch for standard FM (h=0) after warmup
        warmup_frac     : fraction of steps to train FM-only before MV consistency;
                          e.g. 0.4 means first 40% of steps use fm_ratio=1.0
        t_eps           : clamp t to [t_eps, 1-t_eps] for numerical stability
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
    warmup_steps = int(n_steps * warmup_frac)

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

        # Pure FM warmup: ignore h-dependence until FM sub-term has converged
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
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step":            step,
        "loss_history":    loss_history,
    }, path / "latest.pt")
    tqdm.write(f"[Checkpoint] MeanFlow saved at step {step}  →  {path / 'latest.pt'}")
