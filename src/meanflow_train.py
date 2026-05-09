"""
MeanFlow training loop (Part 4).

This trainer supports two MeanFlow parameterizations:
    pred_type='v': model(z_t, t, h) -> u_theta(z_t, t, h)
    pred_type='x': model(z_t, t, h) -> x_hat, converted internally to
                   u_theta = (z_t - x_hat) / t

For the linear interpolant z_t = (1 - t) * x + t * eps, the instantaneous
velocity is v_true = eps - x. MeanFlow trains a finite-horizon mean velocity
with the self-consistency identity:
    u_target = v_true - h * d/dt u_theta(z_t(t), t, h(t)).

The paper writes the model as u(z, r, t), where h = t - r. To avoid ambiguity,
the JVP below differentiates a wrapper over (z, r, t) with tangent
(v_true, 0, 1). This is equivalent to differentiating (z, t, h) with tangent
(v_true, 1, 1), but it makes the "r fixed, t moving" derivative explicit.

Important training details used here:
  - Logit-normal time sampling with mean=-0.4 and std=1.0.
  - A small nonzero-horizon ratio (default 25%), so the model keeps strong
    h=0 FM boundary coverage while learning finite jumps.
  - Optional adaptive per-sample loss weighting.
  - Optional x-pred initialization from the Part 2 x/x FM model.
  - EMA weights saved in checkpoints and applied to the returned model.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

T_EPS: float = 1e-4


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _validate_pred_type(pred_type: str) -> str:
    pred_type = pred_type.lower()
    if pred_type not in {"v", "x"}:
        raise ValueError(f"MeanFlow pred_type must be 'v' or 'x'; got {pred_type!r}.")
    return pred_type


def initialise_meanflow_from_fm(meanflow_model: nn.Module, fm_model: nn.Module) -> nn.Module:
    """
    Copy a trained FlowMatchingMLP into a MeanFlowMLP.

    MeanFlow has one extra horizon embedding. The first-layer weights for
    [z_t, e_t] are copied from the FM model, while the e_h block is zeroed.
    Therefore the initialized MeanFlow model exactly matches the FM model and
    initially ignores h, then finite-horizon training can learn h-dependence.
    """
    with torch.no_grad():
        meanflow_model.time_embedding.load_state_dict(fm_model.time_embedding.state_dict())

        fm_first = fm_model.hidden[0]
        mf_first = meanflow_model.hidden[0]
        copy_cols = fm_first.weight.shape[1]
        mf_first.weight.zero_()
        mf_first.weight[:, :copy_cols].copy_(fm_first.weight)
        mf_first.bias.copy_(fm_first.bias)

        for layer_idx in (2, 4, 6, 8):
            meanflow_model.hidden[layer_idx].weight.copy_(fm_model.hidden[layer_idx].weight)
            meanflow_model.hidden[layer_idx].bias.copy_(fm_model.hidden[layer_idx].bias)

        meanflow_model.output_layer.weight.copy_(fm_model.output_layer.weight)
        meanflow_model.output_layer.bias.copy_(fm_model.output_layer.bias)

    return meanflow_model


def _sample_logit_normal_times(
    batch_size: int,
    *,
    r_neq_t_ratio: float,
    time_dist_mean: float,
    time_dist_std: float,
    t_eps: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (r, t) with mostly r == t and some r < t intervals.

    This mirrors the public MeanFlow training recipe: draw two logit-normal
    times, copy one onto the other for the FM-boundary samples, then sort.
    """
    if not 0.0 <= r_neq_t_ratio <= 1.0:
        raise ValueError(f"r_neq_t_ratio must be in [0, 1], got {r_neq_t_ratio}")
    if not 0.0 < t_eps < 0.5:
        raise ValueError(f"t_eps must be in (0, 0.5), got {t_eps}")

    times = torch.randn(2, batch_size, device=device)
    times = times * time_dist_std + time_dist_mean

    use_interval = torch.rand(batch_size, device=device) < r_neq_t_ratio
    times[1] = torch.where(use_interval, times[1], times[0])

    times = torch.sigmoid(times).clamp(t_eps, 1.0 - t_eps)
    times = times.sort(dim=0).values
    return times[0], times[1]


def meanflow_loss(
    model: nn.Module,
    x: torch.Tensor,
    *,
    pred_type: str = "v",
    r_neq_t_ratio: float = 0.25,
    time_dist_mean: float = -0.4,
    time_dist_std: float = 1.0,
    loss_scale: float = 1.0,
    norm_eps: float = 1e-3,
    device: str = "cpu",
    t_eps: float = T_EPS,
) -> torch.Tensor:
    """
    Compute the MeanFlow loss.

    Args:
        model            : MeanFlowMLP
        x                : clean data batch, shape (B, D)
        pred_type        : 'v' for direct mean velocity, 'x' for clean endpoint
        r_neq_t_ratio    : fraction of samples with r != t; the rest are h=0 FM
        time_dist_mean   : logit-normal time mean
        time_dist_std    : logit-normal time std
        loss_scale       : adaptive weighting exponent; 0.0 disables it
        norm_eps         : adaptive weighting floor
        device           : torch device string
        t_eps            : time clamp floor

    Returns:
        scalar loss
    """
    pred_type = _validate_pred_type(pred_type)

    x = x.to(device).float()
    batch_size = x.shape[0]

    eps = torch.randn_like(x)
    v_true = (eps - x).detach()

    r, t = _sample_logit_normal_times(
        batch_size,
        r_neq_t_ratio=r_neq_t_ratio,
        time_dist_mean=time_dist_mean,
        time_dist_std=time_dist_std,
        t_eps=t_eps,
        device=device,
    )
    h = t - r
    z = (1.0 - t.view(batch_size, 1)) * x + t.view(batch_size, 1) * eps

    pred = model(z, t, h)

    if r_neq_t_ratio <= 0.0:
        target = v_true if pred_type == "v" else x
    else:
        def model_u(
            z_in: torch.Tensor,
            r_in: torch.Tensor,
            t_in: torch.Tensor,
        ) -> torch.Tensor:
            h_in = (t_in - r_in).clamp_min(0.0)
            out = model(z_in, t_in, h_in)
            if pred_type == "v":
                return out
            return (z_in - out) / t_in.clamp_min(t_eps).view(-1, 1)

        zeros = torch.zeros(batch_size, device=device)
        ones = torch.ones(batch_size, device=device)
        _, du_dt = torch.func.jvp(
            model_u,
            (z.detach(), r.detach(), t.detach()),
            (v_true, zeros, ones),
        )
        if pred_type == "v":
            target = (v_true - h.view(batch_size, 1) * du_dt).detach()
        else:
            target = (x + t.view(batch_size, 1) * h.view(batch_size, 1) * du_dt).detach()

    if pred_type == "v":
        per_sample = (pred - target).pow(2).sum(dim=1)
    else:
        per_sample = (pred - target).pow(2).mean(dim=1)
    if loss_scale > 0.0:
        weights = 1.0 / (per_sample.detach() + norm_eps).pow(loss_scale)
        return (weights * per_sample).mean()
    return per_sample.mean()


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
    """Cosine-annealed LR for the MV phase: mv_lr -> mv_lr_min."""
    mv_step = step - warmup_steps
    mv_total = max(1, n_steps - warmup_steps)
    phase = min(max(mv_step / mv_total, 0.0), 1.0)
    return mv_lr_min + 0.5 * (mv_lr - mv_lr_min) * (1.0 + math.cos(math.pi * phase))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_meanflow(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    n_steps: int = 25_000,
    lr: float = 1e-3,
    mv_lr: float = 1e-4,
    mv_lr_min: float = 1e-5,
    pred_type: str = "v",
    r_neq_t_ratio: float = 0.25,
    time_dist_mean: float = -0.4,
    time_dist_std: float = 1.0,
    loss_scale: float = 1.0,
    norm_eps: float = 1e-3,
    warmup_frac: float = 0.1,
    ema_decay: float = 0.9999,
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

    Phase 1: optional FM warmup with r == t for every sample.
    Phase 2: paper-style mixed batches with r != t for r_neq_t_ratio of samples.

    The returned model has EMA weights applied when ema_decay > 0.
    """
    _validate_pred_type(pred_type)

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    warmup_steps = int(n_steps * warmup_frac)
    start_step = 0
    loss_history: list[float] = []
    mv_phase_started = warmup_steps == 0
    ema_shadow: dict[str, torch.Tensor] | None = None

    if mv_phase_started and ema_decay > 0.0:
        ema_shadow = {k: v.float().clone() for k, v in model.state_dict().items()}

    # Resume
    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / run_name / "latest.pt"
        if resume and ckpt_path.exists():
            print(f"[Resume] Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = ckpt["step"]
            loss_history = ckpt.get("loss_history", [])
            mv_phase_started = ckpt.get("mv_phase_started", start_step >= warmup_steps)
            ema_raw = ckpt.get("ema_shadow")
            if ema_raw is not None:
                ema_shadow = {k: v.float().to(device) for k, v in ema_raw.items()}
            print(f"[Resume] Continuing from step {start_step}/{n_steps}")

    if start_step >= n_steps:
        print(f"[Skip] MeanFlow already trained to {start_step} steps.")
        if ema_shadow is not None:
            model.load_state_dict(ema_shadow)
        return model, loss_history

    if mv_phase_started:
        for group in optimizer.param_groups:
            group["lr"] = _cosine_lr(start_step, warmup_steps, n_steps, mv_lr, mv_lr_min)

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

        if not mv_phase_started and step >= warmup_steps:
            mv_phase_started = True
            tqdm.write(
                f"[MeanFlow] Switching to MV phase at step {step}: "
                f"r!=t ratio={r_neq_t_ratio:.2f}, AdamW lr={mv_lr:.2e}"
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=mv_lr,
                betas=(0.9, 0.95),
                weight_decay=0.0,
            )
            if ema_decay > 0.0:
                ema_shadow = {
                    k: v.float().clone()
                    for k, v in model.state_dict().items()
                }

        if mv_phase_started:
            for group in optimizer.param_groups:
                group["lr"] = _cosine_lr(step, warmup_steps, n_steps, mv_lr, mv_lr_min)

        current_ratio = r_neq_t_ratio if mv_phase_started else 0.0

        optimizer.zero_grad(set_to_none=True)
        loss = meanflow_loss(
            model,
            x,
            pred_type=pred_type,
            r_neq_t_ratio=current_ratio,
            time_dist_mean=time_dist_mean,
            time_dist_std=time_dist_std,
            loss_scale=loss_scale,
            norm_eps=norm_eps,
            t_eps=t_eps,
            device=device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if ema_shadow is not None:
            with torch.no_grad():
                for key, value in model.state_dict().items():
                    ema_shadow[key].mul_(ema_decay).add_(
                        value.float(),
                        alpha=1.0 - ema_decay,
                    )

        loss_value = loss.item()
        loss_history.append(loss_value)

        if step % log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss_value:.5f}",
                "phase": "mv" if mv_phase_started else "fm",
                "r!=t": f"{current_ratio:.2f}",
            })

        if checkpoint_dir is not None and (step + 1) % checkpoint_every == 0:
            _save_ckpt(
                model,
                optimizer,
                step + 1,
                loss_history,
                checkpoint_dir,
                run_name,
                mv_phase_started=mv_phase_started,
                ema_shadow=ema_shadow,
            )

    if checkpoint_dir is not None:
        _save_ckpt(
            model,
            optimizer,
            n_steps,
            loss_history,
            checkpoint_dir,
            run_name,
            mv_phase_started=mv_phase_started,
            ema_shadow=ema_shadow,
        )

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
    ema_shadow: dict[str, torch.Tensor] | None = None,
) -> None:
    path = Path(checkpoint_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "loss_history": loss_history,
            "mv_phase_started": mv_phase_started,
            "ema_shadow": (
                {key: value.cpu() for key, value in ema_shadow.items()}
                if ema_shadow is not None else None
            ),
        },
        path / "latest.pt",
    )
    tqdm.write(f"[Checkpoint] MeanFlow saved at step {step} -> {path / 'latest.pt'}")
