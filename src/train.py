"""
Training loop for Flow Matching (Parts 1 – 3).

Supports all 4 prediction × loss combinations:
  pred_type ∈ {'v', 'x'}  ×  loss_type ∈ {'v', 'x'}

Conversion formulas derived from the forward process z_t = (1−t)x + t·ε:
  • z_t = x + t·v        where  v = ε − x  (velocity)
  • x   = z_t − t·v_hat  (from v_hat → x_hat)
  • v   = (z_t − x_hat) / t  (from x_hat → v_hat, careful near t≈0)

Checkpointing:
  Saves to  <checkpoint_dir>/<run_name>/latest.pt
  Auto-resumes if the checkpoint exists and resume=True.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Small epsilon to avoid division by zero and clamp t away from boundaries
T_EPS: float = 1e-4


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def flow_matching_loss(
    model: nn.Module,
    x: torch.Tensor,
    pred_type: str = "v",
    loss_type: str = "v",
    device: str = "cpu",
    t_eps: float = T_EPS,
) -> torch.Tensor:
    """
    Compute the flow matching loss for a single batch.

    Args:
        model     : FlowMatchingMLP (or any model with signature model(z, t))
        x         : clean data batch  (B, D)
        pred_type : what the model predicts — 'v' (velocity) or 'x' (clean data)
        loss_type : in which space the MSE is computed — 'v' or 'x'
        device    : torch device string
        t_eps     : clip t to [t_eps, 1−t_eps] for numerical stability

    Returns:
        scalar loss tensor
    """
    x = x.to(device).float()
    B, D = x.shape

    # ── Sample noise ε ~ N(0, I) and time t ~ Uniform(0, 1)
    eps = torch.randn_like(x)
    t   = torch.rand(B, device=device).clamp(t_eps, 1.0 - t_eps)
    t4  = t.view(B, 1)  # (B, 1) broadcast-friendly

    # ── Forward process: z_t = (1−t)·x + t·ε  ──  equivalently  z_t = x + t·v
    z_t = (1.0 - t4) * x + t4 * eps

    # ── Ground-truth targets in both spaces
    v_target = eps - x   # velocity  v = ε − x
    x_target = x         # clean data

    # ── Model prediction (B, D)
    pred = model(z_t, t)

    # ── Compute loss according to pred_type and loss_type
    if pred_type == "v" and loss_type == "v":
        # Model predicts v;  loss directly in v-space
        loss = nn.functional.mse_loss(pred, v_target)

    elif pred_type == "v" and loss_type == "x":
        # Model predicts v;  convert to x̂ = z_t − t·v̂  and compute x-loss
        x_hat = z_t - t4 * pred
        loss = nn.functional.mse_loss(x_hat, x_target)

    elif pred_type == "x" and loss_type == "v":
        # Model predicts x;  convert to v̂ = (z_t − x̂) / t  and compute v-loss
        v_hat = (z_t - pred) / t4.clamp(min=t_eps)
        loss = nn.functional.mse_loss(v_hat, v_target)

    elif pred_type == "x" and loss_type == "x":
        # Model predicts x;  loss directly in x-space
        loss = nn.functional.mse_loss(pred, x_target)

    else:
        raise ValueError(
            f"Unknown combination pred_type='{pred_type}', loss_type='{loss_type}'"
        )

    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_flow_matching(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    n_steps: int = 25_000,
    lr: float = 1e-3,
    pred_type: str = "v",
    loss_type: str = "v",
    device: str = "cpu",
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 5_000,
    resume: bool = True,
    run_name: str = "run",
    log_every: int = 500,
) -> tuple[nn.Module, list[float]]:
    """
    Train a FlowMatchingMLP with automatic checkpointing and resume support.

    Args:
        model           : FlowMatchingMLP to train (will be moved to device)
        dataloader      : DataLoader over ToyDiffusionDataset
        n_steps         : total gradient steps
        lr              : Adam learning rate
        pred_type       : 'v' or 'x'
        loss_type       : 'v' or 'x'
        device          : 'cuda', 'cpu', or 'mps'
        checkpoint_dir  : directory for saving checkpoints (None = no saving)
        checkpoint_every: save frequency in steps
        resume          : whether to resume from an existing checkpoint
        run_name        : sub-directory name under checkpoint_dir
        log_every       : how often to update the progress-bar postfix

    Returns:
        (trained model, loss_history)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Attempt resume ────────────────────────────────────────────────────
    start_step   = 0
    loss_history: list[float] = []

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / run_name / "latest.pt"
        if resume and ckpt_path.exists():
            print(f"[Resume] Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step   = ckpt["step"]
            loss_history = ckpt.get("loss_history", [])
            print(f"[Resume] Continuing from step {start_step}/{n_steps}")

    if start_step >= n_steps:
        print(f"[Skip] Already trained to {start_step} steps — nothing to do.")
        return model, loss_history

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    data_iter = iter(dataloader)

    pbar = tqdm(
        range(start_step, n_steps),
        desc=f"{run_name}",
        dynamic_ncols=True,
        leave=True,
    )

    for step in pbar:
        # Cycle through dataset
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x = next(data_iter)

        optimizer.zero_grad()
        loss = flow_matching_loss(
            model, x,
            pred_type=pred_type,
            loss_type=loss_type,
            device=device,
        )
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % log_every == 0:
            pbar.set_postfix({"loss": f"{loss_val:.5f}"})

        # Periodic checkpoint
        if (
            checkpoint_dir is not None
            and (step + 1) % checkpoint_every == 0
        ):
            _save_checkpoint(
                model, optimizer, step + 1, loss_history,
                checkpoint_dir, run_name,
            )

    # Final checkpoint
    if checkpoint_dir is not None:
        _save_checkpoint(
            model, optimizer, n_steps, loss_history,
            checkpoint_dir, run_name,
        )

    return model, loss_history


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss_history: list[float],
    checkpoint_dir: str | Path,
    run_name: str,
) -> None:
    path = Path(checkpoint_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step":            step,
        "loss_history":    loss_history,
    }
    torch.save(ckpt, path / "latest.pt")
    tqdm.write(f"[Checkpoint] Saved at step {step}  →  {path / 'latest.pt'}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
    run_name: str,
    device: str = "cpu",
) -> tuple[nn.Module, int, list[float]]:
    """
    Load model weights from a saved checkpoint.

    Returns:
        (model_with_loaded_weights, step, loss_history)

    Raises:
        FileNotFoundError if the checkpoint does not exist.
    """
    ckpt_path = Path(checkpoint_dir) / run_name / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt.get("step", 0), ckpt.get("loss_history", [])


def checkpoint_exists(
    checkpoint_dir: str | Path,
    run_name: str,
) -> bool:
    """Return True if a checkpoint file already exists for this run."""
    return (Path(checkpoint_dir) / run_name / "latest.pt").exists()
