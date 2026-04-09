"""
Model architectures for Flow Matching Assignment (COMP8650).

Includes:
  - SinusoidalEmbedding : positional embedding for time/horizon scalars
  - FlowMatchingMLP     : standard MLP used in Parts 1-3
  - MeanFlowMLP         : extended MLP with horizon input for Part 4
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """
    Maps a scalar t ∈ [0,1] to a fixed 128-dimensional embedding.

    Following the DiT reference:
      k = d/2 = 64 frequencies
      ω_i = exp(−i · ln(10000) / (k−1)),  i = 0, …, k−1
      e_t = [sin(t·ω_0), …, sin(t·ω_{k-1}), cos(t·ω_0), …, cos(t·ω_{k-1})]

    The embedding dimension is 2k = 128 by default.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        assert dim % 2 == 0, "Embedding dim must be even"
        k = dim // 2
        # Frequencies — registered as a buffer so they move with the model
        i = torch.arange(k, dtype=torch.float32)
        freqs = torch.exp(-i * math.log(10000.0) / max(k - 1, 1))
        self.register_buffer("freqs", freqs)  # (k,)
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: shape (B,) — time values in [0, 1]
        Returns:
            embedding: shape (B, dim)
        """
        t = t.view(-1, 1).float()                        # (B, 1)
        args = t * self.freqs.view(1, -1)                # (B, k)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2k)


# ---------------------------------------------------------------------------
# Standard Flow Matching MLP  (Parts 1 – 3)
# ---------------------------------------------------------------------------

class FlowMatchingMLP(nn.Module):
    """
    MLP denoiser for flow matching.

    Architecture (6 linear layers total):
      Input      : [z_t ; e_t]  ∈  R^{D + 128}
      Hidden ×5  : Linear → ReLU,  256 units
                   (first maps R^{D+128} → R^{256}, rest R^{256} → R^{256})
      Output     : Linear  R^{256} → R^D  (no activation)

    The output dimension D equals data_dim, and its *interpretation* (velocity v
    or clean data x) is determined outside this class by pred_type.

    Args:
        data_dim   : ambient dimension D of the data
        hidden_dim : width of hidden layers (default 256)
        time_emb_dim: dimension of sinusoidal time embedding (default 128)
    """

    def __init__(self, data_dim: int, hidden_dim: int = 256, time_emb_dim: int = 128):
        super().__init__()
        self.data_dim = data_dim
        self.time_embedding = SinusoidalEmbedding(time_emb_dim)

        input_dim = data_dim + time_emb_dim  # D + 128

        # Build 5 hidden layers
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(4):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.hidden = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_dim, data_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, D)  — noisy sample z_t
            t : (B,) or scalar — time value(s) in [0, 1]
        Returns:
            out : (B, D)  — predicted v or x depending on pred_type used during training
        """
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        e_t = self.time_embedding(t)                   # (B, 128)
        inp = torch.cat([z.float(), e_t], dim=-1)      # (B, D+128)
        h = self.hidden(inp)                           # (B, 256)
        return self.output_layer(h)                    # (B, D)


# ---------------------------------------------------------------------------
# MeanFlow MLP  (Part 4)
# ---------------------------------------------------------------------------

class MeanFlowMLP(nn.Module):
    """
    Extended MLP for MeanFlow (Part 4).

    Adds a second sinusoidal embedding for the horizon h = t − r (separate
    parameters from the time embedding).

    Architecture:
      Input      : [z_t ; e_t ; e_h]  ∈  R^{D + 256}
      Hidden ×5  : Linear → ReLU,  256 units
                   (first maps R^{D+256} → R^{256}, rest R^{256} → R^{256})
      Output     : Linear  R^{256} → R^D

    Forward pass: model(z, t, h)

    Args:
        data_dim    : ambient dimension D
        hidden_dim  : width of hidden layers (default 256)
        time_emb_dim: per-embedding dimension (default 128; total emb = 256)
    """

    def __init__(self, data_dim: int, hidden_dim: int = 256, time_emb_dim: int = 128):
        super().__init__()
        self.data_dim = data_dim
        self.time_embedding    = SinusoidalEmbedding(time_emb_dim)   # for t
        self.horizon_embedding = SinusoidalEmbedding(time_emb_dim)   # for h (separate params)

        input_dim = data_dim + 2 * time_emb_dim  # D + 256

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(4):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.hidden = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_dim, data_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, D)  — noisy sample z_t
            t : (B,) — time in [0, 1]
            h : (B,) — horizon = t − r ≥ 0
        Returns:
            out : (B, D)  — predicted mean velocity u_θ(z_t, t, h)
        """
        if t.dim() == 0:
            t = t.expand(z.shape[0])
        if h.dim() == 0:
            h = h.expand(z.shape[0])
        e_t = self.time_embedding(t)                        # (B, 128)
        e_h = self.horizon_embedding(h)                     # (B, 128)
        inp = torch.cat([z.float(), e_t, e_h], dim=-1)     # (B, D+256)
        hidden = self.hidden(inp)                           # (B, 256)
        return self.output_layer(hidden)                    # (B, D)
