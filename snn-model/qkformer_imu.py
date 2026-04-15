"""
QKFormer-IMU: an IMU-friendly spiking transformer with Q-K attention.

This is a lightweight adaptation inspired by the upstream QKFormer repository,
but tailored for IMU tensors:

Input:  (B, T, C, V)  where V is the number of IMU nodes (tokens)
Output: (B, num_classes)

We treat V as tokens and keep T as spiking time steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, functional


class MLP1x1(nn.Module):
    """Token-wise MLP using 1x1 Conv1d on the token axis."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0, lif_backend: str = "torch"):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv1d(dim, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.lif1 = neuron.LIFNode(step_mode="m", v_threshold=1.0, backend=lif_backend)

        self.fc2 = nn.Conv1d(hidden, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)
        self.lif2 = neuron.LIFNode(step_mode="m", v_threshold=1.0, backend=lif_backend)

        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.hidden = hidden
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, D, V)
        Returns:
            (T, B, D, V)
        """
        T, B, D, V = x.shape
        y = x.flatten(0, 1)  # (T*B, D, V)
        y = self.fc1(y)
        y = self.bn1(y)
        y = y.reshape(T, B, self.hidden, V).contiguous()
        y = self.lif1(y)
        y = self.drop(y)

        y2 = y.flatten(0, 1)  # (T*B, hidden, V)
        y2 = self.fc2(y2)
        y2 = self.bn2(y2)
        y2 = y2.reshape(T, B, D, V).contiguous()
        y2 = self.lif2(y2)
        return y2


class TokenQKAttentionIMU(nn.Module):
    """
    Q-K attention for IMU tokens (V).

    Inspired by QKFormer Token_QK_Attention:
    - Build spiking Q and K via 1x1 Conv + BN + LIF.
    - Collapse Q across channel groups to build a binary gate (attn_lif).
    - Gate * K, then project back.

    Shapes:
      x: (T, B, D, V)
      out: (T, B, D, V)
    """

    def __init__(self, dim: int, num_heads: int = 4, lif_backend: str = "torch", attn_threshold: float = 0.5):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(step_mode="m", v_threshold=1.0, backend=lif_backend)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(step_mode="m", v_threshold=1.0, backend=lif_backend)

        # binary gate on reduced q
        self.attn_lif = neuron.LIFNode(step_mode="m", v_threshold=attn_threshold, backend=lif_backend)

        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(step_mode="m", v_threshold=1.0, backend=lif_backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, D, V = x.shape
        x_flat = x.flatten(0, 1)  # (T*B, D, V)

        q = self.q_conv(x_flat)
        q = self.q_bn(q).reshape(T, B, D, V).contiguous()
        q = self.q_lif(q)

        k = self.k_conv(x_flat)
        k = self.k_bn(k).reshape(T, B, D, V).contiguous()
        k = self.k_lif(k)

        # reshape into heads: (T,B,H,head_dim,V)
        qh = q.reshape(T, B, self.num_heads, self.head_dim, V)
        kh = k.reshape(T, B, self.num_heads, self.head_dim, V)

        # collapse head_dim -> binary gate per head/token: (T,B,H,1,V)
        q_sum = qh.sum(dim=3, keepdim=True)
        gate = self.attn_lif(q_sum)

        # apply gate to K and merge heads back
        out = (gate * kh).reshape(T, B, D, V)

        out2 = self.proj(out.flatten(0, 1))
        out2 = self.proj_bn(out2).reshape(T, B, D, V).contiguous()
        out2 = self.proj_lif(out2)
        return out2


class QKBlockIMU(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        lif_backend: str = "torch",
    ):
        super().__init__()
        self.attn = TokenQKAttentionIMU(dim=dim, num_heads=num_heads, lif_backend=lif_backend)
        self.mlp = MLP1x1(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout, lif_backend=lif_backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class QKFormerIMU(nn.Module):
    """
    Minimal IMU QKFormer:
      (B,T,C,V) -> embed to D -> QK blocks over V tokens per time step -> pool -> logits
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 12,
        embed_dims: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        lif_backend: str = "torch",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.depth = depth

        # token embedding: (T*B, C, V) -> (T*B, D, V)
        self.embed = nn.Conv1d(input_channels, embed_dims, kernel_size=1, bias=False)
        self.embed_bn = nn.BatchNorm1d(embed_dims)
        self.embed_lif = neuron.LIFNode(step_mode="m", v_threshold=1.0, backend=lif_backend)

        self.blocks = nn.ModuleList(
            [
                QKBlockIMU(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    lif_backend=lif_backend,
                )
                for _ in range(depth)
            ]
        )

        self.head = nn.Linear(embed_dims, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, V)
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, V = x.shape

        # convert to time-first: (T,B,C,V)
        x_t = x.permute(1, 0, 2, 3).contiguous()

        # embed on token axis V
        y = self.embed(x_t.flatten(0, 1))  # (T*B, D, V)
        y = self.embed_bn(y).reshape(T, B, self.embed_dims, V).contiguous()
        y = self.embed_lif(y)  # (T,B,D,V)

        # blocks
        for blk in self.blocks:
            y = blk(y)

        # pool over tokens V then time T
        y = y.mean(dim=3)  # (T,B,D)
        y = y.mean(dim=0)  # (B,D)
        return self.head(y)


def create_qkformer_imu_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})

    return QKFormerIMU(
        input_channels=model_cfg.get("input_channels", 3),
        num_classes=model_cfg.get("num_classes", 12),
        embed_dims=model_cfg.get("embed_dims", 128),
        depth=model_cfg.get("depth", 4),
        num_heads=model_cfg.get("num_heads", 4),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        dropout=model_cfg.get("dropout", 0.0),
        lif_backend=model_cfg.get("lif_backend", "torch"),
    )

