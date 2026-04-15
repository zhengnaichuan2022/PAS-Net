"""
SeqSNN-style models adapted for IMU HAR classification.

These implementations keep the model family spirit (SNN/RNN/GRU/TCN)
while adapting input/output contracts to this project:
  input : (B, T, C, V)
  output: (B, num_classes)
"""
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron


def _lif_backend() -> str:
    try:
        # cupy backend can be faster if available.
        _ = __import__("cupy")
        return "cupy"
    except Exception:
        return "torch"


class _BaseHARModel(nn.Module):
    def __init__(self, input_channels: int, num_imus: int, num_classes: int):
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_imus = int(num_imus)
        self.num_classes = int(num_classes)

    @property
    def in_dim(self) -> int:
        return self.input_channels * self.num_imus

    @staticmethod
    def _flatten_imu(x: torch.Tensor) -> torch.Tensor:
        # (B, T, C, V) -> (B, T, C*V)
        b, t, c, v = x.shape
        return x.reshape(b, t, c * v).contiguous()


class SpikeRNNHAR(_BaseHARModel):
    def __init__(
        self,
        input_channels: int = 3,
        num_imus: int = 1,
        num_classes: int = 12,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.1,
        v_threshold: float = 1.0,
    ):
        super().__init__(input_channels, num_imus, num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.in_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.cells = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        backend = _lif_backend()
        self.lifs = nn.ModuleList(
            [
                neuron.LIFNode(step_mode="s", v_threshold=float(v_threshold), backend=backend)
                for _ in range(self.num_layers)
            ]
        )
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._flatten_imu(x)  # (B, T, D)
        b, t, _ = x.shape
        h = [x.new_zeros((b, self.hidden_dim)) for _ in range(self.num_layers)]
        readout_steps = []

        for i in range(t):
            z = self.in_proj(x[:, i, :])
            for l in range(self.num_layers):
                z = z + self.cells[l](h[l])
                s = self.lifs[l](z)
                h[l] = s
                z = s
            readout_steps.append(z)

        feat = torch.stack(readout_steps, dim=1).mean(dim=1)  # (B, H)
        feat = self.dropout(feat)
        return self.head(feat)


class SpikeGRUHAR(_BaseHARModel):
    def __init__(
        self,
        input_channels: int = 3,
        num_imus: int = 1,
        num_classes: int = 12,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.1,
        v_threshold: float = 1.0,
    ):
        super().__init__(input_channels, num_imus, num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.in_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.gru_cells = nn.ModuleList(
            [nn.GRUCell(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        backend = _lif_backend()
        self.lifs = nn.ModuleList(
            [
                neuron.LIFNode(step_mode="s", v_threshold=float(v_threshold), backend=backend)
                for _ in range(self.num_layers)
            ]
        )
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._flatten_imu(x)  # (B, T, D)
        b, t, _ = x.shape
        h = [x.new_zeros((b, self.hidden_dim)) for _ in range(self.num_layers)]
        readout_steps = []

        for i in range(t):
            z = self.in_proj(x[:, i, :])
            for l in range(self.num_layers):
                h[l] = self.gru_cells[l](z, h[l])
                s = self.lifs[l](h[l])
                h[l] = s
                z = s
            readout_steps.append(z)

        feat = torch.stack(readout_steps, dim=1).mean(dim=1)
        feat = self.dropout(feat)
        return self.head(feat)


class TSSNNHAR(_BaseHARModel):
    def __init__(
        self,
        input_channels: int = 3,
        num_imus: int = 1,
        num_classes: int = 12,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.1,
        v_threshold: float = 1.0,
    ):
        super().__init__(input_channels, num_imus, num_classes)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        dims = [self.in_dim] + [self.hidden_dim] * self.num_layers
        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(self.num_layers)]
        )
        backend = _lif_backend()
        self.lifs = nn.ModuleList(
            [
                neuron.LIFNode(step_mode="m", v_threshold=float(v_threshold), backend=backend)
                for _ in range(self.num_layers)
            ]
        )
        self.head = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._flatten_imu(x)  # (B, T, D)
        # Convert to sequence-first for multi-step LIF.
        x = x.transpose(0, 1).contiguous()  # (T, B, D)

        for fc, lif in zip(self.fcs, self.lifs):
            t, b, d = x.shape
            z = fc(x.reshape(t * b, d)).reshape(t, b, -1).contiguous()
            x = lif(z)

        feat = x.mean(dim=0)  # (B, H)
        feat = self.dropout(feat)
        return self.head(feat)


class _SpikeTemporalBlock2D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        v_threshold: float,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=(1, kernel_size), dilation=(1, dilation), padding=(0, pad)
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=(1, kernel_size), dilation=(1, dilation), padding=(0, pad)
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        backend = _lif_backend()
        self.lif1 = neuron.LIFNode(step_mode="s", v_threshold=float(v_threshold), backend=backend)
        self.lif2 = neuron.LIFNode(step_mode="s", v_threshold=float(v_threshold), backend=backend)
        self.lif_out = neuron.LIFNode(step_mode="s", v_threshold=float(v_threshold), backend=backend)

    def _causal_crop(self, x: torch.Tensor, target_t: int) -> torch.Tensor:
        if x.shape[-1] > target_t:
            return x[..., -target_t:]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_t = x.shape[-1]
        y = self.bn1(self.conv1(x))
        y = self._causal_crop(y, target_t)
        y = self.lif1(y)
        y = self.bn2(self.conv2(y))
        y = self._causal_crop(y, target_t)
        y = self.lif2(y)
        r = self.down(x)
        return self.lif_out(y + r)


class SpikeTemporalConvNet2DHAR(_BaseHARModel):
    def __init__(
        self,
        input_channels: int = 3,
        num_imus: int = 1,
        num_classes: int = 12,
        hidden_dim: int = 64,
        num_levels: int = 3,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.1,
        v_threshold: float = 1.0,
    ):
        super().__init__(input_channels, num_imus, num_classes)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        # Input as feature-map: (B, C, V, T)
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
        )
        backend = _lif_backend()
        self.stem_lif = neuron.LIFNode(step_mode="s", v_threshold=float(v_threshold), backend=backend)

        levels = []
        in_ch = hidden_dim
        for i in range(int(num_levels)):
            out_ch = hidden_dim
            levels.append(
                _SpikeTemporalBlock2D(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel_size=int(kernel_size),
                    dilation=int(dilation_base) ** i,
                    v_threshold=float(v_threshold),
                )
            )
            in_ch = out_ch
        self.tcn = nn.Sequential(*levels)
        self.head = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C, V) -> (B, C, V, T)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.stem(x)
        x = self.stem_lif(x)
        x = self.tcn(x)
        # Global average pooling on (V, T)
        feat = x.mean(dim=(-1, -2))
        feat = self.dropout(feat)
        return self.head(feat)


def _cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("model", {})


def create_spike_rnn_har_model(config: Dict[str, Any]) -> nn.Module:
    m = _cfg(config)
    return SpikeRNNHAR(
        input_channels=m.get("input_channels", 3),
        num_imus=m.get("num_imus", m.get("V_nodes", 1)),
        num_classes=m.get("num_classes", 12),
        hidden_dim=m.get("hidden_dim", 192),
        num_layers=m.get("num_layers", 2),
        dropout=m.get("drop_rate", 0.1),
        v_threshold=m.get("v_threshold", 1.0),
    )


def create_spike_gru_har_model(config: Dict[str, Any]) -> nn.Module:
    m = _cfg(config)
    return SpikeGRUHAR(
        input_channels=m.get("input_channels", 3),
        num_imus=m.get("num_imus", m.get("V_nodes", 1)),
        num_classes=m.get("num_classes", 12),
        hidden_dim=m.get("hidden_dim", 192),
        num_layers=m.get("num_layers", 2),
        dropout=m.get("drop_rate", 0.1),
        v_threshold=m.get("v_threshold", 1.0),
    )


def create_tssnn_har_model(config: Dict[str, Any]) -> nn.Module:
    m = _cfg(config)
    return TSSNNHAR(
        input_channels=m.get("input_channels", 3),
        num_imus=m.get("num_imus", m.get("V_nodes", 1)),
        num_classes=m.get("num_classes", 12),
        hidden_dim=m.get("hidden_dim", 192),
        num_layers=m.get("num_layers", 2),
        dropout=m.get("drop_rate", 0.1),
        v_threshold=m.get("v_threshold", 1.0),
    )


def create_spike_tcn2d_har_model(config: Dict[str, Any]) -> nn.Module:
    m = _cfg(config)
    return SpikeTemporalConvNet2DHAR(
        input_channels=m.get("input_channels", 3),
        num_imus=m.get("num_imus", m.get("V_nodes", 1)),
        num_classes=m.get("num_classes", 12),
        hidden_dim=m.get("hidden_dim", 64),
        num_levels=m.get("num_levels", 3),
        kernel_size=m.get("kernel_size", 3),
        dilation_base=m.get("dilation_base", 2),
        dropout=m.get("drop_rate", 0.1),
        v_threshold=m.get("v_threshold", 1.0),
    )
