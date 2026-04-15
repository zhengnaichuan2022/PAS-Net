"""
Minimal SNN baselines for debugging (B, T, C, V) IMU tensors).
Inspired by a SpikingMLP-style stack.

- ``SimpleSNNModel1D``: mode 1 — Conv1d over (C, V) at each time step T.
- ``SimpleSNNModel2D``: mode 2 — Conv2d over (T, V) as a 2D spatial map.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional


class SimpleSNNModel1D(nn.Module):
    """
    Mode 1: Conv1d MLP on (C, V) with T as the SNN time dimension.

    Input: (B, T, C, V)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_imus: int = 1,
        num_classes: int = 12,
        hidden_dim: int = 128,
        v_threshold: float = 0.5,
        tau: float = 2.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_channels: C (e.g. 3 for acc)
            num_imus: V (number of IMU nodes)
            num_classes: classification classes
            hidden_dim: hidden width
            v_threshold: LIF threshold
            tau: LIF time constant (unused if node uses defaults)
            dropout: dropout probability
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_imus = num_imus
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.fc1_conv = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_dim)

        try:
            self.fc1_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='cupy')
        except Exception:
            self.fc1_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='torch')

        self.fc2_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim)

        try:
            self.fc2_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='cupy')
        except Exception:
            self.fc2_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='torch')

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hook=None):
        """
        Args:
            x: (B, T, C, V)
            hook: unused placeholder for API compatibility

        Returns:
            (B, num_classes)
        """
        B, T, C, V = x.shape

        x = x.permute(1, 0, 2, 3).contiguous()

        x_flat = x.flatten(0, 1)
        x = self.fc1_conv(x_flat)
        x = self.fc1_bn(x)
        x = x.reshape(T, B, self.hidden_dim, V).contiguous()
        x = self.fc1_lif(x)

        x_flat = x.flatten(0, 1)
        x = self.fc2_conv(x_flat)
        x = self.fc2_bn(x)
        x = x.reshape(T, B, self.hidden_dim, V).contiguous()
        x = self.fc2_lif(x)

        x = self.dropout(x)

        x = x.mean(0).mean(-1)

        output = self.classifier(x)

        return output


class SimpleSNNModel2D(nn.Module):
    """
    Mode 2: treat (T, V) as 2D spatial map, repeat an extra SNN time dimension S.

    Input: (B, T, C, V)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_imus: int = 1,
        num_classes: int = 12,
        hidden_dim: int = 128,
        time_steps: int = 4,
        v_threshold: float = 0.5,
        tau: float = 2.0,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_channels: C
            num_imus: V (sanity check only in this minimal model)
            num_classes: classes
            hidden_dim: hidden channels
            time_steps: extra SNN steps S
            v_threshold: LIF threshold
            tau: unused placeholder
            dropout: dropout probability
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps

        self.fc1_conv = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_dim)

        try:
            self.fc1_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='cupy')
        except Exception:
            self.fc1_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='torch')

        self.fc2_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.fc2_bn = nn.BatchNorm2d(hidden_dim)

        try:
            self.fc2_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='cupy')
        except Exception:
            self.fc2_lif = neuron.LIFNode(step_mode='m', v_threshold=v_threshold, backend='torch')

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hook=None):
        B, T, C, V = x.shape

        x = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)

        x = x.permute(0, 1, 3, 2, 4).contiguous()

        S, B, C, T, V = x.shape
        x_flat = x.flatten(0, 1)
        x_flat = self.fc1_conv(x_flat)
        x_flat = self.fc1_bn(x_flat)
        x = x_flat.reshape(S, B, self.hidden_dim, T, V).contiguous()

        x = self.fc1_lif(x)

        x_flat = x.flatten(0, 1)
        x_flat = self.fc2_conv(x_flat)
        x_flat = self.fc2_bn(x_flat)
        x = x_flat.reshape(S, B, self.hidden_dim, T, V).contiguous()

        x = self.fc2_lif(x)

        x = self.dropout(x)

        x = x.mean(0).mean(-1).mean(-1)

        output = self.classifier(x)

        return output


class SimpleSNNModel(nn.Module):
    """
    Dispatch wrapper: ``mode=1`` -> SimpleSNNModel1D, ``mode=2`` -> SimpleSNNModel2D.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_imus: int = 1,
        num_classes: int = 12,
        hidden_dim: int = 128,
        time_steps: int = 4,
        mode: int = 1,
        v_threshold: float = 0.5,
        tau: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode

        if mode == 1:
            self.model = SimpleSNNModel1D(
                input_channels=input_channels,
                num_imus=num_imus,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                v_threshold=v_threshold,
                tau=tau,
                dropout=dropout,
            )
        elif mode == 2:
            self.model = SimpleSNNModel2D(
                input_channels=input_channels,
                num_imus=num_imus,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                time_steps=time_steps,
                v_threshold=v_threshold,
                tau=tau,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 1 or 2.")

    def forward(self, x, hook=None):
        return self.model(x, hook=hook)


def create_simple_model(config: dict) -> nn.Module:
    """Build ``SimpleSNNModel`` from ``config['model']``."""
    model_config = config.get('model', {})

    num_classes = model_config.get('num_classes', 12)

    model = SimpleSNNModel(
        input_channels=model_config.get('input_channels', 3),
        num_imus=model_config.get('num_imus', 1),
        num_classes=num_classes,
        hidden_dim=model_config.get('hidden_dim', 128),
        time_steps=model_config.get('time_steps', 4),
        mode=model_config.get('btcv_mode', 1),
    )

    return model
