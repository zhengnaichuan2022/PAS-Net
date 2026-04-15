import math
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        p = self.conv.padding[0]
        return x[:, :, :-p] if p > 0 else x


class IMUFusionBlock(nn.Module):
    def __init__(self, num_channels=3, hidden_dim=64):
        super().__init__()
        self.posture_conv1 = CausalConv1d(num_channels, hidden_dim, kernel_size=3)
        self.posture_conv2 = CausalConv1d(num_channels, hidden_dim, kernel_size=3, dilation=1)
        self.posture_conv3 = CausalConv1d(num_channels, hidden_dim, kernel_size=4)
        self.posture_conv4 = CausalConv1d(num_channels, hidden_dim, kernel_size=4, dilation=1)
        self.motion_conv1 = CausalConv1d(num_channels, hidden_dim, kernel_size=3)
        self.motion_conv2 = CausalConv1d(num_channels, hidden_dim, kernel_size=5)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * 6),
            nn.Softmax(dim=-1),
        )

    def forward(self, grav_ang, gyro, linear_acc):
        p1 = self.posture_conv1(grav_ang)
        p2 = self.posture_conv2(grav_ang)
        p3 = self.posture_conv3(gyro)
        p4 = self.posture_conv4(gyro)
        m1 = self.motion_conv1(linear_acc)
        m2 = self.motion_conv2(linear_acc)
        features = torch.cat([p1, p2, p3, p4, m1, m2], dim=1)
        bf = features.transpose(1, 2).contiguous()
        attn = self.attention(bf)
        return (bf * attn).transpose(1, 2).contiguous()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class ConvTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, seq_len=128, dropout=0.1):
        super().__init__()
        self.conv_embed = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.pos = PositionalEncoding(hidden_dim, dropout=dropout, max_len=max(5000, seq_len + 8))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.conv_embed(x).transpose(1, 2).contiguous()
        x = self.pos(x)
        x = self.transformer(x)
        return x.mean(dim=1)


class IFConvTransformerANN(nn.Module):
    """IF-ConvTransformer wrapper for (B,T,C,V) input."""

    def __init__(self, input_channels=3, num_imus=1, num_classes=12, hidden_dim=64, transformer_dim=128, seq_len=128):
        super().__init__()
        self.num_sensors = int(input_channels * num_imus)
        if self.num_sensors != 9:
            self.sensor_projection = nn.Conv1d(self.num_sensors, 9, kernel_size=1)
        else:
            self.sensor_projection = None
        self.imu_fusion = IMUFusionBlock(num_channels=3, hidden_dim=hidden_dim)
        self.convtransformer = ConvTransformer(
            input_dim=hidden_dim * 6,
            hidden_dim=transformer_dim,
            num_heads=4,
            num_layers=2,
            seq_len=seq_len,
            dropout=0.1,
        )
        self.classifier = nn.Linear(transformer_dim, num_classes)

    def forward(self, x):
        # x: (B,T,C,V) -> (B,C*V,T)
        b, t, c, v = x.shape
        x = x.reshape(b, t, c * v).transpose(1, 2).contiguous()
        if self.sensor_projection is not None:
            x = self.sensor_projection(x)
        grav_ang, gyro, linear_acc = x[:, :3, :], x[:, 3:6, :], x[:, 6:9, :]
        fused = self.imu_fusion(grav_ang, gyro, linear_acc)
        feat = self.convtransformer(fused)
        return self.classifier(feat)


def create_if_convtransformer_ann_model(config):
    m = config.get('model', {})
    return IFConvTransformerANN(
        input_channels=m.get('input_channels', 3),
        num_imus=m.get('num_imus', 1),
        num_classes=m.get('num_classes', 12),
        hidden_dim=m.get('hidden_dim', 64),
        transformer_dim=m.get('transformer_dim', 128),
        seq_len=m.get('seq_len', 128),
    )
