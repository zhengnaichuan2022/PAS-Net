import torch
import torch.nn as nn


class DeepConvLSTMANN(nn.Module):
    """DeepConvLSTM-style ANN baseline for IMU HAR."""

    def __init__(self, input_channels=3, num_imus=1, num_classes=12, hidden_dim=128, dropout=0.5):
        super().__init__()
        in_ch = input_channels * num_imus
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        # x: (B, T, C, V)
        b, t, c, v = x.shape
        x = x.reshape(b, t, c * v).transpose(1, 2).contiguous()  # (B, C*V, T)
        x = self.conv(x).transpose(1, 2).contiguous()  # (B, T, 64)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.head(x)


def create_deep_conv_lstm_ann_model(config):
    m = config.get('model', {})
    return DeepConvLSTMANN(
        input_channels=m.get('input_channels', 3),
        num_imus=m.get('num_imus', 1),
        num_classes=m.get('num_classes', 12),
        hidden_dim=m.get('hidden_dim', 128),
        dropout=m.get('dropout', 0.5),
    )
