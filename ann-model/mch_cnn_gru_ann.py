import torch
import torch.nn as nn


class MCHCNNGRUANN(nn.Module):
    """Multi-kernel CNN + GRU ANN baseline."""

    def __init__(self, input_channels=3, num_imus=1, num_classes=12, hidden_dim=128, dropout=0.5):
        super().__init__()
        in_ch = input_channels * num_imus
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_ch, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv1d(in_ch, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv1d(in_ch, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(inplace=True)),
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=1),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(input_size=96, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        b, t, c, v = x.shape
        x = x.reshape(b, t, c * v).transpose(1, 2).contiguous()  # (B, C*V, T)
        feats = [br(x) for br in self.branches]
        x = torch.cat(feats, dim=1)
        x = self.proj(x).transpose(1, 2).contiguous()  # (B, T, 96)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.head(x)


def create_mch_cnn_gru_ann_model(config):
    m = config.get('model', {})
    return MCHCNNGRUANN(
        input_channels=m.get('input_channels', 3),
        num_imus=m.get('num_imus', 1),
        num_classes=m.get('num_classes', 12),
        hidden_dim=m.get('hidden_dim', 128),
        dropout=m.get('dropout', 0.5),
    )
