import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class ResidualSE1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.se = SEBlock(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.short = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.short = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        out = out + self.short(x)
        return self.act(out)


class ResNetSEANN(nn.Module):
    def __init__(self, input_channels=3, num_imus=1, num_classes=12, dropout=0.4):
        super().__init__()
        in_ch = input_channels * num_imus
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualSE1D(64, 64)
        self.layer2 = ResidualSE1D(64, 128, stride=2)
        self.layer3 = ResidualSE1D(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        b, t, c, v = x.shape
        x = x.reshape(b, t, c * v).transpose(1, 2).contiguous()
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


def create_resnet_se_ann_model(config):
    m = config.get('model', {})
    return ResNetSEANN(
        input_channels=m.get('input_channels', 3),
        num_imus=m.get('num_imus', 1),
        num_classes=m.get('num_classes', 12),
        dropout=m.get('dropout', 0.4),
    )
