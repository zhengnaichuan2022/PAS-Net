import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdaptiveBatchNorm(nn.Module):
    def __init__(self, num_features, num_domains=1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.domain_bns = nn.ModuleList([nn.BatchNorm1d(num_features) for _ in range(max(1, num_domains))])

    def forward(self, x, domain_id=0):
        if self.training and 0 <= domain_id < len(self.domain_bns):
            return self.domain_bns[domain_id](x)
        return self.bn(x)


class UniHARANN(nn.Module):
    """UniHAR ANN wrapper for (B,T,C,V) input."""

    def __init__(self, input_channels=3, num_imus=1, num_classes=12, hidden_dim=128, num_domains=1, use_dabn=True):
        super().__init__()
        self.num_sensors = int(input_channels * num_imus)
        self.use_dabn = use_dabn
        if use_dabn:
            self.bn1 = DomainAdaptiveBatchNorm(self.num_sensors, num_domains)
            self.bn2 = DomainAdaptiveBatchNorm(hidden_dim, num_domains)
            self.bn3 = DomainAdaptiveBatchNorm(hidden_dim, num_domains)
        else:
            self.bn1 = nn.BatchNorm1d(self.num_sensors)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.conv1 = nn.Conv1d(self.num_sensors, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, domain_id=0):
        # x: (B,T,C,V) -> (B,C*V,T)
        b, t, c, v = x.shape
        x = x.reshape(b, t, c * v).transpose(1, 2).contiguous()

        x = self.bn1(x, domain_id) if self.use_dabn else self.bn1(x)
        x = F.relu(self.conv1(x))
        x = self.bn2(x, domain_id) if self.use_dabn else self.bn2(x)
        x = F.relu(self.conv2(x))
        x = self.bn3(x, domain_id) if self.use_dabn else self.bn3(x)
        x = F.relu(self.conv3(x))

        x = x.transpose(1, 2).contiguous()
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.classifier(x)


def create_unihar_ann_model(config):
    m = config.get('model', {})
    return UniHARANN(
        input_channels=m.get('input_channels', 3),
        num_imus=m.get('num_imus', 1),
        num_classes=m.get('num_classes', 12),
        hidden_dim=m.get('hidden_dim', 128),
        num_domains=m.get('num_domains', 1),
        use_dabn=m.get('use_dabn', True),
    )
