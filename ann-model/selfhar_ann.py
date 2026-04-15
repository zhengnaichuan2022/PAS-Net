import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfHARANN(nn.Module):
    """SelfHAR ANN wrapper for (B,T,C,V) input."""

    def __init__(self, input_channels=3, num_imus=1, num_classes=12, hidden_dim=128, temperature=0.5):
        super().__init__()
        self.num_sensors = int(input_channels * num_imus)
        self.temperature = temperature

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(self.num_sensors, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, return_features=False):
        # x: (B,T,C,V) -> (B,C*V,T)
        b, t, c, v = x.shape
        x = x.reshape(b, t, c * v).transpose(1, 2).contiguous()

        features = self.feature_extractor(x)
        features_seq = features.transpose(1, 2).contiguous()
        lstm_out, _ = self.lstm(features_seq)
        features_final = lstm_out[:, -1, :]

        if return_features:
            return features_final
        return self.classifier(self.dropout(features_final))

    def get_pseudo_labels(self, x, threshold=0.9):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits / self.temperature, dim=-1)
            confidence, pseudo_labels = torch.max(probs, dim=-1)
            mask = confidence >= threshold
            return pseudo_labels, confidence, mask


def create_selfhar_ann_model(config):
    m = config.get('model', {})
    return SelfHARANN(
        input_channels=m.get('input_channels', 3),
        num_imus=m.get('num_imus', 1),
        num_classes=m.get('num_classes', 12),
        hidden_dim=m.get('hidden_dim', 128),
        temperature=m.get('temperature', 0.5),
    )
