"""
Abstract base class for windowed IMU / time-series feeders.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Dict, Any

# Lazy import so environments without torch can still import feeders for static checks
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass
    class Tensor:
        pass
    torch = type('torch', (), {'Tensor': Tensor, 'FloatTensor': Tensor, 'LongTensor': Tensor})()


def _imu_small_rotation_matrix_xyz(max_deg: float) -> "np.ndarray":
    """Random small Euler XYZ rotation for wearable IMU misalignment augmentation."""
    rx, ry, rz = np.random.uniform(-max_deg, max_deg, 3) * (np.pi / 180.0)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


class BaseFeeder(Dataset, ABC):
    """Shared Dataset API for IMU HAR feeders."""

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """
        Args:
            config: Full YAML-derived dict
            split: ``train`` | ``val`` | ``test``
        """
        self.config = config
        self.split = split
        self.data = []
        self.labels = []
        self._load_data()

    def _split_seed(self) -> int:
        """RNG seed for splits (``project.seed`` in config)."""
        return int(self.config.get("project", {}).get("seed", 42))

    @abstractmethod
    def _load_data(self):
        """Populate ``self.data`` / ``self.labels``; implemented by subclasses."""
        pass

    @abstractmethod
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Per-window preprocessing before tensor conversion."""
        pass

    def __len__(self) -> int:
        return len(self.data)

    def _imu_train_augment(self, data: np.ndarray) -> np.ndarray:
        """
        Train-time IMU augmentation: noise, scale, small 3-axis rotation per triplet of channels.
        Controlled by ``training.imu_augmentation``; disabled by default.

        Implemented once here so all feeders inherit the same behavior.
        """
        cfg = (self.config or {}).get('training', {}).get('imu_augmentation', {})
        if self.split != 'train' or not cfg.get('enabled', False):
            return data
        if np.random.random() > float(cfg.get('prob', 0.4)):
            return data
        x = np.asarray(data, dtype=np.float32, order="C")
        if x.ndim != 3:
            return data
        T, C, V = x.shape
        lo, hi = cfg.get('scale_range', [0.9, 1.1])
        x = x * float(np.random.uniform(float(lo), float(hi)))
        ns = float(cfg.get('noise_std', 0.02))
        if ns > 0.0:
            x = x + np.random.randn(T, C, V).astype(np.float32) * ns
        max_deg = float(cfg.get('rotation_max_deg', 15.0))
        if max_deg > 0.0 and C >= 3:
            R = _imu_small_rotation_matrix_xyz(max_deg)
            for g in range(0, C // 3):
                sl = slice(g * 3, (g + 1) * 3)
                for v in range(V):
                    x[:, sl, v] = (R @ x[:, sl, v].T).T
        return x

    def __getitem__(self, idx: int):
        """
        Returns:
            ``(data, label)`` with ``data`` shape ``(T, C, V)`` (or ``(T, C, 1)`` for single IMU).
            Collate stacks to ``(B, T, C, V)``.
        """
        data = self.data[idx]
        label = self.labels[idx]

        data = self._preprocess(data)

        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]

        data = self._imu_train_augment(data)

        if TORCH_AVAILABLE:
            data = torch.FloatTensor(data)
            label = torch.LongTensor([label])[0]
        else:
            label = np.int64(label)

        return data, label

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score per feature channel."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)
        return (data - mean) / std

    def min_max_normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-max to [0,1] per channel."""
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        return (data - min_val) / range_val

    def sliding_window(self, data: np.ndarray, window_size: int, stride: int) -> List[np.ndarray]:
        """Sliding windows along time axis."""
        windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            windows.append(data[i:i+window_size])
        return windows

    def get_num_classes(self) -> int:
        return len(set(self.labels)) if self.labels else 0

    def get_class_weights(self):
        """Inverse-frequency weights for class imbalance."""
        if not self.labels:
            return None

        labels = np.array(self.labels)
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = total / (len(unique) * counts)

        weights = weights / weights.sum() * len(unique)

        if TORCH_AVAILABLE:
            return torch.FloatTensor(weights)
        else:
            return weights.astype(np.float32)
