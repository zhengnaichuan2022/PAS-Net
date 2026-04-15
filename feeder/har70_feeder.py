"""
HAR70 dataset feeder.
HAR70+: activity recognition for older adults; back and thigh each have 3-axis acceleration.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import random
from .base_feeder import BaseFeeder
from .split_utils import subject_independent_indices


class HAR70_Feeder(BaseFeeder):
    """HAR70 dataset feeder"""

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """
        Args:
            config: config dict
            split: ``train`` | ``val`` | ``test``
        """
        self.data_root = Path(config['dataset']['data_root']) / "har70" / "har70plus"
        self.subjects = config['dataset']['har70']['subjects']  # e.g. [501, 502, ..., 518]
        self.activities = config['dataset']['har70']['activities']  # e.g. [1, 3, 4, 5, 6, 7, 8]
        self.window_size = config['dataset']['har70']['window_size']
        self.window_stride = config['dataset']['har70']['window_stride']
        self.normalize = config['dataset']['har70']['normalize']
        self.train_split = config['dataset']['train_split']
        self.val_split = config['dataset']['val_split']
        self.test_split = config['dataset']['test_split']
        self.subject_independent_split = config['dataset'].get('subject_independent_split', True)

        # Mount: 'back', 'thigh', 'both'
        self.sensor_position = config['dataset']['har70'].get('sensor_position', 'both')

        super().__init__(config, split)

    def _load_data(self):
        all_samples = []
        all_labels = []
        all_units: List[Any] = []

        for subject in self.subjects:
            csv_file = self.data_root / f"{subject}.csv"
            if not csv_file.exists():
                print(f"Warning: File not found {csv_file}")
                continue

            try:
                df = pd.read_csv(csv_file)

                if self.sensor_position == 'back':
                    sensor_data = df[['back_x', 'back_y', 'back_z']].values
                    sensor_data = sensor_data[:, :, np.newaxis]
                elif self.sensor_position == 'thigh':
                    sensor_data = df[['thigh_x', 'thigh_y', 'thigh_z']].values
                    sensor_data = sensor_data[:, :, np.newaxis]
                elif self.sensor_position == 'both':
                    back_data = df[['back_x', 'back_y', 'back_z']].values
                    thigh_data = df[['thigh_x', 'thigh_y', 'thigh_z']].values
                    sensor_data = np.stack([back_data, thigh_data], axis=2)  # (n_samples, 3, 2)
                else:
                    raise ValueError(f"Unknown sensor_position: {self.sensor_position}")

                labels = df['label'].values

                mask = np.isin(labels, self.activities)
                sensor_data = sensor_data[mask]
                labels = labels[mask]

                windows, window_labels = self.sliding_window_with_labels(
                    sensor_data, labels, self.window_size, self.window_stride
                )

                label_map = {label: idx for idx, label in enumerate(sorted(self.activities))}
                for window, label in zip(windows, window_labels):
                    all_samples.append(window)  # window shape: (T, C, V)
                    all_labels.append(label_map[label])
                    all_units.append(subject)

            except Exception as e:
                print(f"Failed to load file {csv_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if self.subject_independent_split:
            indices, info = subject_independent_indices(
                all_units,
                self.split,
                self.train_split,
                self.val_split,
                self.test_split,
                seed=self._split_seed(),
            )
            if self.split == 'train':
                print(
                    f"[Subject-independent] HAR70 | n_subjects={info['n_units']} | "
                    f"train={info['train_units']} | val={info['val_units']} | test={info['test_units']}"
                )
            self.data = [all_samples[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]
        else:
            indices = list(range(len(all_samples)))
            random.Random(self._split_seed()).shuffle(indices)
            n_total = len(all_samples)
            n_train = int(n_total * self.train_split)
            n_val = int(n_total * self.val_split)
            if self.split == 'train':
                indices = indices[:n_train]
            elif self.split == 'val':
                indices = indices[n_train : n_train + n_val]
            elif self.split == 'test':
                indices = indices[n_train + n_val :]
            self.data = [all_samples[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]

        print(f"HAR70 {self.split}: {len(self.data)} windows, {self.get_num_classes()} classes")

    def sliding_window_with_labels(self, data: np.ndarray, labels: np.ndarray,
                                   window_size: int, stride: int) -> tuple:
        """
        Sliding-window segmentation with per-window labels.

        Args:
            data: (n_samples, C, V)
            labels: (n_samples,)
            window_size: window length
            stride: hop size

        Returns:
            windows: list of (T, C, V)
            window_labels: list of labels
        """
        windows = []
        window_labels = []

        n_samples = data.shape[0]

        for start_idx in range(0, n_samples - window_size + 1, stride):
            end_idx = start_idx + window_size

            window = data[start_idx:end_idx]  # (T, C, V)

            window_label = labels[start_idx + window_size // 2]

            window_labels_in_range = labels[start_idx:end_idx]
            if len(np.unique(window_labels_in_range)) == 1:
                windows.append(window)
                window_labels.append(window_label)

        return windows, window_labels

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess"""
        if self.normalize:
            data = super().normalize(data)
        return data

    def get_num_classes(self) -> int:
        """Number of classes."""
        return len(self.activities)
