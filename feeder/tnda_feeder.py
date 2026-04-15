"""
TNDA dataset feeder.
TNDA-HAR: activity recognition with multiple sensor mounts; each mount has Acc, Gyr, Mag with X/Y/Z axes.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import random
from .base_feeder import BaseFeeder
from .split_utils import subject_independent_indices


class TNDA_Feeder(BaseFeeder):
    """TNDA dataset feeder"""

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """
        Args:
            config: config dict
            split: ``train`` | ``val`` | ``test``
        """
        self.data_root = Path(config['dataset']['data_root']) / "TNDADATASET"
        self.subjects = config['dataset']['tnda']['subjects']  # e.g. [1, 2, ..., 50]
        self.activities = config['dataset']['tnda']['activities']  # activity id list
        self.window_size = config['dataset']['tnda']['window_size']
        self.window_stride = config['dataset']['tnda']['window_stride']
        self.normalize = config['dataset']['tnda']['normalize']
        self.train_split = config['dataset']['train_split']
        self.val_split = config['dataset']['val_split']
        self.test_split = config['dataset']['test_split']
        self.subject_independent_split = config['dataset'].get('subject_independent_split', True)

        # Mount selection: 'all' or a list such as ['arm', 'leg', 'wri', 'ank', 'bac']
        self.sensor_positions = config['dataset']['tnda'].get('sensor_positions', 'all')
        # Sensor modality: 'acc', 'gyr', 'mag', 'all'
        self.sensor_type = config['dataset']['tnda'].get('sensor_type', 'acc')

        super().__init__(config, split)

    def _load_data(self):
        all_samples = []
        all_labels = []
        all_units: List[Any] = []

        for subject in self.subjects:
            subject_str = f"Subject{subject:02d}"
            csv_file = self.data_root / f"{subject_str}.csv"

            if not csv_file.exists():
                print(f"Warning: File not found {csv_file}")
                continue

            try:
                df = pd.read_csv(csv_file)

                label_col = None
                for col in df.columns:
                    if col.lower() in ['class', 'label', 'activity']:
                        label_col = col
                        break

                if label_col is None:
                    print(f"Warning: no label column in {csv_file}")
                    continue

                labels = df[label_col].values

                if self.activities:
                    mask = np.isin(labels, self.activities)
                    df_filtered = df[mask].copy()
                    labels_filtered = labels[mask]
                else:
                    df_filtered = df.copy()
                    labels_filtered = labels

                if len(df_filtered) == 0:
                    continue

                sensor_data = self._extract_sensor_data(df_filtered, label_col)

                if sensor_data is None or sensor_data.shape[1] == 0:
                    print(f"Warning: no matching sensor columns in {csv_file}")
                    continue

                windows, window_labels = self.sliding_window_with_labels(
                    sensor_data, labels_filtered, self.window_size, self.window_stride
                )

                if self.activities:
                    label_map = {label: idx for idx, label in enumerate(sorted(self.activities))}
                else:
                    unique_labels = sorted(np.unique(labels_filtered))
                    label_map = {label: idx for idx, label in enumerate(unique_labels)}

                for window, label in zip(windows, window_labels):
                    if label in label_map:
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
                    f"[Subject-independent] TNDA | n_subjects={info['n_units']} | "
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

        print(f"TNDA {self.split}: {len(self.data)} windows, {self.get_num_classes()} classes")

    def _extract_sensor_data(self, df: pd.DataFrame, label_col: str) -> np.ndarray:
        """
        Extract sensor columns into a tensor.

        Args:
            df: input table
            label_col: name of the label column

        Returns:
            sensor_data: shape (n_samples, C, V); C is usually 3 (X,Y,Z); V is number of mounts.
        """
        sensor_cols = [col for col in df.columns if col != label_col]

        sensor_dict = {}  # {position: {sensor_type: {axis: col}}}

        for col in sensor_cols:
            # Column pattern: {position}_{sensor_type}_{axis}, e.g. arm_Acc_X, leg_Gyr_Y
            parts = col.split('_')
            if len(parts) < 3:
                continue

            position = parts[0].lower()
            sensor_type = parts[1].lower()
            axis = parts[2].lower()

            if self.sensor_positions != 'all':
                if position not in [p.lower() for p in self.sensor_positions]:
                    continue

            if self.sensor_type != 'all':
                if sensor_type != self.sensor_type.lower():
                    continue

            if position not in sensor_dict:
                sensor_dict[position] = {}
            if sensor_type not in sensor_dict[position]:
                sensor_dict[position][sensor_type] = {}

            sensor_dict[position][sensor_type][axis] = col

        selected_channels = []
        for position in sorted(sensor_dict.keys()):
            for sensor_type in sorted(sensor_dict[position].keys()):
                axes = sensor_dict[position][sensor_type]
                if 'x' in axes and 'y' in axes and 'z' in axes:
                    x_col = axes['x']
                    y_col = axes['y']
                    z_col = axes['z']
                    if x_col in df.columns and y_col in df.columns and z_col in df.columns:
                        selected_channels.append([x_col, y_col, z_col])

        if len(selected_channels) == 0:
            return None

        channel_data_list = []
        for channel_cols in selected_channels:
            channel_data = df[channel_cols].values  # (n_samples, 3)
            channel_data_list.append(channel_data)

        sensor_data = np.stack(channel_data_list, axis=2)  # (n_samples, 3, V)

        return sensor_data

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
        return len(self.activities) if self.activities else 0
