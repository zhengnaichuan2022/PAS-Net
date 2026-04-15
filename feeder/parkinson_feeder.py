"""
Parkinson Freezing of Gait dataset feeder
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import random
from .base_feeder import BaseFeeder
from .split_utils import subject_independent_indices


class Parkinson_Feeder(BaseFeeder):
    """Parkinson Freezing of Gait dataset feeder"""

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """
        Args:
            config: config dict
            split: ``train`` | ``val`` | ``test``
        """
        self.data_root = Path(config['dataset']['data_root']) / "Parkinson" / "dataset"
        self.subjects = config['dataset']['parkinson']['subjects']
        self.recordings = config['dataset']['parkinson']['recordings']
        self.window_size = config['dataset']['parkinson']['window_size']
        self.window_stride = config['dataset']['parkinson']['window_stride']
        self.normalize = config['dataset']['parkinson']['normalize']
        self.train_split = config['dataset']['train_split']
        self.val_split = config['dataset']['val_split']
        self.test_split = config['dataset']['test_split']
        self.subject_independent_split = config['dataset'].get('subject_independent_split', True)

        # Task: 'binary' (normal vs freeze) or 'regression' (freeze severity)
        self.task_type = config['dataset']['parkinson'].get('task_type', 'binary')
        # Whether to use extra sensor columns (cols 6-11)
        self.use_other_sensors = config['dataset']['parkinson'].get('use_other_sensors', False)
        # 'acc' only or 'all' (accel + other sensors when enabled)
        self.sensor_type = config['dataset']['parkinson'].get('sensor_type', 'acc')

        super().__init__(config, split)

    def _load_data(self):
        all_samples = []
        all_labels = []
        all_units: List[Any] = []

        for subject in self.subjects:
            if not self.recordings:
                available_recordings = []
                for rec_file in self.data_root.glob(f"S{subject:02d}R*.txt"):
                    rec_num = int(rec_file.stem.split('R')[1])
                    available_recordings.append(rec_num)
                recordings_to_use = sorted(available_recordings)
            else:
                recordings_to_use = self.recordings

            for recording in recordings_to_use:
                filepath = self.data_root / f"S{subject:02d}R{recording:02d}.txt"
                if not filepath.exists():
                    continue

                try:
                    # 11 columns: timestamp, 3-axis accel, label, 6 other sensors
                    data = []
                    labels = []

                    with open(filepath, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 11:
                                accel = [int(parts[1]), int(parts[2]), int(parts[3])]
                                label = int(parts[4])  # 0 = normal, nonzero = freeze-related
                                other_sensors = [int(parts[i]) for i in range(5, 11)]

                                if self.sensor_type == 'acc':
                                    sample_data = accel
                                elif self.sensor_type == 'all':
                                    if self.use_other_sensors:
                                        sample_data = accel + other_sensors  # 9-D
                                    else:
                                        sample_data = accel
                                else:
                                    raise ValueError(f"Unknown sensor_type: {self.sensor_type}")

                                data.append(sample_data)
                                labels.append(label)

                    if not data:
                        continue

                    data = np.array(data)  # (n_samples, C)
                    labels = np.array(labels)

                    if self.task_type == 'binary':
                        labels = (labels != 0).astype(int)
                    elif self.task_type == 'regression':
                        labels = labels / 1000.0  # scale to roughly [-1, 1]
                    else:
                        raise ValueError(f"Unknown task_type: {self.task_type}")

                    windows = self.sliding_window(data, self.window_size, self.window_stride)
                    window_labels = []

                    for i in range(0, len(labels) - self.window_size + 1, self.window_stride):
                        window_label_values = labels[i:i + self.window_size]
                        if self.task_type == 'binary':
                            window_label = 1 if np.any(window_label_values == 1) else 0
                        else:
                            window_label = np.mean(window_label_values)
                        window_labels.append(window_label)

                    min_len = min(len(windows), len(window_labels))
                    windows = windows[:min_len]
                    window_labels = window_labels[:min_len]

                    for window, label in zip(windows, window_labels):
                        window = window[:, :, np.newaxis]  # (T, C, 1)
                        all_samples.append(window)
                        all_labels.append(label)
                        all_units.append(subject)

                except Exception as e:
                    print(f"Failed to load file {filepath}: {e}")
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
                    f"[Subject-independent] Parkinson | n_subjects={info['n_units']} | "
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

        if self.task_type == 'binary':
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print(f"Parkinson {self.split}: {len(self.data)} windows, {self.get_num_classes()} classes")
            for label, count in zip(unique_labels, counts):
                label_name = "normal" if label == 0 else "freeze"
                print(f"  {label_name}: {count} windows ({count/len(self.data)*100:.2f}%)")
        else:
            print(f"Parkinson {self.split}: {len(self.data)} windows (regression)")
            print(f"  label range: {min(self.labels):.4f} to {max(self.labels):.4f}")
            print(f"  label mean: {np.mean(self.labels):.4f}, std: {np.std(self.labels):.4f}")

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess"""
        if self.normalize:
            data = super().normalize(data)
        return data
