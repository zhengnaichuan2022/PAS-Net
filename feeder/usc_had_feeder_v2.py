"""
USC-HAD dataset feeder V2.
Uses ``sensor_location`` metadata to stack multiple IMU mounts when available.
"""
import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import Dict, Any, List
import random
from collections import defaultdict
from .base_feeder import BaseFeeder


class USC_HAD_Feeder_V2(BaseFeeder):
    """
    USC-HAD feeder with optional multi-mount stacking.

    MAT contents:
    - ``sensor_readings``: (n_samples, 6) — acc_xyz then gyro_xyz
    - ``sensor_location``: mount name (e.g. 'front-right-hip'); different files may differ
    """

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """
        Args:
            config: config dict
            split: ``train`` | ``val`` | ``test``
        """
        self.data_root = Path(config['dataset']['data_root']) / "USC-HAD" / "USC-HAD"
        self.subjects = config['dataset']['usc_had']['subjects']
        self.activities = config['dataset']['usc_had']['activities']
        self.trials = config['dataset']['usc_had']['trials']
        self.window_size = config['dataset']['usc_had']['window_size']
        self.window_stride = config['dataset']['usc_had']['window_stride']
        self.normalize = config['dataset']['usc_had']['normalize']
        self.train_split = config['dataset']['train_split']
        self.val_split = config['dataset']['val_split']
        self.test_split = config['dataset']['test_split']

        self.use_multiple_locations = config['dataset']['usc_had'].get('use_multiple_locations', False)
        self.sensor_type = config['dataset']['usc_had'].get('sensor_type', 'acc')  # 'acc', 'gyro', 'both'

        super().__init__(config, split)

    def _load_data(self):
        all_samples = []
        all_labels = []

        if self.use_multiple_locations:
            location_data = defaultdict(list)  # {location: [(window, label), ...]}

            for subject in self.subjects:
                for activity in self.activities:
                    for trial in self.trials:
                        filepath = self.data_root / f"Subject{subject}" / f"a{activity}t{trial}.mat"
                        if not filepath.exists():
                            continue

                        try:
                            data = loadmat(str(filepath))
                            sensor_readings = data['sensor_readings']  # (n_samples, 6)
                            sensor_location = str(data.get('sensor_location', ['unknown'])[0])

                            if self.sensor_type == 'acc':
                                sensor_data = sensor_readings[:, :3]
                            elif self.sensor_type == 'gyro':
                                sensor_data = sensor_readings[:, 3:6]
                            elif self.sensor_type == 'both':
                                sensor_data = sensor_readings
                            else:
                                raise ValueError(f"Unknown sensor_type: {self.sensor_type}")

                            windows = self.sliding_window(sensor_data, self.window_size, self.window_stride)

                            for window in windows:
                                location_data[sensor_location].append((window, activity - 1))
                        except Exception as e:
                            print(f"Failed to load file {filepath}: {e}")
                            continue

            if len(location_data) > 1:
                locations = sorted(location_data.keys())
                print(f"Found {len(locations)} IMU mounts: {locations}")

                location_samples = {loc: [] for loc in locations}
                location_labels = {loc: [] for loc in locations}

                for loc, samples in location_data.items():
                    for window, label in samples:
                        location_samples[loc].append(window)
                        location_labels[loc].append(label)

                min_len = min(len(samples) for samples in location_samples.values())

                for loc in locations:
                    location_samples[loc] = location_samples[loc][:min_len]
                    location_labels[loc] = location_labels[loc][:min_len]

                for i in range(min_len):
                    windows = [location_samples[loc][i] for loc in locations]
                    combined_window = np.stack(windows, axis=2)  # (T, C, V)
                    label = location_labels[locations[0]][i]
                    all_samples.append(combined_window)
                    all_labels.append(label)
            else:
                loc = list(location_data.keys())[0]
                for window, label in location_data[loc]:
                    window = window[:, :, np.newaxis] if len(window.shape) == 2 else window
                    all_samples.append(window)
                    all_labels.append(label)
        else:
            for subject in self.subjects:
                for activity in self.activities:
                    for trial in self.trials:
                        filepath = self.data_root / f"Subject{subject}" / f"a{activity}t{trial}.mat"
                        if not filepath.exists():
                            continue

                        try:
                            data = loadmat(str(filepath))
                            sensor_readings = data['sensor_readings']

                            if self.sensor_type == 'acc':
                                sensor_data = sensor_readings[:, :3]
                            elif self.sensor_type == 'gyro':
                                sensor_data = sensor_readings[:, 3:6]
                            elif self.sensor_type == 'both':
                                sensor_data = sensor_readings
                            else:
                                raise ValueError(f"Unknown sensor_type: {self.sensor_type}")

                            if len(sensor_data.shape) == 2:
                                sensor_data = sensor_data[:, :, np.newaxis]  # (n_samples, C, 1)

                            windows = self.sliding_window(sensor_data, self.window_size, self.window_stride)

                            for window in windows:
                                all_samples.append(window)
                                all_labels.append(activity - 1)
                        except Exception as e:
                            print(f"Failed to load file {filepath}: {e}")
                            continue

        indices = list(range(len(all_samples)))
        random.Random(self._split_seed()).shuffle(indices)

        n_total = len(all_samples)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)

        if self.split == 'train':
            indices = indices[:n_train]
        elif self.split == 'val':
            indices = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            indices = indices[n_train + n_val:]

        self.data = [all_samples[i] for i in indices]
        self.labels = [all_labels[i] for i in indices]

        if len(self.data) > 0:
            sample_shape = self.data[0].shape
            num_imus = sample_shape[2] if len(sample_shape) == 3 else 1
            print(
                f"USC-HAD {self.split}: {len(self.data)} windows, "
                f"{self.get_num_classes()} classes, {num_imus} IMU mounts"
            )
        else:
            print(f"USC-HAD {self.split}: {len(self.data)} windows, {self.get_num_classes()} classes")

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess"""
        if self.normalize:
            data = super().normalize(data)
        return data
