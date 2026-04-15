"""
Daily and Sports Activities dataset feeder
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import random
from .base_feeder import BaseFeeder
from .split_utils import subject_independent_indices


class DailySports_Feeder(BaseFeeder):
    """Daily and Sports Activities dataset feeder"""

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """
        Args:
            config: config dict
            split: ``train`` | ``val`` | ``test``
        """
        self.data_root = Path(config['dataset']['data_root']) / "daily+and+sports+activities" / "data"
        self.activities = config['dataset']['daily_sports']['activities']
        self.subjects = config['dataset']['daily_sports']['subjects']
        self.segments = config['dataset']['daily_sports']['segments']
        self.normalize = config['dataset']['daily_sports']['normalize']
        self.train_split = config['dataset']['train_split']
        self.val_split = config['dataset']['val_split']
        self.test_split = config['dataset']['test_split']
        self.subject_independent_split = config['dataset'].get('subject_independent_split', True)

        # Sensor unit: 'torso', 'right_arm', 'left_arm', 'right_leg', 'left_leg', 'all'
        self.sensor_unit = config['dataset']['daily_sports'].get('sensor_unit', 'torso')
        # Modality: 'acc', 'gyro', 'mag', 'all'
        self.sensor_type = config['dataset']['daily_sports'].get('sensor_type', 'acc')

        super().__init__(config, split)

    def _load_data(self):
        all_samples = []
        all_labels = []
        all_units: List[Any] = []

        for activity in self.activities:
            for subject in self.subjects:
                for segment in self.segments:
                    filepath = self.data_root / f"a{activity:02d}" / f"p{subject}" / f"s{segment:02d}.txt"
                    if not filepath.exists():
                        continue

                    try:
                        # CSV-like: 125 rows, 45 columns = 5 units x 9 channels per unit
                        # Per unit: xacc, yacc, zacc, xgyro, ygyro, zgyro, xmag, ymag, zmag
                        data = pd.read_csv(filepath, header=None).values

                        sensor_data = self._extract_sensor_data(data)

                        if len(sensor_data.shape) == 2:
                            sensor_data = sensor_data[:, :, np.newaxis]  # (T, C, 1)

                        all_samples.append(sensor_data)
                        all_labels.append(activity - 1)  # activity ids from 0
                        all_units.append(subject)
                    except Exception as e:
                        print(f"Failed to load file {filepath}: {e}")
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
                    f"[Subject-independent] Daily-Sports | n_subjects={info['n_units']} | "
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

        print(f"Daily-Sports {self.split}: {len(self.data)} windows, {self.get_num_classes()} classes")

    def _extract_sensor_data(self, data: np.ndarray) -> np.ndarray:
        """
        Extract selected unit and modality from 45-column rows.

        Layout (0-based columns):
        - 0-8:   Torso (T)
        - 9-17:  Right arm (RA)
        - 18-26: Left arm (LA)
        - 27-35: Right leg (RL)
        - 36-44: Left leg (LL)
        Each block: xacc, yacc, zacc, xgyro, ygyro, zgyro, xmag, ymag, zmag

        Returns:
            (T, C) or (T, C, V)
        """
        unit_map = {
            'torso': (0, 9),
            'right_arm': (9, 18),
            'left_arm': (18, 27),
            'right_leg': (27, 36),
            'left_leg': (36, 45),
        }

        sensor_type_map = {
            'acc': (0, 3),
            'gyro': (3, 6),
            'mag': (6, 9),
            'all': (0, 9),
        }

        if self.sensor_unit == 'all':
            all_units_data = []
            for _unit_name, (start_col, end_col) in unit_map.items():
                unit_data = data[:, start_col:end_col]  # (T, 9)

                sensor_start, sensor_end = sensor_type_map[self.sensor_type]
                unit_sensor_data = unit_data[:, sensor_start:sensor_end]
                all_units_data.append(unit_sensor_data)

            return np.stack(all_units_data, axis=2)  # (T, C, 5)
        else:
            start_col, end_col = unit_map[self.sensor_unit]
            unit_data = data[:, start_col:end_col]  # (T, 9)

            sensor_start, sensor_end = sensor_type_map[self.sensor_type]
            sensor_data = unit_data[:, sensor_start:sensor_end]  # (T, C)

            return sensor_data  # (T, C); _load_data may add V dimension

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess"""
        if self.normalize:
            data = super().normalize(data)
        return data
