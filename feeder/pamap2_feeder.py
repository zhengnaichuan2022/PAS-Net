"""
PAMAP2 dataset feeder.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import random
from .base_feeder import BaseFeeder
from .split_utils import subject_independent_indices


class PAMAP2_Feeder(BaseFeeder):
    """PAMAP2 Protocol / windowed IMU loader."""

    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """
        Args:
            config: Full config dict
            split: ``train`` | ``val`` | ``test``
        """
        self.data_root = Path(config['dataset']['data_root']) / "pamap2+physical+activity+monitoring"
        self.subjects = config['dataset']['pamap2']['subjects']
        self.protocol_type = config['dataset']['pamap2']['protocol_type']
        self.activities = config['dataset']['pamap2']['activities']
        self.window_size = config['dataset']['pamap2']['window_size']
        self.window_stride = config['dataset']['pamap2']['window_stride']
        self.imu_position = config['dataset']['pamap2']['imu_position']
        self.sensor_type = config['dataset']['pamap2']['sensor_type']
        self.normalize = config['dataset']['pamap2']['normalize']
        self.train_split = config['dataset']['train_split']
        self.val_split = config['dataset']['val_split']
        self.test_split = config['dataset']['test_split']
        self.subject_independent_split = config['dataset'].get('subject_independent_split', True)
        
        super().__init__(config, split)
    
    def _load_data(self):
        all_samples = []
        all_labels = []
        all_raw_activity_ids: List[int] = []
        all_units: List[Any] = []
        
        # Per-subject .dat files
        for subject_id in self.subjects:
            filename = f"subject{subject_id}.dat"
            filepath = (self.data_root / "PAMAP2_Dataset" / "PAMAP2_Dataset" / 
                       self.protocol_type / filename)
            
            if not filepath.exists():
                continue
            
            try:
                # Read CSV-like .dat; NaN as missing
                data = pd.read_csv(filepath, sep=' ', header=None, na_values='NaN')
                data = data.values
                
                # Drop transient activity (id 0)
                valid_mask = (data[:, 1] != 0) & np.isin(data[:, 1], self.activities)
                data = data[valid_mask]
                
                if len(data) == 0:
                    continue
                
                # IMU columns for configured position / sensor
                imu_data = self._extract_imu_data(data)
                
                if imu_data is None or len(imu_data) == 0:
                    continue
                
                # Sliding windows
                windows = self.sliding_window(imu_data, self.window_size, self.window_stride)
                activities = data[:, 1].astype(int)

                # Class indices from config activity list only (sorted IDs), stable across subjects.
                # Protocol PAMAP2: 12 classes in self.activities when using protocol_type: Protocol.
                sorted_acts = sorted(int(x) for x in self.activities)
                activity_to_idx = {act_id: idx for idx, act_id in enumerate(sorted_acts)}
                
                for i, window in enumerate(windows):
                    # Label at window center
                    window_center = i * self.window_stride + self.window_size // 2
                    if window_center < len(activities):
                        activity_id = activities[window_center]
                        if activity_id in activity_to_idx:
                            all_samples.append(window)
                            # Map raw activity id to class index
                            activity_idx = activity_to_idx[activity_id]
                            all_labels.append(activity_idx)
                            all_raw_activity_ids.append(int(activity_id))
                            all_units.append(subject_id)
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
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
                    f"[Subject-independent] PAMAP2 | n_subjects={info['n_units']} | "
                    f"train={info['train_units']} | val={info['val_units']} | test={info['test_units']}"
                )
            self.data = [all_samples[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]
            self.raw_activity_ids = [all_raw_activity_ids[i] for i in indices]
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
            self.raw_activity_ids = [all_raw_activity_ids[i] for i in indices]

        print(f"PAMAP2 {self.split}: {len(self.data)} windows, {self.get_num_classes()} classes")
    
    def _extract_imu_data(self, data: np.ndarray) -> np.ndarray:
        """
        Extract IMU columns from PAMAP2 row format (54+ label columns).

        Returns:
            ``(n_samples, 3, V)`` or ``(n_samples, 6, V)`` for acc_gyro.
        """
        position_map = {
            'hand': (3, 19),
            'chest': (20, 36),
            'ankle': (37, 53)
        }

        if self.imu_position == 'all':
            hand_data = self._extract_sensor_data(data, 'hand')  # (n_samples, 3)
            chest_data = self._extract_sensor_data(data, 'chest')  # (n_samples, 3)
            ankle_data = self._extract_sensor_data(data, 'ankle')  # (n_samples, 3)
            return np.stack([hand_data, chest_data, ankle_data], axis=2)
        else:
            sensor_data = self._extract_sensor_data(data, self.imu_position)  # (n_samples, 3)
            return sensor_data[:, :, np.newaxis]  # (n_samples, 3, 1)
    
    def _extract_sensor_data(self, data: np.ndarray, position: str) -> np.ndarray:
        """
        Sensor slice for one IMU mount (hand / chest / ankle).
        """
        position_map = {
            'hand': (3, 19),
            'chest': (20, 36),
            'ankle': (37, 53)
        }
        
        start_col, end_col = position_map[position]
        
        if self.sensor_type == 'acc_16g':
            cols = [start_col + 1, start_col + 2, start_col + 3]
        elif self.sensor_type == 'acc_6g':
            cols = [start_col + 4, start_col + 5, start_col + 6]
        elif self.sensor_type == 'gyro':
            cols = [start_col + 7, start_col + 8, start_col + 9]
        elif self.sensor_type == 'acc_gyro':
            cols_acc = [start_col + 1, start_col + 2, start_col + 3]
            cols_gyro = [start_col + 7, start_col + 8, start_col + 9]
            acc = data[:, cols_acc]
            gyr = data[:, cols_gyro]
            df_a = pd.DataFrame(acc)
            df_a = df_a.ffill().bfill().fillna(0)
            df_g = pd.DataFrame(gyr)
            df_g = df_g.ffill().bfill().fillna(0)
            return np.concatenate([df_a.values, df_g.values], axis=1)  # (n_samples, 6)
        elif self.sensor_type == 'mag':
            cols = [start_col + 10, start_col + 11, start_col + 12]
        else:
            raise ValueError(f"Unknown sensor_type: {self.sensor_type}")
        
        sensor_data = data[:, cols]  # (n_samples, 3)
        
        df = pd.DataFrame(sensor_data)
        df = df.ffill().bfill().fillna(0)
        sensor_data = df.values  # (n_samples, 3)
        
        return sensor_data
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        if self.normalize:
            data = super().normalize(data)
        return data

