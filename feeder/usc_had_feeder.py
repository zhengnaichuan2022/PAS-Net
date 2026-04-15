"""
USC-HAD dataset feeder
"""
import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random
from .base_feeder import BaseFeeder
from .split_utils import subject_independent_indices


class USC_HAD_Feeder(BaseFeeder):
    """USC-HAD dataset feeder"""
    
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
        self.subject_independent_split = config['dataset'].get('subject_independent_split', True)
        
        self.sensor_type = config['dataset']['usc_had'].get('sensor_type', 'acc')  # 'acc', 'gyro', 'both'
        self.use_multiple_locations = config['dataset']['usc_had'].get('use_multiple_locations', False)
        
        super().__init__(config, split)
    
    def _load_data(self):
        all_samples = []
        all_labels = []
        all_units: List[Any] = []
        
        for subject in self.subjects:
            for activity in self.activities:
                for trial in self.trials:
                    filepath = self.data_root / f"Subject{subject}" / f"a{activity}t{trial}.mat"
                    if not filepath.exists():
                        continue
                    
                    try:
                        data = loadmat(str(filepath))
                        sensor_readings = data['sensor_readings']  # shape: (n_samples, 6)
                        # sensor_location may list mount name (e.g. 'front-right-hip')
                        sensor_location = data.get('sensor_location', ['unknown'])
                        
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
                        
                        # Sliding-window segmentation
                        windows = self.sliding_window(sensor_data, self.window_size, self.window_stride)
                        
                        for window in windows:
                            all_samples.append(window)  # window shape: (T, C, V) = (window_size, C, 1)
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
                    f"[Subject-independent] USC-HAD | n_subjects={info['n_units']} | "
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
        
        print(f"USC-HAD {self.split}: {len(self.data)} windows, {self.get_num_classes()} classes")
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess"""
        if self.normalize:
            data = super().normalize(data)
        return data

