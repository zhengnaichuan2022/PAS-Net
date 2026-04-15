"""
OPPORTUNITY Activity Recognition (UCI / Roggen et al.)

Raw ``.dat`` files live under ``{data_root}/{relative_root}/``; column names in ``dataset/column_names.txt``.

Default task: Locomotion 4-class (column 244, raw codes 1/2/4/5 → 0..3). Sensor columns match the
body-IMU subset in the official benchmark ``prepareData.m`` for UCIbegin1–3; subject S4 uses the
Motion Jacket subset (different ``selectedCol`` in scripts).

Splits:
- ``subject_independent`` (default): same as other datasets — subject IDs 1–4 via ``subject_independent_indices``.
- ``benchmark_runs``: train = ADL1–3 + Drill, test = ADL4–5; val is a fraction of train windows.
"""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .base_feeder import BaseFeeder
from .split_utils import subject_independent_indices


def _uci_sensor_columns_s123() -> List[int]:
    """MATLAB [1:46 51:59 ... 103:134] as 0-based indices (label column excluded)."""
    c: List[int] = []
    c.extend(range(0, 46))
    c.extend(range(50, 59))
    c.extend(range(63, 72))
    c.extend(range(76, 85))
    c.extend(range(89, 98))
    c.extend(range(102, 134))
    return c


def _uci_sensor_columns_s4() -> List[int]:
    """MATLAB [1 38:46 ... 90:98] as 0-based (Jacket-only subset)."""
    c = [0]
    c.extend(range(37, 46))
    c.extend(range(50, 59))
    c.extend(range(63, 72))
    c.extend(range(76, 85))
    c.extend(range(89, 98))
    return c


def _imu_accgyro_5nodes_specs() -> List[Tuple[str, List[int]]]:
    """
    Five on-body IMU nodes × 6 channels (acc+gyro). Column numbers from ``column_names.txt`` (1-based),
    returned here as 0-based indices: BACK, RUA, RLA, LUA, LLA.
    """
    return [
        ("BACK", [37, 38, 39, 40, 41, 42]),
        ("RUA", [50, 51, 52, 53, 54, 55]),
        ("RLA", [63, 64, 65, 66, 67, 68]),
        ("LUA", [76, 77, 78, 79, 80, 81]),
        ("LLA", [89, 90, 91, 92, 93, 94]),
    ]


def _parse_subject_from_name(name: str) -> int:
    m = re.match(r"S(\d+)-", name, re.I)
    if not m:
        raise ValueError(f"Cannot parse subject id from filename: {name}")
    return int(m.group(1))


def _reference_channel_count() -> int:
    return len(_uci_sensor_columns_s123())


def _pad_channels_to(X: np.ndarray, target_c: int) -> np.ndarray:
    """Pad channel dim to ``target_c`` with zeros on the right."""
    if X.ndim == 2:
        n, c = X.shape
        if c == target_c:
            return X
        if c > target_c:
            return X[:, :target_c]
        pad = np.zeros((n, target_c - c), dtype=X.dtype)
        return np.concatenate([X, pad], axis=1)
    if X.ndim == 3:
        n, c, v = X.shape
        if c == target_c:
            return X
        if c > target_c:
            return X[:, :target_c, :]
        pad = np.zeros((n, target_c - c, v), dtype=X.dtype)
        return np.concatenate([X, pad], axis=1)
    raise ValueError(f"unexpected ndim {X.ndim}")


class OpportunityFeeder(BaseFeeder):
    """OPPORTUNITY windowed loader (whitespace-separated ``.dat`` tables)."""

    # Column 244 (1-based) = Locomotion label; 0-based 243
    LOCOMOTION_COL_0 = 243

    def __init__(self, config: Dict[str, Any], split: str = "train"):
        ds = config["dataset"]
        self.opp = ds.get("opportunity", {})
        self.data_root = Path(ds["data_root"])
        self.rel_root = Path(self.opp.get("relative_root", "OpportunityUCIDataset/dataset"))
        self.dataset_dir = self.data_root / self.rel_root

        self.split_mode = str(self.opp.get("split_mode", "subject_independent")).lower()
        self.label_task = str(self.opp.get("label_task", "locomotion")).lower()
        self.sensor_preset = str(self.opp.get("sensor_preset", "uci_s123")).lower()

        self.window_size = int(self.opp.get("window_size", 30))
        self.window_stride = int(self.opp.get("window_stride", 15))
        self.normalize = bool(self.opp.get("normalize", True))
        self.train_split = float(ds.get("train_split", 0.7))
        self.val_split = float(ds.get("val_split", 0.15))
        self.test_split = float(ds.get("test_split", 0.15))
        self.subject_independent_split = bool(ds.get("subject_independent_split", True))

        # Raw locomotion code → class index
        raw_map = self.opp.get("locomotion_class_map", None)
        if raw_map is None:
            self.locomotion_map = {1: 0, 2: 1, 4: 2, 5: 3}
        else:
            self.locomotion_map = {int(k): int(v) for k, v in raw_map.items()}
        self.activities_raw = [int(x) for x in self.opp.get("activities", [1, 2, 4, 5])]

        self.missing_fill = str(self.opp.get("missing_fill", "ffill")).lower()
        self.val_fraction_of_train = float(self.opp.get("val_fraction_of_train", 0.15))
        self.benchmark_val_seed = int(self.opp.get("benchmark_val_seed", 42))
        self.pad_to_reference_channels = bool(self.opp.get("pad_to_reference_channels", True))
        self._ref_c = _reference_channel_count()

        super().__init__(config, split)

    def _label_col_0based(self) -> int:
        if self.label_task == "locomotion":
            return int(self.opp.get("locomotion_column_0based", self.LOCOMOTION_COL_0))
        raise ValueError(f"Unknown label_task: {self.label_task}")

    def _sensor_columns_for_subject(self, subject_id: int) -> List[int]:
        custom = self.opp.get("sensor_columns_0based", None)
        if custom is not None:
            return [int(x) for x in custom]
        if self.sensor_preset == "uci_s4" or (self.sensor_preset == "auto" and subject_id == 4):
            return _uci_sensor_columns_s4()
        return _uci_sensor_columns_s123()

    def _extract_imu_accgyro_5nodes(self, arr: np.ndarray) -> np.ndarray:
        """
        Build (n, 6, 5) from five IMU nodes. Out-of-range columns become NaN then filled.
        """
        n = arr.shape[0]
        node_feats: List[np.ndarray] = []
        for _name, cols in _imu_accgyro_5nodes_specs():
            if max(cols) < arr.shape[1]:
                x_node = arr[:, cols]
            else:
                x_node = np.full((n, 6), np.nan, dtype=np.float64)
            x_node = self._fill_missing(x_node).astype(np.float32)
            node_feats.append(x_node)
        return np.stack(node_feats, axis=2)  # (n, 6, 5)

    def _load_dat(self, path: Path) -> np.ndarray:
        """Load whitespace-separated ``.dat``; NaN = missing."""
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python", na_values=["NaN", "nan", "NA"])
        return df.values.astype(np.float64)

    def _fill_missing(self, X: np.ndarray) -> np.ndarray:
        if self.missing_fill == "none":
            return X
        dfp = pd.DataFrame(X)
        if self.missing_fill == "ffill":
            dfp = dfp.ffill(axis=0).bfill(axis=0)
        elif self.missing_fill == "zero":
            dfp = dfp.fillna(0.0)
        else:
            dfp = dfp.ffill(axis=0).bfill(axis=0)
        return dfp.values.astype(np.float32)

    def _iter_run_files(self) -> List[Path]:
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"OPPORTUNITY directory not found: {self.dataset_dir}")
        files = sorted(self.dataset_dir.glob("S*.dat"))
        if not files:
            raise FileNotFoundError(f"No S*.dat under: {self.dataset_dir}")
        return files

    def _filter_benchmark(self, paths: List[Path], train_part: bool) -> List[Path]:
        train_names = ("ADL1", "ADL2", "ADL3", "Drill")
        test_names = ("ADL4", "ADL5")
        out: List[Path] = []
        for p in paths:
            name = p.stem.upper()
            if train_part and any(t in name for t in train_names):
                out.append(p)
            if not train_part and any(t in name for t in test_names):
                out.append(p)
        return out

    def _load_data(self) -> None:
        all_samples: List[np.ndarray] = []
        all_labels: List[int] = []
        all_units: List[Any] = []

        label_col = self._label_col_0based()
        files = self._iter_run_files()

        if self.split_mode == "benchmark_runs":
            train_files = self._filter_benchmark(files, train_part=True)
            test_files = self._filter_benchmark(files, train_part=False)
            if self.split == "test":
                to_load = test_files
            elif self.split == "train":
                to_load = train_files
            else:
                to_load = train_files
        else:
            to_load = files

        for fp in to_load:
            subject_id = _parse_subject_from_name(fp.name)
            try:
                arr = self._load_dat(fp)
            except Exception as e:
                print(f"Warning: read failed {fp}: {e}")
                continue
            if arr.shape[1] < 244:
                print(f"Warning: not enough columns {arr.shape} {fp}")
                continue

            if label_col >= arr.shape[1]:
                print(f"Warning: label column out of range {fp}")
                continue

            if self.sensor_preset == "imu_accgyro_5nodes":
                X = self._extract_imu_accgyro_5nodes(arr)
            else:
                scols = self._sensor_columns_for_subject(subject_id)
                if max(scols) >= arr.shape[1]:
                    print(f"Warning: sensor column index out of range {fp}")
                    continue
                X = arr[:, scols]
                X = self._fill_missing(X)
                X = X.astype(np.float32)
                if self.pad_to_reference_channels and X.shape[1] != self._ref_c:
                    X = _pad_channels_to(X, self._ref_c)
                # (n, C) -> (n, C, 1)
                X = X[:, :, np.newaxis]

            y_raw = arr[:, label_col]

            labels = np.full(y_raw.shape[0], -1, dtype=np.int64)
            for i in range(len(y_raw)):
                v = y_raw[i]
                if np.isnan(v):
                    continue
                iv = int(round(float(v)))
                if iv == 0 or iv not in self.locomotion_map:
                    continue
                if iv not in self.activities_raw:
                    continue
                labels[i] = self.locomotion_map[iv]

            valid = labels >= 0
            X = X[valid]
            labels = labels[valid]
            if len(X) < self.window_size:
                continue

            windows, wlabs = self.sliding_window_with_labels(X, labels, self.window_size, self.window_stride)
            run_tag = f"{subject_id}:{fp.stem}"
            for w, wl in zip(windows, wlabs):
                all_samples.append(w)
                all_labels.append(int(wl))
                all_units.append(subject_id if self.split_mode == "subject_independent" else run_tag)

        if len(all_samples) == 0:
            print("Warning: OPPORTUNITY has no valid windows; check path, split, and labels.")
            self.data = []
            self.labels = []
            return

        if self.split_mode == "benchmark_runs":
            self._apply_benchmark_split(all_samples, all_labels)
            print(
                f"OPPORTUNITY benchmark_runs | {self.split} | windows {len(self.data)} | classes {self.get_num_classes()}"
            )
            return

        if self.subject_independent_split:
            indices, info = subject_independent_indices(
                all_units,
                self.split,
                self.train_split,
                self.val_split,
                self.test_split,
                seed=self._split_seed(),
            )
            if self.split == "train":
                print(
                    f"[Subject-independent] OPPORTUNITY | n_subjects={info['n_units']} | "
                    f"train={info['train_units']} | val={info['val_units']} | test={info['test_units']}"
                )
            self.data = [all_samples[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]
        else:
            idx = list(range(len(all_samples)))
            random.Random(self._split_seed()).shuffle(idx)
            n = len(idx)
            n_train = int(n * self.train_split)
            n_val = int(n * self.val_split)
            if self.split == "train":
                sel = idx[:n_train]
            elif self.split == "val":
                sel = idx[n_train : n_train + n_val]
            else:
                sel = idx[n_train + n_val :]
            self.data = [all_samples[i] for i in sel]
            self.labels = [all_labels[i] for i in sel]

        print(
            f"OPPORTUNITY {self.split_mode} | {self.split} | windows {len(self.data)} | classes {self.get_num_classes()}"
        )

    def _apply_benchmark_split(
        self,
        all_samples: List[np.ndarray],
        all_labels: List[int],
    ) -> None:
        """benchmark_runs: test = ADL4–5 only; train/val windows from ADL1–3+Drill with fixed seed."""
        if self.split == "test":
            self.data = all_samples
            self.labels = all_labels
            return

        n = len(all_samples)
        rng = random.Random(self.benchmark_val_seed)
        order = list(range(n))
        rng.shuffle(order)
        n_val = max(1, int(n * self.val_fraction_of_train))
        val_set = set(order[:n_val])
        if self.split == "val":
            self.data = [all_samples[i] for i in range(n) if i in val_set]
            self.labels = [all_labels[i] for i in range(n) if i in val_set]
        else:
            self.data = [all_samples[i] for i in range(n) if i not in val_set]
            self.labels = [all_labels[i] for i in range(n) if i not in val_set]

    def sliding_window_with_labels(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        window_size: int,
        stride: int,
    ) -> Tuple[List[np.ndarray], List[int]]:
        windows: List[np.ndarray] = []
        window_labels: List[int] = []
        n = data.shape[0]
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            seg = labels[start:end]
            if np.any(seg < 0):
                continue
            if len(np.unique(seg)) != 1:
                continue
            wl = int(seg[window_size // 2])
            windows.append(data[start:end])
            window_labels.append(wl)
        return windows, window_labels

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        if self.normalize:
            data = super().normalize(data)
        return data

    def get_num_classes(self) -> int:
        return len(set(self.locomotion_map.values()))

