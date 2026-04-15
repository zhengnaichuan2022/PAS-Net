"""
HuGaDB dataset feeder.

Data: ``{data_root}/{relative_root}/Data/*.txt``
Format: first 4 lines are metadata/header; tab-separated rows; last column is activity id ``act``.
"""
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_feeder import BaseFeeder
from .split_utils import subject_independent_indices


class HuGaDB_Feeder(BaseFeeder):
    """HuGaDB windowed IMU loader."""

    SENSOR_ORDER = ["RF", "RS", "RT", "LF", "LS", "LT"]

    def __init__(self, config: Dict[str, Any], split: str = "train"):
        ds = config["dataset"]
        self.hcfg = ds["hugadb"]

        self.data_root = Path(ds["data_root"])
        self.relative_root = Path(self.hcfg.get("relative_root", "HuGaDB"))
        self.data_dir = Path(self.hcfg.get("data_dir", "Data"))
        self.subjects = list(self.hcfg.get("subjects", list(range(1, 19))))
        self.activities = list(self.hcfg.get("activities", list(range(1, 13))))
        self.window_size = int(self.hcfg.get("window_size", 128))
        self.window_stride = int(self.hcfg.get("window_stride", 64))
        self.normalize = bool(self.hcfg.get("normalize", True))
        self.sensor_type = str(self.hcfg.get("sensor_type", "both")).lower()  # acc|gyro|both
        self.sensor_positions = self.hcfg.get("sensor_positions", "all")  # all|[RF, RS, ...]
        self.require_window_label_consensus = bool(self.hcfg.get("require_window_label_consensus", True))
        self.min_label_ratio = float(self.hcfg.get("min_label_ratio", 1.0))

        self.train_split = float(ds["train_split"])
        self.val_split = float(ds["val_split"])
        self.test_split = float(ds["test_split"])
        self.subject_independent_split = bool(ds.get("subject_independent_split", True))

        super().__init__(config, split)

    def _dataset_dir(self) -> Path:
        return (self.data_root / self.relative_root / self.data_dir).resolve()

    @staticmethod
    def _extract_subject_id(filename: str) -> Optional[int]:
        # HuGaDB_v1_activity_<participant>_<counter>.txt
        m = re.search(r"_([0-9]+)_[0-9]+\.txt$", filename)
        if not m:
            return None
        return int(m.group(1))

    def _selected_positions(self) -> List[str]:
        if self.sensor_positions == "all":
            return list(self.SENSOR_ORDER)
        positions = [str(x).upper() for x in self.sensor_positions]
        valid = [p for p in positions if p in self.SENSOR_ORDER]
        return valid

    def _parse_one_file(self, fp: Path) -> Tuple[np.ndarray, np.ndarray]:
        # Line 4 is the column header (0-based index 3)
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            _ = f.readline()
            _ = f.readline()
            _ = f.readline()
            header_line = f.readline().strip()
        if not header_line:
            raise ValueError(f"HuGaDB empty header: {fp}")

        columns = header_line.split("\t")
        raw = np.genfromtxt(str(fp), delimiter="\t", skip_header=4)
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        if raw.shape[1] != len(columns):
            raise ValueError(
                f"Column count mismatch: file={fp.name}, data_cols={raw.shape[1]}, header_cols={len(columns)}"
            )

        idx = {c: i for i, c in enumerate(columns)}
        act_idx = idx.get("act")
        if act_idx is None:
            raise ValueError(f"Missing act column: {fp}")

        positions = self._selected_positions()
        selected_cols: List[int] = []
        for pos in positions:
            if self.sensor_type in ("acc", "both"):
                for ax in ("x", "y", "z"):
                    key = f"{pos}_acc_{ax}"
                    if key not in idx:
                        raise ValueError(f"Missing column {key}: {fp}")
                    selected_cols.append(idx[key])
            if self.sensor_type in ("gyro", "both"):
                for ax in ("x", "y", "z"):
                    key = f"{pos}_gyro_{ax}"
                    if key not in idx:
                        raise ValueError(f"Missing column {key}: {fp}")
                    selected_cols.append(idx[key])

        x = raw[:, selected_cols].astype(np.float32)  # (T, C*V)
        y = raw[:, act_idx].astype(np.int64)  # (T,)

        # reshape to (T, C, V)
        n_pos = len(positions)
        c_per_pos = (3 if self.sensor_type in ("acc", "gyro") else 6)
        x = x.reshape(x.shape[0], n_pos, c_per_pos).transpose(0, 2, 1)
        return x, y

    def _window_with_labels(self, data: np.ndarray, labels: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        xs: List[np.ndarray] = []
        ys: List[int] = []
        n = data.shape[0]
        for s in range(0, n - self.window_size + 1, self.window_stride):
            e = s + self.window_size
            w = data[s:e]
            seg = labels[s:e]
            vals, counts = np.unique(seg, return_counts=True)
            top = int(vals[np.argmax(counts)])
            ratio = float(np.max(counts)) / float(len(seg))
            if self.require_window_label_consensus and ratio < self.min_label_ratio:
                continue
            if top not in self.activities:
                continue
            xs.append(w)
            ys.append(top)
        return xs, ys

    def _load_data(self):
        root = self._dataset_dir()
        if not root.exists():
            raise FileNotFoundError(f"HuGaDB data directory not found: {root}")

        files = sorted(root.glob("*.txt"))
        all_samples: List[np.ndarray] = []
        all_labels: List[int] = []
        all_units: List[Any] = []

        label_map = {lab: i for i, lab in enumerate(sorted(self.activities))}

        for fp in files:
            sid = self._extract_subject_id(fp.name)
            if sid is None or sid not in self.subjects:
                continue
            try:
                x, y = self._parse_one_file(fp)
                wx, wy = self._window_with_labels(x, y)
                for w, lab in zip(wx, wy):
                    all_samples.append(w)
                    all_labels.append(label_map[lab])
                    all_units.append(sid)
            except Exception as e:
                print(f"HuGaDB skip file {fp.name}: {e}")
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
            if self.split == "train":
                print(
                    f"[Subject-independent] HuGaDB | n_subjects={info['n_units']} | "
                    f"train={info['train_units']} | val={info['val_units']} | test={info['test_units']}"
                )
            self.data = [all_samples[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]
        else:
            idxs = list(range(len(all_samples)))
            random.Random(self._split_seed()).shuffle(idxs)
            n = len(idxs)
            n_train = int(n * self.train_split)
            n_val = int(n * self.val_split)
            if self.split == "train":
                sel = idxs[:n_train]
            elif self.split == "val":
                sel = idxs[n_train : n_train + n_val]
            else:
                sel = idxs[n_train + n_val :]
            self.data = [all_samples[i] for i in sel]
            self.labels = [all_labels[i] for i in sel]

        print(f"HuGaDB {self.split}: {len(self.data)} windows, {self.get_num_classes()} classes")

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        if self.normalize:
            data = super().normalize(data)
        return data

    def get_num_classes(self) -> int:
        return len(self.activities)
