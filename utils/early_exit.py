"""
Runtime early-exit inference (PAS-Net [T,B,C] logits): threshold on softmax max per timestep.

Used by train.py final test and by ``tools/eval_early_exit.py``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from spikingjelly.activation_based import functional

from utils.model_flags import is_imuphysics_aware_spikeformer_config


def macro_f1_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 in [0, 100], same scale as accuracy in training scripts."""
    if y_true.size == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro")) * 100.0


def early_exit_predict(
    outputs: torch.Tensor,
    conf_threshold: float,
    mode: str = "softmax_max",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runtime early-exit inference on full-sequence logits [T, B, C].

    Scans timesteps t = 0 .. T-1; stops at the first t where the confidence rule is met.
    If never met, uses the last timestep (same cost as full sequence, conservative).

    Args:
        outputs: Float logits, shape [T, B, C].
        conf_threshold: In (0, 1]; for ``softmax_max``, compare max softmax prob to this.
        mode: ``softmax_max`` only (max class probability after softmax).

    Returns:
        exit_t: LongTensor [B], index in 0 .. T-1 where inference stopped.
        predictions: LongTensor [B], predicted class at ``exit_t``.
    """
    if mode != "softmax_max":
        raise ValueError(f"Unsupported early_exit_predict mode: {mode}")
    if outputs.dim() != 3:
        raise ValueError(f"Expected logits [T, B, C], got {tuple(outputs.shape)}")
    T, B, _ = outputs.shape
    probs = torch.softmax(outputs.float(), dim=-1)
    conf, pred_t = probs.max(dim=-1)
    if T == 1:
        exit_t = torch.zeros(B, dtype=torch.long, device=outputs.device)
        return exit_t, pred_t[0]

    t_idx = torch.arange(T, device=outputs.device, dtype=torch.long).view(T, 1).expand(T, B)
    sentinel = torch.full_like(t_idx, T, dtype=torch.long)
    masked_t = torch.where(conf >= conf_threshold, t_idx, sentinel)
    exit_t = masked_t.min(dim=0).values
    no_hit = exit_t >= T
    exit_t = torch.where(no_hit, torch.full_like(exit_t, T - 1), exit_t)
    predictions = pred_t.gather(0, exit_t.unsqueeze(0)).squeeze(0)
    return exit_t, predictions


def evaluate_early_exit_inference(
    model,
    loader,
    device,
    config: dict,
    conf_threshold: float,
    split_name: str = "test",
) -> Optional[Dict]:
    """
    Measure *runtime* early exit: threshold-based stop on [T,B,C] logits.

    Aggregates exit-step histogram, mean exit index, mean observation ratio (exit_t+1)/T,
    mean saved-step ratio (T - exit_t - 1) / T, accuracy and macro-F1 of predictions at exit.

    Returns None if model is not PAS-Net or outputs are not temporal [T,B,C] with T>1.
    """
    if not is_imuphysics_aware_spikeformer_config(config):
        return None

    model.eval()
    all_pred: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_exit_t: List[np.ndarray] = []
    sum_exit = 0.0
    sum_obs = 0.0
    sum_saved = 0.0
    n_seq = 0
    skipped = 0
    max_t_seen = 1

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            functional.reset_net(model)
            outputs = model(data)
            if outputs.dim() != 3 or outputs.shape[0] <= 1:
                skipped += int(labels.size(0))
                continue
            T = int(outputs.shape[0])
            max_t_seen = max(max_t_seen, T)
            exit_t, pred = early_exit_predict(outputs.detach(), conf_threshold, mode="softmax_max")
            all_pred.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            et_np = exit_t.cpu().numpy()
            all_exit_t.append(et_np)
            bs = labels.size(0)
            n_seq += bs
            et = exit_t.float()
            sum_exit += et.sum().item()
            sum_obs += ((et + 1.0) / float(T)).sum().item()
            sum_saved += ((float(T) - 1.0 - et) / float(T)).sum().item()

    if n_seq == 0:
        return {
            "n_windows": 0,
            "skipped_non_temporal": skipped,
            "mean_exit_step": float("nan"),
            "mean_observation_ratio": float("nan"),
            "mean_saved_step_ratio": float("nan"),
            "accuracy_percent": 0.0,
            "macro_f1_percent": 0.0,
            "conf_threshold": conf_threshold,
            "split": split_name,
            "exit_distribution": {},
            "max_timesteps": max_t_seen,
        }

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    exit_flat = np.concatenate(all_exit_t, axis=0).astype(np.int64)
    unique, counts = np.unique(exit_flat, return_counts=True)
    exit_distribution = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "n_windows": n_seq,
        "skipped_non_temporal": skipped,
        "mean_exit_step": sum_exit / n_seq,
        "mean_observation_ratio": sum_obs / n_seq,
        "mean_saved_step_ratio": sum_saved / n_seq,
        "accuracy_percent": 100.0 * float((y_pred == y_true).mean()),
        "macro_f1_percent": macro_f1_percent(y_true, y_pred),
        "conf_threshold": conf_threshold,
        "split": split_name,
        "exit_distribution": exit_distribution,
        "max_timesteps": max_t_seen,
    }
