"""
Training script (SNN reset_net pattern aligned with spiking-topo-transformer/train.py).
"""
import argparse
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # mixed precision
import os
import time
from pathlib import Path
import numpy as np
from spikingjelly.activation_based import functional

from utils.config_loader import load_config
from feeder.feeder_factory import create_feeder
from snn_model.model_factory import create_model
from utils.model_analysis import (
    analyze_model,
    analyze_model_structure_reference,
    format_structure_reference_analysis,
    infer_energy_mode_from_model_type,
)
from utils.energy_proxy_report import write_energy_proxy_analysis_file
from utils.model_flags import is_imuphysics_aware_spikeformer_config
from utils.early_exit import evaluate_early_exit_inference, macro_f1_percent


def _is_tse_criterion(criterion) -> bool:
    return criterion.__class__.__name__.lower() == "tseloss"


def _reduce_temporal_logits(outputs: torch.Tensor, pred_time_mode: str) -> torch.Tensor:
    if len(outputs.shape) == 3 and outputs.shape[0] > 1:
        if pred_time_mode == 'mean':
            return outputs.mean(dim=0)
        return outputs[-1]
    return outputs


def train_epoch(model, train_loader, criterion, optimizer, device, config, epoch, log_dir=None, scaler=None, use_amp=False):
    """Run one training epoch (reset_net each batch, same as spiking-topo-transformer)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pred_rows = []
    label_rows = []

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        # Reset SNN state every batch
        functional.reset_net(model)
        
        # Data already in IMU layout; forward as-is
        
        optimizer.zero_grad()
        
        topology_l1_loss = 0.0
        if hasattr(config, 'config'):
            config_dict = config.config
        else:
            config_dict = config
        lambda_l1 = config_dict.get('training', {}).get('topology_l1_weight', 0.0)
        # YAML may parse 1e-4 as str; coerce to float
        try:
            lambda_l1 = float(lambda_l1)
        except Exception:
            lambda_l1 = 0.0
        if lambda_l1 > 0 and hasattr(model, 'get_topology_l1_loss'):
            try:
                topology_l1_loss = model.get_topology_l1_loss()
            except:
                topology_l1_loss = 0.0
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(data)
                pred_time_mode = config_dict.get('training', {}).get('pred_time_mode', 'last')
                outputs_for_loss = outputs if _is_tse_criterion(criterion) else _reduce_temporal_logits(outputs, pred_time_mode)
                task_loss = criterion(outputs_for_loss, labels)
                total_loss = task_loss + lambda_l1 * topology_l1_loss
            
            scaler.scale(total_loss).backward()
            # grad clip after unscale
            clip_val = float(config_dict.get('training', {}).get('clip_grad', 0.0) or 0.0)
            if clip_val > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=clip_val)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data)
            pred_time_mode = config_dict.get('training', {}).get('pred_time_mode', 'last')
            outputs_for_loss = outputs if _is_tse_criterion(criterion) else _reduce_temporal_logits(outputs, pred_time_mode)
            task_loss = criterion(outputs_for_loss, labels)
            total_loss = task_loss + lambda_l1 * topology_l1_loss
            total_loss.backward()
            clip_val = float(config_dict.get('training', {}).get('clip_grad', 0.0) or 0.0)
            if clip_val > 0:
                clip_grad_norm_(model.parameters(), max_norm=clip_val)
            optimizer.step()
        
        loss = total_loss
        
        running_loss += loss.item()
        
        # Temporal logits: TSELoss uses [T, B, num_classes]
        pred_time_mode = config_dict.get('training', {}).get('pred_time_mode', 'last')  # last | mean
        outputs_for_pred = _reduce_temporal_logits(outputs, pred_time_mode)
        
        _, predicted = outputs_for_pred.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pred_rows.append(predicted.detach().cpu().numpy())
        label_rows.append(labels.detach().cpu().numpy())

        if (batch_idx + 1) % config['training']['print_freq'] == 0:
            y_p = np.concatenate(pred_rows, axis=0)
            y_t = np.concatenate(label_rows, axis=0)
            run_f1 = macro_f1_percent(y_t, y_p)
            print(
                f'Epoch [{epoch}], Batch [{batch_idx+1}/{len(train_loader)}], '
                f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%, Macro-F1: {run_f1:.2f}%'
            )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    y_pred = np.concatenate(pred_rows, axis=0) if pred_rows else np.array([], dtype=np.int64)
    y_true = np.concatenate(label_rows, axis=0) if label_rows else np.array([], dtype=np.int64)
    epoch_macro_f1 = macro_f1_percent(y_true, y_pred)

    if log_dir is not None:
        log_file = log_dir / 'training.log'
        if hasattr(config, 'config') and 'log_file' in config.config.get('project', {}):
            log_file_path = Path(config.config['project']['log_file'])
            if not log_file_path.is_absolute():
                log_file = log_dir / log_file_path.name
            else:
                log_file = log_file_path
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(
                f'Epoch [{epoch}] Train - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, '
                f'Macro-F1: {epoch_macro_f1:.2f}%\n'
            )

    return epoch_loss, epoch_acc, epoch_macro_f1


def _write_early_exit_aggregates(
    f,
    n_correct_t,
    n_frac,
    n_cum_prefix,
    total: int,
    val_acc: float,
    power_thr: float,
):
    """
    Offline diagnostic only (oracle analysis on full [T,B,C] val logits after one forward).

    Per-timestep argmax accuracy and prefix-mean curves — upper-bound style analysis, not the
    same as runtime threshold early-exit (see ``evaluate_early_exit_inference`` / final_test.log).
    n_cum_prefix[t]: cumulative correct count for prefix-mean logits up to length t+1.
    """
    if total <= 0 or not n_correct_t:
        return
    T = len(n_correct_t)
    f.write(
        "EarlyExit OFFLINE diagnostic (per-step / prefix oracle on full sequence; not runtime exit)\n"
    )
    acc_ts = [100.0 * n_correct_t[t] / total for t in range(T)]
    acc_str = ", ".join(f"{x:.2f}" for x in acc_ts)
    frac_parts = []
    for k in range(8):
        acc_k = 100.0 * n_frac[k] / total
        frac_parts.append(f"{k + 1}/8:{acc_k:.2f}%")
    f.write(f"  Acc over time (%): {acc_str}\n")
    f.write(f"  Final ensembled Acc (pred_time reduction): {val_acc:.2f}%\n")
    f.write(f"  @Fraction 1/8..8/8: {' | '.join(frac_parts)}\n")
    if n_cum_prefix is not None and len(n_cum_prefix) == T:
        cum_accs = [100.0 * n_cum_prefix[t] / total for t in range(T)]
        cum_str = ", ".join(f"{x:.2f}" for x in cum_accs)
        f.write(f"  PrefixMean Acc cum. steps 1..{T} (%): {cum_str}\n")
    f.write(
        f"  (legacy power_thr ref={power_thr:.2f}% — runtime confidence exit uses "
        "training.early_exit_conf_threshold in evaluate_early_exit_inference)\n"
    )


def evaluate_split(
    model,
    loader,
    criterion,
    device,
    config,
    split_name: str = "val",
    log_dir=None,
):
    """
    Unified eval for train/val/test: loss, accuracy, macro-F1, sample count.
    For split_name='val' with log_dir, appends to training.log (including EarlyExit lines).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pred_time_mode = config.get('training', {}).get('pred_time_mode', 'last')  # last | mean

    pred_list = []
    label_list = []

    # EarlyExit block only for PAS-Net on val; omit on test to keep the held-out test report clean
    imuphysics = is_imuphysics_aware_spikeformer_config(config)
    log_early = (
        split_name == "val"
        and imuphysics
        and bool(config.get('training', {}).get('log_early_exit', True))
    )
    power_thr = float(config.get('training', {}).get('early_exit_power_threshold', 90.0))

    n_correct_t = None
    n_frac = [0] * 8
    n_cum_prefix = None
    t_ref = None

    n_batches = len(loader)
    if n_batches == 0:
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "num_samples": 0,
        }

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)

            functional.reset_net(model)

            outputs = model(data)
            outputs_for_loss = outputs if _is_tse_criterion(criterion) else _reduce_temporal_logits(outputs, pred_time_mode)
            loss = criterion(outputs_for_loss, labels)

            running_loss += loss.item()

            outputs_for_pred = _reduce_temporal_logits(outputs, pred_time_mode)

            _, predicted = outputs_for_pred.max(1)
            bs = labels.size(0)
            total += bs
            correct += predicted.eq(labels).sum().item()

            pred_list.append(predicted.detach().cpu().numpy())
            label_list.append(labels.detach().cpu().numpy())

            if (
                log_early
                and outputs.dim() == 3
                and outputs.shape[0] > 1
            ):
                tb = outputs.shape[0]
                if t_ref is None:
                    t_ref = tb
                    n_correct_t = [0] * tb
                    n_cum_prefix = [0] * tb
                if tb == t_ref:
                    for t in range(tb):
                        n_correct_t[t] += (outputs[t].argmax(1) == labels).sum().item()
                    for k in range(1, 9):
                        end = max(1, (k * tb + 7) // 8)
                        m = outputs[:end].float().mean(dim=0)
                        n_frac[k - 1] += (m.argmax(1) == labels).sum().item()
                    for end in range(1, tb + 1):
                        m = outputs[:end].float().mean(dim=0)
                        n_cum_prefix[end - 1] += (m.argmax(1) == labels).sum().item()

    split_loss = running_loss / n_batches
    split_acc = 100.0 * correct / total if total > 0 else 0.0

    y_pred = np.concatenate(pred_list, axis=0) if pred_list else np.array([], dtype=np.int64)
    y_true = np.concatenate(label_list, axis=0) if label_list else np.array([], dtype=np.int64)
    macro_f1 = macro_f1_percent(y_true, y_pred) if total > 0 else 0.0

    split_label = {"val": "Validation", "test": "Test", "train": "Train"}.get(split_name, split_name)

    if log_dir is not None and split_name == "val":
        log_file = log_dir / 'training.log'
        if 'log_file' in config.get('project', {}):
            log_file_path = Path(config['project']['log_file'])
            if not log_file_path.is_absolute():
                log_file = log_dir / log_file_path.name
            else:
                log_file = log_file_path

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(
                f'{split_label} - Loss: {split_loss:.4f}, Acc: {split_acc:.2f}%, Macro-F1: {macro_f1:.2f}%\n'
            )
            if log_early and n_correct_t is not None and t_ref is not None:
                _write_early_exit_aggregates(
                    f, n_correct_t, n_frac, n_cum_prefix, total, split_acc, power_thr
                )

    return {
        "loss": split_loss,
        "accuracy": split_acc,
        "macro_f1": macro_f1,
        "num_samples": total,
    }


def validate(model, val_loader, criterion, device, config, log_dir=None):
    """Validation: same as evaluate_split(split_name='val'); returns full metric dict."""
    return evaluate_split(
        model, val_loader, criterion, device, config, split_name="val", log_dir=log_dir
    )


def _val_score_for_selection(metrics: dict, selection_metric: str) -> float:
    """Scalar used for checkpoint / early stopping (both metrics on 0–100 scale)."""
    if selection_metric == "macro_f1":
        return float(metrics["macro_f1"])
    return float(metrics["accuracy"])


def _normalize_selection_metric(name: Optional[str]) -> str:
    if name is None or (isinstance(name, str) and not name.strip()):
        return "macro_f1"
    m = str(name).lower().strip()
    if m in ("accuracy", "acc"):
        return "accuracy"
    if m in ("macro_f1", "macro-f1", "f1", "macrof1"):
        return "macro_f1"
    return "macro_f1"


def training_selection_metric(training_cfg: dict) -> str:
    """
    Validation metric to maximize for checkpointing / early stopping.
    Prefer ``training.selection_metric``; fall back to legacy ``training.model_selection``.
    Default: ``macro_f1``.
    """
    raw = training_cfg.get("selection_metric")
    if raw is None:
        raw = training_cfg.get("model_selection")
    return _normalize_selection_metric(raw)


def imuphysics_eval_temporal_aggregates(model, val_loader, device, config: dict):
    """
    PAS-Net only: temporal / segment / prefix-mean accuracy on val (no log write here).
    Returns None if outputs are not [T,B,C].
    """
    if not is_imuphysics_aware_spikeformer_config(config):
        return None
    pred_time_mode = config.get('training', {}).get('pred_time_mode', 'last')
    model.eval()
    total = 0
    correct = 0
    n_correct_t = None
    n_frac = [0] * 8
    n_cum_prefix = None
    t_ref = None
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            functional.reset_net(model)
            outputs = model(data)
            if len(outputs.shape) == 3 and outputs.shape[0] > 1:
                if pred_time_mode == 'mean':
                    outputs_for_pred = outputs.mean(dim=0)
                else:
                    outputs_for_pred = outputs[-1]
            else:
                outputs_for_pred = outputs
            _, predicted = outputs_for_pred.max(1)
            bs = labels.size(0)
            total += bs
            correct += predicted.eq(labels).sum().item()
            if outputs.dim() != 3 or outputs.shape[0] <= 1:
                continue
            tb = outputs.shape[0]
            if t_ref is None:
                t_ref = tb
                n_correct_t = [0] * tb
                n_cum_prefix = [0] * tb
            if tb != t_ref:
                continue
            for t in range(tb):
                n_correct_t[t] += (outputs[t].argmax(1) == labels).sum().item()
            for k in range(1, 9):
                end = max(1, (k * tb + 7) // 8)
                m = outputs[:end].float().mean(dim=0)
                n_frac[k - 1] += (m.argmax(1) == labels).sum().item()
            for end in range(1, tb + 1):
                m = outputs[:end].float().mean(dim=0)
                n_cum_prefix[end - 1] += (m.argmax(1) == labels).sum().item()
    if total <= 0 or n_correct_t is None:
        return None
    val_acc = 100.0 * correct / total
    return {
        'total': total,
        'val_acc': val_acc,
        'n_correct_t': n_correct_t,
        'n_frac': n_frac,
        'n_cum_prefix': n_cum_prefix,
        'T': len(n_correct_t),
    }


def append_best_imuphysics_temporal_report(log_file: Path, config: dict, aggregates: dict, best_epoch: int):
    """Append full temporal-step report for the best checkpoint on the validation set."""
    power_thr = float(config.get('training', {}).get('early_exit_power_threshold', 90.0))
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(
            "Best checkpoint — PAS-Net OFFLINE temporal metrics (full validation set; oracle analysis)\n"
        )
        f.write(
            f"best_epoch={best_epoch}, val_acc={aggregates['val_acc']:.2f}%, "
            f"T={aggregates['T']}, n_val={aggregates['total']}\n"
        )
        _write_early_exit_aggregates(
            f,
            aggregates['n_correct_t'],
            aggregates['n_frac'],
            aggregates['n_cum_prefix'],
            aggregates['total'],
            aggregates['val_acc'],
            power_thr,
        )
        f.write("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='SNN IMU Activity Recognition Training')
    parser.add_argument('--config', type=str, default='snn-config/default_config.yaml',
                        help='Path to YAML config')
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint path to resume training')
    args = parser.parse_args()
    
    config = load_config(args.config)
    legacy_energy_off = str(config["training"].get("energy_analysis_when", "after")).lower() == "none"

    run_ts = int(time.time())
    log_dir = Path(config['project']['exp_dir']) / f"{config['project']['name']}_{run_ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {log_dir}")
    
    if torch.cuda.is_available() and config['device']['cuda']:
        gpu_ids = config['device'].get('gpu_ids', [0])
        if isinstance(gpu_ids, list) and len(gpu_ids) > 0:
            gpu_id = int(gpu_ids[0])
            n_dev = torch.cuda.device_count()
            # Single visible device (e.g. CUDA_VISIBLE_DEVICES): use cuda:0
            if n_dev == 1:
                gpu_id = 0
            elif gpu_id >= n_dev:
                print(
                    f'Warning: gpu_ids[0]={gpu_ids[0]} exceeds visible device count {n_dev}; using cuda:0'
                )
                gpu_id = 0
            torch.cuda.set_device(gpu_id)
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')
    
    torch.manual_seed(config['project']['seed'])
    np.random.seed(config['project']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['project']['seed'])
    
    def collate_fn_btcv(batch):
        """Stack feeder samples (T, C, V) into batch (B, T, C, V)."""
        data_list, label_list = zip(*batch)
        
        data = torch.stack(data_list, dim=0)  # (B, T, C, V)
        labels = torch.stack(label_list, dim=0)  # (B,)
        
        return data, labels
    
    print("Loading train split...")
    train_feeder = create_feeder(config.config, split='train')
    train_loader = DataLoader(
        train_feeder,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn_btcv
    )
    
    print("Loading val split...")
    val_feeder = create_feeder(config.config, split='val')
    val_loader = DataLoader(
        val_feeder,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn_btcv
    )
    
    num_classes = train_feeder.get_num_classes()
    config.set('model.num_classes', num_classes)
    print(f'Num classes: {num_classes}')
    
    sample_data, _ = next(iter(train_loader))
    if len(sample_data.shape) == 4:
        B, T, C, V = sample_data.shape
        config.set('model.input_channels', C)
        config.set('model.num_imus', V)
        config.set('model.V_nodes', V)
        config.set('model.num_tokens', V)
        print(f'Input shape: B={B}, T={T}, C={C}, V={V}')
    
    print("Building model...")
    model_config = config.config.get('model', {})
    print(f"Model type: {model_config.get('type', 'simple_snn')}")
    print(f"Model file: {model_config.get('model_file', 'N/A')}")
    print(f"Model class: {model_config.get('model_class', 'N/A')}")
    
    model = create_model(config.config)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    if 'log_file' in config['project']:
        log_file_path = Path(config['project']['log_file'])
        if not log_file_path.is_absolute():
            log_file_path = log_dir / log_file_path.name
    else:
        log_file_path = log_dir / 'training.log'
    
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"Train start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {config['dataset']['name']}\n")
        f.write(f"Project: {config['project']['name']}\n")
        f.write(f"Model: {config['model']}\n")
        f.write(
            f"Training: batch_size={config['training']['batch_size']}, "
            f"epochs={config['training']['num_epochs']}, lr={config['training']['optimizer']['lr']}\n"
        )
        f.write(f"Checkpoint dir: {config['project']['checkpoint_dir']}\n")
        f.write("="*80 + "\n")
    
    pretrain_structure_proxy_path = log_dir / "pretrain_structure_proxy.log"
    energy_proxy_analysis_path = log_dir / "energy_proxy_analysis.log"

    print(f"Training log: {log_file_path}")
    print(f"Pre-train structure reference (proxy naming): {pretrain_structure_proxy_path}")
    print(f"Post-train compute energy proxy log (after best checkpoint): {energy_proxy_analysis_path}")
    print(f"Checkpoints: {config['project']['checkpoint_dir']}")

    do_pretrain_structure = bool(config["training"].get("energy_analysis_pretrain", True)) and not legacy_energy_off

    if do_pretrain_structure:
        print("\nPre-train structure analysis (parameters + static FLOPs only; untrained weights)...")
        try:
            sample_data, _ = next(iter(train_loader))
            input_shape = sample_data.shape
            _energy_mode = infer_energy_mode_from_model_type(model_config.get("type", ""))
            struct_results = analyze_model_structure_reference(
                model=model,
                input_shape=input_shape,
                device=device,
                energy_mode=_energy_mode,
            )
            pretrain_str = format_structure_reference_analysis(struct_results)
            print(pretrain_str)
            with open(pretrain_structure_proxy_path, "w", encoding="utf-8") as f:
                f.write("pretrain_structure_proxy.log — parameters + static FLOPs only (untrained weights)\n")
                f.write("NOT activation-based compute energy proxy (see energy_proxy_analysis.log after training).\n")
                f.write("=" * 80 + "\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {config['dataset']['name']}\n")
                f.write(f"Project: {config['project']['name']}\n")
                f.write(f"Weights: initial / untrained (not best_model.pth)\n")
                f.write("\n")
                f.write(pretrain_str)
                f.write("\n")
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write("Pre-train structure reference written to: pretrain_structure_proxy.log\n")
                f.write(
                    f"Parameters: {struct_results['num_parameters']:,} "
                    f"({struct_results['num_parameters_millions']:.2f}M)\n"
                )
                if struct_results.get("flops") is not None:
                    f.write(
                        f"Theoretical FLOPs: {struct_results['flops']:,.0f} "
                        f"({struct_results['flops_g']:.2f}G)\n"
                    )
                f.write(
                    "Spike-domain compute energy proxy / firing rates: see energy_proxy_analysis.log after training.\n"
                )
                f.write("=" * 80 + "\n\n")
            print(f"  Wrote: {pretrain_structure_proxy_path}\n")
        except Exception as e:
            print(f"Pre-train structure analysis failed: {e}")
            print("Continuing training.\n")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping pre-train structure analysis (energy_analysis_pretrain=false or energy_analysis_when=none).\n")
    
    use_tse_loss = config['training']['loss'].get('use_tse', False)
    model_type = config.config.get('model', {}).get('type', '')
    force_ce = bool(config['training']['loss'].get('force_ce_for_imu_physics', False))
    
    if (not force_ce) and (
        use_tse_loss
        or model_type in ('pas_net', 'imu_physics_spikeformer', 'imu_physics_aware_spikeformer')
    ):
        from snn_model.PAS_Net import TSELoss
        base_criterion = nn.CrossEntropyLoss()
        if config['training']['loss'].get('label_smoothing', 0.0) > 0:
            from timm.loss import LabelSmoothingCrossEntropy
            base_criterion = LabelSmoothingCrossEntropy(
                smoothing=config['training']['loss']['label_smoothing']
            )
        loss_cfg = config['training']['loss']
        criterion = TSELoss(
            criterion=base_criterion,
            weighting_type=loss_cfg.get('tse_weighting', 'linear'),
            min_weight=float(loss_cfg.get('tse_min_weight', 0.1)),
            warmup_ratio=float(loss_cfg.get('tse_warmup_ratio', 0.2)),
        )
        print("Using TSELoss (Temporal Spike Error Loss)")
    elif force_ce and model_type in ('pas_net', 'imu_physics_spikeformer', 'imu_physics_aware_spikeformer'):
        criterion = nn.CrossEntropyLoss()
        print("IMUPhysics ablation: CrossEntropyLoss (forced)")
    elif config['training']['loss']['name'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['training']['loss']['name'] == 'label_smoothing':
        from timm.loss import LabelSmoothingCrossEntropy
        criterion = LabelSmoothingCrossEntropy(smoothing=config['training']['loss']['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer_config = config['training']['optimizer']
    if optimizer_config['name'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['name'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    elif optimizer_config['name'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config['momentum'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=optimizer_config['lr'])
    
    scheduler_config = config['training']['scheduler']
    warmup_epochs = scheduler_config.get('warmup_epochs', 0)
    warmup_lr = scheduler_config.get('warmup_lr', 1e-6)
    base_lr = optimizer_config['lr']
    
    warmup_lr = float(warmup_lr)
    base_lr = float(base_lr)
    
    def set_lr(optimizer, lr):
        """Set optimizer learning rate."""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # CosineAnnealingLR snapshots optimizer.lr at construction; build it at base_lr, then warmup overrides per epoch.
    if warmup_epochs > 0:
        set_lr(optimizer, warmup_lr)
        print(f"Warmup: start lr {warmup_lr:.6f}, ramp to {base_lr:.6f}")
    
    if scheduler_config['name'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        min_lr = float(scheduler_config.get('min_lr', 1e-5))
        T_max = config['training']['num_epochs'] - warmup_epochs
        T_max = max(T_max, 1)
        set_lr(optimizer, base_lr)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=min_lr
        )
        if warmup_epochs > 0:
            set_lr(optimizer, warmup_lr)
    elif scheduler_config['name'] == 'step':
        set_lr(optimizer, base_lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config['step_size'],
            gamma=scheduler_config['gamma']
        )
        if warmup_epochs > 0:
            set_lr(optimizer, warmup_lr)
    else:
        scheduler = None
    
    use_amp = config['training'].get('amp', False)
    scaler = GradScaler() if use_amp else None
    
    selection_metric = training_selection_metric(config['training'])

    start_epoch = 0
    best_metric_value = 0.0
    best_epoch = 0
    best_val_acc_at_best = 0.0
    best_val_macro_f1_at_best = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        selection_metric = _normalize_selection_metric(
            checkpoint.get("best_metric_name")
            or checkpoint.get("model_selection")
            or selection_metric
        )
        best_metric_value = float(
            checkpoint.get(
                "best_metric_value",
                checkpoint.get("best_val_score", checkpoint.get("best_acc", 0.0)),
            )
        )
        best_epoch = int(checkpoint.get('best_epoch', checkpoint.get('epoch', 0) or 0))
        best_val_acc_at_best = float(checkpoint.get('best_val_acc', checkpoint.get('best_acc', 0.0)))
        best_val_macro_f1_at_best = float(checkpoint.get('best_val_macro_f1', 0.0))
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(
            f"Resuming from epoch {start_epoch}, best validation metric ({selection_metric})="
            f"{best_metric_value:.2f}% "
            f"(val_acc={best_val_acc_at_best:.2f}%, val_macro_f1={best_val_macro_f1_at_best:.2f}%)"
        )
    
    early_stop_patience = int(config['training'].get('early_stopping_patience', 0) or 0)
    early_stop_min_delta = float(config['training'].get('early_stopping_min_delta', 0.0) or 0.0)
    eval_rounds_no_improve = 0

    print("Training...")
    print(
        f"Validation: reporting accuracy + macro-F1 (sklearn f1_score, average='macro'); "
        f"checkpoint maximizes: {selection_metric} (training.selection_metric or model_selection)"
    )
    print(f"LR schedule: warmup_epochs={warmup_epochs}, base_lr={base_lr}, warmup_lr={warmup_lr}")
    if early_stop_patience > 0:
        print(
            f"Early stopping: patience={early_stop_patience} (val evals), "
            f"min_delta={early_stop_min_delta:.4f}"
        )

    for epoch in range(start_epoch, config['training']['num_epochs']):
        if epoch < warmup_epochs:
            current_lr = warmup_lr + (base_lr - warmup_lr) * ((epoch + 1) / max(warmup_epochs, 1))
            set_lr(optimizer, current_lr)
        else:
            if scheduler:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        train_loss, train_acc, train_macro_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, config.config, epoch,
            log_dir=log_dir, scaler=scaler, use_amp=use_amp
        )
        
        if (epoch + 1) % config['training']['eval_freq'] == 0:
            val_metrics = validate(
                model, val_loader, criterion, device, config.config, log_dir=log_dir
            )
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            val_macro_f1 = val_metrics['macro_f1']
            val_score = _val_score_for_selection(val_metrics, selection_metric)
            print(
                f'Epoch [{epoch+1}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Train Macro-F1: {train_macro_f1:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Macro-F1: {val_macro_f1:.2f}%, '
                f'LR: {current_lr:.6f}'
            )
            
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(
                    f'Epoch [{epoch+1}/{config["training"]["num_epochs"]}], '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Macro-F1: {train_macro_f1:.2f}%, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Macro-F1: {val_macro_f1:.2f}%, '
                    f'LR: {current_lr:.6f}\n'
                )
            
            if val_score > (best_metric_value + early_stop_min_delta):
                best_metric_value = val_score
                best_epoch = epoch + 1
                best_val_acc_at_best = val_acc
                best_val_macro_f1_at_best = val_macro_f1
                eval_rounds_no_improve = 0
                checkpoint_dir = Path(config['project']['checkpoint_dir'])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    'epoch': epoch + 1,
                    'best_epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric_name': selection_metric,
                    'best_metric_value': best_metric_value,
                    'best_val_score': best_metric_value,
                    'model_selection': selection_metric,
                    'best_val_acc': best_val_acc_at_best,
                    'best_val_macro_f1': best_val_macro_f1_at_best,
                    'best_acc': best_val_acc_at_best,
                    'config': config.config,
                }
                if scaler is not None:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                torch.save(checkpoint, checkpoint_dir / 'best_model.pth')
                print(
                    f'Saved best_model.pth (metric={selection_metric}, value={best_metric_value:.2f}%; '
                    f'val acc={best_val_acc_at_best:.2f}%, val macro_f1={best_val_macro_f1_at_best:.2f}%)'
                )
            else:
                eval_rounds_no_improve += 1

            if early_stop_patience > 0 and eval_rounds_no_improve >= early_stop_patience:
                stop_msg = (
                    f"Early stopping: no val improvement ({selection_metric}) for {eval_rounds_no_improve} eval(s); "
                    f"best metric {best_metric_value:.2f}% (epoch {best_epoch}), "
                    f"val acc={best_val_acc_at_best:.2f}%, val macro_f1={best_val_macro_f1_at_best:.2f}%"
                )
                print(stop_msg)
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(stop_msg + '\n')
                break
        
        if (epoch + 1) % config['training']['save_freq'] == 0:
            checkpoint_dir = Path(config['project']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric_name': selection_metric,
                'best_metric_value': best_metric_value,
                'best_val_score': best_metric_value,
                'model_selection': selection_metric,
                'best_val_acc': best_val_acc_at_best,
                'best_val_macro_f1': best_val_macro_f1_at_best,
                'best_acc': best_val_acc_at_best,
                'config': config.config,
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    print(
        f'Training finished. Best validation metric ({selection_metric})={best_metric_value:.2f}% '
        f'(val_acc={best_val_acc_at_best:.2f}%, val_macro_f1={best_val_macro_f1_at_best:.2f}%) at epoch {best_epoch}'
    )

    checkpoint_dir = Path(config['project']['checkpoint_dir'])
    best_ckpt_path = checkpoint_dir / 'best_model.pth'
    run_final_test = bool(config['training'].get('run_final_test', True))

    test_loader = None
    if run_final_test:
        try:
            print("Building test_loader (held-out test; used only for post-train final eval)...")
            test_feeder = create_feeder(config.config, split='test')
            test_loader = DataLoader(
                test_feeder,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['training']['num_workers'],
                pin_memory=config['training']['pin_memory'],
                collate_fn=collate_fn_btcv,
            )
            print(f"Test windows: {len(test_feeder)}")
        except Exception as e:
            print(f"Warning: could not create test_loader; skipping final test: {e}")
            import traceback
            traceback.print_exc()

    final_test_log_path = log_dir / 'final_test.log'

    best_ckpt_loaded = False
    if best_ckpt_path.is_file():
        try:
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            ep = int(ckpt.get('best_epoch', ckpt.get('epoch', best_epoch)))
            best_ckpt_loaded = True

            do_post_energy = (
                bool(config["training"].get("energy_analysis_posttrain", True))
                and not legacy_energy_off
            )
            if do_post_energy:
                try:
                    split_req = str(config["training"].get("energy_analysis_split", "val")).lower()
                    if split_req not in ("val", "test"):
                        split_req = "val"
                    split_note = ""
                    if split_req == "test" and (
                        test_loader is None or len(test_loader) == 0
                    ):
                        energy_loader = val_loader
                        analysis_split = "val"
                        split_note = (
                            "requested test split but test_loader missing/empty; used val"
                        )
                    elif split_req == "test":
                        energy_loader = test_loader
                        analysis_split = "test"
                    else:
                        energy_loader = val_loader
                        analysis_split = "val"

                    sample_data_e, _ = next(iter(train_loader))
                    input_shape_e = sample_data_e.shape
                    _energy_mode = infer_energy_mode_from_model_type(
                        model_config.get("type", "")
                    )
                    n_batches_e = min(10, len(energy_loader))
                    post_analysis = analyze_model(
                        model=model,
                        input_shape=input_shape_e,
                        device=device,
                        sample_batches=n_batches_e,
                        dataloader=energy_loader,
                        energy_mode=_energy_mode,
                        emac=4.6e-12,
                        eac=0.1e-12,
                        snn_activation_agg="mean",
                    )
                    write_energy_proxy_analysis_file(
                        energy_proxy_analysis_path,
                        post_analysis,
                        checkpoint_path=best_ckpt_path,
                        analysis_split=analysis_split,
                        split_note=split_note,
                        dataloader_batches_sampled=n_batches_e,
                        dataset_name=config["dataset"]["name"],
                        project_name=config["project"]["name"],
                        model_cfg=config["model"],
                    )
                    print(f"Post-train compute energy proxy analysis written to: {energy_proxy_analysis_path}")
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write(
                            f"Post-train compute energy proxy (trained checkpoint): {energy_proxy_analysis_path.name}\n"
                        )
                        f.write(
                            f"checkpoint={best_ckpt_path.resolve()}, "
                            f"analysis_split={analysis_split}\n"
                        )
                        if split_note:
                            f.write(f"{split_note}\n")
                        f.write("=" * 80 + "\n")
                except Exception as e_e:
                    err_e = f"Post-train compute energy proxy analysis failed: {e_e}\n"
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(err_e)
                    print(err_e.strip())
                    import traceback
                    traceback.print_exc()

            if is_imuphysics_aware_spikeformer_config(config.config):
                agg = imuphysics_eval_temporal_aggregates(
                    model, val_loader, device, config.config
                )
                if agg is not None:
                    append_best_imuphysics_temporal_report(
                        log_file_path, config.config, agg, ep
                    )
                    print(f"Wrote PAS-Net temporal diagnostics (epoch={ep}) to training log.")
        except Exception as e:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"Best checkpoint load / temporal report skipped: {e}\n")
            print(f"Best checkpoint load or temporal report failed: {e}")

        if (
            best_ckpt_loaded
            and run_final_test
            and test_loader is not None
            and len(test_loader) > 0
        ):
            try:
                test_metrics = evaluate_split(
                    model,
                    test_loader,
                    criterion,
                    device,
                    config.config,
                    split_name="test",
                    log_dir=None,
                )
                ee_run = bool(config['training'].get('early_exit_inference', True))
                ee_thr = float(config['training'].get('early_exit_conf_threshold', 0.9))
                ee_metrics = None
                if ee_run and is_imuphysics_aware_spikeformer_config(config.config):
                    ee_metrics = evaluate_early_exit_inference(
                        model,
                        test_loader,
                        device,
                        config.config,
                        ee_thr,
                        split_name="test",
                    )

                with open(final_test_log_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("Final independent test evaluation\n")
                    f.write(
                        "(Held-out test split; not used for checkpoint selection or early stopping.)\n"
                    )
                    f.write("=" * 80 + "\n")
                    f.write(f"Dataset: {config['dataset']['name']}\n")
                    f.write(f"Config: {args.config}\n")
                    f.write(f"Best checkpoint (selected on validation): {best_ckpt_path.resolve()}\n")
                    f.write(
                        f"Best epoch (validation): {best_epoch} "
                        f"(best_metric_name={selection_metric}, best_metric_value={best_metric_value:.2f}%; "
                        f"val_acc={best_val_acc_at_best:.2f}%, val_macro_f1={best_val_macro_f1_at_best:.2f}%)\n"
                    )
                    f.write(f"Evaluated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"test_loss: {test_metrics['loss']:.6f}\n")
                    f.write(f"test_accuracy (%): {test_metrics['accuracy']:.4f}\n")
                    f.write(f"test_macro_f1 (%): {test_metrics['macro_f1']:.4f}\n")
                    f.write(f"test_sample_count (windows): {test_metrics['num_samples']}\n")
                    f.write("=" * 80 + "\n")

                    if ee_metrics is not None:
                        f.write("\n" + "=" * 80 + "\n")
                        f.write("Runtime early-exit inference (test set)\n")
                        f.write(
                            "Stop at first timestep t where max softmax probability >= threshold; "
                            "prediction = argmax at that t. If never, use t = T-1.\n"
                        )
                        f.write("=" * 80 + "\n")
                        if ee_metrics.get("n_windows", 0) > 0:
                            f.write("mode: softmax_max\n")
                            f.write(
                                f"early_exit_conf_threshold: {ee_metrics['conf_threshold']:.6f}\n"
                            )
                            f.write(f"n_windows: {ee_metrics['n_windows']}\n")
                            f.write(
                                f"skipped_non_temporal_batches (T<=1): "
                                f"{ee_metrics['skipped_non_temporal']}\n"
                            )
                            f.write(f"mean_exit_step: {ee_metrics['mean_exit_step']:.4f}\n")
                            f.write(
                                f"mean_observation_ratio: {ee_metrics['mean_observation_ratio']:.6f} "
                                f"(mean (exit_t+1)/T)\n"
                            )
                            f.write(
                                f"mean_saved_step_ratio: {ee_metrics['mean_saved_step_ratio']:.6f} "
                                f"(mean (T-exit_t-1)/T)\n"
                            )
                            f.write(
                                f"early_exit_test_accuracy (%): "
                                f"{ee_metrics['accuracy_percent']:.4f}\n"
                            )
                            f.write(
                                f"early_exit_test_macro_f1 (%): "
                                f"{ee_metrics['macro_f1_percent']:.4f}\n"
                            )
                        else:
                            f.write(
                                "Skipped or no temporal outputs: PAS-Net [T,B,C] with T>1 required.\n"
                            )
                        f.write("=" * 80 + "\n")

                print(
                    f"Final TEST — loss={test_metrics['loss']:.4f}, "
                    f"acc={test_metrics['accuracy']:.2f}%, "
                    f"macro-F1={test_metrics['macro_f1']:.2f}%, "
                    f"n={test_metrics['num_samples']}"
                )
                if ee_metrics is not None and ee_metrics.get("n_windows", 0) > 0:
                    print(
                        f"Runtime early-exit (thr={ee_thr:.3f}): "
                        f"mean_exit_step={ee_metrics['mean_exit_step']:.3f}, "
                        f"mean_saved_step_ratio={ee_metrics['mean_saved_step_ratio']:.4f}, "
                        f"acc={ee_metrics['accuracy_percent']:.2f}%"
                    )
                print(f"Final test metrics written to: {final_test_log_path}")

                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write("Final independent test (best val checkpoint)\n")
                    f.write(f"See: {final_test_log_path.name}\n")
                    f.write(
                        f"test_acc={test_metrics['accuracy']:.4f}%, "
                        f"test_macro_f1={test_metrics['macro_f1']:.4f}%, "
                        f"test_loss={test_metrics['loss']:.6f}, "
                        f"n={test_metrics['num_samples']}\n"
                    )
                    f.write("=" * 80 + "\n")
            except Exception as e:
                err_msg = f"Final test evaluation failed: {e}\n"
                with open(final_test_log_path, 'w', encoding='utf-8') as f:
                    f.write(err_msg)
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(err_msg)
                print(f"Final test evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        elif best_ckpt_loaded and run_final_test and (
            test_loader is None or len(test_loader) == 0
        ):
            msg = (
                "Final test skipped: test_loader missing or empty "
                "(check dataset test_split / feeder or run_final_test).\n"
            )
            with open(final_test_log_path, 'w', encoding='utf-8') as f:
                f.write(msg)
                f.write(
                    f"best_epoch (val)={best_epoch}, best_metric_name={selection_metric}, "
                    f"best_metric_value={best_metric_value:.2f}%, val_acc={best_val_acc_at_best:.2f}%, "
                    f"val_macro_f1={best_val_macro_f1_at_best:.2f}%\n"
                )
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(msg)
            print(msg.strip())
    else:
        skip = (
            f"No best checkpoint at {best_ckpt_path} (e.g. never saved or no val eval); "
            "skipping load, PAS-Net val temporal report, and final test.\n"
        )
        print(skip)
        with open(final_test_log_path, 'w', encoding='utf-8') as f:
            f.write(skip)
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(skip)

    with open(log_file_path, 'a', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"Train finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"Checkpoint selection (validation): best_metric_name={selection_metric}, "
            f"best_metric_value={best_metric_value:.2f}% "
            f"(val_acc={best_val_acc_at_best:.2f}%, val_macro_f1={best_val_macro_f1_at_best:.2f}%) "
            f"at epoch {best_epoch}\n"
        )
        f.write(f"Epochs configured: {config['training']['num_epochs']}\n")
        f.write(f"Best checkpoint: {Path(config['project']['checkpoint_dir']) / 'best_model.pth'}\n")
        if run_final_test:
            f.write(
                f"Held-out test metrics: {final_test_log_path.name} "
                f"(use test_accuracy / test_macro_f1 from this file for test-set reporting.)\n"
            )
        if bool(config["training"].get("energy_analysis_posttrain", True)) and not legacy_energy_off:
            f.write(
                f"Trained-model compute energy proxy: {energy_proxy_analysis_path.name} "
                f"(checkpoint best_model.pth; see analysis_split in file header)\n"
            )
        if bool(config["training"].get("energy_analysis_pretrain", True)) and not legacy_energy_off:
            f.write(
                f"Structure reference (untrained, not activation energy): {pretrain_structure_proxy_path.name}\n"
            )
    
    print(f"Training log: {log_file_path}")
    print(f"Best checkpoint: {Path(config['project']['checkpoint_dir']) / 'best_model.pth'}")


if __name__ == '__main__':
    main()

