#!/usr/bin/env python3
"""
Standalone early-exit evaluation for PAS-Net (IMU Physics Spikeformer) checkpoints.

Does not run training; loads weights and measures threshold-based exit on [T,B,C] logits.

Example:
  python tools/eval_early_exit.py \\
    --config snn-config/pamap2/pas_net.yaml \\
    --checkpoint logs/.../best_model.pth \\
    --split test \\
    --conf-threshold 0.9 \\
    --json-out early_exit_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feeder.feeder_factory import create_feeder
from snn_model.model_factory import create_model
from utils.config_loader import load_config
from utils.early_exit import evaluate_early_exit_inference
from utils.model_flags import is_imuphysics_aware_spikeformer_config


def collate_fn_btcv(batch):
    data_list, label_list = zip(*batch)
    data = torch.stack(data_list, dim=0)
    labels = torch.stack(label_list, dim=0)
    return data, labels


def resolve_device(cfg) -> torch.device:
    if torch.cuda.is_available() and cfg["device"]["cuda"]:
        gpu_ids = cfg["device"].get("gpu_ids", [0])
        if isinstance(gpu_ids, list) and len(gpu_ids) > 0:
            gpu_id = int(gpu_ids[0])
            n_dev = torch.cuda.device_count()
            if n_dev == 1:
                gpu_id = 0
            elif gpu_id >= n_dev:
                gpu_id = 0
            torch.cuda.set_device(gpu_id)
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate runtime early-exit on a PAS-Net checkpoint")
    p.add_argument("--config", type=str, required=True, help="Training YAML (same as train.py)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth (e.g. best_model.pth)")
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="Dataset split",
    )
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.9,
        help="Stop at first timestep with max softmax prob >= this (same as training early_exit_conf_threshold)",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write full metrics JSON (includes exit_distribution)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if not is_imuphysics_aware_spikeformer_config(cfg.config):
        print(
            "error: config is not PAS-Net (model.type pas_net + model_file PAS_Net.py); "
            "early-exit evaluation only applies to temporal [T,B,C] outputs.",
            file=sys.stderr,
        )
        sys.exit(2)

    device = resolve_device(cfg)
    torch.manual_seed(int(cfg["project"].get("seed", 42)))
    np.random.seed(int(cfg["project"].get("seed", 42)))

    print(f"Loading {args.split} split...")
    feeder = create_feeder(cfg.config, split=args.split)
    loader = DataLoader(
        feeder,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        collate_fn=collate_fn_btcv,
    )

    num_classes = feeder.get_num_classes()
    cfg.set("model.num_classes", num_classes)
    sample_data, _ = next(iter(loader))
    if len(sample_data.shape) == 4:
        _, _, C, V = sample_data.shape
        cfg.set("model.input_channels", C)
        cfg.set("model.num_imus", V)
        cfg.set("model.V_nodes", V)
        cfg.set("model.num_tokens", V)

    print("Building model and loading checkpoint...")
    model = create_model(cfg.config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    metrics = evaluate_early_exit_inference(
        model,
        loader,
        device,
        cfg.config,
        float(args.conf_threshold),
        split_name=args.split,
    )
    if metrics is None:
        print("error: evaluate_early_exit_inference returned None.", file=sys.stderr)
        sys.exit(3)

    dist = metrics.get("exit_distribution") or {}
    max_t = int(metrics.get("max_timesteps", 0) or 0)

    lines = [
        "=" * 72,
        "Early-exit evaluation (runtime threshold on softmax max per timestep)",
        "=" * 72,
        f"config:          {args.config}",
        f"checkpoint:      {args.checkpoint}",
        f"split:           {args.split}",
        f"conf_threshold:  {metrics['conf_threshold']:.6f}",
        f"n_windows:       {metrics['n_windows']}",
        f"skipped (T<=1):  {metrics['skipped_non_temporal']}",
        f"max_timesteps T: {max_t}",
        "-" * 72,
        "Exit step distribution (exit_t -> count of windows):",
    ]
    if dist:
        for t in sorted(dist.keys()):
            lines.append(f"  t={t:3d}:  {dist[t]:6d}")
    else:
        lines.append("  (empty — no valid temporal batches)")
    lines.extend(
        [
            "-" * 72,
            f"mean_exit_step:              {metrics['mean_exit_step']:.6f}",
            f"mean_observation_ratio:      {metrics['mean_observation_ratio']:.6f}  (mean (exit_t+1)/T)",
            f"mean_saved_step_ratio:       {metrics['mean_saved_step_ratio']:.6f}  "
            f"(est. compute saved vs full sequence, E[(T-1-exit_t)/T])",
            f"accuracy (%):                {metrics['accuracy_percent']:.4f}",
            f"macro-F1 (%):                {metrics['macro_f1_percent']:.4f}",
            "=" * 72,
        ]
    )
    print("\n".join(lines))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        print(f"Wrote JSON: {out_path.resolve()}")


if __name__ == "__main__":
    main()
