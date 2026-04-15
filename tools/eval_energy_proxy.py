#!/usr/bin/env python3
"""
Standalone compute-energy proxy analysis (params, FLOPs, SOPs, firing rates, layerwise proxy).

Uses the same ``analyze_model`` path as post-train energy in train.py — not measured silicon power.

Example:
  python tools/eval_energy_proxy.py \\
    --config snn-config/pamap2/pas_net_lite.yaml \\
    --checkpoint path/to/best_model.pth \\
    --split val \\
    --sample-batches 10 \\
    --log-out energy_proxy_analysis.log
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
from utils.energy_proxy_report import write_energy_proxy_analysis_file
from utils.model_analysis import analyze_model, infer_energy_mode_from_model_type


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


def build_loader(cfg, split: str):
    feeder = create_feeder(cfg.config, split=split)
    loader = DataLoader(
        feeder,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        collate_fn=collate_fn_btcv,
    )
    return feeder, loader


def print_console_summary(results: dict) -> None:
    print("-" * 72)
    print("Parameters")
    print("-" * 72)
    print(
        f"  {results['num_parameters']:,} "
        f"({results['num_parameters_millions']:.2f} M)"
    )
    print("-" * 72)
    print("FLOPs (single-sample theoretical / static graph where applicable)")
    print("-" * 72)
    if results.get("flops") is not None:
        print(f"  {results['flops']:,.0f}  ({results['flops_g']:.2f} G)")
    else:
        print("  (not available — install thop or ptflops, or see model-specific estimates)")
    print("-" * 72)
    print("Total SOPs (SNN: sparse accumulates; None for pure ANN mode)")
    print("-" * 72)
    ts = results.get("total_sops")
    if ts is not None:
        print(f"  {ts:,.0f}")
    else:
        print("  N/A")
    print("-" * 72)
    print("Estimated total compute energy proxy (pJ) — op counts × assumed EMAC/EAC")
    print("-" * 72)
    print(
        f"  {results['total_energy_pj']:.2f} pJ  "
        f"({results['total_energy_joule']:.2e} J)"
    )
    print(f"  energy_mode: {results.get('energy_mode', 'snn')}")
    print(f"  num_timesteps (inferred): {results.get('num_timesteps', 'N/A')}")
    print("-" * 72)
    print("LIF spike rates (layer -> rate)")
    print("-" * 72)
    lif = results.get("lif_activation_rates") or {}
    if not lif:
        print("  (none — ANN mode or no LIF hooks)")
    else:
        for _k, st in lif.items():
            pct = st.get("spike_rate_percent", st.get("spike_rate", 0.0) * 100.0)
            print(f"  {st.get('name', _k)}: {st.get('spike_rate', 0.0):.6f} ({pct:.4f}%)")
    print("-" * 72)
    print("Compute-layer input firing rates (layer -> rate)")
    print("-" * 72)
    comp = results.get("compute_input_firing_rates") or {}
    if not comp:
        print("  (none)")
    else:
        for _k, st in comp.items():
            pct = st.get("input_firing_rate_percent", st.get("input_firing_rate", 0.0) * 100.0)
            lt = st.get("layer_type", "?")
            print(
                f"  {st.get('name', _k)} ({lt}): "
                f"{st.get('input_firing_rate', 0.0):.6f} ({pct:.4f}%)"
            )
    print("-" * 72)
    print("Per-layer energy proxy (pJ) — see --log-out for full detail")
    print("-" * 72)
    ec = results.get("energy_consumption") or {}
    for name, st in ec.items():
        print(f"  {name}: {st.get('energy_pj', 0.0):.2f} pJ")
    print("-" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute-energy proxy (FLOPs, SOPs, firing, energy) on a checkpoint")
    p.add_argument("--config", type=str, required=True, help="Training YAML (same as train.py)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth (e.g. best_model.pth)")
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val", "test"),
        help="Dataloader split for activation sampling",
    )
    p.add_argument(
        "--sample-batches",
        type=int,
        default=10,
        help="Max number of batches to sample for firing-rate aggregation (capped by loader length)",
    )
    p.add_argument(
        "--snn-activation-agg",
        type=str,
        default="mean",
        choices=("mean", "min", "last"),
        help="How to combine LIF/compute firing stats across batches (same as train.py analyze_model)",
    )
    p.add_argument(
        "--emac",
        type=float,
        default=4.6e-12,
        help="Dense MAC energy in Joules (default 4.6e-12 = 4.6 pJ)",
    )
    p.add_argument(
        "--eac",
        type=float,
        default=0.1e-12,
        help="Sparse AC energy in Joules (default 0.1e-12 = 0.1 pJ)",
    )
    p.add_argument(
        "--log-out",
        type=str,
        default="",
        help="Optional path to write full energy_proxy_analysis.log-style report",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write full analyze_model dict as JSON",
    )
    p.add_argument(
        "--print-full-format",
        action="store_true",
        help="Also print full format_analysis_results() text to stdout (verbose)",
    )
    args = p.parse_args()

    if args.sample_batches < 1:
        print("error: --sample-batches must be >= 1", file=sys.stderr)
        sys.exit(2)

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    torch.manual_seed(int(cfg["project"].get("seed", 42)))
    np.random.seed(int(cfg["project"].get("seed", 42)))

    split_note = ""
    analysis_split = args.split
    try:
        feeder, loader = build_loader(cfg, args.split)
    except Exception as e:
        if args.split == "test":
            print(f"Warning: could not build test split ({e}); falling back to val.", file=sys.stderr)
            feeder, loader = build_loader(cfg, "val")
            analysis_split = "val"
            split_note = "requested test split failed; used val"
        else:
            raise

    if len(feeder) == 0 or len(loader) == 0:
        print("error: chosen split has no samples.", file=sys.stderr)
        sys.exit(3)

    num_classes = feeder.get_num_classes()
    cfg.set("model.num_classes", num_classes)
    sample_data, _ = next(iter(loader))
    if len(sample_data.shape) == 4:
        _, _, C, V = sample_data.shape
        cfg.set("model.input_channels", C)
        cfg.set("model.num_imus", V)
        cfg.set("model.V_nodes", V)
        cfg.set("model.num_tokens", V)

    model_cfg = cfg.config.get("model", {})
    _energy_mode = infer_energy_mode_from_model_type(model_cfg.get("type", ""))

    print("Building model and loading checkpoint...")
    model = create_model(cfg.config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    input_shape = tuple(sample_data.shape)
    n_batches = min(int(args.sample_batches), max(1, len(loader)))

    print(
        f"Running analyze_model (split={analysis_split}, batches={n_batches}, "
        f"energy_mode={_energy_mode})..."
    )
    results = analyze_model(
        model=model,
        input_shape=input_shape,
        device=device,
        sample_batches=n_batches,
        dataloader=loader,
        energy_mode=_energy_mode,
        emac=args.emac,
        eac=args.eac,
        snn_activation_agg=args.snn_activation_agg,
    )

    print("\n" + "=" * 72)
    print("Compute energy proxy (NOT measured device power)")
    print("=" * 72)
    print(f"config:     {args.config}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"split:      {analysis_split}")
    if split_note:
        print(f"note:       {split_note}")
    print(f"sample_batches (cap): {n_batches}")
    print("=" * 72 + "\n")

    print_console_summary(results)

    if args.print_full_format:
        from utils.model_analysis import format_analysis_results

        print("\n" + format_analysis_results(results))

    ckpt_path = Path(args.checkpoint)
    if args.log_out:
        write_energy_proxy_analysis_file(
            Path(args.log_out),
            results,
            checkpoint_path=ckpt_path,
            analysis_split=analysis_split,
            split_note=split_note,
            dataloader_batches_sampled=n_batches,
            dataset_name=cfg.config.get("dataset", {}).get("name", ""),
            project_name=cfg.config.get("project", {}).get("name", ""),
            model_cfg=cfg.config.get("model"),
        )
        print(f"\nWrote log: {Path(args.log_out).resolve()}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"Wrote JSON: {out_path.resolve()}")


if __name__ == "__main__":
    main()
