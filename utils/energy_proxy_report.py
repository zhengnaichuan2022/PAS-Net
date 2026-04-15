"""
Write compute-energy proxy reports (same content as legacy train.py post-train log).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from utils.model_analysis import format_analysis_results, snn_total_energy_uniform_pj


def write_energy_proxy_analysis_file(
    path: Path,
    analysis_results: dict,
    *,
    checkpoint_path: Path,
    analysis_split: str,
    split_note: str,
    dataloader_batches_sampled: int,
    dataset_name: str,
    project_name: str,
    model_cfg: Any,
) -> None:
    """
    Write compute-energy proxy report (not measured system/on-chip power).

    MAC/AC assumptions and exclusions are repeated in format_analysis_results() body text.
    """
    e01_pj = None
    if analysis_results.get("energy_mode", "snn") == "snn":
        e01_pj = snn_total_energy_uniform_pj(analysis_results, emac_pj=0.1, eac_pj=0.1)
    emac_assumption = float(analysis_results.get("emac_pj", 4.6))
    eac_assumption = float(analysis_results.get("eac_pj", 0.1))
    analysis_str = format_analysis_results(analysis_results)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("energy_proxy_analysis.log — compute energy proxy (NOT measured device/system energy)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Project: {project_name}\n")
        f.write("computed_on_trained_checkpoint: yes\n")
        f.write(f"checkpoint: {checkpoint_path.resolve()}\n")
        f.write(f"analysis_split: {analysis_split}\n")
        if split_note:
            f.write(f"split_note: {split_note}\n")
        f.write(f"dataloader_batches_sampled (cap): {dataloader_batches_sampled}\n")
        f.write(f"Model: {model_cfg}\n")
        f.write("MAC energy assumption (this run): EMAC = {:.6g} pJ per dense MAC\n".format(emac_assumption))
        f.write(
            "AC / SOP energy assumption (this run): EAC = {:.6g} pJ per sparse accumulate (SNN only)\n".format(
                eac_assumption
            )
        )
        f.write(
            "Overhead excluded from proxy: DRAM & memory hierarchy, off-chip I/O, host control, "
            "clock/control logic not modeled as MAC/SOP, analog/mixed-signal beyond EMAC/EAC mapping.\n"
        )
        f.write("=" * 80 + "\n\n")
        f.write(analysis_str)
        f.write("\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("Detailed metrics\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"Parameters: {analysis_results['num_parameters']:,} "
            f"({analysis_results['num_parameters_millions']:.2f}M)\n"
        )
        if analysis_results["flops"] is not None:
            f.write(f"FLOPs: {analysis_results['flops']:,.0f} ({analysis_results['flops_g']:.2f}G)\n")
        f.write(
            f"Total estimated compute energy proxy: {analysis_results['total_energy_pj']:.2f} pJ "
            f"({analysis_results['total_energy_joule']:.2e} J)\n"
        )
        if e01_pj is not None:
            f.write(
                f"Total proxy (sensitivity scan, EMAC=0.1pJ, EAC=0.1pJ): {e01_pj:.2f} pJ "
                f"({e01_pj * 1e-12:.2e} J)\n"
            )
        tsops = analysis_results.get("total_sops")
        if tsops is not None:
            f.write(f"Total SOPs: {tsops:,.0f}\n")
        f.write(f"Timesteps: {analysis_results.get('num_timesteps', 'N/A')}\n")

        f.write("\nLIF spike rate (fraction of outputs equal to 1)\n")
        f.write("-" * 80 + "\n")
        for module_name, stats in analysis_results["lif_activation_rates"].items():
            spike_rate_pct = stats.get("spike_rate_percent", stats.get("spike_rate", 0.0) * 100.0)
            f.write(
                f"  {stats['name']}: {stats['spike_rate']:.6f} ({spike_rate_pct:.4f}%) "
                f"[spikes: {stats['spikes']:,} / elements: {stats['total']:,}]\n"
            )

        f.write("\nCompute-layer input firing rate\n")
        f.write("-" * 80 + "\n")
        if "compute_input_firing_rates" in analysis_results:
            for module_name, stats in analysis_results["compute_input_firing_rates"].items():
                input_fr_pct = stats.get(
                    "input_firing_rate_percent", stats.get("input_firing_rate", 0.0) * 100.0
                )
                f.write(f"  {stats['name']} ({stats.get('layer_type', 'Unknown')}):\n")
                f.write(
                    f"    input firing rate: {stats['input_firing_rate']:.6f} ({input_fr_pct:.4f}%) "
                    f"[spikes: {stats['input_spikes']:,} / elements: {stats['input_total']:,}]\n"
                )

        f.write("\nPer-layer estimated compute energy proxy\n")
        f.write("-" * 80 + "\n")
        for module_name, stats in analysis_results["energy_consumption"].items():
            layer_type = (
                "First Layer (MAC)" if stats.get("is_first_layer", False) else "Subsequent Layer (AC)"
            )
            layer_type_info = stats.get("layer_type", "Unknown")
            input_fr = stats.get("input_firing_rate", 0.0)
            input_fr_pct = stats.get("input_firing_rate_percent", input_fr * 100.0)
            f.write(f"  {module_name} ({layer_type}, {layer_type_info}):\n")
            f.write(f"    layer_proxy: {stats['energy_pj']:.2f} pJ ({stats['energy_joule']:.2e} J)\n")
            f.write(f"    FLOPs: {stats.get('flops', 0):,.0f}\n")
            if not stats.get("is_first_layer", False):
                f.write(f"    SOPs: {stats.get('sops', 0):,.0f}\n")
            f.write(f"    input firing rate (for SOPs): {input_fr:.6f} ({input_fr_pct:.4f}%)\n")
        f.write("\n" + "=" * 80 + "\n")
