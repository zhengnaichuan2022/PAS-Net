"""
SNN model analysis: FLOPs, LIF firing rates, and theoretical energy proxy.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np
from spikingjelly.activation_based import neuron, functional
try:
    import snntorch as snn  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    snn = None


def _is_spike_neuron_module(module: nn.Module) -> bool:
    """
    Spike neurons aligned with the data path: LIF/PLIF and TrueDynamicThresholdLIF in the main model.
    Used for activation-rate stats and identifying the \"previous LIF\" in forward order.
    """
    if isinstance(module, neuron.LIFNode):
        return True
    plif = getattr(neuron, "ParametricLIFNode", None)
    if plif is not None and isinstance(module, plif):
        return True
    if module.__class__.__name__ == "TrueDynamicThresholdLIF":
        return True
    # SeqSNN / other ports may use snntorch neurons.
    if snn is not None:
        leaky = getattr(snn, "Leaky", None)
        if leaky is not None and isinstance(module, leaky):
            return True
    # Name-based fallback for custom spike neurons not inheriting standard classes.
    cname = module.__class__.__name__.lower()
    if any(k in cname for k in ("lif", "ifnode", "spikenode")):
        return True
    return False


def _extract_spike_tensor(output):
    """
    Normalize spike-neuron outputs to a tensor for FR counting.
    Supports:
      - Tensor
      - tuple/list like (spk, mem)
      - dict with key 'spk' / 'spike'
    """
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
        return None
    if isinstance(output, dict):
        for k in ("spk", "spike", "out"):
            v = output.get(k)
            if isinstance(v, torch.Tensor):
                return v
    return None


def _count_spikes(t: torch.Tensor) -> int:
    """
    Robust spike counting:
      - Binary-like outputs (0/1 or {-1,1}) -> non-zero count.
      - Analog outputs -> threshold at 0.5 to avoid treating dense positives as spikes.
    """
    if t.numel() == 0:
        return 0
    # Fast binary-likeness test on a small sample.
    flat = t.detach().flatten()
    sample = flat if flat.numel() <= 4096 else flat[:: max(1, flat.numel() // 4096)]
    uniq = torch.unique(sample)
    if uniq.numel() <= 4:
        # Typical spike tensors are binary/signed-binary.
        return int((t != 0).sum().item())
    return int((t > 0.5).sum().item())


def trace_compute_upstream_lif(
    model: nn.Module,
    sample_input: torch.Tensor,
) -> Tuple[Dict[str, Optional[str]], Set[str]]:
    """
    One forward pass, execution order: for each Conv/Linear, record the most recent spike
    neuron module that finished output before this layer ran.

    SOP convention: for spike-domain compute layers, fr(upstream LIF) × MACs(this layer),
    i.e. multiply ops between adjacent LIFs by the driving LIF's rate (last_lif updates per layer if
    there are intermediate LIFs in the segment).
    """
    last_lif: Optional[str] = None
    seen_spike: bool = False
    upstream: Dict[str, Optional[str]] = {}
    pre_spike_compute_names: Set[str] = set()
    handles = []

    def make_lif_hook(name: str):
        def hook(module, inp, out):
            nonlocal last_lif, seen_spike
            last_lif = name
            seen_spike = True

        return hook

    def make_conv_hook(name: str):
        def hook(module, inp, out):
            upstream[name] = last_lif
            if not seen_spike:
                pre_spike_compute_names.add(name)

        return hook

    for name, m in model.named_modules():
        if _is_spike_neuron_module(m):
            handles.append(m.register_forward_hook(make_lif_hook(name)))
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(make_conv_hook(name)))

    model.eval()
    with torch.no_grad():
        functional.reset_net(model)
        model(sample_input)
    for h in handles:
        h.remove()
    return upstream, pre_spike_compute_names


def estimate_rtsfnet_ann_flops(model: nn.Module, input_shape: Tuple) -> Optional[float]:
    """
    Coarse-to-fine RTSFNetANN FLOPs estimate:
    - Linear layers from structure parameters
    - Rodrigues rotation and TSF stats (main part missed by generic hooks)
    Returns FLOPs per single sample (B=1).
    """
    try:
        if model.__class__.__name__ != 'RTSFNetANN':
            return None

        # input_shape: (B, T, C, V)
        if len(input_shape) != 4:
            return None
        _, T, C, V = input_shape
        B = 1

        H = int(getattr(model, 'heads', 4))
        block_sizes = tuple(int(x) for x in getattr(model, 'block_sizes', (16, 32, 64)))
        in_feat = int(getattr(model, 'in_feat', C * V))

        F = int(C * V)
        pad = (3 - (F % 3)) % 3
        Fp = F + pad
        G = max(1, Fp // 3)

        # 1) rot_param_mlp: [in_feat->rot_hidden->rot_hidden->H*4]
        # Read Linear dims from the module to avoid drift from hand-written sizes
        linear_layers = [m for m in model.rot_param_mlp.modules() if isinstance(m, nn.Linear)]
        flops_rot_mlp = 0.0
        for m in linear_layers:
            flops_rot_mlp += float(m.in_features * m.out_features * B)

        # 2) Rodrigues rotation core: ~38 FLOPs per 3D vector at each (B,H,T,G)
        flops_rodrigues = float(B * H * T * G * 38)

        # 3) TSF stats: _extract_tsf counts 7 features per chunk; ~13*bs FLOPs per element
        flops_tsf = 0.0
        for bs in block_sizes:
            bs_eff = max(1, min(int(bs), int(T)))
            # Match code: step=bs, number of chunks
            n_chunks = max(1, (int(T) - bs_eff) // bs_eff + 1)
            # Stats over (B, N, bs, Fp)
            flops_per_chunk_feat = float(13 * bs_eff)  # mean/std/min/max/rms/mac/energy approx.
            flops_tsf += float(B * H * n_chunks * Fp * flops_per_chunk_feat)
            # tsf.mean(dim=1) reduction (B,F,7 -> B,7)
            flops_tsf += float(B * H * Fp * 7)

        # 4) feature_proj Linears
        feature_linear_layers = [m for m in model.feature_proj.modules() if isinstance(m, nn.Linear)]
        flops_head = 0.0
        for m in feature_linear_layers:
            flops_head += float(m.in_features * m.out_features * B)

        total_flops = flops_rot_mlp + flops_rodrigues + flops_tsf + flops_head
        return float(total_flops)
    except Exception:
        return None


def estimate_rtsfnet_ann_special_fp_ops(
    model: nn.Module,
    input_shape: Tuple,
    trig_pair_mode: str = "spatiotemporal",
) -> Optional[Dict[str, float]]:
    """
    Non-MAC FP op counts for RTSFNetANN (B=1), for literature-style pJ add-ons:
    - FP div/sqrt: counted from formulas
    - FP sin/cos pairs (one CORDIC outputs both): controlled by ``trig_pair_mode``:
        - ``spatiotemporal`` (default): one pair per 3D vector at each (t, g), i.e. B×H×T×G
        - ``per_head``: fused graph semantics, cos/sin once per head, i.e. B×H pairs

    Matches rtsfnet_ann.py forward (axis norm, Rodrigues, _extract_tsf, etc.).
    """
    try:
        if model.__class__.__name__ != "RTSFNetANN":
            return None
        if len(input_shape) != 4:
            return None
        _, T, C, V = input_shape
        B = 1
        T = int(T)
        H = int(getattr(model, "heads", 4))
        block_sizes = tuple(int(x) for x in getattr(model, "block_sizes", (16, 32, 64)))
        in_feat = int(getattr(model, "in_feat", C * V))
        F = int(C * V)
        pad = (3 - (F % 3)) % 3
        Fp = F + pad
        G = max(1, Fp // 3)  # x_xyz: (B, T, G, 3)

        def _n_chunks(t: int, bs: int) -> int:
            if t <= 0:
                return 1
            bs = max(1, min(int(bs), int(t)))
            hi = max(1, t - bs + 1)
            return len(range(0, hi, bs))

        n_div = 0.0
        n_sqrt = 0.0
        n_trig_pair = 0.0  # one CORDIC (sin+cos) per angle

        # global_feat = x.mean(dim=1)[:, :in_feat]
        n_div += float(B * in_feat)

        # axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        n_sqrt += float(B * H)  # norm = sqrt(sum of squares)
        n_div += float(3 * B * H)  # divide each of 3 components by norm

        # _rodrigues_rotation: sin/cos pair count vs theta
        mode = (trig_pair_mode or "spatiotemporal").strip().lower()
        if mode in ("per_head", "head", "fused"):
            n_trig_pair += float(B * H)
        else:
            # default: one CORDIC pair per (b,h,t,g)
            n_trig_pair += float(B * H * T * G)

        # one _extract_tsf per head × block_size
        per_extract_div = 0.0
        per_extract_sqrt = 0.0
        for bs in block_sizes:
            bs_eff = max(1, min(int(bs), T))
            N = _n_chunks(T, bs_eff)
            # z: (B, N, bs, Fp)
            # mean(dim=2), std(dim=2), rms=sqrt(mean(z^2)), mac=mean(|diff|), energy=sum(z^2)/bs
            # feat.mean(dim=1): (B,N,Fp,7)->(B,Fp,7)
            per_extract_div += float(4 * B * N * Fp + B * Fp * 7)
            per_extract_sqrt += float(2 * B * N * Fp)  # sqrt in std + rms

        n_div += float(H * per_extract_div)
        n_sqrt += float(H * per_extract_sqrt)

        return {
            "n_fp_div": n_div,
            "n_fp_sqrt": n_sqrt,
            "n_fp_sin_cos_pair": n_trig_pair,
        }
    except Exception:
        return None


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops_thop(model: nn.Module, input_shape: Tuple, device: str = 'cuda') -> Optional[float]:
    """
    Estimate FLOPs with thop.
    Args:
        model: PyTorch module
        input_shape: (B, T, C, V) or (B, C, T, V)
        device: torch device string
    Returns:
        FLOP count (float) or None
    """
    try:
        from thop import profile, clever_format
        
        model.eval()
        dummy_input = torch.randn(*input_shape).to(device)

        functional.reset_net(model)
        
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return None
    except Exception as e:
        print(f"Error calculating FLOPs with thop: {e}")
        return None


def count_flops_ptflops(model: nn.Module, input_shape: Tuple, device: str = 'cuda') -> Optional[float]:
    """
    Estimate FLOPs with ptflops.
    Args:
        model: PyTorch module
        input_shape: input shape
        device: torch device string
    Returns:
        FLOPs as MACs * 2, or None
    """
    try:
        from ptflops import get_model_complexity_info
        
        model.eval()
        # ptflops expects (C, H, W) or (C, T, ...)
        if len(input_shape) == 4:
            input_shape_ptflops = input_shape[2:] + input_shape[1:2]  # (C, V, T) from (B,T,C,V)
        else:
            input_shape_ptflops = input_shape[1:]  # drop batch
        
        macs, params = get_model_complexity_info(
            model,
            tuple(input_shape_ptflops),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        # MACs to FLOPs (often MACs * 2)
        return macs * 2
    except ImportError:
        print("Warning: ptflops not installed. Install with: pip install ptflops")
        return None
    except Exception as e:
        print(f"Error calculating FLOPs with ptflops: {e}")
        return None


class LIFActivationTracker:
    """Track LIF output rates and Conv/Linear input firing rates."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.lif_hooks = []
        self.compute_hooks = []
        # LIF output stats
        self.lif_stats = defaultdict(lambda: {
            'spikes': 0,
            'total': 0,
            'name': '',
            'is_first_layer': False
        })
        # Conv/Linear input stats
        self.compute_stats = defaultdict(lambda: {
            'input_spikes': 0,
            'input_total': 0,
            'name': '',
            'layer_type': ''
        })
        self.layer_order = []  # module visit order
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on LIF and compute layers."""
        # LIF modules
        first_lif_found = False
        for name, module in self.model.named_modules():
            if _is_spike_neuron_module(module):
                is_first = not first_lif_found
                if is_first:
                    first_lif_found = True
                self.layer_order.append(('lif', name))
                
                def make_lif_hook(lif_name, is_first_layer):
                    def hook(module, input, output):
                        # LIF output spikes (binary); rate = count(1) / numel
                        out_t = _extract_spike_tensor(output)
                        if isinstance(out_t, torch.Tensor):
                            spike_count = _count_spikes(out_t)
                            total_elements = out_t.numel()

                            self.lif_stats[lif_name]['spikes'] += spike_count
                            self.lif_stats[lif_name]['total'] += total_elements
                            self.lif_stats[lif_name]['name'] = lif_name
                            self.lif_stats[lif_name]['is_first_layer'] = is_first_layer
                    return hook
                
                handle = module.register_forward_hook(make_lif_hook(name, is_first))
                self.lif_hooks.append(handle)
        
        # Conv/Linear: input firing rate
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                self.layer_order.append(('compute', name))

                def make_compute_hook(compute_name, layer_type):
                    def hook(module, input, output):
                        # Count input spikes to this layer
                        if input and isinstance(input[0], torch.Tensor):
                            input_tensor = input[0]
                            input_spike_count = _count_spikes(input_tensor)
                            input_total_elements = input_tensor.numel()
                            
                            self.compute_stats[compute_name]['input_spikes'] += input_spike_count
                            self.compute_stats[compute_name]['input_total'] += input_total_elements
                            self.compute_stats[compute_name]['name'] = compute_name
                            self.compute_stats[compute_name]['layer_type'] = layer_type
                    return hook
                
                layer_type = type(module).__name__
                handle = module.register_forward_hook(make_compute_hook(name, layer_type))
                self.compute_hooks.append(handle)
    
    def get_activation_rates(self) -> Dict[str, Dict]:
        """
        Per-LIF spike rate in [0, 1]: fraction of outputs equal to 1.

        Returns:
            Mapping lif_name -> spike_rate, counts, etc.
        """
        results = {}
        for module_name, stats in self.lif_stats.items():
            if stats['total'] > 0:
                spike_rate = stats['spikes'] / stats['total']
                spike_rate = max(0.0, min(1.0, spike_rate))
                results[module_name] = {
                    'spike_rate': spike_rate,
                    'spike_rate_percent': spike_rate * 100.0,
                    'spikes': stats['spikes'],
                    'total': stats['total'],
                    'name': stats['name'],
                    'is_first_layer': stats.get('is_first_layer', False)
                }
            else:
                results[module_name] = {
                    'spike_rate': 0.0,
                    'spike_rate_percent': 0.0,
                    'spikes': 0,
                    'total': 0,
                    'name': stats['name'],
                    'is_first_layer': stats.get('is_first_layer', False)
                }
        return results

    def get_compute_layer_input_fr(self) -> Dict[str, Dict]:
        """
        Per compute layer: input firing rate in [0, 1] (fraction of input elements counted as spikes).

        Returns:
            Mapping compute_name -> input_firing_rate, counts, etc.
        """
        results = {}
        for module_name, stats in self.compute_stats.items():
            if stats['input_total'] > 0:
                input_fr = stats['input_spikes'] / stats['input_total']
                input_fr = max(0.0, min(1.0, input_fr))
                results[module_name] = {
                    'input_firing_rate': input_fr,
                    'input_firing_rate_percent': input_fr * 100.0,
                    'input_spikes': stats['input_spikes'],
                    'input_total': stats['input_total'],
                    'name': stats['name'],
                    'layer_type': stats['layer_type']
                }
            else:
                results[module_name] = {
                    'input_firing_rate': 0.0,
                    'input_firing_rate_percent': 0.0,
                    'input_spikes': 0,
                    'input_total': 0,
                    'name': stats['name'],
                    'layer_type': stats['layer_type']
                }
        return results

    def reset(self):
        """Reset running statistics."""
        self.lif_stats = defaultdict(lambda: {
            'spikes': 0, 
            'total': 0, 
            'name': '',
            'is_first_layer': False
        })
        self.compute_stats = defaultdict(lambda: {
            'input_spikes': 0,
            'input_total': 0,
            'name': '',
            'layer_type': ''
        })
    
    def remove_hooks(self):
        """Remove all forward hooks."""
        for hook in self.lif_hooks:
            hook.remove()
        for hook in self.compute_hooks:
            hook.remove()
        self.lif_hooks = []
        self.compute_hooks = []
    


class LayerFLOPsTracker:
    """Per-layer MAC counts via forward hooks."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.layer_flops = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks on Conv / Linear layers."""
        def make_hook(name):
            def hook(module, input, output):
                flops = 0
                if isinstance(module, nn.Conv1d):
                    # FLOPs = kernel_size * in_channels * out_channels * output_size
                    if input[0] is not None and output is not None:
                        batch_size = input[0].shape[0]
                        kernel_flops = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                        output_size = output.shape[2] if len(output.shape) > 2 else 1
                        flops = kernel_flops * module.in_channels * module.out_channels * output_size * batch_size
                elif isinstance(module, nn.Linear):
                    # FLOPs = in_features * out_features * batch_size
                    if input[0] is not None:
                        batch_size = input[0].shape[0] if len(input[0].shape) > 1 else 1
                        flops = module.in_features * module.out_features * batch_size
                
                if flops > 0:
                    if name not in self.layer_flops:
                        self.layer_flops[name] = 0
                    self.layer_flops[name] += flops
            return hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(make_hook(name))
                self.hooks.append(handle)
    
    def get_layer_flops(self) -> Dict[str, float]:
        """Return accumulated per-layer MACs."""
        return self.layer_flops.copy()

    def remove_hooks(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def _merge_activation_dicts_min(
    dicts: List[Dict[str, Dict]], rate_key: str
) -> Dict[str, Dict]:
    """Across forwards: per layer take the minimum of ``rate_key`` (not mean)."""
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0]
    all_names = set()
    for d in dicts:
        all_names |= set(d.keys())
    out: Dict[str, Dict] = {}
    for name in all_names:
        candidates = []
        for d in dicts:
            if name not in d:
                continue
            st = d[name]
            r = float(st.get(rate_key, 0.0))
            candidates.append((r, st))
        if not candidates:
            continue
        _, best = min(candidates, key=lambda x: x[0])
        out[name] = dict(best)
    return out


def _merge_activation_dicts_mean(
    dicts: List[Dict[str, Dict]], rate_key: str
) -> Dict[str, Dict]:
    """Across forwards: per layer average ``rate_key``; merge count fields."""
    if not dicts:
        return {}
    if len(dicts) == 1:
        return dicts[0]
    all_names = set()
    for d in dicts:
        all_names |= set(d.keys())
    out: Dict[str, Dict] = {}
    for name in all_names:
        items = [d[name] for d in dicts if name in d]
        if not items:
            continue
        rates = [float(it.get(rate_key, 0.0)) for it in items]
        merged = dict(items[-1])
        mean_r = float(sum(rates) / max(1, len(rates)))
        merged[rate_key] = max(0.0, min(1.0, mean_r))
        if rate_key == 'spike_rate':
            merged['spike_rate_percent'] = merged['spike_rate'] * 100.0
            spikes_sum = sum(int(it.get('spikes', 0)) for it in items)
            total_sum = sum(int(it.get('total', 0)) for it in items)
            merged['spikes'] = spikes_sum
            merged['total'] = total_sum
        elif rate_key == 'input_firing_rate':
            merged['input_firing_rate_percent'] = merged['input_firing_rate'] * 100.0
            spikes_sum = sum(int(it.get('input_spikes', 0)) for it in items)
            total_sum = sum(int(it.get('input_total', 0)) for it in items)
            merged['input_spikes'] = spikes_sum
            merged['input_total'] = total_sum
        out[name] = merged
    return out


def calculate_layer_flops(model: nn.Module, input_shape: Tuple, device: str = 'cuda', 
                          sample_input: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Per-layer MAC counts from one forward.

    Args:
        model: PyTorch module
        input_shape: Input tensor shape
        device: Device string
        sample_input: If set, use it; else random tensor

    Returns:
        ``{layer_name: flops}``
    """
    tracker = LayerFLOPsTracker(model)
    model.eval()
    
    with torch.no_grad():
        if sample_input is not None:
            functional.reset_net(model)
            _ = model(sample_input)
        else:
            dummy_input = torch.randn(*input_shape).to(device)
            functional.reset_net(model)
            _ = model(dummy_input)
    
    layer_flops = tracker.get_layer_flops()
    tracker.remove_hooks()
    
    return layer_flops


def _is_snn_first_compute_layer(compute_name: str) -> bool:
    """
    Only treat analog front-end / dense encoding as \"first\" (EMAC × full MACs).
    Do not infer first layer from low input_fr, or layers like fc2 with ~0 input rate
    get misclassified and blow up dense MAC / energy.
    """
    n = compute_name.lower().replace("model.", "")
    if "embed" in n and "proj" in n:
        return True
    if "patch_embed" in n and "proj" in n:
        return True
    return False


def _calibrate_lif_rates_with_compute_hooks(
    lif_activation_rates: Dict[str, Dict],
    compute_input_fr: Dict[str, Dict],
    compute_upstream_lif: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Dict]:
    """
    Conservative LIF rate fix: if spike_rate is 0 but downstream compute hooks show
    positive input firing, backfill from downstream evidence (mean of hooks).
    """
    if not lif_activation_rates or not compute_upstream_lif or not compute_input_fr:
        return lif_activation_rates

    hook_by_lif: Dict[str, List[float]] = defaultdict(list)
    for compute_name, up_lif in compute_upstream_lif.items():
        if not up_lif:
            continue
        st = compute_input_fr.get(compute_name)
        if not st:
            continue
        fr = float(st.get("input_firing_rate", 0.0))
        if fr > 0:
            hook_by_lif[up_lif].append(fr)

    if not hook_by_lif:
        return lif_activation_rates

    out: Dict[str, Dict] = {}
    for lif_name, lif_st in lif_activation_rates.items():
        st = dict(lif_st)
        lif_fr = float(st.get("spike_rate", 0.0))
        hooks = hook_by_lif.get(lif_name, [])
        if lif_fr <= 0.0 and hooks:
            corrected_fr = max(0.0, min(1.0, float(sum(hooks) / len(hooks))))
            st["spike_rate"] = corrected_fr
            st["spike_rate_percent"] = corrected_fr * 100.0
            total = int(st.get("total", 0))
            if total > 0:
                st["spikes"] = int(round(corrected_fr * total))
            st["calibrated_from_compute_hook"] = True
        out[lif_name] = st
    return out


def calculate_energy_consumption_snn(
    lif_activation_rates: Dict[str, Dict],
    compute_input_fr: Dict[str, Dict],
    layer_flops: Dict[str, float],
    num_timesteps: int,
    emac: float = 4.6e-12,  # dense MAC: 4.6 pJ
    eac: float = 0.1e-12,  # sparse AC (spike domain): 0.1 pJ
    compute_upstream_lif: Optional[Dict[str, Optional[str]]] = None,
    pre_spike_compute_names: Optional[Set[str]] = None,
) -> Dict[str, Dict]:
    """
    SNN energy proxy from MAC/SOP counts.

    Spike domain: SOPs(l) = fr(upstream LIF) × MACs_total(l); fr from
    ``trace_compute_upstream_lif`` (consistent with FLOPs × fr between LIFs).
    Analog front-end (e.g. embed.proj): no upstream LIF → Energy = EMAC × MACs_total.

    Args:
        lif_activation_rates: Per-LIF spike_rate stats
        compute_input_fr: Per-compute input firing rate (logging; energy uses LIF upstream)
        layer_flops: Per-layer MAC count
        compute_upstream_lif: Upstream LIF name per compute module; if None, legacy hook-only fr
        num_timesteps: T (kept for API compatibility)
    """
    energy_stats = {}

    def _resolve_flops(name: str) -> float:
        v = layer_flops.get(name, 0)
        if v > 0:
            return float(v)
        for layer_name, flops in layer_flops.items():
            if layer_name in name or name in layer_name:
                return float(flops)
        return 0.0

    def _resolve_upstream(name: str) -> Optional[str]:
        if not compute_upstream_lif:
            return None
        u = compute_upstream_lif.get(name)
        if u is not None or name in compute_upstream_lif:
            return u
        for k, v in compute_upstream_lif.items():
            if k in name or name in k:
                return v
        return None

    def _is_pre_spike_compute(name: str) -> bool:
        if not pre_spike_compute_names:
            return False
        if name in pre_spike_compute_names:
            return True
        # tolerate submodule name prefix differences
        for k in pre_spike_compute_names:
            if k in name or name in k:
                return True
        return False

    use_lif_upstream = compute_upstream_lif is not None
    compute_names = set(layer_flops.keys())
    compute_names |= set(compute_input_fr.keys())
    if compute_upstream_lif:
        compute_names |= set(compute_upstream_lif.keys())

    for compute_name in sorted(compute_names):
        layer_flops_value = _resolve_flops(compute_name)
        compute_stats = compute_input_fr.get(compute_name, {})
        layer_type = compute_stats.get("layer_type", "Unknown")

        input_fr_hook = max(0.0, min(1.0, float(compute_stats.get("input_firing_rate", 0.0))))

        # First dense segment: (1) compute before first spike in trace; (2) name hints embed.proj
        is_first_layer = _is_pre_spike_compute(compute_name) or _is_snn_first_compute_layer(compute_name)
        upstream_lif = _resolve_upstream(compute_name) if use_lif_upstream else None

        if use_lif_upstream:
            if layer_flops_value == 0:
                energy = 0.0
                sops = 0.0
                fr_used = 0.0
                fr_source = "none"
            elif is_first_layer and upstream_lif is None:
                energy = emac * layer_flops_value
                sops = 0.0
                fr_used = 1.0
                fr_source = "first_layer_dense"
            elif upstream_lif is not None:
                lif_st = lif_activation_rates.get(upstream_lif, {})
                fr_used = max(0.0, min(1.0, float(lif_st.get("spike_rate", 0.0))))
                fr_source = "upstream_lif"
                # if upstream LIF is 0 but input hook > 0, prefer hook to avoid false zero
                if fr_used <= 0.0 and input_fr_hook > 0.0:
                    fr_used = input_fr_hook
                    fr_source = "hook_fallback"
                sops = fr_used * layer_flops_value
                energy = eac * sops
            else:
                # no upstream LIF and not known analog front: do not assume dense
                fr_used = 0.0
                sops = 0.0
                energy = 0.0
                fr_source = "none"
        else:
            input_fr = input_fr_hook
            fr_used = input_fr
            fr_source = "hook_only"
            if layer_flops_value == 0:
                energy = 0.0
                sops = 0.0
            elif is_first_layer:
                energy = emac * layer_flops_value
                sops = 0.0
            else:
                sops = input_fr * layer_flops_value
                energy = eac * sops

        energy_stats[compute_name] = {
            "energy_joule": energy,
            "energy_pj": energy * 1e12,
            "sops": sops,
            "flops": layer_flops_value,
            "input_firing_rate": fr_used,
            "input_firing_rate_percent": fr_used * 100.0,
            "is_first_layer": is_first_layer,
            "layer_type": layer_type,
            "upstream_lif": upstream_lif if use_lif_upstream else None,
            "input_fr_hook": input_fr_hook,
            "fr_source": fr_source,
        }
    
    # LIF-only rows for display when no compute row exists
    for lif_name, lif_stats in lif_activation_rates.items():
        if lif_name not in energy_stats:
            energy_stats[f'{lif_name}_lif'] = {
                'energy_joule': 0.0,
                'energy_pj': 0.0,
                'sops': 0.0,
                'flops': 0.0,
                'input_firing_rate': 0.0,
                'spike_rate': lif_stats.get('spike_rate', 0.0),
                'is_first_layer': lif_stats.get('is_first_layer', False),
                'layer_type': 'LIF'
            }
    
    return energy_stats


def analyze_model_structure_reference(
    model: nn.Module,
    input_shape: Tuple,
    device: str = "cuda",
    energy_mode: str = "snn",
    rtsfnet_trig_pair_mode: str = "spatiotemporal",
    fp_div_pj: float = 20.0,
    fp_sqrt_pj: float = 25.0,
    fp_sin_cos_pair_pj: float = 40.0,
) -> Dict:
    """
    Pre-training structural reference only: parameter count and static / theoretical FLOPs.

    Does not run activation tracking or spike-domain energy (those require representative
    forward passes; use :func:`analyze_model` after loading a trained checkpoint).
    """
    results: Dict = {}
    if len(input_shape) >= 1 and input_shape[0] != 1:
        input_shape = (1,) + tuple(input_shape[1:])
    results["analysis_stage"] = "structure_reference"
    results["note"] = (
        "Structure-only reference before training: parameters + theoretical FLOPs. "
        "Spike rates / SOPs / energy proxy belong in post-train analysis on trained weights."
    )
    results["energy_batch_size"] = 1
    results["snn_activation_agg"] = "n/a"
    results["emac_pj"] = 4.6e-12 * 1e12
    results["eac_pj"] = 0.1e-12 * 1e12

    num_params = count_parameters(model)
    results["num_parameters"] = num_params
    results["num_parameters_millions"] = num_params / 1e6

    flops_thop = count_flops_thop(model, input_shape, device)
    if flops_thop is not None:
        results["flops"] = flops_thop
        results["flops_g"] = flops_thop / 1e9
    else:
        flops_ptflops = count_flops_ptflops(model, input_shape, device)
        if flops_ptflops is not None:
            results["flops"] = flops_ptflops
            results["flops_g"] = flops_ptflops / 1e9
        else:
            results["flops"] = None
            results["flops_g"] = None

    rtsfnet_flops = estimate_rtsfnet_ann_flops(model, input_shape)
    if rtsfnet_flops is not None:
        results["flops"] = rtsfnet_flops
        results["flops_g"] = rtsfnet_flops / 1e9
        results["rtsfnet_analytic_flops"] = float(rtsfnet_flops)

    special_ops = estimate_rtsfnet_ann_special_fp_ops(
        model, input_shape, trig_pair_mode=rtsfnet_trig_pair_mode
    )
    results["rtsfnet_ann_fp_special_ops"] = special_ops
    results["rtsfnet_trig_pair_mode"] = rtsfnet_trig_pair_mode
    results["fp_energy_constants_pj"] = {
        "fp_div": fp_div_pj,
        "fp_sqrt": fp_sqrt_pj,
        "fp_sin_cos_pair": fp_sin_cos_pair_pj,
    }

    if len(input_shape) >= 2:
        num_timesteps = input_shape[1] if len(input_shape) == 4 else 1
    else:
        num_timesteps = 1
    results["num_timesteps"] = num_timesteps
    results["energy_mode"] = energy_mode
    results["lif_activation_rates"] = {}
    results["compute_input_firing_rates"] = {}
    results["layer_flops"] = {}
    results["compute_upstream_lif"] = {}
    results["pre_spike_compute_names"] = []
    results["energy_consumption"] = {}
    results["total_energy_joule"] = 0.0
    results["total_energy_pj"] = 0.0
    results["total_sops"] = None
    return results


def format_structure_reference_analysis(results: Dict) -> str:
    """Human-readable summary for pretrain structure-only analysis."""
    lines = [
        "=" * 80,
        "Structure reference (untrained weights) — parameters & theoretical FLOPs only",
        "=" * 80,
        "Not device-measured power. No MAC/AC energy proxy totals in this file (structure/FLOPs only).",
        results.get("note", ""),
        "",
        f"Parameters: {results['num_parameters']:,} ({results['num_parameters_millions']:.2f}M)",
    ]
    if results.get("flops") is not None:
        lines.append(f"Theoretical FLOPs (static graph): {results['flops']:,.0f} ({results['flops_g']:.2f} G)")
    else:
        lines.append("Theoretical FLOPs: not available (install thop or ptflops)")
    sp = results.get("rtsfnet_ann_fp_special_ops")
    if sp:
        lines.append(
            f"RTSFNet FP op counts (analytic): div={sp['n_fp_div']:,.0f}, "
            f"sqrt={sp['n_fp_sqrt']:,.0f}, sin_cos_pair={sp['n_fp_sin_cos_pair']:,.0f}"
        )
    lines.append("")
    lines.append("No LIF firing rates, SOPs, or spike-domain energy in this file (see post-train energy log).")
    lines.append("=" * 80)
    return "\n".join(lines)


def analyze_model(
    model: nn.Module,
    input_shape: Tuple,
    device: str = 'cuda',
    sample_batches: int = 10,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    energy_mode: str = 'snn',
    emac: float = 4.6e-12,  # dense MAC: 4.6 pJ
    eac: float = 0.1e-12,  # sparse AC: 0.1 pJ
    snn_activation_agg: str = 'mean',
    fp_div_pj: float = 20.0,
    fp_sqrt_pj: float = 25.0,
    fp_sin_cos_pair_pj: float = 40.0,
    rtsfnet_trig_pair_mode: str = "spatiotemporal",
) -> Dict:
    """
    Full model analysis: parameters, FLOPs, LIF rates, energy proxy.

    Energy and op counts use effective batch size 1.
    - ANN: MAC × EMAC; RTSFNetANN adds FP div/sqrt/sin-cos pairs (default 20/25/40 pJ).
    - RTSFNet sin/cos pair count: ``rtsfnet_trig_pair_mode`` = ``spatiotemporal`` (B×H×T×G) or ``per_head`` (B×H).
    - SNN: analog front-end MACs×EMAC; else SOPs×EAC with SOPs = fr(upstream LIF) × MACs_total (no extra ×T).
    If sample_batches>1, aggregate per-layer rates with ``snn_activation_agg`` (default mean).
    """
    results = {}
    # single-sample energy / FLOPs convention
    if len(input_shape) >= 1 and input_shape[0] != 1:
        input_shape = (1,) + tuple(input_shape[1:])
    results['energy_batch_size'] = 1
    results['snn_activation_agg'] = snn_activation_agg
    results['emac_pj'] = emac * 1e12
    results['eac_pj'] = eac * 1e12

    # 1. Parameters
    num_params = count_parameters(model)
    results['num_parameters'] = num_params
    results['num_parameters_millions'] = num_params / 1e6
    
    # 2. FLOPs
    flops_thop = count_flops_thop(model, input_shape, device)
    if flops_thop is not None:
        results['flops'] = flops_thop
        results['flops_g'] = flops_thop / 1e9
    else:
        flops_ptflops = count_flops_ptflops(model, input_shape, device)
        if flops_ptflops is not None:
            results['flops'] = flops_ptflops
            results['flops_g'] = flops_ptflops / 1e9
        else:
            results['flops'] = None
            results['flops_g'] = None

    # RTSFNetANN: hooks/thop often miss functional stats; prefer analytic estimate
    rtsfnet_flops = estimate_rtsfnet_ann_flops(model, input_shape)
    if rtsfnet_flops is not None:
        results['flops'] = rtsfnet_flops
        results['flops_g'] = rtsfnet_flops / 1e9
        results['rtsfnet_analytic_flops'] = float(rtsfnet_flops)

    special_ops = estimate_rtsfnet_ann_special_fp_ops(
        model, input_shape, trig_pair_mode=rtsfnet_trig_pair_mode
    )
    results['rtsfnet_ann_fp_special_ops'] = special_ops
    results['rtsfnet_trig_pair_mode'] = rtsfnet_trig_pair_mode
    results['fp_energy_constants_pj'] = {
        'fp_div': fp_div_pj,
        'fp_sqrt': fp_sqrt_pj,
        'fp_sin_cos_pair': fp_sin_cos_pair_pj,
    }

    # 3. Mode: snn = LIF stats + SOP energy; ann = MAC×EMAC only
    model.eval()

    # Infer T from input_shape (IMU often (B, T, C, V))
    if len(input_shape) >= 2:
        num_timesteps = input_shape[1] if len(input_shape) == 4 else 1
    else:
        num_timesteps = 1
    
    if energy_mode == 'snn':
        lif_list: List[Dict[str, Dict]] = []
        compute_list: List[Dict[str, Dict]] = []
        if dataloader is not None:
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(dataloader):
                    if batch_idx >= sample_batches:
                        break
                    data = data[:1].to(device)
                    if len(data.shape) >= 2:
                        num_timesteps = data.shape[1] if len(data.shape) == 4 else 1
                    tracker = LIFActivationTracker(model)
                    functional.reset_net(model)
                    _ = model(data)
                    lif_list.append(tracker.get_activation_rates())
                    compute_list.append(tracker.get_compute_layer_input_fr())
                    tracker.remove_hooks()
        else:
            with torch.no_grad():
                for _ in range(max(1, sample_batches)):
                    dummy_input = torch.randn(*input_shape).to(device)
                    tracker = LIFActivationTracker(model)
                    functional.reset_net(model)
                    _ = model(dummy_input)
                    lif_list.append(tracker.get_activation_rates())
                    compute_list.append(tracker.get_compute_layer_input_fr())
                    tracker.remove_hooks()

        if snn_activation_agg == 'min':
            lif_activation_rates = _merge_activation_dicts_min(lif_list, 'spike_rate')
            compute_input_fr = _merge_activation_dicts_min(compute_list, 'input_firing_rate')
        elif snn_activation_agg == 'mean':
            lif_activation_rates = _merge_activation_dicts_mean(lif_list, 'spike_rate')
            compute_input_fr = _merge_activation_dicts_mean(compute_list, 'input_firing_rate')
        else:
            lif_activation_rates = lif_list[-1] if lif_list else {}
            compute_input_fr = compute_list[-1] if compute_list else {}
        results['lif_activation_rates'] = lif_activation_rates
        results['compute_input_firing_rates'] = compute_input_fr
    else:
        # ANN: no spike activation stats
        lif_activation_rates = {}
        compute_input_fr = {}
        results['lif_activation_rates'] = {}
        results['compute_input_firing_rates'] = {}
    results['num_timesteps'] = num_timesteps
    results['energy_mode'] = energy_mode  # 'snn' | 'ann'
    
    # 4. Per-layer MACs (single sample)
    if dataloader is not None:
        sample_data, _ = next(iter(dataloader))
        sample_data = sample_data[:1].to(device)
    else:
        sample_data = torch.randn(*input_shape).to(device)
    
    functional.reset_net(model)
    layer_flops = calculate_layer_flops(model, input_shape, device, sample_input=sample_data)
    if energy_mode == 'ann' and rtsfnet_flops is not None:
        # RTSFNet: if special FP ops split out, use hook MACs + div/sqrt/trig pJ; else one analytic FLOPs×EMAC line
        if special_ops is None:
            layer_flops = {'rtsfnet_ann_total': float(rtsfnet_flops)}
    results['layer_flops'] = layer_flops

    compute_upstream_lif: Optional[Dict[str, Optional[str]]] = None
    pre_spike_compute_names: Set[str] = set()
    if energy_mode == 'snn':
        functional.reset_net(model)
        compute_upstream_lif, pre_spike_compute_names = trace_compute_upstream_lif(model, sample_data)
        lif_activation_rates = _calibrate_lif_rates_with_compute_hooks(
            lif_activation_rates, compute_input_fr, compute_upstream_lif
        )
    results['compute_upstream_lif'] = compute_upstream_lif or {}
    results['pre_spike_compute_names'] = sorted(pre_spike_compute_names)
    
    # 5. Energy proxy
    if energy_mode == 'snn':
        energy_stats = calculate_energy_consumption_snn(
            lif_activation_rates,
            compute_input_fr,
            layer_flops,
            num_timesteps=num_timesteps,
            emac=emac,
            eac=eac,
            compute_upstream_lif=compute_upstream_lif,
            pre_spike_compute_names=pre_spike_compute_names,
        )
    else:
        # ANN: Linear layers MAC×EMAC; RTSFNet adds FP div/sqrt/sin-cos energy
        energy_stats = {}
        for layer_name, flops in layer_flops.items():
            energy = emac * flops
            energy_stats[layer_name] = {
                'energy_joule': energy,
                'energy_pj': energy * 1e12,
                'sops': 0.0,
                'flops': flops,
                'input_firing_rate': 0.0,
                'input_firing_rate_percent': 0.0,
                'is_first_layer': False,
                'layer_type': 'ANN',
            }
        if special_ops is not None:
            nd = float(special_ops['n_fp_div'])
            ns = float(special_ops['n_fp_sqrt'])
            nt = float(special_ops['n_fp_sin_cos_pair'])
            ed = nd * fp_div_pj * 1e-12
            es = ns * fp_sqrt_pj * 1e-12
            et = nt * fp_sin_cos_pair_pj * 1e-12
            energy_stats['rtsfnet_fp_div'] = {
                'energy_joule': ed,
                'energy_pj': ed * 1e12,
                'sops': 0.0,
                'flops': nd,
                'input_firing_rate': 0.0,
                'input_firing_rate_percent': 0.0,
                'is_first_layer': False,
                'layer_type': 'RTSFNet FP Div',
            }
            energy_stats['rtsfnet_fp_sqrt'] = {
                'energy_joule': es,
                'energy_pj': es * 1e12,
                'sops': 0.0,
                'flops': ns,
                'input_firing_rate': 0.0,
                'input_firing_rate_percent': 0.0,
                'is_first_layer': False,
                'layer_type': 'RTSFNet FP Sqrt',
            }
            energy_stats['rtsfnet_fp_sin_cos_pair'] = {
                'energy_joule': et,
                'energy_pj': et * 1e12,
                'sops': 0.0,
                'flops': nt,
                'input_firing_rate': 0.0,
                'input_firing_rate_percent': 0.0,
                'is_first_layer': False,
                'layer_type': 'RTSFNet FP Sin/Cos (CORDIC pair)',
            }
            # Residual FLOPs after subtracting hooked Linear MACs (Rodrigues + TSF etc.) at EMAC
            if rtsfnet_flops is not None:
                hooked_linear = float(sum(layer_flops.values()))
                residual_flops = max(0.0, float(rtsfnet_flops) - hooked_linear)
                er = residual_flops * emac
                results['rtsfnet_linear_macs_hooked'] = hooked_linear
                results['rtsfnet_residual_flops'] = residual_flops
                energy_stats['rtsfnet_residual_flops'] = {
                    'energy_joule': er,
                    'energy_pj': er * 1e12,
                    'sops': 0.0,
                    'flops': residual_flops,
                    'input_firing_rate': 0.0,
                    'input_firing_rate_percent': 0.0,
                    'is_first_layer': False,
                    'layer_type': 'RTSFNet residual (analytic FLOPs − Linear MACs) × EMAC',
                }
    results['energy_consumption'] = energy_stats
    
    # Sum layer energy proxies to total
    total_energy = sum(stats['energy_joule'] for stats in energy_stats.values())
    results['total_energy_joule'] = total_energy
    results['total_energy_pj'] = total_energy * 1e12
    if energy_mode == 'snn':
        results['total_sops'] = sum(stats.get('sops', 0) for stats in energy_stats.values())
    else:
        results['total_sops'] = None

    return results


def infer_energy_mode_from_model_type(model_type: str) -> str:
    """
    Training/analysis: model types ending with ``_ann`` use ANN energy (all MAC×EMAC, no SOPs).
    """
    t = (model_type or "").strip().lower()
    if t.endswith("_ann"):
        return "ann"
    return "snn"


def snn_total_energy_uniform_pj(results: Dict, emac_pj: float = 0.1, eac_pj: float = 0.1) -> float:
    """
    Recompute total SNN proxy from per-layer stats with uniform EMAC/EAC (pJ), single-sample; sensitivity scan.

    First layers: MACs×EMAC; later: SOPs×EAC (same structure as calculate_energy_consumption_snn).
    If energy_mode is ann, return existing total_energy_pj unchanged.
    """
    if results.get('energy_mode') != 'snn':
        return float(results.get('total_energy_pj', 0.0))
    emac = emac_pj * 1e-12
    eac = eac_pj * 1e-12
    total_j = 0.0
    for st in results.get('energy_consumption', {}).values():
        if st.get('is_first_layer'):
            total_j += float(st.get('flops', 0) or 0) * emac
        else:
            total_j += float(st.get('sops', 0) or 0) * eac
    return total_j * 1e12


def format_analysis_results(results: Dict) -> str:
    """
    Format analyze_model output for logs.

    Terminology: compute-energy *proxy* from op counts × assumed pJ/MAC and pJ/SOP — not measured
    silicon or system power.
    """
    energy_mode = results.get('energy_mode', 'snn')
    lines = []
    lines.append("=" * 80)
    sp_ops = results.get("rtsfnet_ann_fp_special_ops")
    if energy_mode == 'ann':
        if sp_ops:
            lines.append(
                "ANN compute energy proxy (MAC × EMAC + RTSFNet FP Div/Sqrt/Sin-Cos at assumed pJ)"
            )
        else:
            lines.append("ANN compute energy proxy (all layers: MACs × EMAC at assumed pJ)")
    else:
        lines.append("SNN compute energy proxy & activation-derived statistics")
    lines.append("=" * 80)
    eac_pj_disp = float(results.get("eac_pj", 0.1))
    emac_pj = float(results.get('emac_pj', 4.6))
    lines.append("DISCLAIMER: Estimated COMPUTE energy proxy — NOT measured device/board/system energy.")
    lines.append(
        "Excluded from this model: DRAM & memory-hierarchy energy, data movement beyond MAC/SOP accounting, "
        "control/clock overhead, host I/O, analog front-end beyond modeled EMAC/EAC."
    )
    lines.append(f"MAC energy assumption (dense MAC / EMAC): {emac_pj:g} pJ per MAC")
    lines.append(
        f"AC / sparse accumulate energy assumption (SOP, SNN spike domain): {eac_pj_disp:g} pJ per SOP"
        if energy_mode != "ann"
        else "SNN sparse term N/A in ANN mode."
    )
    lines.append("=" * 80)
    if energy_mode == 'ann':
        lines.append(
            "Proxy accounting: per-sample (batch_size=1); "
            f"ANN: MAC at EMAC ({emac_pj:g} pJ); RTSFNet adds FP special ops at separate pJ (see below)."
        )
        lines.append(
            f"Formula: ANN E_MAC = sum_l (MACs(l) × {emac_pj:g} pJ)"
            + (
                "; E_residual_FLOPs = (analytic_total − hooked_Linear_MACs) × EMAC (Rodrigues+TSF); "
                "E_FP = n_div×pJ_div + n_sqrt×pJ_sqrt + n_trig_pair×pJ_trig."
                if sp_ops
                else "; no sparse/SOPs term."
            )
        )
        if sp_ops:
            fpc = results.get("fp_energy_constants_pj") or {}
            lines.append(
                f"RTSFNet FP constants: Div={fpc.get('fp_div', 20.0):g} pJ, "
                f"Sqrt={fpc.get('fp_sqrt', 25.0):g} pJ, Sin/Cos pair={fpc.get('fp_sin_cos_pair', 40.0):g} pJ "
                f"(Galal et al. TC 2011–style orders of magnitude; CORDIC pair for sin+cos)."
            )
            tpm = results.get("rtsfnet_trig_pair_mode", "spatiotemporal")
            lines.append(
                f"RTSFNet FP op counts (single-sample); sin/cos pair mode={tpm!r}: "
                f"n_div={sp_ops['n_fp_div']:,.0f}, n_sqrt={sp_ops['n_fp_sqrt']:,.0f}, "
                f"n_sin_cos_pair={sp_ops['n_fp_sin_cos_pair']:,.0f} "
                f"(spatiotemporal: B×H×T×G; per_head: B×H)"
            )
            af = results.get("rtsfnet_analytic_flops") or results.get("flops")
            hl = results.get("rtsfnet_linear_macs_hooked")
            rf = results.get("rtsfnet_residual_flops")
            if af is not None and hl is not None and rf is not None:
                lines.append(
                    f"RTSFNet FLOPs reconciliation: analytic_total={float(af):,.0f}; "
                    f"hooked_Linear_MACs={float(hl):,.0f}; "
                    f"residual_FLOPs={float(rf):,.0f} (charged at EMAC={emac_pj:g} pJ as non-GEMM arithmetic)."
                )
    else:
        agg = results.get('snn_activation_agg', 'mean')
        lines.append(
            f"Proxy accounting: per-sample (batch_size=1); "
            f"SNN layer rates aggregated by {agg} over passes (not averaged)."
        )
        lines.append(
            "Proxy formula: SNN analog front-end E_proxy=MACs*EMAC; "
            "spike-domain SOPs=fr(upstream LIF)*MACs_total, E_proxy=SOPs*EAC."
        )
        lines.append(
            f"Default constants in this run: EMAC={emac_pj:g} pJ (dense MAC), "
            f"EAC={eac_pj_disp:g} pJ (sparse accumulate per SOP)."
        )
    
    lines.append(f"\nParameters: {results['num_parameters']:,} ({results['num_parameters_millions']:.2f}M)")

    # FLOPs
    if results['flops'] is not None:
        lines.append(f"FLOPs: {results['flops']:,.0f} ({results['flops_g']:.2f}G)")
    else:
        lines.append("FLOPs: Not available (install thop or ptflops)")
    
    if energy_mode != 'ann':
        lines.append("\nLIF Neuron Activation Rates (fraction of outputs equal to 1):")
        lines.append("-" * 80)
        for module_name, stats in results['lif_activation_rates'].items():
            spike_rate_pct = stats.get('spike_rate_percent', stats.get('spike_rate', 0.0) * 100.0)
            lines.append(f"  {stats['name']}: {stats['spike_rate']:.6f} ({spike_rate_pct:.4f}%) "
                        f"[spikes=1: {stats['spikes']:,} / elements: {stats['total']:,}]")

        if 'compute_input_firing_rates' in results:
            lines.append("\nCompute Layer Input Firing Rates (fraction of input elements counted as spikes):")
            lines.append("-" * 80)
            for module_name, stats in results['compute_input_firing_rates'].items():
                input_fr_pct = stats.get('input_firing_rate_percent', stats.get('input_firing_rate', 0.0) * 100.0)
                lines.append(f"  {stats['name']} ({stats.get('layer_type', 'Unknown')}): "
                            f"{stats['input_firing_rate']:.6f} ({input_fr_pct:.4f}%) "
                            f"[spikes=1: {stats['input_spikes']:,} / elements: {stats['input_total']:,}]")
    
    if energy_mode == 'ann':
        lines.append("\nEstimated compute energy proxy (per layer, ANN: MACs × EMAC):")
    else:
        lines.append("\nEstimated compute energy proxy (per layer, SNN MAC/SOP formulas):")
    lines.append("-" * 80)
    for module_name, stats in results['energy_consumption'].items():
        layer_type_info = stats.get('layer_type', 'Unknown')
        if energy_mode == 'ann':
            lt = str(layer_type_info)
            if "residual" in lt.lower():
                layer_type_str = lt
                flops_info = f"Residual_FLOPs={stats.get('flops', 0):,.0f}"
            elif lt.startswith("RTSFNet"):
                layer_type_str = lt
                flops_info = f"FP_ops={stats.get('flops', 0):,.0f}"
            else:
                layer_type_str = f"ANN dense MAC (EMAC={emac_pj:g}pJ)"
                flops_info = f"MACs={stats.get('flops', 0):,.0f}"
        else:
            layer_type_str = "First Layer (MAC)" if stats.get('is_first_layer', False) else "Subsequent Layer (AC)"
            flops_info = f"MACs={stats.get('flops', 0):,.0f}"
        input_fr = stats.get('input_firing_rate', 0.0)
        input_fr_pct = stats.get('input_firing_rate_percent', input_fr * 100.0)
        lines.append(f"  {module_name} ({layer_type_str}, {layer_type_info}): {stats['energy_pj']:.2f} pJ")
        if energy_mode == 'ann':
            lines.append(f"    {flops_info}")
        else:
            sops_info = f"SOPs={stats.get('sops', 0):,.0f}" if not stats.get('is_first_layer', False) else ""
            up = stats.get("upstream_lif")
            src = stats.get("fr_source", "")
            up_s = f", upstream_LIF={up}" if up else ""
            src_s = f", fr_source={src}" if src else ""
            hook_fr = stats.get("input_fr_hook", input_fr)
            lines.append(
                f"    {flops_info}, {sops_info}, fr(LIF)={input_fr:.6f} ({input_fr_pct:.4f}%){up_s} "
                f"[hook_input_fr={hook_fr:.6f}{src_s}]"
            )
    
    lines.append(
        f"\nTotal estimated compute energy proxy (sum of layer proxies): {results['total_energy_pj']:.2f} pJ "
        f"({results['total_energy_joule']:.2e} J)"
    )
    ts = results.get('total_sops', None)
    if ts is not None:
        lines.append(f"Total SOPs: {ts:,.0f}")
    lines.append(f"Time Steps (T): {results.get('num_timesteps', 'N/A')}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)
