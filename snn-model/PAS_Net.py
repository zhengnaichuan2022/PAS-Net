"""
PAS-Net (Physics-Aware Spiking Network for HAR).

IMU physics-aware spikeformer with adaptive topology, causal neuromodulation, and spike residuals;
typically trained with TSE loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, base, functional
from timm.models.layers import DropPath, trunc_normal_


# ==========================================
# 1. Utilities and dynamics
# ==========================================
def vector_magnitude(x_xyz: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Euclidean norm of last dimension (e.g. 3D vectors)."""
    return torch.sqrt((x_xyz ** 2).sum(dim=-1, keepdim=True) + eps)


class TemporalBatchNorm1d(nn.Module):
    """
    Temporal-aware normalization for spiking/time-series tensors.

    We keep the time axis explicit and compute statistics across (B, T, N),
    i.e. normalize each channel C using samples pooled from all time steps.

    Input shape:  [T, B, C, N]
    Output shape: [T, B, C, N]
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, C, N = x.shape
        # [T, B, C, N] -> [B*N, C, T]
        y = x.permute(1, 3, 2, 0).reshape(B * N, C, T).contiguous()
        y = self.bn(y)
        # [B*N, C, T] -> [T, B, C, N]
        y = y.reshape(B, N, C, T).permute(3, 0, 2, 1).contiguous()
        return y


class TrueDynamicThresholdLIF(base.MemoryModule):
    """
    LIF with dynamic threshold driven by a gate (neuromodulation).

    Optional learnable per-channel decay (PLIF-style) for long sequences.
    """
    def __init__(
        self,
        dim: int,
        base_thr: float = 0.8,
        v_reset: float = 0.0,
        decay: float = 0.8,
        learnable_decay: bool = False,
        surrogate_function=surrogate.ATan(),
    ):
        super().__init__()
        self.dim = dim
        self.base_thr = base_thr
        self.v_reset = v_reset
        self.decay = float(decay)
        self.learnable_decay = learnable_decay
        self.surrogate_function = surrogate_function
        self.register_memory('v', 0.0)

        if self.learnable_decay:
            # learnable decay in (0, 1) with sigmoid; per-channel
            init = torch.logit(torch.tensor(self.decay).clamp(1e-4, 1 - 1e-4))
            self.logit_decay = nn.Parameter(init.repeat(dim).view(1, dim, 1))  # [1, C, 1]

    def forward(self, x: torch.Tensor, gate: torch.Tensor):
        """
        Args:
            x: [T, B, C, N] input current
            gate: [T, B, C, N] neuromodulation (higher gate -> lower threshold)
        Returns:
            spikes: [T, B, C, N]
        """
        T, B, C, N = x.shape
        spikes = []
        
        if not isinstance(self.v, torch.Tensor):
            self.v = torch.zeros((B, C, N), device=x.device, dtype=x.dtype)
            
        for t in range(T):
            if self.learnable_decay:
                decay = torch.sigmoid(self.logit_decay)  # [1, C, 1]
                self.v = self.v * decay + x[t]
            else:
                self.v = self.v * self.decay + x[t]
            # Neuromodulation: larger gate -> lower threshold -> easier to fire
            current_thr = self.base_thr * (2.0 - gate[t])
            spike = self.surrogate_function(self.v - current_thr)
            self.v = self.v * (1.0 - spike) + self.v_reset * spike
            spikes.append(spike)
            
        return torch.stack(spikes, dim=0)


class CausalNeuromodulation(nn.Module):
    """Causal rhythm energy from causal conv over time (past-only)."""
    def __init__(self, dim: int, history_len: int = 15):
        super().__init__()
        self.history_len = history_len
        self.causal_pool = nn.Conv1d(dim, dim, kernel_size=history_len, 
                                     groups=dim, padding=history_len - 1, bias=False)
        # Initialize as average pooling
        nn.init.constant_(self.causal_pool.weight, 1.0 / history_len)
        self.causal_pool.weight.requires_grad = False

    def forward(self, x_spike: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_spike: [T, B, C, N] input spikes
        Returns:
            energy: [T, B, C, N] gating signal in [0,1]
        """
        T, B, C, N = x_spike.shape
        x_flat = x_spike.permute(1, 3, 2, 0).reshape(B * N, C, T)
        energy = self.causal_pool(x_flat)[..., :T]
        energy = torch.clamp(energy, 0.0, 1.0)
        return energy.reshape(B, N, C, T).permute(3, 0, 2, 1).contiguous()


class EMA_CausalNeuromodulation(base.MemoryModule):
    """O(1) causal EMA neuromodulation over spikes."""
    def __init__(self, dim: int, alpha: float = 0.9):
        super().__init__()
        init_alpha = float(max(1e-4, min(1.0 - 1e-4, alpha)))
        self.logit_alpha = nn.Parameter(torch.logit(torch.tensor(init_alpha)))
        self.register_memory('energy', 0.0)

    def forward(self, x_spike: torch.Tensor) -> torch.Tensor:
        T, B, C, N = x_spike.shape
        alpha = torch.sigmoid(self.logit_alpha)
        if not isinstance(self.energy, torch.Tensor):
            self.energy = torch.zeros((B, C, N), device=x_spike.device, dtype=x_spike.dtype)
        outs = []
        for t in range(T):
            self.energy = self.energy * alpha + x_spike[t] * (1.0 - alpha)
            outs.append(self.energy)
        return torch.stack(outs, dim=0)


# ==========================================
# 2. IMU tokenizer and adaptive physics mixer
# ==========================================
class IMUInvariantTokenizer(nn.Module):
    """Invariant IMU features + optional local mean/max/var downsampling."""
    def __init__(self, c_in: int, local_stat_stride: int = 1, use_local_stats: bool = False):
        super().__init__()
        self.c_in = c_in
        self.local_stat_stride = max(1, int(local_stat_stride))
        self.use_local_stats = bool(use_local_stats)
        # Match _downsample_local_stats: channels become 3*C when stride>1
        eff_c = c_in * (3 if (self.use_local_stats and self.local_stat_stride > 1) else 1)
        # Invariant branch for 3/6/9/18 ch; high-D datasets use 1x1 Conv1d projection
        self._use_generic = eff_c not in (3, 6, 9, 18)
        if self._use_generic:
            triple = self.use_local_stats and self.local_stat_stride > 1
            self.out_channels = 8 * (3 if triple else 1)
            self.generic_proj = nn.Conv1d(eff_c, self.out_channels, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.generic_proj.weight, nonlinearity="linear")
        else:
            self.generic_proj = None
            # 3ch->4 tokens, 6ch->8; local stats triples channels when enabled
            base_out = 4 if c_in == 3 else 8
            self.out_channels = base_out * (3 if self.use_local_stats else 1)

    def _downsample_local_stats(self, x: torch.Tensor) -> torch.Tensor:
        """
        Local mean/max/var pooling along time.
        Input x: [B, T, C, V]
        Output: [B, T', 3C, V] with T' = floor(T / stride)
        """
        if (not self.use_local_stats) or self.local_stat_stride <= 1:
            return x

        B, T, C, V = x.shape
        # [B,T,C,V] -> [B*C*V,1,T]
        xv = x.permute(0, 2, 3, 1).contiguous().reshape(B * C * V, 1, T)
        s = self.local_stat_stride
        mean = F.avg_pool1d(xv, kernel_size=s, stride=s)
        maxv = F.max_pool1d(xv, kernel_size=s, stride=s)
        mean_sq = F.avg_pool1d(xv * xv, kernel_size=s, stride=s)
        var = torch.clamp(mean_sq - mean * mean, min=0.0)

        t_new = mean.shape[-1]
        mean = mean.reshape(B, C, V, t_new).permute(0, 3, 1, 2).contiguous()  # [B,T',C,V]
        maxv = maxv.reshape(B, C, V, t_new).permute(0, 3, 1, 2).contiguous()
        var = var.reshape(B, C, V, t_new).permute(0, 3, 1, 2).contiguous()
        return torch.cat([mean, maxv, var], dim=2)  # [B,T',3C,V]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, V] raw IMU
        Returns:
            tokens: [B, T, 4 or 8 or more, V] invariant features
        """
        x = self._downsample_local_stats(x)
        c_now = x.shape[2]

        if c_now == 3:
            # Acc only: [ax, ay, az, |a|]
            acc_xyz = x.permute(0, 1, 3, 2).contiguous()      
            acc_mag = vector_magnitude(acc_xyz)               
            return torch.cat([acc_xyz, acc_mag], dim=-1).permute(0, 1, 3, 2).contiguous()

        # Acc + gyro: 8-D per node. With local stats, 6-channel input becomes mean/max/var triples first.
        if c_now == 6:
            acc = x[:, :, 0:3, :].permute(0, 1, 3, 2).contiguous()
            gyr = x[:, :, 3:6, :].permute(0, 1, 3, 2).contiguous()
            acc_mag, gyr_mag = vector_magnitude(acc), vector_magnitude(gyr)
            acc_4 = torch.cat([acc, acc_mag], dim=-1)
            gyr_4 = torch.cat([gyr, gyr_mag], dim=-1)
            # Concat on channel dim; V unchanged
            tokens = torch.cat([acc_4, gyr_4], dim=-1)
            return tokens.permute(0, 1, 3, 2).contiguous()

        if c_now == 18:
            toks = []
            for i in range(3):
                seg = x[:, :, i * 6:(i + 1) * 6, :]  # [B,T,6,V]
                acc = seg[:, :, 0:3, :].permute(0, 1, 3, 2).contiguous()
                gyr = seg[:, :, 3:6, :].permute(0, 1, 3, 2).contiguous()
                acc_mag, gyr_mag = vector_magnitude(acc), vector_magnitude(gyr)
                acc_4 = torch.cat([acc, acc_mag], dim=-1)
                gyr_4 = torch.cat([gyr, gyr_mag], dim=-1)
                tok = torch.cat([acc_4, gyr_4], dim=-1).permute(0, 1, 3, 2).contiguous()  # [B,T,8,V]
                toks.append(tok)
            return torch.cat(toks, dim=2)  # [B,T,24,V]

        if c_now == 9:
            toks = []
            for i in range(3):
                seg = x[:, :, i * 3:(i + 1) * 3, :]  # [B,T,3,V]
                acc_xyz = seg.permute(0, 1, 3, 2).contiguous()
                acc_mag = vector_magnitude(acc_xyz)
                tok = torch.cat([acc_xyz, acc_mag], dim=-1).permute(0, 1, 3, 2).contiguous()  # [B,T,4,V]
                toks.append(tok)
            return torch.cat(toks, dim=2)  # [B,T,12,V]

        if self._use_generic and self.generic_proj is not None:
            B, T, C, V = x.shape
            y = self.generic_proj(x.permute(0, 1, 3, 2).reshape(B * T, C, V))
            return y.reshape(B, T, self.out_channels, V).contiguous()

        raise ValueError(f"Unexpected tokenizer input channels after local stats: {c_now}")


class IMUAdaptivePhysicTokenMixer(nn.Module):
    """
    Adaptive graph-style mixing on IMU nodes: fusion / topology conv + norm + LIF keeps spike semantics.
    """
    def __init__(
        self,
        num_tokens: int,
        num_channels: int,
        use_adaptive_topology: bool = True,
        V_nodes: int = 5,
        st_kernel_size: int = 1,
        st_causal: bool = False,
        lif_backend: str = 'torch',
    ):
        super().__init__()
        self.V = V_nodes
        self.use_adaptive_topology = use_adaptive_topology
        self.st_kernel_size = int(st_kernel_size)
        self.st_causal = bool(st_causal)
        self.lif_backend = lif_backend
        self.is_dual_modality = (num_tokens % 2 == 0 and num_tokens == V_nodes * 2)

        if self.is_dual_modality:
            self.modality_fusion = nn.Conv1d(num_channels * 2, num_channels, kernel_size=1, bias=False)
            self.fusion_gn = nn.GroupNorm(1, num_channels)
            self.lif_fusion = neuron.LIFNode(step_mode='m', v_threshold=0.8, backend=lif_backend)

        if self.V > 1:
            k = max(1, self.st_kernel_size)
            self.node_mixer = nn.Conv1d(self.V, self.V, kernel_size=k, bias=False)
            self.mixer_gn = nn.GroupNorm(1, self.V)
            self.lif_topo = neuron.LIFNode(step_mode='m', v_threshold=0.8, backend=lif_backend)

            if self.use_adaptive_topology:
                init_adj = torch.eye(self.V) + torch.randn(self.V, self.V) * 0.01
                self.learnable_adj = nn.Parameter(init_adj)

    def forward(self, x_spike: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_spike: [T, B, C, N] with N=V or N=2V (dual modality)
        Returns:
            [T, B, C, N] mixed spikes, same shape
        """
        T, B, C, N = x_spike.shape

        dual_mode = self.is_dual_modality and (N == self.V * 2)
        if dual_mode:
            acc, gyr = x_spike[..., :self.V], x_spike[..., self.V:]
            fused_flat = torch.cat([acc, gyr], dim=2).reshape(T * B, 2 * C, self.V)
            x_nodes_fp32 = self.fusion_gn(self.modality_fusion(fused_flat))
            x_nodes_fp32 = x_nodes_fp32.reshape(T, B, C, self.V)
            x_nodes_spike = self.lif_fusion(x_nodes_fp32).reshape(T * B, C, self.V)
        else:
            x_nodes_spike = x_spike.reshape(T * B, C, self.V)

        if self.V > 1:
            x_nodes_tb = x_nodes_spike.reshape(T, B, C, self.V).contiguous()
            x_bt = x_nodes_tb.permute(1, 2, 3, 0).reshape(B * C, self.V, T).contiguous()

            k = int(self.node_mixer.kernel_size[0])
            if k > 1:
                if self.st_causal:
                    x_bt = F.pad(x_bt, (k - 1, 0))
                else:
                    pad = k // 2
                    x_bt = F.pad(x_bt, (pad, pad))

            if self.use_adaptive_topology:
                # Symmetric adjacency for bidirectional joints
                sym_adj = (self.learnable_adj + self.learnable_adj.transpose(0, 1)) * 0.5
                adj_mask = torch.sigmoid(sym_adj).unsqueeze(-1)
                masked_weight = self.node_mixer.weight * adj_mask
                y_bt = F.conv1d(x_bt, masked_weight, bias=None)
            else:
                y_bt = self.node_mixer(x_bt)

            y_bt = y_bt[..., :T].contiguous()
            y_bt_fp32 = self.mixer_gn(y_bt)
            y_mixed_fp32 = y_bt_fp32.reshape(B, C, self.V, T).permute(3, 0, 1, 2).contiguous()
            y_mixed_spike = self.lif_topo(y_mixed_fp32)
        else:
            y_mixed_spike = x_nodes_spike.reshape(T, B, C, self.V).contiguous()

        return torch.cat([y_mixed_spike, y_mixed_spike], dim=-1) if dual_mode else y_mixed_spike


# ==========================================
# 3. Spike residual backbone
# ==========================================
class SpikingEmbed(nn.Module):
    """Conv1d embed + norm + LIF."""
    def __init__(self, in_channels: int, embed_dims: int, norm_type: str = 'gn', lif_backend: str = 'torch', use_plif: bool = False):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, embed_dims, 1, 1, bias=False)
        self.norm_type = norm_type
        if norm_type == 'tbn':
            self.norm = TemporalBatchNorm1d(embed_dims)
        else:
            self.norm = nn.GroupNorm(8, embed_dims)

        if use_plif and hasattr(neuron, 'ParametricLIFNode'):
            self.lif = neuron.ParametricLIFNode(step_mode='m', v_threshold=0.8, backend=lif_backend)
        else:
            self.lif = neuron.LIFNode(step_mode='m', v_threshold=0.8, backend=lif_backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, C_in, N] tokens
        Returns:
            spikes: [T, B, embed_dims, N]
        """
        T, B, C, N = x.shape
        y = self.proj(x.flatten(0, 1))  # [T*B, embed_dims, N]
        y = y.reshape(T, B, -1, N).contiguous()
        y = self.norm(y)  # temporal-aware or groupnorm
        return self.lif(y)


class IMURhythmNeuromodBlock(nn.Module):
    """IMU block: neuromodulated token mix + MLP with optional causal dilated depthwise conv."""
    def __init__(self, dim: int, num_tokens: int, V_nodes: int, mlp_ratio: float = 4.0, 
                 base_thr: float = 0.8, use_adaptive_topology: bool = True, drop_path: float = 0.0,
                 norm_type: str = 'gn', lif_backend: str = 'torch', use_plif: bool = False,
                 learnable_decay: bool = False, st_kernel_size: int = 1, st_causal: bool = False,
                 residual_mode: str = 'clamp', use_causal_dw_mlp: bool = True, mlp_time_kernel: int = 3,
                 neuromod_mode: str = 'ema', neuromod_alpha: float = 0.9,
                 layer_idx: int = 0, dilation_base: int = 2,
                 use_lightweight_mlp_dw: bool = False,
                 use_dynamic_lif: bool = True):
        super().__init__()
        self.use_dynamic_lif = bool(use_dynamic_lif)
        if self.use_dynamic_lif:
            if neuromod_mode == 'conv':
                self.rhythm_extractor = CausalNeuromodulation(dim=dim)
            else:
                self.rhythm_extractor = EMA_CausalNeuromodulation(dim=dim, alpha=neuromod_alpha)
        else:
            self.rhythm_extractor = None
        self.residual_mode = residual_mode
        self.use_causal_dw_mlp = bool(use_causal_dw_mlp)
        self.use_lightweight_mlp_dw = bool(use_lightweight_mlp_dw)
        
        # Token-mix branch
        self.conv1 = nn.Conv1d(dim, dim, 1, 1, bias=False)
        if norm_type == 'tbn':
            self.norm1 = TemporalBatchNorm1d(dim)
        else:
            self.norm1 = nn.GroupNorm(8, dim)
        self.token_mixer = IMUAdaptivePhysicTokenMixer(
            num_tokens=num_tokens,
            num_channels=dim,
            use_adaptive_topology=use_adaptive_topology,
            V_nodes=V_nodes,
            st_kernel_size=st_kernel_size,
            st_causal=st_causal,
            lif_backend=lif_backend,
        )
        if self.use_dynamic_lif:
            self.dynamic_lif = TrueDynamicThresholdLIF(dim=dim, base_thr=base_thr, learnable_decay=learnable_decay)
            self.static_lif = None
        else:
            self.dynamic_lif = None
            self.static_lif = neuron.LIFNode(step_mode='m', v_threshold=base_thr, backend=lif_backend)
        
        # MLP branch
        hidden_dim = int(dim * mlp_ratio)
        if self.use_lightweight_mlp_dw:
            self.dw_time = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.fc1 = nn.Conv1d(dim, hidden_dim, 1, 1, bias=False)
        if self.use_causal_dw_mlp:
            k = max(1, int(mlp_time_kernel))
            self.mlp_time_kernel = k
            # Dilated depthwise conv: receptive field grows per layer (1,2,4,...)
            self.time_dilation = int(max(1, dilation_base) ** max(0, int(layer_idx)))
            self.time_tcn_pad = (k - 1) * self.time_dilation
            self.time_tcn = nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=k,
                groups=hidden_dim,
                dilation=self.time_dilation,
                padding=self.time_tcn_pad,
                bias=False
            )
        if norm_type == 'tbn':
            self.norm_mlp1 = TemporalBatchNorm1d(hidden_dim)
        else:
            self.norm_mlp1 = nn.GroupNorm(8, hidden_dim)
        if use_plif and hasattr(neuron, 'ParametricLIFNode'):
            self.lif_mlp1 = neuron.ParametricLIFNode(step_mode='m', v_threshold=base_thr, backend=lif_backend)
        else:
            self.lif_mlp1 = neuron.LIFNode(step_mode='m', v_threshold=base_thr, backend=lif_backend)
        
        self.fc2 = nn.Conv1d(hidden_dim, dim, 1, 1, bias=False)
        if norm_type == 'tbn':
            self.norm_mlp2 = TemporalBatchNorm1d(dim)
        else:
            self.norm_mlp2 = nn.GroupNorm(8, dim)
        if use_plif and hasattr(neuron, 'ParametricLIFNode'):
            self.lif_mlp2 = neuron.ParametricLIFNode(step_mode='m', v_threshold=base_thr, backend=lif_backend)
        else:
            self.lif_mlp2 = neuron.LIFNode(step_mode='m', v_threshold=base_thr, backend=lif_backend)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x_spike: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_spike: [T, B, C, N] input spikes
        Returns:
            out: [T, B, C, N] output spikes (residual)
        """
        T, B, C, N = x_spike.shape

        # 1. Causal rhythm gate
        gate = self.rhythm_extractor(x_spike) if self.rhythm_extractor is not None else None  # [T, B, C, N]
        
        # 2. Spatial / topology mixing
        mixed_spike = self.token_mixer(x_spike)
        current1 = self.conv1(mixed_spike.flatten(0, 1)).reshape(T, B, C, N)
        current1 = self.norm1(current1)
        if self.use_dynamic_lif:
            x_spike_mixed = self.dynamic_lif(current1, gate)
        else:
            x_spike_mixed = self.static_lif(current1)
        
        # Residual: add mode (SEW-style) or clamped sum
        res1 = self.drop_path(x_spike_mixed)
        if self.residual_mode == 'add':
            x_spike_out1 = x_spike + res1
        else:
            x_spike_out1 = torch.clamp(x_spike + res1, 0.0, 1.0)
        
        # 3. MLP
        mlp_in = x_spike_out1.flatten(0, 1)
        if self.use_lightweight_mlp_dw:
            mlp_in = self.dw_time(mlp_in)
        cur_mlp1 = self.fc1(mlp_in).reshape(T, B, -1, N)
        if self.use_causal_dw_mlp:
            # Causal dilated depthwise: trim future padding
            h = cur_mlp1.shape[2]
            cur_bt = cur_mlp1.permute(1, 3, 2, 0).reshape(B * N, h, T).contiguous()
            cur_bt = self.time_tcn(cur_bt)
            if self.time_tcn_pad > 0:
                cur_bt = cur_bt[..., :-self.time_tcn_pad]
            cur_bt = cur_bt[..., :T].contiguous()
            cur_mlp1 = cur_bt.reshape(B, N, h, T).permute(3, 0, 2, 1).contiguous()
        cur_mlp1 = self.norm_mlp1(cur_mlp1)
        spike_mlp1 = self.lif_mlp1(cur_mlp1)
        
        cur_mlp2 = self.fc2(spike_mlp1.flatten(0, 1)).reshape(T, B, C, N)
        cur_mlp2 = self.norm_mlp2(cur_mlp2)
        spike_mlp2 = self.lif_mlp2(cur_mlp2)
        
        res2 = self.drop_path(spike_mlp2)
        if self.residual_mode == 'add':
            return x_spike_out1 + res2
        return torch.clamp(x_spike_out1 + res2, 0.0, 1.0)


class IMUSpikeformer(nn.Module):
    """
    IMU Physics-Aware Spikeformer.

    Returns per-timestep logits [T, B, num_classes] for TSELoss (no temporal average before head).
    """
    def __init__(
        self,
        input_channels=6,
        num_classes=12,
        embed_dims=128,
        depth=4,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        V_nodes=5,
        use_adaptive_topology=True,
        norm_type: str = 'gn',          # 'gn' | 'tbn'
        lif_backend: str = 'torch',     # 'torch' | 'cupy'
        use_plif: bool = False,
        learnable_decay: bool = False,
        st_kernel_size: int = 1,        # 1 means static topology; >1 enables spatio-temporal topology mixing
        st_causal: bool = False,
        residual_mode: str = 'clamp',   # 'clamp' | 'add'
        local_stat_stride: int = 1,
        use_local_stats: bool = False,
        use_causal_dw_mlp: bool = True,
        mlp_time_kernel: int = 3,
        neuromod_mode: str = 'ema',     # 'ema' | 'conv'
        neuromod_alpha: float = 0.9,
        use_dynamic_lif: bool = True,
        dilation_base: int = 2,
        use_lightweight_mlp_dw: bool = False,
        share_block_weights: bool = False,
        spatial_pool_mode: str = "mean_max",  # pool over node dim V: mean | max | mean+max
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.tokenizer = IMUInvariantTokenizer(
            c_in=input_channels,
            local_stat_stride=local_stat_stride,
            use_local_stats=use_local_stats,
        )
        self.latest_sparsity_loss = None
        # Tokenizer width: default 3->4, 6->8; with local stats 3->12, 6->24
        self.embed = SpikingEmbed(
            in_channels=self.tokenizer.out_channels,
            embed_dims=embed_dims,
            norm_type=norm_type,
            lif_backend=lif_backend,
            use_plif=use_plif,
        )
        
        # N aligns with V_nodes; acc+gyro fused in channel dim, not by doubling V
        num_tokens = V_nodes
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.depth = int(depth)
        self.share_block_weights = bool(share_block_weights)
        self.spatial_pool_mode = str(spatial_pool_mode).lower()
        if self.spatial_pool_mode not in ("mean", "max", "mean_max"):
            self.spatial_pool_mode = "mean_max"
        dp = float(dropout)
        self.embed_dropout = nn.Dropout(dp) if dp > 0 else nn.Identity()
        self.head_dropout = nn.Dropout(dp) if dp > 0 else nn.Identity()

        if self.share_block_weights:
            self.shared_block = IMURhythmNeuromodBlock(
                dim=embed_dims,
                num_tokens=num_tokens,
                V_nodes=V_nodes,
                mlp_ratio=mlp_ratio,
                use_adaptive_topology=use_adaptive_topology,
                drop_path=max(dpr) if len(dpr) > 0 else 0.0,
                norm_type=norm_type,
                lif_backend=lif_backend,
                use_plif=use_plif,
                learnable_decay=learnable_decay,
                st_kernel_size=st_kernel_size,
                st_causal=st_causal,
                residual_mode=residual_mode,
                use_causal_dw_mlp=use_causal_dw_mlp,
                mlp_time_kernel=mlp_time_kernel,
                neuromod_mode=neuromod_mode,
                neuromod_alpha=neuromod_alpha,
                use_dynamic_lif=use_dynamic_lif,
                layer_idx=0,
                dilation_base=dilation_base,
                use_lightweight_mlp_dw=use_lightweight_mlp_dw,
            )
            self.blocks = nn.ModuleList()
        else:
            self.blocks = nn.ModuleList([
                IMURhythmNeuromodBlock(
                    dim=embed_dims,
                    num_tokens=num_tokens,
                    V_nodes=V_nodes,
                    mlp_ratio=mlp_ratio,
                    use_adaptive_topology=use_adaptive_topology,
                    drop_path=dpr[i],
                    norm_type=norm_type,
                    lif_backend=lif_backend,
                    use_plif=use_plif,
                    learnable_decay=learnable_decay,
                    st_kernel_size=st_kernel_size,
                    st_causal=st_causal,
                    residual_mode=residual_mode,
                    use_causal_dw_mlp=use_causal_dw_mlp,
                    mlp_time_kernel=mlp_time_kernel,
                    neuromod_mode=neuromod_mode,
                    neuromod_alpha=neuromod_alpha,
                    use_dynamic_lif=use_dynamic_lif,
                    layer_idx=i,
                    dilation_base=dilation_base,
                    use_lightweight_mlp_dw=use_lightweight_mlp_dw,
                )
                for i in range(depth)
            ])
        
        self.head = nn.Linear(embed_dims, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, V] IMU batch
        Returns:
            temporal_logits: [T, B, num_classes]
        """
        x = self.tokenizer(x)
        x = x.permute(1, 0, 2, 3).contiguous()
        x_spike = self.embed(x)
        x_spike = self.embed_dropout(x_spike)

        all_spikes = []
        if self.share_block_weights:
            for _ in range(self.depth):
                x_spike = self.shared_block(x_spike)
                all_spikes.append(x_spike)
        else:
            for blk in self.blocks:
                x_spike = blk(x_spike)
                all_spikes.append(x_spike)

        if len(all_spikes) > 0:
            self.latest_sparsity_loss = sum(s.mean() for s in all_spikes) / len(all_spikes)
        else:
            self.latest_sparsity_loss = None

        # Pool over nodes V; keep T for TSELoss
        if self.spatial_pool_mode == "max":
            x_pool = x_spike.amax(dim=3)
        elif self.spatial_pool_mode == "mean_max":
            x_pool = x_spike.mean(dim=3) + x_spike.amax(dim=3)
        else:
            x_pool = x_spike.mean(dim=3)
        x_pool = self.head_dropout(x_pool)

        # Per-timestep classifier
        temporal_logits = self.head(x_pool)
        return temporal_logits

    def get_topology_l1_loss(self):
        """
        L1 on learnable adjacency masks (topology sparsity regularizer).

        Returns:
            Scalar sum of L1 norms.
        """
        l1_loss = 0.0
        mixers = []
        if getattr(self, "share_block_weights", False) and hasattr(self, "shared_block"):
            mixers.append(self.shared_block.token_mixer)
        else:
            for blk in self.blocks:
                mixers.append(blk.token_mixer)
        for tm in mixers:
            if hasattr(tm, "learnable_adj"):
                la = tm.learnable_adj
                sym_adj = (la + la.transpose(0, 1)) * 0.5
                adj_mask = torch.sigmoid(sym_adj)
                l1_loss += torch.norm(adj_mask, p=1)
        return l1_loss

    def get_sparsity_loss(self):
        if self.latest_sparsity_loss is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.latest_sparsity_loss


# ==========================================
# 4. TSELoss
# ==========================================
class TSELoss(nn.Module):
    """
    Temporal supervision: average CE over time steps (optionally skip early steps for warm-up).

    Encourages correct predictions throughout the window and stabilizes BPTT through time.
    """
    def __init__(
        self,
        criterion=nn.CrossEntropyLoss(),
        weighting_type: str = 'linear',
        min_weight: float = 0.1,
        warmup_ratio: float = 0.2,
    ):
        super().__init__()
        self.criterion = criterion
        self.weighting_type = weighting_type
        self.min_weight = float(min_weight)
        self.warmup_ratio = float(warmup_ratio)

    def forward(self, temporal_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_logits: [T, B, num_classes]
            targets: [B] class indices
        Returns:
            scalar weighted mean loss over time
        """
        T, B, C = temporal_logits.shape
        if self.weighting_type == 'linear':
            weights = torch.linspace(self.min_weight, 1.0, steps=T, device=temporal_logits.device, dtype=temporal_logits.dtype)
        elif self.weighting_type == 'exp':
            w = torch.linspace(0.0, 1.0, steps=T, device=temporal_logits.device, dtype=temporal_logits.dtype)
            weights = self.min_weight + (1.0 - self.min_weight) * (torch.exp(w) - 1.0) / (torch.e - 1.0)
        else:
            weights = torch.ones(T, device=temporal_logits.device, dtype=temporal_logits.dtype)

        # Optional warmup: zero weight on early timesteps
        warmup_steps = int(T * self.warmup_ratio)
        warmup_steps = max(0, min(warmup_steps, max(T - 1, 0)))
        weights[:warmup_steps] = 0.0
        wsum = weights.sum()
        if wsum < 1e-12:
            weights = torch.ones(T, device=temporal_logits.device, dtype=temporal_logits.dtype)
            weights[:warmup_steps] = 0.0
            wsum = weights.sum().clamp_min(1e-12)
        else:
            wsum = wsum.clamp_min(1e-12)

        loss = temporal_logits.new_tensor(0.0)
        for t in range(warmup_steps, T):
            loss = loss + weights[t] * self.criterion(temporal_logits[t], targets)
        return loss / wsum


# ==========================================
# 5. Factory
# ==========================================
class IMUSpikeformerWrapper(nn.Module):
    """Thin wrapper: accepts (B, T, C, V) like the training loop."""
    def __init__(self, model: IMUSpikeformer):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C, V]
        Returns:
            temporal_logits: [T, B, num_classes]
        """
        return self.model(x)

    def get_topology_l1_loss(self):
        return self.model.get_topology_l1_loss()

    def get_sparsity_loss(self):
        return self.model.get_sparsity_loss()


def create_imu_physics_spikeformer(config: dict) -> nn.Module:
    """
    Build IMUSpikeformer from ``config['model']`` and wrap for training API.

    Prefer YAML ``model.type: pas_net``; this name is kept for backward compatibility.

    Args:
        config: full training config
    Returns:
        ``IMUSpikeformerWrapper`` module
    """
    model_cfg = config.get("model", {})

    # V_nodes: prefer num_imus (set from batch/feeder in train.py)
    V_nodes = model_cfg.get("num_imus", model_cfg.get("V_nodes", 1))
    if V_nodes is None or V_nodes == 0:
        V_nodes = 1
    
    model = IMUSpikeformer(
        input_channels=model_cfg.get("input_channels", 3),
        num_classes=model_cfg.get("num_classes", 12),
        embed_dims=model_cfg.get("embed_dims", 128),
        depth=model_cfg.get("depth", 4),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
        V_nodes=V_nodes,
        use_adaptive_topology=model_cfg.get("use_adaptive_topology", True),
        norm_type=model_cfg.get("norm_type", "gn"),
        lif_backend=model_cfg.get("lif_backend", "torch"),
        use_plif=bool(model_cfg.get("use_plif", False)),
        learnable_decay=bool(model_cfg.get("learnable_decay", False)),
        st_kernel_size=int(model_cfg.get("st_kernel_size", 1)),
        st_causal=bool(model_cfg.get("st_causal", False)),
        residual_mode=model_cfg.get("residual_mode", "clamp"),
        local_stat_stride=int(model_cfg.get("local_stat_stride", 4)),
        use_local_stats=bool(model_cfg.get("use_local_stats", True)),
        use_causal_dw_mlp=bool(model_cfg.get("use_causal_dw_mlp", True)),
        mlp_time_kernel=int(model_cfg.get("mlp_time_kernel", 3)),
        neuromod_mode=model_cfg.get("neuromod_mode", "ema"),
        neuromod_alpha=float(model_cfg.get("neuromod_alpha", 0.9)),
        use_dynamic_lif=bool(model_cfg.get("use_dynamic_lif", True)),
        dilation_base=int(model_cfg.get("dilation_base", 2)),
        use_lightweight_mlp_dw=bool(model_cfg.get("use_lightweight_mlp_dw", False)),
        share_block_weights=bool(model_cfg.get("share_block_weights", False)),
        spatial_pool_mode=str(model_cfg.get("spatial_pool_mode", "mean_max")),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    
    return IMUSpikeformerWrapper(model)


# Canonical factory alias: YAML ``type: pas_net`` maps to this file (PAS_Net.py).
create_pas_net = create_imu_physics_spikeformer
