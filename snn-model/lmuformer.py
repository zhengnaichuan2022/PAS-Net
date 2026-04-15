import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional
from torch import fft
from scipy.signal import cont2discrete
import numpy as np

__all__ = ['LMUFormer']


class SpikingLMUFFTCell(nn.Module):
    """
    Spiking Legendre Memory Unit using FFT
    Adapted for T B C V format where V is the sequence length
    """
    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):
        super(SpikingLMUFFTCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.seq_len = seq_len
        self.theta = theta

        self.W_u = nn.Linear(in_features=input_size, out_features=1)
        self.bn_u = nn.BatchNorm1d(1)
        self.f_u = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='torch')
        
        self.W_h = nn.Linear(in_features=memory_size + input_size, out_features=hidden_size)
        self.bn_m = nn.BatchNorm1d(memory_size)
        self.f_m = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='torch')
        self.bn_h = nn.BatchNorm1d(hidden_size)
        self.f_h = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='torch')
        self.act_loss = 0.0

        A, B = self.stateSpaceMatrices()
        self.register_buffer("A", A)  # [memory_size, memory_size]
        self.register_buffer("B", B)  # [memory_size, 1]

        H, fft_H = self.impulse()
        self.register_buffer("H", H)  # [memory_size, seq_len]
        self.register_buffer("fft_H", fft_H)  # [memory_size, seq_len + 1]

        # Cache for dynamically computed fft_H
        self._fft_H_cache = {}

    def stateSpaceMatrices(self):
        """ Returns the discretized state space matrices A and B """
        Q = np.arange(self.memory_size, dtype=np.float64).reshape(-1, 1)
        R = (2*Q + 1) / self.theta
        i, j = np.meshgrid(Q, Q, indexing="ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = np.ones((1, self.memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system=(A, B, C, D),
            dt=1.0,
            method="zoh"
        )

        # To torch.tensor
        A = torch.from_numpy(A).float()  # [memory_size, memory_size]
        B = torch.from_numpy(B).float()  # [memory_size, 1]
        
        return A, B

    def impulse(self, seq_len=None, device=None):
        """ Returns the matrices H and the 1D Fourier transform of H """
        if seq_len is None:
            seq_len = self.seq_len
        
        # Get device from A if not provided
        if device is None:
            device = self.A.device
        
        H = []
        A_i = torch.eye(self.memory_size, device=device, dtype=self.A.dtype)
        A = self.A.to(device)
        B = self.B.to(device)
        for t in range(seq_len):
            H.append(A_i @ B)
            A_i = A @ A_i

        H = torch.cat(H, dim=-1)  # [memory_size, seq_len]
        fft_H = fft.rfft(H, n=2*seq_len, dim=-1)  # [memory_size, seq_len + 1]

        return H, fft_H

    def forward(self, x):
        """
        Parameters:
            x (torch.tensor):
                [batch_size, seq_len, input_size] = (B, V, C).
                V is the IMU window length (vertices / time steps); C is ``dim`` (embed size).
                Callers must not pass (B, C, V) — use ``x.permute(0, 2, 1)`` before calling.
        """
        batch_size, seq_len, input_size = x.shape  # B, V, C

        # Equation 18: u = f_u(W_u(x))
        u_spike = self.f_u(self.bn_u(self.W_u(x).transpose(-1, -2)).permute(2, 0, 1).contiguous())
        u = u_spike.permute(1, 0, 2).contiguous()  # [B, V, 1]

        # Equation 26: FFT-based memory computation
        fft_input = u.permute(0, 2, 1)  # [batch_size, 1, seq_len]
        fft_u = fft.rfft(fft_input, n=2*seq_len, dim=-1)  # [batch_size, 1, seq_len+1]

        # Dynamically compute fft_H for the actual sequence length
        if seq_len != self.seq_len:
            # Use cache to avoid recomputing for the same seq_len
            cache_key = (seq_len, str(x.device))
            if cache_key not in self._fft_H_cache:
                _, fft_H = self.impulse(seq_len, device=x.device)
                self._fft_H_cache[cache_key] = fft_H
            else:
                fft_H = self._fft_H_cache[cache_key]
        else:
            fft_H = self.fft_H.to(x.device)

        # Element-wise multiplication
        # fft_u: [batch_size, 1, seq_len+1]
        # fft_H: [memory_size, seq_len+1]
        # fft_H.unsqueeze(0): [1, memory_size, seq_len+1]
        # Result: [batch_size, memory_size, seq_len+1]
        temp = fft_u * fft_H.unsqueeze(0)  # [batch_size, memory_size, seq_len+1]

        m = fft.irfft(temp, n=2*seq_len, dim=-1)  # [batch_size, memory_size, seq_len+1]
        m = m[:, :, :seq_len]  # [batch_size, memory_size, seq_len]
        m = self.f_m(self.bn_m(m).permute(2, 1, 0).contiguous()).permute(2, 1, 0).contiguous()
        m = m.permute(0, 2, 1)  # [batch_size, seq_len, memory_size]

        # Equation 20: h = f_h(W_h([m; x]))
        input_h = torch.cat((m, x), dim=-1)  # [batch_size, seq_len, memory_size + input_size]
        h = self.f_h(self.bn_h(self.W_h(input_h).transpose(-1, -2)).permute(2, 0, 1).contiguous())
        h = h.permute(1, 0, 2).contiguous()  # [batch_size, seq_len, hidden_size]

        h_n = h[:, -1, :].unsqueeze(-1)  # [batch_size, hidden_size, 1]

        return h, h_n


class SLMUMs(nn.Module):
    """
    Multi-Step Spiking LMU
    Adapted for T B C V format
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_all_h=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.hidden_size = dim
        self.memory_size = int(dim * 2.0)
        self.use_all_h = use_all_h
        
        # For T B C V format, seq_len is V
        self.lmu = SpikingLMUFFTCell(
            input_size=dim,
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
            seq_len=128,  # Default seq_len, can be adjusted
            theta=128
        )

        self.prev_bn = nn.BatchNorm1d(dim)
        self.prev_lif = neuron.LIFNode(step_mode='m', tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.act_loss = 0.0

    def cal_act_loss(self, x):
        return torch.sum(torch.abs(x))
    
    def forward(self, x):
        """
        Args:
            x: T B C V
        Returns:
            x: T B C V
        """
        T, B, C, V = x.shape
        
        # Process each time step
        outputs = []
        for t in range(T):
            x_t = x[t]  # B, C, V
            
            # Pre-processing
            x_t = self.prev_bn(x_t).permute(2, 1, 0).contiguous()  # B, C, V -> V, C, B
            x_t = self.prev_lif(x_t).permute(2, 1, 0).contiguous()  # V, C, B -> B, C, V (inv of above)
            
            # LMU forward: SpikingLMUFFTCell expects (B, V, C) with V=sequence, C=embed dim;
            # the backbone uses (B, C, V) everywhere else — permute so W_u / FFT see correct axes.
            h, h_n = self.lmu(x_t.permute(0, 2, 1).contiguous())
            
            # Select output
            if self.use_all_h:
                x_out = h.transpose(-1, -2).contiguous()  # B, H, V  (H == embed dim)
            else:
                x_out = h_n  # B, C, 1
            
            # Projection
            x_out = self.proj_conv(x_out)
            x_out = self.proj_bn(x_out)
            
            outputs.append(x_out)
        
        x = torch.stack(outputs, dim=0)  # T, B, C, V
        return x


class ConvFFNMs(nn.Module):
    """
    Multi-Step Spiking MLP
    Adapted for T B C V format
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., act_type='spike'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_bn = nn.BatchNorm1d(in_features)
        self.fc1_lif = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='torch')
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(hidden_features)
        self.fc2_lif = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='torch')
        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.act_loss = 0.0

    def cal_act_loss(self, x):
        return torch.sum(torch.abs(x))
    
    def forward(self, x):
        """
        Args:
            x: T B C V
        Returns:
            x: T B C V
        """
        T, B, C, V = x.shape
        
        # Process each time step
        outputs = []
        for t in range(T):
            x_t = x[t]  # B, C, V
            
            x_t = self.fc1_bn(x_t).permute(2, 1, 0).contiguous()  # B, C, V -> V, C, B
            x_t = self.fc1_lif(x_t).permute(2, 1, 0).contiguous()  # V, C, B -> B, C, V
            x_t = self.fc1_conv(x_t)
            
            x_t = self.fc2_bn(x_t).permute(2, 1, 0).contiguous()  # B, C, V -> V, C, B
            x_t = self.fc2_lif(x_t).permute(2, 1, 0).contiguous()  # V, C, B -> B, C, V
            x_t = self.fc2_conv(x_t)
            
            outputs.append(x_t)
        
        x = torch.stack(outputs, dim=0)  # T, B, C, V
        return x


class MS_SPS(nn.Module):
    """
    Simplified Patch Embedding for LMUFormer
    Input: T B C V (Time, Batch, Channels, Vertices)
    Output: T B embed_dims V (no downsampling, only channel mapping)
    """
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        spike_mode="lif"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.spike_mode = spike_mode

        # Single layer mapping: C -> embed_dims
        self.proj_conv = nn.Conv1d(in_channels, embed_dims, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dims)
        if spike_mode == "lif" or spike_mode == "spike":
            self.proj_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

    def forward(self, x):
        """
        Args:
            x: T B C V
        Returns:
            x: T B embed_dims V
        """
        T, B, C, V = x.shape
        
        # Process each time step
        outputs = []
        for t in range(T):
            x_t = x[t]  # B, C, V
            x_t = self.proj_conv(x_t)  # B, embed_dims, V
            x_t = self.proj_bn(x_t)
            # Only apply LIF if it exists
            if hasattr(self, 'proj_lif') and self.proj_lif is not None:
                x_t = self.proj_lif(x_t.permute(2, 1, 0).contiguous()).permute(2, 1, 0).contiguous()
            outputs.append(x_t)
        
        x = torch.stack(outputs, dim=0)  # T, B, embed_dims, V
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, act_type='spike', attn=SLMUMs, mlp=ConvFFNMs):
        super().__init__()

        self.attn = attn(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        drop_path = 0.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, act_type=act_type)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class LMUFormer(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_classes=11,
        embed_dims=256,
        num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=nn.LayerNorm,
        depth=6,
        sr_ratio=1,
        T=4,
        act_type='spike',
        with_head_lif=False,
    ):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depth = depth
        self.with_head_lif = with_head_lif

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Patch embedding: T B C V -> T B embed_dims V
        self.patch_embed = MS_SPS(
            in_channels=in_channels,
            embed_dims=embed_dims,
            spike_mode=act_type,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratio, act_type=act_type, attn=SLMUMs, mlp=ConvFFNMs
            )
            for j in range(depth)
        ])

        # classification head
        if self.with_head_lif:
            self.head_bn = nn.BatchNorm1d(embed_dims)
            self.head_lif = neuron.LIFNode(step_mode='m', tau=2.0, detach_reset=True, backend='torch')

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        Args:
            x: T B C V
        Returns:
            x: T B embed_dims (averaged over V dimension)
        """
        # Patch embedding: T B C V -> T B embed_dims V
        x = self.patch_embed(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Average over vertices: T B embed_dims V -> T B embed_dims
        x = x.mean(3)
        return x

    def forward(self, x):
        """
        Args:
            x: T B C V (Time, Batch, Channels, Vertices)
        Returns:
            x: B num_classes
        """
        if len(x.shape) < 4:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1)
        elif len(x.shape) == 4:
            # Already in T B C V format
            pass
        else:
            # If input is in different format, transpose to T B C V
            x = x.transpose(0, 1).contiguous()

        x = self.forward_features(x)

        if self.with_head_lif:
            x = self.head_bn(x)
            x = self.head_lif(x.permute(2, 1, 0).contiguous()).permute(2, 1, 0).contiguous()
        
        # x is (T, B, embed_dims); Linear applies on the last dim — do not permute to (T, embed_dims, B)
        x = self.head(x)
        x = x.mean(dim=0)  # Average over time: T B num_classes -> B num_classes
        return x


@register_model
def lmuformer(**kwargs):
    model = LMUFormer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


class LMUFormerWrapper(nn.Module):
    """
    Adapts (B, T, C, V) IMU batches to internal (T, B, C, V).
    """
    def __init__(self, lmuformer_model):
        super().__init__()
        self.model = lmuformer_model

    def forward(self, x):
        """
        Input: (B, T, C, V)
        Output: (B, num_classes)
        """
        x = x.permute(1, 0, 2, 3).contiguous()
        return self.model(x)


def create_lmuformer_model(config: dict) -> nn.Module:
    """
    Build LMUFormer from ``config['model']`` and wrap for (B, T, C, V).
    """
    model_config = config.get('model', {})

    num_classes = model_config.get('num_classes', 12)
    input_channels = model_config.get('input_channels', 3)
    embed_dims = model_config.get('embed_dims', 256)
    num_heads = model_config.get('num_heads', 8)
    mlp_ratio = model_config.get('mlp_ratio', 4.0)
    depth = model_config.get('depth', 6)
    drop_rate = model_config.get('drop_rate', 0.0)
    attn_drop_rate = model_config.get('attn_drop_rate', 0.0)
    drop_path_rate = model_config.get('drop_path_rate', 0.0)
    T = model_config.get('T', 4)
    act_type = model_config.get('act_type', 'spike')
    with_head_lif = model_config.get('with_head_lif', False)
    
    model = LMUFormer(
        in_channels=input_channels,
        num_classes=num_classes,
        embed_dims=embed_dims,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=False,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=nn.LayerNorm,
        depth=depth,
        sr_ratio=1,
        T=T,
        act_type=act_type,
        with_head_lif=with_head_lif,
    )
    
    return LMUFormerWrapper(model)

