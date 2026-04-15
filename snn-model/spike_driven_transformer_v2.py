from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.activation_based import neuron, layer, functional

__all__ = ['SpikeDrivenTransformerV2']


class RepConv1d(nn.Module):
    """
    Simplified RepConv for 1D: Conv1d version
    Original RepConv uses Conv2d with BNAndPadLayer, here we simplify to Conv1d
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # Simplified: use single Conv1d instead of complex RepConv structure
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        """
        Args:
            x: T B C V
        Returns:
            x: T B C V
        """
        T, B, C, V = x.shape
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, V).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, V).contiguous()

        return x


class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        # RepConv for q, k, v - adapted to Conv1d
        self.q_conv = nn.Sequential(RepConv1d(dim, dim, bias=False), nn.BatchNorm1d(dim))
        self.k_conv = nn.Sequential(RepConv1d(dim, dim, bias=False), nn.BatchNorm1d(dim))
        self.v_conv = nn.Sequential(RepConv1d(dim, dim, bias=False), nn.BatchNorm1d(dim))

        self.q_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')
        self.k_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')
        self.v_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.attn_lif = neuron.LIFNode(
            step_mode='m', v_threshold=0.5, backend='torch'
        )

        self.proj_conv = nn.Sequential(
            RepConv1d(dim, dim, bias=False), nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        """
        Args:
            x: T B C V
        Returns:
            x: T B C V
        """
        T, B, C, V = x.shape
        N = V

        x = self.head_lif(x)

        x_for_qkv = x.flatten(0, 1)
        q = self.q_conv(x_for_qkv).reshape(T, B, C, V)
        k = self.k_conv(x_for_qkv).reshape(T, B, C, V)
        v = self.v_conv(x_for_qkv).reshape(T, B, C, V)

        q = self.q_lif(q)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        # Attention computation: k^T @ v, then q @ (k^T @ v)
        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, V)

        return x


class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class MS_SPS(nn.Module):
    """
    Simplified Patch Embedding for SpikeDrivenTransformerV2
    Input: T B C V (Time, Batch, Channels, Vertices)
    Output: T B embed_dims V (no downsampling, only channel mapping)
    Based on MS_DownSampling but simplified to single layer mapping
    """
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        first_layer=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        # Single layer mapping: C -> embed_dims (no downsampling)
        self.proj_conv = nn.Conv1d(
            in_channels,
            embed_dims,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.proj_bn = nn.BatchNorm1d(embed_dims)
        if not first_layer:
            self.proj_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')
        else:
            self.proj_lif = None

    def forward(self, x):
        """
        Args:
            x: T B C V
        Returns:
            x: T B embed_dims V
        """
        T, B, C, V = x.shape

        if self.proj_lif is not None:
            x = self.proj_lif(x)
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, self.embed_dims, V).contiguous()

        return x


class SpikeDrivenTransformerV2(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_classes=11,
        embed_dims=256,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depth=6,
        sr_ratio=1,
        T=4,
        kd=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.T = T
        self.kd = kd

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # Patch embedding: T B C V -> T B embed_dims V
        # Simplified: single layer mapping instead of multiple downsampling stages
        self.patch_embed = MS_SPS(
            in_channels=in_channels,
            embed_dims=embed_dims,
            first_layer=True,
        )

        # Transformer blocks (similar to block3 in original)
        self.blocks = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratio,
                )
                for j in range(depth)
            ]
        )

        self.head_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )

        if self.kd:
            self.head_kd = (
                nn.Linear(embed_dims, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
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
        x_lif = self.head_lif(x)
        x = self.head(x_lif).mean(0)  # Average over time: T B num_classes -> B num_classes

        if self.kd:
            x_kd = self.head_kd(x_lif).mean(0)
            if self.training:
                return x, x_kd
            else:
                return (x + x_kd) / 2
        return x


@register_model
def sdt_v2(**kwargs):
    model = SpikeDrivenTransformerV2(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


class SpikeDrivenTransformerV2Wrapper(nn.Module):
    """
    Adapts (B, T, C, V) IMU batches to internal (T, B, C, V).
    """
    def __init__(self, spike_driven_transformer_v2_model):
        super().__init__()
        self.model = spike_driven_transformer_v2_model

    def forward(self, x):
        """
        Input: (B, T, C, V)
        Output: (B, num_classes)
        """
        x = x.permute(1, 0, 2, 3).contiguous()
        return self.model(x)


def create_spike_driven_transformer_v2_model(config: dict) -> nn.Module:
    """
    Build SpikeDrivenTransformerV2 from ``config['model']`` and wrap for (B, T, C, V).
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
    kd = model_config.get('kd', False)
    
    model = SpikeDrivenTransformerV2(
        in_channels=input_channels,
        num_classes=num_classes,
        embed_dims=embed_dims,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=False,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depth=depth,
        sr_ratio=1,
        T=T,
        kd=kd,
    )
    
    return SpikeDrivenTransformerV2Wrapper(model)
