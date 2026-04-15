import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

__all__ = ['Spikeformer']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, N = x.shape
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.attn_drop = nn.Dropout(0.2)
        self.res_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')
        self.attn_lif = neuron.LIFNode(step_mode='m', v_threshold=0.5, backend='torch')

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

    def forward(self, x):
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T, B, C, N))

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + (self.mlp(x))
        return x


class SPS(nn.Module):
    """
    Simplified Patch Embedding for Spikeformer
    Input: T B C V (Time, Batch, Channels, Vertices)
    Output: T B embed_dims V (no downsampling, only channel mapping)
    """
    def __init__(self, in_channels, embed_dims=256):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        
        # Single layer mapping: C -> embed_dims
        self.proj_conv = nn.Conv1d(in_channels, embed_dims, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dims)
        self.proj_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

    def forward(self, x):
        """
        Args:
            x: T B C V
        Returns:
            x: T B embed_dims V
        """
        T, B, C, V = x.shape
        x = self.proj_conv(x.flatten(0, 1))  # (T*B, C, V) -> (T*B, embed_dims, V)
        x = self.proj_bn(x).reshape(T, B, self.embed_dims, V).contiguous()
        x = self.proj_lif(x)
        return x


class Spikeformer(nn.Module):
    def __init__(self,
                 in_channels=2, num_classes=11,
                 embed_dims=256, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depth=2, sr_ratio=1
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.embed_dims = embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Patch embedding: T B C V -> T B embed_dims V
        self.patch_embed = SPS(in_channels=in_channels, embed_dims=embed_dims)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
                norm_layer=norm_layer, sr_ratio=sr_ratio
            )
            for j in range(depth)
        ])

        self.norm = norm_layer(embed_dims)
        # Classification head
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
        # Forward features: T B C V -> T B embed_dims
        x = self.forward_features(x)
        # Average over time and classify: T B embed_dims -> B num_classes
        x = self.head(x.mean(0))
        return x


@register_model
def spikeformer(pretrained=False, **kwargs):
    model = Spikeformer(
        in_channels=2, num_classes=10,
        embed_dims=256, num_heads=16, mlp_ratio=4., qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depth=2, sr_ratio=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


class SpikeformerWrapper(nn.Module):
    """
    Adapts batch-first IMU tensors (B, T, C, V) to the internal (T, B, C, V) convention.
    """
    def __init__(self, spikeformer_model):
        super().__init__()
        self.model = spikeformer_model

    def forward(self, x):
        """
        Input: (B, T, C, V)
        Output: (B, num_classes)
        """
        x = x.permute(1, 0, 2, 3).contiguous()
        return self.model(x)


def create_spikformer_model(config: dict) -> nn.Module:
    """
    Instantiate Spikeformer from ``config['model']`` and wrap for (B, T, C, V) inputs.
    """
    model_config = config.get('model', {})

    num_classes = model_config.get('num_classes', 12)
    input_channels = model_config.get('input_channels', 3)
    embed_dims = model_config.get('embed_dims', 256)
    num_heads = model_config.get('num_heads', 16)
    mlp_ratio = model_config.get('mlp_ratio', 4.0)
    depth = model_config.get('depth', 2)
    drop_rate = model_config.get('drop_rate', 0.0)
    attn_drop_rate = model_config.get('attn_drop_rate', 0.0)
    drop_path_rate = model_config.get('drop_path_rate', 0.0)
    
    model = Spikeformer(
        in_channels=input_channels,
        num_classes=num_classes,
        embed_dims=embed_dims,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        depth=depth,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sr_ratio=1,
    )
    
    return SpikeformerWrapper(model)

