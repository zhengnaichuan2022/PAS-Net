from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.activation_based import neuron, layer, functional

__all__ = ['SpikeDrivenTransformer']


class MS_MLP_Conv(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')
        # Note: plif mode not supported in activation_based, using lif as default

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer

    def forward(self, x):
        T, B, C, V = x.shape
        identity = x

        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, V).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, V).contiguous()

        x = x + identity
        return x


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        if spike_mode == "lif":
            self.q_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        if spike_mode == "lif":
            self.k_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        if spike_mode == "lif":
            self.v_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')

        if spike_mode == "lif":
            self.attn_lif = neuron.LIFNode(
                step_mode='m', v_threshold=0.5, backend='torch'
            )

        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = neuron.LIFNode(
                step_mode='m', v_threshold=0.5, backend='torch'
            )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = neuron.LIFNode(
                step_mode='m', v_threshold=1.0, backend='torch'
            )

        self.mode = mode
        self.layer = layer

    def forward(self, x):
        T, B, C, V = x.shape
        identity = x
        N = V
        x = self.shortcut_lif(x)

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, V).contiguous()
        q_conv_out = self.q_lif(q_conv_out)

        q = (
            q_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, V).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = (
            k_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, V).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = (
            v_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        x = q.mul(kv)

        x = x.transpose(3, 4).reshape(T, B, C, V).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, V)
            .contiguous()
        )

        x = x + identity
        return x, v


class MS_Block_Conv(nn.Module):
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
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP_Conv(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
            spike_mode=spike_mode,
            layer=layer,
        )

    def forward(self, x):
        x_attn, attn = self.attn(x)
        x = self.mlp(x_attn)
        return x, attn


class MS_SPS(nn.Module):
    """
    Simplified Patch Embedding for SpikeDrivenTransformer
    Input: T B C V (Time, Batch, Channels, Vertices)
    Output: T B embed_dims V (no downsampling, only channel mapping)
    """
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        spike_mode="lif",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        
        # Single layer mapping: C -> embed_dims
        self.proj_conv = nn.Conv1d(
            in_channels, embed_dims, kernel_size=1, stride=1, bias=False
        )
        self.proj_bn = nn.BatchNorm1d(embed_dims)
        if spike_mode == "lif":
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


class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depth=6,
        sr_ratios=1,
        T=4,
        attn_mode="direct_xor",
        spike_mode="lif",
        get_embed=False,
        dvs_mode=False,
        TET=False,
        cml=False,
        pretrained=False,
        pretrained_cfg=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # Patch embedding: T B C V -> T B embed_dims V
        self.patch_embed = MS_SPS(
            in_channels=in_channels,
            embed_dims=embed_dims,
            spike_mode=spike_mode,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                )
                for j in range(depth)
            ]
        )

        self.block = blocks

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = neuron.LIFNode(step_mode='m', v_threshold=1.0, backend='torch')
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
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
        x = self.patch_embed(x)
        for blk in self.block:
            x, _ = blk(x)

        x = x.mean(3)  # Average over vertices: T B embed_dims V -> T B embed_dims
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
        x = self.head_lif(x)
        x = self.head(x)
        if not self.TET:
            x = x.mean(0)  # Average over time: T B num_classes -> B num_classes
        return x


@register_model
def sdt(**kwargs):
    model = SpikeDrivenTransformer(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


class SpikeDrivenTransformerWrapper(nn.Module):
    """
    Adapts (B, T, C, V) IMU batches to internal (T, B, C, V).
    """
    def __init__(self, spike_driven_transformer_model):
        super().__init__()
        self.model = spike_driven_transformer_model

    def forward(self, x):
        """
        Input: (B, T, C, V)
        Output: (B, num_classes)
        """
        x = x.permute(1, 0, 2, 3).contiguous()
        return self.model(x)


def create_spike_driven_transformer_model(config: dict) -> nn.Module:
    """
    Build SpikeDrivenTransformer from ``config['model']`` and wrap for (B, T, C, V).
    """
    model_config = config.get('model', {})

    num_classes = model_config.get('num_classes', 12)
    input_channels = model_config.get('input_channels', 3)
    embed_dims = model_config.get('embed_dims', 512)
    num_heads = model_config.get('num_heads', 8)
    mlp_ratios = model_config.get('mlp_ratios', 4.0)
    depth = model_config.get('depth', 6)
    drop_rate = model_config.get('drop_rate', 0.0)
    attn_drop_rate = model_config.get('attn_drop_rate', 0.0)
    drop_path_rate = model_config.get('drop_path_rate', 0.0)
    T = model_config.get('T', 4)
    attn_mode = model_config.get('attn_mode', 'direct_xor')
    spike_mode = model_config.get('spike_mode', 'lif')
    dvs_mode = model_config.get('dvs_mode', False)
    TET = model_config.get('TET', False)
    
    model = SpikeDrivenTransformer(
        in_channels=input_channels,
        num_classes=num_classes,
        embed_dims=embed_dims,
        num_heads=num_heads,
        mlp_ratios=mlp_ratios,
        qkv_bias=False,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depth=depth,
        sr_ratios=1,
        T=T,
        attn_mode=attn_mode,
        spike_mode=spike_mode,
        dvs_mode=dvs_mode,
        TET=TET,
    )
    
    return SpikeDrivenTransformerWrapper(model)

