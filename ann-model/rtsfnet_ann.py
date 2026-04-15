import math
import torch
import torch.nn as nn


def _rodrigues_rotation(x_xyz: torch.Tensor, axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate xyz vectors with Rodrigues formula.
    x_xyz: (B, T, G, 3), axis: (B, H, 3), theta: (B, H)
    return: (B, H, T, G, 3)
    """
    b, t, g, _ = x_xyz.shape
    h = axis.shape[1]

    v = x_xyz.unsqueeze(1)  # (B,1,T,G,3)
    k = axis.unsqueeze(2).unsqueeze(2)  # (B,H,1,1,3)
    th = theta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1,1)

    cos_t = torch.cos(th)
    sin_t = torch.sin(th)
    cross = torch.cross(k.expand_as(v.expand(b, h, t, g, 3)), v.expand(b, h, t, g, 3), dim=-1)
    dot = (k * v).sum(dim=-1, keepdim=True)
    rot = v * cos_t + cross * sin_t + k * dot * (1.0 - cos_t)
    return rot


class RTSFNetANN(nn.Module):
    """PyTorch ANN adaptation of rTsfNet core ideas.

    Keeps the key mechanisms:
    - multi-head 3D rotation on grouped xyz axes
    - handcrafted-ish time-series features from blocks
    - MLP classifier
    """

    def __init__(
        self,
        input_channels=3,
        num_imus=1,
        num_classes=12,
        heads=4,
        rot_hidden=128,
        mlp_hidden=256,
        dropout=0.5,
        block_sizes=(16, 32, 64),
    ):
        super().__init__()
        self.in_feat = int(input_channels * num_imus)
        self.heads = int(heads)
        self.block_sizes = tuple(int(x) for x in block_sizes)

        self.rot_param_mlp = nn.Sequential(
            nn.Linear(self.in_feat, rot_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(rot_hidden, rot_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(rot_hidden, self.heads * 4),
            nn.Tanh(),
        )

        # feature projection after TSF extraction
        self.feature_proj = nn.Sequential(
            nn.Linear(self.heads * 7 * len(self.block_sizes), mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

    def _extract_tsf(self, x: torch.Tensor, block_size: int) -> torch.Tensor:
        """x: (B, T, F) -> pooled block-level statistics: (B, F, 7)"""
        b, t, f = x.shape
        bs = min(block_size, t)
        step = bs
        chunks = []
        for s in range(0, max(1, t - bs + 1), step):
            chunks.append(x[:, s:s + bs, :])
        if not chunks:
            chunks = [x]
        z = torch.stack(chunks, dim=1)  # (B, N, bs, F)

        mean = z.mean(dim=2)
        std = z.std(dim=2, unbiased=False)
        z_min = z.min(dim=2).values
        z_max = z.max(dim=2).values
        rms = torch.sqrt((z ** 2).mean(dim=2) + 1e-8)
        diff = z[:, :, 1:, :] - z[:, :, :-1, :]
        mac = diff.abs().mean(dim=2) if diff.numel() > 0 else torch.zeros_like(mean)
        energy = (z ** 2).sum(dim=2) / float(bs)

        feat = torch.stack([mean, std, z_min, z_max, rms, mac, energy], dim=-1)  # (B, N, F, 7)
        return feat.mean(dim=1)  # (B, F, 7)

    def forward(self, x):
        # x: (B, T, C, V)
        b, t, c, v = x.shape
        x = x.reshape(b, t, c * v)

        # pad to multiples of 3 for xyz grouping
        f = x.shape[-1]
        pad = (3 - (f % 3)) % 3
        if pad > 0:
            x = torch.cat([x, torch.zeros(b, t, pad, device=x.device, dtype=x.dtype)], dim=-1)
        g = x.shape[-1] // 3
        x_xyz = x.view(b, t, g, 3)

        # derive rotation params from global context
        global_feat = x.mean(dim=1)[:, :self.in_feat]  # (B, C*V)
        rot = self.rot_param_mlp(global_feat).view(b, self.heads, 4)
        axis = rot[..., :3]
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        theta = rot[..., 3] * math.pi

        rotated = _rodrigues_rotation(x_xyz, axis, theta)  # (B,H,T,G,3)
        rotated = rotated.reshape(b, self.heads, t, g * 3)

        # TSF on each head and each block size
        per_head_feats = []
        for h in range(self.heads):
            xh = rotated[:, h, :, :]  # (B,T,F)
            blk_feats = []
            for bs in self.block_sizes:
                tsf = self._extract_tsf(xh, bs)  # (B,F,7)
                tsf = tsf.mean(dim=1)  # axis aggregation -> (B,7)
                blk_feats.append(tsf)
            per_head_feats.append(torch.cat(blk_feats, dim=-1))  # (B, 7*nb)

        feat = torch.cat(per_head_feats, dim=-1)  # (B, H*7*nb)
        return self.feature_proj(feat)


def create_rtsfnet_ann_model(config):
    m = config.get('model', {})
    return RTSFNetANN(
        input_channels=m.get('input_channels', 3),
        num_imus=m.get('num_imus', 1),
        num_classes=m.get('num_classes', 12),
        heads=m.get('heads', 4),
        rot_hidden=m.get('rot_hidden', 128),
        mlp_hidden=m.get('hidden_dim', 256),
        dropout=m.get('dropout', 0.5),
        block_sizes=tuple(m.get('block_sizes', [16, 32, 64])),
    )
