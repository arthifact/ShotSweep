"""Painter networks: one update step of the state, conditioned on the exact adjoint image."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from wave import NZ, NX, NS, NT, NR
from scene import K, BGZ, BGX

TMAX = 8
PZ, PX = 8, 8
NPZ, NPX = NZ // PZ, NX // PX


def norm_img(g):
    return g / (g.abs().amax(dim=(1, 2), keepdim=True) + 1e-12)


def norm_v(v):
    return (v - 2500.0) / 1000.0


def _rms_norm(x, dim):
    return x / (x.pow(2).mean(dim=dim, keepdim=True).sqrt() + 1e-12)


class GaussPainter(nn.Module):
    """Slot tokens + background-node tokens + image-patch tokens -> deltas for slots and background."""

    def __init__(self, d=96, layers=2, heads=4):
        super().__init__()
        self.slot_in = nn.Linear(5 + 5 + 2, d)
        self.bg_in = nn.Linear(4, d)
        self.patch_in = nn.Linear(2 * PZ * PX + 2, d)
        self.type_emb = nn.Parameter(torch.randn(3, d) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(TMAX, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers)
        self.slot_out = nn.Linear(d, 5)
        self.bg_out = nn.Linear(d, 1)
        for m in (self.slot_out, self.bg_out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.register_buffer("slot_scale", torch.tensor([0.1, 0.1, 0.3, 0.3, 0.3]))
        bz, bx = torch.meshgrid(torch.linspace(0, 1, BGZ), torch.linspace(0, 1, BGX), indexing="ij")
        self.register_buffer("bg_pos", torch.stack([bz.reshape(-1), bx.reshape(-1)], -1))
        pz, px = torch.meshgrid(torch.linspace(0, 1, NPZ), torch.linspace(0, 1, NPX), indexing="ij")
        self.register_buffer("patch_pos", torch.stack([pz.reshape(-1), px.reshape(-1)], -1))

    def forward(self, bg, slots, gb, gs, g, v, t):
        B = bg.shape[0]
        gn, vn = norm_img(g), norm_v(v)
        img = torch.stack([gn, vn], 1)  # (B,2,NZ,NX)
        # sample image at slot centres
        grid = torch.stack([2 * slots[..., 1] - 1, 2 * slots[..., 0] - 1], -1)[:, :, None, :]  # (B,K,1,2) x,y
        samp = F.grid_sample(img, grid, align_corners=True)[:, :, :, 0].transpose(1, 2)  # (B,K,2)
        gs_n = _rms_norm(gs, dim=1)
        slot_tok = self.slot_in(torch.cat([slots, gs_n, samp], -1)) + self.type_emb[0]
        gb_n = _rms_norm(gb.reshape(B, -1, 1), dim=1)
        bg_tok = self.bg_in(torch.cat([bg.reshape(B, -1, 1), gb_n, self.bg_pos.expand(B, -1, -1)], -1)) + self.type_emb[1]
        patches = img.unfold(2, PZ, PZ).unfold(3, PX, PX)  # (B,2,NPZ,NPX,PZ,PX)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, NPZ * NPX, -1)
        patch_tok = self.patch_in(torch.cat([patches, self.patch_pos.expand(B, -1, -1)], -1)) + self.type_emb[2]
        x = torch.cat([slot_tok, bg_tok, patch_tok], 1) + self.step_emb[t]
        x = self.tf(x)
        d_slots = self.slot_out(x[:, :K]) * self.slot_scale
        d_bg = self.bg_out(x[:, K:K + BGZ * BGX])[..., 0].reshape(B, BGZ, BGX) * 0.3
        return d_bg, d_slots


class PixelPainter(nn.Module):
    """Small U-Net: (gradient image, current field, step) -> delta velocity field."""

    def __init__(self, c=32):
        super().__init__()
        cin = 2 + TMAX
        self.e1 = nn.Sequential(nn.Conv2d(cin, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1), nn.GELU())
        self.e2 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.GELU())
        self.e3 = nn.Sequential(nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.GELU())
        self.d2 = nn.Sequential(nn.Conv2d(4 * c, 2 * c, 3, padding=1), nn.GELU())
        self.d1 = nn.Sequential(nn.Conv2d(3 * c, c, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(c, 1, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, g, v, t):
        B = g.shape[0]
        step = F.one_hot(torch.full((B,), t), TMAX).float()[:, :, None, None].expand(B, TMAX, NZ, NX)
        x = torch.cat([norm_img(g)[:, None], norm_v(v)[:, None], step], 1)
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        u2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], 1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], 1))
        return self.out(d1)[:, 0] * 0.3  # km/s


class OneShotNet(nn.Module):
    """The thing the proposal argues against: raw gathers -> velocity in one shot, no physics in the loop."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(NS, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 4)), nn.Flatten(),
            nn.Linear(64 * 32, 512), nn.GELU(),
            nn.Linear(512, NZ * NX),
        )

    def forward(self, d_obs):
        d = d_obs / (d_obs.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-12)
        return 2.5 + self.enc(d).reshape(-1, NZ, NX)  # km/s
