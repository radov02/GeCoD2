import torch
from torch import nn
from torch.nn import functional as F


class DensityHead(nn.Module):
    """Density head on the C_base adapted map (image_size // 2, same res as the
    GT density). Softplus output, spatial sum = predicted count."""

    def __init__(self, in_ch=256, hidden=128):
        super(DensityHead, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 1, kernel_size=1),
        )
        self.act = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # negative bias keeps the initial predicted count small
        nn.init.constant_(self.net[-1].bias, -9.0)

    def forward(self, x):
        return self.act(self.net(x))


class DensityDecoder(nn.Module):
    """Deeper density decoder: fuses adapted_f + adapted_f_aux and upsamples 2x,
    so DensityLoss upsamples the GT to match. Same near-empty init as DensityHead."""

    def __init__(self, in_ch=256, hidden=128):
        super(DensityDecoder, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 2, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),  # 512 -> 1024
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden // 2, 1, kernel_size=1)
        self.act = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        # negative bias keeps the initial predicted count small
        nn.init.constant_(self.out.bias, -9.0)

    def forward(self, main, aux):
        x = torch.cat([main, aux], dim=1)
        x = self.fuse(x)
        x = self.up(x)
        return self.act(self.out(x))


class DensityGuidedDetection(nn.Module):
    """Density-guided detection feedback (IOCFormer-style): the detached predicted
    density modulates adapted_f via a zero-init residual gate before the class/bbox heads."""

    def __init__(self, feat_ch=256, hidden=64):
        super(DensityGuidedDetection, self).__init__()
        # Encode the single-channel density prior into feat_ch modulation features.
        self.encode = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, feat_ch, kernel_size=3, padding=1),
        )
        # Zero-init residual gate -> identity at init.
        self.gamma = nn.Parameter(torch.zeros(1, feat_ch, 1, 1))
        # plain attr (out of state_dict / DDP sync): ||gamma*mod|| / ||feat||
        # from the last forward, read by CNT.density_guide_stats().
        self.last_rel_energy = None
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.encode.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, feat, density):
        # feat (B, C, w, h), density (B, 1, Hd, Wd) (512 'simple', 1024 'fpn')
        d = density.detach()
        if d.shape[-2:] != feat.shape[-2:]:
            d = F.interpolate(d, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        # log1p compresses the density's large dynamic range for the encoder
        d = torch.log1p(d)
        mod = self.encode(d)
        residual = self.gamma * mod
        # gate perturbation ||gamma*mod|| / ||feat|| for the log
        with torch.no_grad():
            num = residual.flatten(1).norm(dim=1)
            den = feat.flatten(1).norm(dim=1).clamp_min(1e-6)
            self.last_rel_energy = (num / den).mean().detach()
        return feat + residual


class UpsamplingLayer(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(UpsamplingLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.reset_parameters()

    def forward(self, x):
        return self.layer(x)


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
