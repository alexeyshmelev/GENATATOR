import torch
from torch import nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int = 2):
        super().__init__()
        convs = [nn.Conv1d(in_channels, out_channels, 3, padding=1)]
        convs += [nn.Conv1d(out_channels, out_channels, 3, padding=1) for _ in range(layers - 1)]
        self.convs = nn.ModuleList(convs)
        self.pool = nn.AvgPool1d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        for conv in self.convs:
            x = F.silu(conv(x))
        skip = x
        return self.pool(x), skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 2, stride=2)
        convs = [nn.Conv1d(out_channels * 2, out_channels, 3, padding=1)]
        convs += [nn.Conv1d(out_channels, out_channels, 3, padding=1) for _ in range(layers - 1)]
        self.convs = nn.ModuleList(convs)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) < skip.size(-1):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
        if x.size(-1) > skip.size(-1):
            x = x[..., : skip.size(-1)]
        x = torch.cat([skip, x], dim=1)
        for conv in self.convs:
            x = F.silu(conv(x))
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels=(192, 384, 768), layers: int = 2):
        super().__init__()
        ch = list(channels)
        self.down = nn.ModuleList([DownBlock(a, b, layers) for a, b in zip([in_channels] + ch[:-1], ch)])
        rev = ch[::-1]
        self.up = nn.ModuleList([UpBlock(a, b, layers) for a, b in zip([ch[-1]] + rev[:-1], rev)])
        self.final = nn.Conv1d(ch[0], out_channels, 3, padding=1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x, skip = block(x)
            skips.append(skip)
        for block, skip in zip(self.up, reversed(skips)):
            x = block(x, skip)
        return self.final(x)
