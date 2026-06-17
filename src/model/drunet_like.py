import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Bias-free residual convolution block.
    """

    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


def make_resblocks(channels, num_blocks):
    return nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])


class Downsample(nn.Module):
    """
    2x2 strided convolution downsampling.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """
    2x2 transposed convolution upsampling.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.tconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )

    def forward(self, x, skip):
        x = self.tconv(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return x + skip


class DRUNetLike(nn.Module):
    """
    DRUNet-like U-Net for pre/post processing.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 116, 128),
        num_blocks=4,
        residual=True,
    ):
        super().__init__()

        c1, c2, c3, c4 = channels

        self.residual = residual

        self.head = nn.Conv2d(
            in_channels,
            c1,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.enc1 = make_resblocks(c1, num_blocks)
        self.down1 = Downsample(c1, c2)

        self.enc2 = make_resblocks(c2, num_blocks)
        self.down2 = Downsample(c2, c3)

        self.enc3 = make_resblocks(c3, num_blocks)
        self.down3 = Downsample(c3, c4)

        self.body = make_resblocks(c4, num_blocks)

        self.up3 = Upsample(c4, c3)
        self.dec3 = make_resblocks(c3, num_blocks)

        self.up2 = Upsample(c3, c2)
        self.dec2 = make_resblocks(c2, num_blocks)

        self.up1 = Upsample(c2, c1)
        self.dec1 = make_resblocks(c1, num_blocks)

        self.tail = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        x0 = x

        x1 = self.head(x)
        x1 = self.enc1(x1)

        x2 = self.down1(x1)
        x2 = self.enc2(x2)

        x3 = self.down2(x2)
        x3 = self.enc3(x3)

        x4 = self.down3(x3)
        x4 = self.body(x4)

        x = self.up3(x4, x3)
        x = self.dec3(x)

        x = self.up2(x, x2)
        x = self.dec2(x)

        x = self.up1(x, x1)
        x = self.dec1(x)

        x = self.tail(x)

        if self.residual and x.shape == x0.shape:
            x = x0 + x

        return x
