import torch
import torch.nn.functional as F


def center_pad(x, out_hw):
    """
    Pad tensor around the center.

    Args:
        x: tensor [..., H, W]
        out_hw: target size (out_h, out_w)

    Returns:
        tensor [..., out_h, out_w]
    """
    h, w = x.shape[-2:]
    out_h, out_w = out_hw

    pad_top = (out_h - h) // 2
    pad_bottom = out_h - h - pad_top
    pad_left = (out_w - w) // 2
    pad_right = out_w - w - pad_left

    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


def center_crop(x, crop_hw):
    """
    Crop tensor around the center.

    Args:
        x: tensor [..., H, W]
        crop_hw: target size (crop_h, crop_w)

    Returns:
        tensor [..., crop_h, crop_w]
    """
    h, w = x.shape[-2:]
    crop_h, crop_w = crop_hw

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    return x[..., top : top + crop_h, left : left + crop_w]


def psf_to_otf(psf, out_hw):
    """
    Convert PSF to Fourier-domain OTF.

    Args:
        psf: tensor [B, C, H, W] or [C, H, W]
        out_hw: padded size (out_h, out_w)

    Returns:
        otf: complex tensor [B, C, out_h, out_w // 2 + 1]
    """
    if psf.ndim == 3:
        psf = psf.unsqueeze(0)

    psf_pad = center_pad(psf, out_hw)
    psf_pad = torch.fft.ifftshift(psf_pad, dim=(-2, -1))

    return torch.fft.rfft2(psf_pad, dim=(-2, -1))


def fft_convolve(x, otf):
    """
    Apply forward convolution Hx using FFT.

    Args:
        x: tensor [B, C, H, W]
        otf: complex tensor [B, C, H, W // 2 + 1]

    Returns:
        y: tensor [B, C, H, W]
    """
    x_f = torch.fft.rfft2(x, dim=(-2, -1))
    y = torch.fft.irfft2(x_f * otf, s=x.shape[-2:], dim=(-2, -1))
    return y


def fft_convolve_adjoint(y, otf):
    """
    Apply adjoint convolution H^T y using FFT.

    Args:
        y: tensor [B, C, H, W]
        otf: complex tensor [B, C, H, W // 2 + 1]

    Returns:
        x: tensor [B, C, H, W]
    """
    y_f = torch.fft.rfft2(y, dim=(-2, -1))
    x = torch.fft.irfft2(y_f * torch.conj(otf), s=y.shape[-2:], dim=(-2, -1))
    return x


def finite_diff(x):
    """
    Circular finite differences for anisotropic TV.

    Args:
        x: tensor [B, C, H, W]

    Returns:
        diff: tensor [B, C, 2, H, W]
              diff[:, :, 0] - vertical differences
              diff[:, :, 1] - horizontal differences
    """
    dx = torch.roll(x, shifts=-1, dims=-2) - x
    dy = torch.roll(x, shifts=-1, dims=-1) - x
    return torch.stack([dx, dy], dim=2)


def finite_diff_adjoint(g):
    """
    Apply adjoint finite-difference operator Psi^T.

    Args:
        g: tensor [B, C, 2, H, W]

    Returns:
        tensor [B, C, H, W]
    """
    dx = g[:, :, 0]
    dy = g[:, :, 1]

    adj_x = torch.roll(dx, shifts=1, dims=-2) - dx
    adj_y = torch.roll(dy, shifts=1, dims=-1) - dy

    return adj_x + adj_y


def normalize_psf_sum(psf, eps=1e-8):
    """
    Normalize PSF by spatial sum.

    Args:
        psf: tensor [B, C, H, W] or [C, H, W]
        eps: small value to avoid division by zero

    Returns:
        normalized psf with the same shape
    """
    denom = psf.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return psf / denom


def finite_diff_otf(channels, out_hw, device=None, dtype=torch.float32):
    """
    Compute Fourier-domain |Psi|^2 term for ADMM x-update.

    Args:
        channels: number of channels C
        out_hw: padded size (H, W)
        device: tensor device
        dtype: tensor dtype

    Returns:
        tensor [1, C, H, W // 2 + 1]
    """
    h, w = out_hw

    impulse = torch.zeros(1, channels, h, w, device=device, dtype=dtype)
    impulse[..., 0, 0] = 1.0

    diff = finite_diff(impulse)
    dx = diff[:, :, 0]
    dy = diff[:, :, 1]

    dx_f = torch.fft.rfft2(dx, dim=(-2, -1))
    dy_f = torch.fft.rfft2(dy, dim=(-2, -1))

    return torch.abs(dx_f) ** 2 + torch.abs(dy_f) ** 2
