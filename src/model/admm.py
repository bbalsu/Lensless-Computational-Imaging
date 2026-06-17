import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.fft_operators import (
    center_crop,
    center_pad,
    fft_convolve,
    finite_diff,
    finite_diff_adjoint,
    finite_diff_otf,
    normalize_psf_sum,
    psf_to_otf,
)


def soft_threshold(x, threshold):
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)


def inverse_softplus(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return torch.log(torch.expm1(x))


class ADMMReconstructor(nn.Module):
    """
    ADMM / Le-ADMM reconstructor for lensless imaging.

    Fixed ADMM-100:
        trainable=False, num_iters=100,
        mu1=mu2=mu3=1e-4, tau=2e-4

    Le-ADMM:
        trainable=True, num_iters=20 or 5,
        parameters are learned per iteration.
    """

    def __init__(
        self,
        num_iters=100,
        padded_hw=(768, 1024),
        mu1_init=1e-4,
        mu2_init=1e-4,
        mu3_init=1e-4,
        tau_init=2e-4,
        trainable=False,
        eps=1e-8,
        return_padded=False,
        psf_gain=5.0,
    ):
        super().__init__()

        self.num_iters = num_iters
        self.padded_hw = tuple(padded_hw)
        self.trainable = trainable
        self.eps = eps
        self.return_padded = return_padded
        self.psf_gain = psf_gain

        if trainable:
            self.raw_mu1 = nn.Parameter(inverse_softplus(mu1_init).repeat(num_iters))
            self.raw_mu2 = nn.Parameter(inverse_softplus(mu2_init).repeat(num_iters))
            self.raw_mu3 = nn.Parameter(inverse_softplus(mu3_init).repeat(num_iters))
            self.raw_tau = nn.Parameter(inverse_softplus(tau_init).repeat(num_iters))
        else:
            self.register_buffer("mu1", torch.tensor(float(mu1_init)))
            self.register_buffer("mu2", torch.tensor(float(mu2_init)))
            self.register_buffer("mu3", torch.tensor(float(mu3_init)))
            self.register_buffer("tau", torch.tensor(float(tau_init)))

    def _get_params(self, k):
        if self.trainable:
            mu1 = F.softplus(self.raw_mu1[k]) + self.eps
            mu2 = F.softplus(self.raw_mu2[k]) + self.eps
            mu3 = F.softplus(self.raw_mu3[k]) + self.eps
            tau = F.softplus(self.raw_tau[k]) + self.eps
        else:
            mu1 = self.mu1
            mu2 = self.mu2
            mu3 = self.mu3
            tau = self.tau

        return mu1, mu2, mu3, tau

    @staticmethod
    def _crop_mask(sensor_hw, padded_hw, device, dtype):
        ones = torch.ones(1, 1, *sensor_hw, device=device, dtype=dtype)
        return center_pad(ones, padded_hw)

    def _x_update(self, rhs, otf, diff_symbol, mu1, mu2, mu3):
        denom = mu1 * torch.abs(otf) ** 2 + mu2 * diff_symbol + mu3 + self.eps
        rhs_f = torch.fft.rfft2(rhs, dim=(-2, -1))
        x = torch.fft.irfft2(
            rhs_f / denom,
            s=rhs.shape[-2:],
            dim=(-2, -1),
        )
        return x

    def forward(self, measurement, psf):
        """
        Args:
            measurement: [B, C, H, W], lensless image b
            psf: [B, C, H, W], PSF

        Returns:
            reconstruction: [B, C, H, W], cropped reconstruction
        """
        b, c, h, w = measurement.shape
        sensor_hw = (h, w)
        padded_hw = self.padded_hw
        device = measurement.device
        dtype = measurement.dtype

        psf = normalize_psf_sum(psf) * self.psf_gain
        otf = psf_to_otf(psf, padded_hw)

        diff_symbol = finite_diff_otf(
            channels=c,
            out_hw=padded_hw,
            device=device,
            dtype=dtype,
        )

        crop_mask = self._crop_mask(
            sensor_hw=sensor_hw,
            padded_hw=padded_hw,
            device=device,
            dtype=dtype,
        )

        ct_b = center_pad(measurement, padded_hw)

        x = torch.zeros(b, c, *padded_hw, device=device, dtype=dtype)
        v = torch.zeros_like(x)
        w_var = torch.zeros_like(x)

        u = torch.zeros(b, c, 2, *padded_hw, device=device, dtype=dtype)

        alpha1 = torch.zeros_like(x)
        alpha2 = torch.zeros_like(u)
        alpha3 = torch.zeros_like(x)

        for k in range(self.num_iters):
            mu1, mu2, mu3, tau = self._get_params(k)

            psi_x = finite_diff(x)
            u = soft_threshold(psi_x + alpha2 / mu2, tau / mu2)

            hx = fft_convolve(x, otf)
            rhs_v = alpha1 + mu1 * hx + ct_b
            v = rhs_v / (crop_mask + mu1)

            w_var = torch.clamp(x + alpha3 / mu3, min=0.0)

            rhs_x = (
                fft_convolve_adjoint_like(mu1 * v - alpha1, otf)
                + finite_diff_adjoint(mu2 * u - alpha2)
                + mu3 * w_var
                - alpha3
            )
            x = self._x_update(rhs_x, otf, diff_symbol, mu1, mu2, mu3)

            hx = fft_convolve(x, otf)
            psi_x = finite_diff(x)

            alpha1 = alpha1 + mu1 * (hx - v)
            alpha2 = alpha2 + mu2 * (psi_x - u)
            alpha3 = alpha3 + mu3 * (x - w_var)

        reconstruction = center_crop(x, sensor_hw)

        return reconstruction


def fft_convolve_adjoint_like(y, otf):
    y_f = torch.fft.rfft2(y, dim=(-2, -1))
    x = torch.fft.irfft2(
        y_f * torch.conj(otf),
        s=y.shape[-2:],
        dim=(-2, -1),
    )
    return x
