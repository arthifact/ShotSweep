"""OpenFWI metrics: MAE/RMSE on [-1, 1], SSIM on [0, 1].

SSIM follows LANL/OpenFWI's pytorch_ssim.py: an 11-pixel Gaussian
window (sigma 1.5), zero padding, population moments and data range 1.
Values remain full precision until the caller formats them for display.
"""
import numpy as np
import torch
import torch.nn.functional as F

from constants import VMIN, VMAX


def norm(v):
    """Scale physical velocity to [-1, 1] without hiding out-of-range errors."""
    return (np.asarray(v, dtype=np.float64) - VMIN) / (VMAX - VMIN) * 2 - 1


def torch_ssim(a, b, window_size=11, size_average=True):
    """Differentiable OpenFWI SSIM for float BHW or B1HW tensors in [0, 1].

    Inputs must have identical shape, dtype and device. No clipping or range
    conversion is performed. Pass ``(normalized_velocity + 1) / 2`` when
    training on velocities normalized to [-1, 1]. With ``size_average=False``
    the result contains one score per image; otherwise it is a scalar.
    """
    if a.shape != b.shape or a.ndim not in (3, 4):
        raise ValueError('SSIM expects matching BHW or B1HW tensors')
    if a.ndim == 3:
        a, b = a[:, None], b[:, None]
    if a.shape[1] != 1 or min(a.shape) < 1:
        raise ValueError('SSIM expects nonempty, single-channel images')
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError('SSIM inputs must share dtype and device')
    if not a.is_floating_point():
        raise TypeError('SSIM inputs must be floating-point tensors')
    if not isinstance(window_size, int) or window_size < 1 or window_size % 2 != 1:
        raise ValueError('SSIM window_size must be a positive odd integer')
    x = torch.arange(window_size, dtype=a.dtype, device=a.device) - window_size // 2
    gaussian = torch.exp(-x.square() / (2 * 1.5 ** 2))
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :])[None, None]
    padding = window_size // 2
    mu_a = F.conv2d(a, window, padding=padding)
    mu_b = F.conv2d(b, window, padding=padding)
    va = F.conv2d(a * a, window, padding=padding) - mu_a.square()
    vb = F.conv2d(b * b, window, padding=padding) - mu_b.square()
    cov = F.conv2d(a * b, window, padding=padding) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a.square() + mu_b.square() + c1) * (va + vb + c2))
    per_image = score.mean(dim=(1, 2, 3))
    return per_image.mean() if size_average else per_image


def ssim(a, b, win=11):
    """SSIM for two 2-D NumPy images already scaled to [0, 1]."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError('SSIM expects matching 2-D images')
    with torch.no_grad():
        return float(torch_ssim(torch.from_numpy(a)[None], torch.from_numpy(b)[None], win))


def metrics(pred, true):
    """Return unrounded OpenFWI scores for two physical velocity maps (m/s)."""
    a, b = norm(pred), norm(true)
    if a.ndim != 2 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Metrics require finite, matching 2-D velocity maps')
    difference = a - b
    return dict(mae=float(np.abs(difference).mean()),
                rmse=float(np.sqrt(np.square(difference).mean())),
                ssim=ssim((a + 1) / 2, (b + 1) / 2))
