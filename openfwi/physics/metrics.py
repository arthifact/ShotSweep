"""OpenFWI metrics on velocity min-max scaled to [-1, 1]: MAE, RMSE, SSIM (7 x 7 window)."""
import numpy as np
from constants import VMIN, VMAX

def norm(v): return (np.clip(v, VMIN, VMAX) - VMIN) / (VMAX - VMIN) * 2 - 1

def ssim(a, b, win=7, K1=0.01, K2=0.03, L=2.0):
    def box(x):
        c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), 0), 1); w = win
        return (c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]) / (w * w)
    mu_a, mu_b = box(a), box(b); va = box(a * a) - mu_a ** 2; vb = box(b * b) - mu_b ** 2; cov = box(a * b) - mu_a * mu_b
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    return float((((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))).mean())

def metrics(pred, true):
    a, b = norm(pred), norm(true); return dict(mae=round(float(np.abs(a - b).mean()), 4), rmse=round(float(np.sqrt(((a - b) ** 2).mean())), 4), ssim=round(ssim(a, b), 4))
