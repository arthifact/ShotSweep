"""Data-derived surface anchors and training-only mean depth trends.

First arrivals at long offsets can be refracted. Estimate the surface anchor
from short offsets, assess curvature and shot-to-shot disagreement, and reject
picks that do not support a single near-surface velocity. Confidence is a
heuristic quality score, not a calibrated probability.
"""
import numpy as np
from constants import DX, DT, NX, NZ, NT, SRC_X
RHOS = (0.6, 1.0, 1.4)


def _line(offsets, times):
    slope, intercept = np.polyfit(offsets, times, 1)
    residual = times - (intercept + slope * offsets)
    return slope, float(np.sqrt(np.mean(residual ** 2)))


def estimate_surface_velocity(obs, frac=0.05):
    """Fit threshold onsets at 10–60 m and return velocity and quality diagnostics.

    Wider offsets only diagnose possible refraction; they do not set velocity.
    Strong curvature in the short-offset picks, excessive shot disagreement,
    zero/nonfinite traces, or an implausible slope cause a clear ValueError.
    A very shallow refractor can still be unresolved at this bandwidth; this
    estimator is an input approximation, not a guaranteed surface measurement.
    """
    obs = np.asarray(obs)
    if obs.shape != (5, NT, NX) or not np.isfinite(obs).all():
        raise ValueError(f'observations must be finite with shape (5, {NT}, {NX})')
    if not 0 < frac < 1:
        raise ValueError('onset threshold frac must lie between zero and one')
    offsets, arrivals, shots = [], [], []
    for s, gather in enumerate(obs):
        amplitude = np.abs(gather)
        peaks = amplitude.max(axis=0)
        for r in range(NX):
            h = abs(r * DX - SRC_X[s])
            if 10 <= h <= 250 and peaks[r] > 0:
                offsets.append(h)
                arrivals.append(np.argmax(amplitude[:, r] > frac * peaks[r]) * DT)
                shots.append(s)
    offsets, arrivals, shots = map(np.asarray, (offsets, arrivals, shots))
    if len(offsets) == 0:
        raise ValueError('surface velocity unavailable: no nonzero onset traces')
    distances = np.unique(offsets)
    times = np.array([np.median(arrivals[offsets == h]) for h in distances])
    near = (distances >= 10) & (distances <= 60)
    if near.sum() != 6:
        raise ValueError('surface velocity unavailable: insufficient 10–60 m offsets')
    slope, rmse = _line(distances[near], times[near])
    if slope <= 0 or not 1400 <= 1 / slope <= 4600:
        raise ValueError('surface velocity unavailable: short-offset slope is outside 1400–4600 m/s')
    first = (distances >= 10) & (distances <= 40)
    last = (distances >= 30) & (distances <= 60)
    early_slope, _ = _line(distances[first], times[first])
    late_slope, _ = _line(distances[last], times[last])
    curvature = abs(early_slope - late_slope) / slope
    shot_slopes = []
    for s in range(5):
        keep = (shots == s) & (offsets >= 10) & (offsets <= 60)
        if len(np.unique(offsets[keep])) >= 4:
            shot_slopes.append(_line(offsets[keep], arrivals[keep])[0])
    if len(shot_slopes) < 3:
        raise ValueError('surface velocity unavailable: fewer than three usable shots')
    spread = float(np.std(shot_slopes) / slope)
    if rmse > 0.002 or curvature > 0.15 or spread > 0.15:
        raise ValueError('surface velocity unreliable: curved or inconsistent short-offset arrivals '
                         f'(RMSE={rmse * 1000:.2f} ms, curvature={curvature:.3f}, shot spread={spread:.3f})')
    far = (distances >= 120) & (distances <= 250)
    far_slope = _line(distances[far], times[far])[0] if far.sum() >= 4 else slope
    far_velocity = 1 / far_slope if far_slope > 0 else float('nan')
    refraction = max(0.0, 1 - far_slope / slope)
    confidence = 1 / (1 + rmse / DT + curvature / 0.15 + spread / 0.15 + refraction)
    return {
        'velocity': float(np.clip(1 / slope, 1500.0, 4500.0)),
        'confidence': float(confidence),
        'fit_rmse_ms': rmse * 1000,
        'short_offset_curvature': float(curvature),
        'shot_slope_spread': spread,
        'far_offset_velocity': float(far_velocity),
        'possible_refraction': bool(refraction > 0.15),
    }


def surface_velocity(obs, frac=0.05):
    """Compatibility wrapper; raises instead of silently clipping unusable fits."""
    return estimate_surface_velocity(obs, frac)['velocity']


def depth_trend(models):
    """Training velocity models (N,70,70) -> mean v(z)-v(0), shape (70,)."""
    m = np.asarray(models, np.float64)
    return (m.mean(2) - m[:, :1, :].mean(2)).mean(0)


def backgrounds(v_surface, trend, rhos=RHOS):
    """Return laterally constant slow/medium/fast backgrounds (K,70,70)."""
    return np.stack([np.tile(np.clip(v_surface + r * trend, 1500.0, 4500.0)[:, None], (1, NX)) for r in rhos])
