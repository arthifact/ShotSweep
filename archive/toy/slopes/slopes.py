"""Slope-field velocity extraction on one OpenFWI FlatVel-A model with its five official shot gathers.

Geometry-first pipeline, no training, no simulation:
  1. direct wave  -> surface velocity v1 and wavelet delay tau (linear fit of first-break times vs offset)
  2. structure tensor on each gather -> local slope p = dt/dh and coherence (an orientation field)
  3. every coherent sample votes, in closed form, for a (t0, v_rms) pair:   v^2 = h/(t p),  t0^2 = t^2 - h t p
  4. ridge + peaks of the vote image -> interfaces (t0_k, vrms_k);  Dix -> interval velocities and depths
  5. compare the recovered v(z) to the true model; error normalised by the dataset-mean MSE (264970 (m/s)^2)

usage: uv run python -I toy/slopes/slopes.py [--model 0] [--out toy/slopes/result_0.png]
"""
import argparse, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DX = 10.0; DT = 1e-3; NX = NZ = 70; SRC_X = np.array([0, 17, 34, 52, 69]) * DX   # OpenFWI FlatVel 5-shot geometry
DATUM = 10.0          # sources and receivers sit one cell below the surface: zero-offset times are 2(z - DATUM)/v (seen as a constant 2*DATUM/v1 bias before this was added)
MEAN_MSE = 264970.0   # MSE of the dataset-mean model over FlatVel-A train file, (m/s)^2
BOTTOM_JUMP = 1.13    # the one dataset-derived prior: median velocity ratio across an interface in FlatVel-A (see scale check)

def gauss1d(a, s, axis):
    if s <= 0: return a
    r = int(3 * s) + 1; k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    a = np.moveaxis(a, axis, -1); pad = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(r, r)], mode='edge')
    out = np.zeros_like(a)
    for i, w in enumerate(k): out += w * pad[..., i:i + a.shape[-1]]
    return np.moveaxis(out, -1, axis)

def envelope(g, axis=0):
    n = g.shape[axis]; F = np.fft.fft(g, axis=axis); h = np.zeros(n); h[0] = 1; h[1:(n + 1) // 2] = 2
    if n % 2 == 0: h[n // 2] = 1
    sh = [1] * g.ndim; sh[axis] = n
    return np.abs(np.fft.ifft(F * h.reshape(sh), axis=axis))

def deconvolve(gathers, eps=0.02, wlen=0.15):
    """Wiener deconvolution of the source wavelet, measured on the five zero-offset traces (receiver under the source,
    first `wlen` seconds). Every event becomes a compact zero-phase spike at its true arrival time, so the wavelet delay
    and the Ricker side lobes disappear before any geometry is extracted."""
    nt = gathers.shape[1]; n = int(wlen / DT)
    w = np.zeros(nt)
    for s in range(len(gathers)): w[:n] += gathers[s, :n, int(round(SRC_X[s] / DX))]
    W = np.fft.rfft(w); H = np.conj(W) / (np.abs(W) ** 2 + eps * np.abs(W).max() ** 2)
    out = np.fft.irfft(np.fft.rfft(gathers, axis=1) * H[None, :, None], n=nt, axis=1)
    return out / np.abs(out).max() * np.abs(gathers).max()

def direct_wave_fit(gathers):
    """Surface velocity v1 from the slope of the direct wave (picking-free), wavelet delay tau from its envelope peak.
    Offsets 100-220 m: far enough from the source for the slope to be clean, near enough to precede the first refraction."""
    inv_p, taus = [], []
    for s, g in enumerate(gathers):
        env = envelope(g, 0); p, coh = slope_field(g)
        for r in range(NX):
            h = abs(r * DX - SRC_X[s])
            if h < 100 or h > 220: continue
            tr = env[:, r]; nwin = int((h / 1300.0 + 0.12) / DT)                  # first-arrival window: a later strong event must not set the threshold
            thr = 0.3 * tr[:nwin].max(); i0 = int(np.argmax(tr > thr)); ipk = i0 + int(np.argmax(tr[i0:i0 + 60]))
            win = slice(max(ipk - 10, 0), ipk + 10); ps = np.abs(p[win, r][coh[win, r] > 0.8])
            if len(ps): inv_p.append(1.0 / np.median(ps))
            taus.append((ipk, h))
    v1 = float(np.median(inv_p)); tau = float(np.median([i * DT - h / v1 for i, h in taus]))
    return v1, tau

def slope_field(g, win=12.0, maxlag=20):
    """Local slope p = dt/dh (s/m) and coherence in [0,1] of a gather g(t, h), from local cross-correlation of each trace with
    its two neighbours (Gaussian window of std `win` samples, lags up to `maxlag` samples, parabolic sub-sample refinement).
    Finite-difference structure tensors are biased here because one trace spacing shifts the wavelet by ~7 samples."""
    env = envelope(g, 0); g = g / (gauss1d(env, 25.0, 0) + 1e-3 * env.max())
    nt, nr = g.shape; lags = np.arange(-maxlag, maxlag + 1)
    E = gauss1d(g * g, win, 0)                                    # local energy per trace
    P = np.zeros(g.shape); C = np.zeros(g.shape); n = np.zeros(g.shape)
    for dr in (1, -1):                                            # correlate trace r with trace r+dr
        corr = np.zeros((len(lags), nt, nr - 1))
        a = g[:, :-1] if dr == 1 else g[:, 1:]; b = g[:, 1:] if dr == 1 else g[:, :-1]
        for i, l in enumerate(lags):                              # b shifted by l samples: event at t in a is at t+l in b
            bs = np.zeros_like(b)
            if l >= 0: bs[:nt - l] = b[l:]
            else: bs[-l:] = b[:nt + l]
            corr[i] = gauss1d(a * bs, win, 0)
        Ea = E[:, :-1] if dr == 1 else E[:, 1:]; Eb = E[:, 1:] if dr == 1 else E[:, :-1]
        corr /= (np.sqrt(Ea * Eb) + 1e-3 * E.max())[None]
        k = np.argmax(corr, 0); kc = np.clip(k, 1, len(lags) - 2)
        cm, cl, cr = np.take_along_axis(corr, kc[None], 0)[0], np.take_along_axis(corr, (kc - 1)[None], 0)[0], np.take_along_axis(corr, (kc + 1)[None], 0)[0]
        denom = cl - 2 * cm + cr; frac = np.where(np.abs(denom) > 1e-12, 0.5 * (cl - cr) / (denom + 1e-30), 0.0)
        lag = (lags[kc] + np.clip(frac, -1, 1)) * DT                # event shift from trace r to trace r+dr, seconds
        slope = lag / (dr * DX)                                     # dt/dh along increasing h
        cols = slice(0, nr - 1) if dr == 1 else slice(1, nr)
        P[:, cols] += slope * np.clip(cm, 0, 1); C[:, cols] += np.clip(cm, 0, 1); n[:, cols] += 1
    coh = C / np.maximum(n, 1); p = P / (C + 1e-9)
    return p, coh ** 2

PICK = 'envelope'   # 'envelope': polarity-free (default, same for every family); 'polarity': the earlier FlatVel-only rule

def oriented_points(gathers, tau, v1, coh_min=0.5, hmin=60.0, amp_min=0.3, mute_s=0.045, lobe_s=0.045, src_x=None):
    """The gather as an oriented point set with local slopes. 'envelope' picking: one point per wavelet, at the local
    maxima of the AGC-normalised envelope, whatever the polarity (no side lobes, since the envelope has one peak per
    wavelet, and no assumption about the sign of velocity jumps). The direct-wave band (|t - |h|/v1| < mute_s) is
    removed geometrically. 'polarity' picking (earlier, FlatVel-specific): peaks of |g| that share the direct wave's sign.
    Returns t (s, delay removed), h (m, signed offset), p (s/m), w (weight = normalized amplitude^2 x coherence)."""
    T, H, P, W = [], [], [], []; srcs = SRC_X if src_x is None else np.asarray(src_x)
    for s, g in enumerate(gathers):
        env = envelope(g, 0); agc = gauss1d(env, 40.0, 0) + 1e-3 * env.max(); gn = g / agc
        a = env / agc if PICK == 'envelope' else np.abs(gn)
        p, coh = slope_field(g)
        peak = (a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] > amp_min)
        it, ir = np.nonzero(peak); it = it + 1
        h = ir * DX - srcs[s]; t = it * DT - tau
        keep = (np.abs(h) >= hmin) & (t > 0) & (coh[it, ir] > coh_min) & (np.abs(t - np.abs(h) / v1) > mute_s)
        if PICK == 'polarity':
            pol = np.sign(g[int(np.argmax(np.abs(g[:200, int(round(srcs[s] / DX))]))), int(round(srcs[s] / DX))])
            keep &= np.sign(gn[it, ir]) == pol
        T.append(t[keep]); H.append(h[keep]); P.append(p[it, ir][keep]); W.append((a[it, ir] ** 2 * coh[it, ir])[keep])
    return np.concatenate(T), np.concatenate(H), np.concatenate(P), np.concatenate(W)

def vote(points, t0_bins, v_bins, max_sin=0.8):
    """Closed-form hyperbolic (t0, v_rms) vote of every oriented point: v^2 = h/(t p), t0^2 = t^2 - h t p.
    Points whose apparent ray angle is grazing (p * v_rms > max_sin) are head waves or post-critical and are dropped."""
    t, h, p, w = points
    v2 = h / (t * p + 1e-30); t02 = t ** 2 - h * t * p
    ok = (h * p > 0) & (v2 > 0) & (t02 >= 0) & (np.abs(p) * np.sqrt(np.maximum(v2, 0)) < max_sin)
    return np.histogram2d(np.sqrt(t02[ok]), np.sqrt(v2[ok]), bins=[t0_bins, v_bins], weights=w[ok])[0]

def reflection_curve(layers_above, q):
    """Exact offset and two-way time of the reflection from the bottom of a flat layer stack, for ray parameter(s) q."""
    h = np.zeros_like(q); t = np.zeros_like(q)
    for zt, zb, v in layers_above:
        c = np.sqrt(np.maximum(1 - (v * q) ** 2, 1e-12)); h += 2 * (zb - zt) * v * q / c; t += 2 * (zb - zt) / (v * c)
    return h, t

def avo_velocity(gathers, tau, layers, hmax=580.0, min_rise=1.4):
    """Velocity of the layer below the deepest picked reflector from its amplitude-versus-offset curve. Along the exact
    ray-traced curve of that reflection the spreading-corrected envelope, pooled over shots and normalized at near offset,
    is compared with the acoustic reflection coefficient |R(theta)| / |R(0)| of the layer stack over a candidate v_below
    (one parameter, monotonic). Returns (v_below, rise) or None when the curve is too flat to say anything."""
    above = layers[:-1]
    if not above: return None
    vk = above[-1][2]; vmax = max(v for _, _, v in above); q = np.linspace(1e-6, 0.999 / vmax, 4000); H, T = reflection_curve(above, q)
    hs, amps = [], []
    for s, g in enumerate(gathers):
        env = envelope(g, 0)
        for r in range(NX):
            h = abs(r * DX - SRC_X[s])
            if h < 60 or h > min(hmax, H.max()): continue
            j = np.argmin(np.abs(H - h)); i = int(round((T[j] + tau) / DT))
            if i + 15 >= g.shape[0]: continue
            hs.append(h); amps.append(env[i - 15:i + 16, r].max() * np.sqrt(T[j]))   # 2D spreading ~ 1/sqrt(t)
    hs, amps = np.array(hs), np.array(amps)
    hb = np.arange(60, hmax, 40.0); A = np.array([np.median(amps[(hs >= a) & (hs < a + 40)]) if np.any((hs >= a) & (hs < a + 40)) else np.nan for a in hb])
    ok = ~np.isnan(A); hb, A = hb[ok] + 20, A[ok]
    if len(A) < 5: return None
    A = A / A[:2].mean(); rise = float(A.max())
    if rise < min_rise: return None
    qh = q[np.argmin(np.abs(H[None, :] - hb[:, None]), 1)]; sin1 = np.clip(qh * vk, 0, 1); cos1 = np.sqrt(1 - sin1 ** 2)
    best = None
    for v2 in np.arange(vk * 1.05, 5000, 10.0):
        sin2 = sin1 * v2 / vk; cos2 = np.sqrt(np.maximum(1 - sin2 ** 2, 0)) + 0j
        cos2 = np.where(sin2 > 1, 1j * np.sqrt(sin2 ** 2 - 1), cos2)
        R = np.abs((v2 * cos1 - vk * cos2) / (v2 * cos1 + vk * cos2)); P = R / ((v2 - vk) / (v2 + vk))
        P = np.convolve(P, [0.25, 0.5, 0.25], mode='same'); P[0] = P[1]; P[-1] = P[-2]     # wavelet smearing over ~1 bin
        mis = float(((np.log(A) - np.log(P / P[:2].mean())) ** 2).mean())
        if best is None or mis < best[0]: best = (mis, v2)
    if best is None: return None                         # layer above already at the top of the search range
    return float(best[1]), rise

def pick_interfaces(V, t0c, vc, v1, nmax=7, min_sep_s=0.04, rel=0.06, t0_min=0.06):
    """Interfaces = 2D local maxima of the smoothed vote image, refined to the local centroid. Also returns the ridge for display."""
    Vs = gauss1d(gauss1d(V, 1.5, 0), 1.5, 1)
    ridge_v = vc[np.argmax(Vs, 1)]; mass = Vs.max(1)
    loc = np.ones_like(Vs, bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj: loc &= Vs >= np.roll(np.roll(Vs, di, 0), dj, 1)
    ii, jj = np.nonzero(loc & (Vs > rel * Vs.max()) & (t0c[:, None] > t0_min))
    order = np.argsort(-Vs[ii, jj]); picks = []
    for k in order:
        i, j = ii[k], jj[k]
        if all(abs(t0c[i] - t0c[pi]) >= min_sep_s for pi, pj in picks): picks.append((i, j))   # two interfaces cannot share a t0
        if len(picks) == nmax: break
    out = []
    for i, j in picks:                                   # centroid within +-3 bins
        sl = (slice(max(i - 3, 0), i + 4), slice(max(j - 3, 0), j + 4)); w = Vs[sl]
        tt, vv = np.meshgrid(t0c[sl[0]], vc[sl[1]], indexing='ij')
        out.append((float((w * tt).sum() / w.sum()), float((w * vv).sum() / w.sum())))
    out.sort()
    return ridge_v, mass, out

def dix(picks, v1):
    """(t0_k, vrms_k) -> layers [(ztop, zbot, v_int)], anchored at the datum by the direct-wave velocity v1 (the first layer starts at z=0)."""
    layers = []; tp, vp, z = 0.0, v1, DATUM
    for t0, vr in picks:
        v2 = (vr ** 2 * t0 - vp ** 2 * tp) / (t0 - tp)
        vi = float(np.sqrt(max(v2, 1300.0 ** 2))); dz = vi * (t0 - tp) / 2
        layers.append((z, z + dz, vi)); z += dz; tp, vp = t0, vr
    layers.append((z, NZ * DX, (layers[-1][2] if layers else v1) * BOTTOM_JUMP))   # bottom layer: no reflection below it -> dataset prior
    return layers

def layers_to_vz(layers):
    zc = (np.arange(NZ) + 0.5) * DX; vz = np.full(NZ, layers[-1][2])
    for zt, zb, v in layers: vz[(zc >= zt) & (zc < zb)] = v
    vz[zc < layers[0][1]] = layers[0][2]                     # first layer extends up to the surface
    return vz

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', type=int, default=0); ap.add_argument('--out', default=None); ap.add_argument('--data', default=None)
    a = ap.parse_args(); path = a.data or f'toy/slopes/flatvel_a_{a.model}.npz'; out = a.out or f'toy/slopes/result_{a.model}.png'
    d = np.load(path); raw = d['gathers'].astype(np.float64); model = d['model'].astype(np.float64)   # (5,1000,70), (70 z,70 x)
    gathers = raw   # deconvolve(raw) was tried: it weakens shallow reflections and wraps the trace end; not used
    v1, tau = direct_wave_fit(gathers)
    t0_bins = np.arange(0, 1.0, 0.004); v_bins = np.arange(1300, 5000, 25.0)
    t0c = 0.5 * (t0_bins[1:] + t0_bins[:-1]); vc = 0.5 * (v_bins[1:] + v_bins[:-1])
    points = oriented_points(gathers, tau, v1); V = vote(points, t0_bins, v_bins)
    ridge_v, mass, picks = pick_interfaces(V, t0c, vc, v1)
    layers = dix(picks, v1)
    crit = avo_velocity(gathers, tau, layers)
    if crit is not None: layers[-1] = (layers[-1][0], layers[-1][1], crit[0])
    vz = layers_to_vz(layers); pred = np.tile(vz[:, None], (1, NX))
    true_vz = model.mean(1)
    mse = float(((pred - model) ** 2).mean()); mse_mean_model = float(((model - model.mean()) ** 2).mean())
    # truth in (t0, v_rms) coordinates, for the vote-image overlay
    dz = DX; dt_cells = 2 * dz / true_vz; t0_true = np.cumsum(dt_cells) - 2 * DATUM / true_vz[0]; vrms_true = np.sqrt(np.cumsum(true_vz ** 2 * dt_cells) / t0_true)
    iface = np.where(np.abs(np.diff(true_vz)) > 1)[0]
    res = dict(model=a.model, v1=round(v1, 1), tau_ms=round(tau * 1e3, 1), picks=[(round(t, 4), round(v, 1)) for t, v in picks],
               layers=[(round(zt), round(zb), round(v)) for zt, zb, v in layers], true_layers=[(int(i + 1) * 10, round(true_vz[i])) for i in iface] + [(700, round(true_vz[-1]))],
               avo=(None if crit is None else dict(v_below=round(crit[0]), rise=round(crit[1], 2))),
               mse=round(mse), mse_over_dataset_mean=round(mse / MEAN_MSE, 3), mse_of_flat_mean_model_over_dataset_mean=round(mse_mean_model / MEAN_MSE, 3))
    print(json.dumps(res, indent=1))

    fig, ax = plt.subplots(2, 3, figsize=(16, 9)); s = 2; g = gathers[s]; gain = (1 + np.arange(1000) * DT * 3)[:, None]  # deconvolved shot gain = (1 + np.arange(1000) * DT * 3)[:, None]
    ext = [0, NX * DX, 1.0, 0]
    ax[0, 0].imshow(g * gain, aspect='auto', cmap='gray', extent=ext, vmin=-np.abs(g * gain).max() * 0.1, vmax=np.abs(g * gain).max() * 0.1); ax[0, 0].set_title(f'shot {s} (gained)'); ax[0, 0].set_xlabel('x (m)'); ax[0, 0].set_ylabel('t (s)')
    p, coh = slope_field(g)
    ax[0, 1].imshow(np.where(coh > 0.5, p * 1e3, np.nan), aspect='auto', cmap='RdBu', extent=ext, vmin=-1, vmax=1); ax[0, 1].set_title('slope dt/dh (ms/m), coherent samples')
    ax[0, 2].imshow(coh, aspect='auto', cmap='magma', extent=ext, vmin=0, vmax=1); ax[0, 2].set_title('coherence')
    ax[1, 0].imshow(np.log1p(V / (V.max() + 1e-9) * 100), aspect='auto', cmap='viridis', extent=[v_bins[0], v_bins[-1], t0_bins[-1], 0]); ax[1, 0].plot(vrms_true, t0_true, 'w--', lw=1, label='true v_rms(t0)')
    ax[1, 0].plot([v for _, v in picks], [t for t, _ in picks], 'r+', ms=14, mew=2, label='picks'); ax[1, 0].set_title('vote image (t0, v_rms), log'); ax[1, 0].set_xlabel('v_rms (m/s)'); ax[1, 0].set_ylabel('t0 (s)'); ax[1, 0].legend(loc='lower right', fontsize=8)
    zc = (np.arange(NZ) + 0.5) * DX
    ax[1, 1].step(true_vz, zc, 'k', lw=2, label='true v(z)'); ax[1, 1].step(vz, zc, 'r', lw=2, label='recovered (slopes+Dix)'); ax[1, 1].axvline(model.mean(), color='gray', ls=':', label='flat mean')
    ax[1, 1].invert_yaxis(); ax[1, 1].set_xlabel('v (m/s)'); ax[1, 1].set_ylabel('z (m)'); ax[1, 1].legend(fontsize=8); ax[1, 1].set_title(f'MSE/dataset-mean = {mse / MEAN_MSE:.3f}  (v1={v1:.0f}, tau={tau * 1e3:.0f} ms)')
    ax[1, 2].plot(t0c, mass / mass.max(), 'b'); [ax[1, 2].axvline(t, color='r', ls='--') for t, _ in picks]; [ax[1, 2].axvline(t0_true[i], color='k', ls=':') for i in iface]
    ax[1, 2].set_title('ridge mass vs t0 (red picks, black true interfaces)'); ax[1, 2].set_xlabel('t0 (s)'); ax[1, 2].set_xlim(0, 0.8)
    plt.tight_layout(); plt.savefig(out, dpi=110); print('wrote', out)

if __name__ == '__main__':
    main()
