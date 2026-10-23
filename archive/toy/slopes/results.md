# Slope pipeline on official FlatVel-A gathers (2026-10-22)

A geometry-first, training-free inversion: the gather is treated as an oriented point set, each point decodes in
closed form to an interface time and a velocity, interfaces are peaks of the vote image, and Dix turns them into a
layered v(z). No simulation, no network. `slopes.py` runs one model and draws the figure, `batch.py` runs many.

## Pipeline

1. **Calibration from the direct wave.** Surface velocity v1 = 1 / median slope of the direct wave at offsets
   100-220 m; wavelet delay tau = median(peak time - h/v1), about 74 ms on this data.
2. **Orientation field.** Local slope p = dt/dh of every sample by cross-correlation of each trace with its two
   neighbours (Gaussian window 12 ms, lags up to 20 ms, parabolic sub-sample refinement), coherence = normalized
   correlation squared. A finite-difference structure tensor was 9 % biased here: one trace spacing shifts the
   wavelet by about 7 samples, so the spatial derivative is under-estimated.
3. **Oriented points.** Wavelet peaks (local maxima of |g| after trace-wise AGC), offset >= 60 m, coherence > 0.5,
   outside a 45 ms band around the direct wave, and with the direct wave's polarity. The polarity rule is a checked
   geological prior: the FlatVel-A train file has 2214 positive and 0 negative velocity jumps, so every reflection
   shares the direct wave's main-lobe sign and every Ricker side lobe has the opposite sign.
4. **Closed-form votes.** For a flat reflector, v^2 = h / (t p) and t0^2 = t^2 - h t p. Votes with a grazing
   apparent ray angle (p v > 0.8) are head waves or post-critical and are dropped. All five shots vote into one
   (t0, v_rms) image.
5. **Interfaces.** 2D local maxima of the smoothed vote image above 6 % of the maximum, refined to the local centroid,
   at least 50 ms or 300 m/s apart.
6. **Layers.** Dix from a 10 m datum (sources and receivers sit one cell below the surface; before the datum was
   added every pick looked 2*10/v1 early). Bottom layer: a one-parameter fit of the acoustic reflection coefficient
   to the deepest reflector's amplitude-versus-offset curve when that curve rises by at least 1.4x, else the last
   interval velocity x 1.13, the median jump across an interface in the dataset.

## Numbers (MSE / dataset-mean MSE, dataset-mean MSE = 264970 (m/s)^2)

| model | ratio | note |
|---|---|---|
| 0 | 0.026 | 4 of 5 interfaces, the 5 % contrast one missed |
| 1 | 0.013 | all interfaces within 10 m |
| 2 | 0.184 | thin slow first layer (90 m at 1555 m/s) hides under the direct wave and is missed |
| 3 | 0.013 | all interfaces within 10 m |
| 7 | 0.003 | the 2 % contrast interface is invisible and does not matter |

48-model validation split (models 452-499, the split the octree-Gaussian artifact used), `batch.json`:

| statistic | value |
|---|---|
| mean ratio | 0.40 |
| median ratio | 0.15 |
| models below 0.1 | 35 % |
| models above 0.3 | 35 % |
| mean over the 44 models with ratio < 1 | 0.21 |
| worst four | 4.7, 2.9, 1.3, 1.1 |

The worst four all have the same cause: the only visible reflector sits on a very large jump (for example 2206 to
3848 m/s) and nothing reflects below it, so the bottom layer's velocity is a null space of the kinematics. The head
wave carries it but is never a first arrival in this geometry and hides inside the direct-wave band. The amplitude
versus offset curve carries it too: a one-parameter fit of the acoustic reflection coefficient to that curve
(`avo_velocity`, used on 22 of 48 models) fixes two of the four and underestimates on the two shallow ones, where the
15 Hz wavelet smears the critical transition. That is an amplitude problem, not a geometry problem, and is left here.

For reference, head B of the octree-Gaussian artifact (free Gaussians predicted by a network from PCA-128 gathers,
trained on 452 models plus mirrors) reached 0.36 on the same 48 models. The octree representation floor is 0.11.

## What broke on the way

- Envelope-weighted votes from every sample merged reflections only 80 ms apart; voting from wavelet peaks fixed it.
- A global amplitude threshold dominated by the direct wave silenced deep reflections; the threshold is now local.
- Head-wave samples voted as hyperbolas and created ghost interfaces with high v_rms; the ray-angle cut removes them.
- A side-lobe rule (drop a peak with a stronger opposite-polarity peak within 45 ms) killed real reflections that sit
  under the direct wave's trailing lobe (model 7); the polarity prior replaced it.
- Wiener deconvolution of the measured wavelet removed the delay but weakened shallow reflections and wrapped the
  trace end; rejected.
- A refraction (head-wave) vote space never fired: with 700 m of offset the head wave is never a first arrival here.
- A critical-offset brightness detector fired late: at 15 Hz the amplitude rises smoothly, so it became the AVO fit.

## Known limits

- Flat layers only: the hyperbola formulas assume no dip. FlatFault-A needs the two-shot slope relation for dip and
  diffraction apexes for the fault.
- Interfaces with contrast under about 5 % do not form a peak.
- The layer below the deepest reflector is a null space of the data; the 1.13 prior is the only learned number.
- Dix amplifies v_rms errors in thin deep layers (model 2: 3381 vs 3619 m/s).
