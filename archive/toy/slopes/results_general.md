# General version: structure-from-waves on a FlatFault-A model (2026-10-22)

`general.py`. Scene = reflector elements + a velocity field; camera = traveltime tables on that field; decoder = generic
kinematic migration of the oriented points; refinement = multi-view consistency of the migrated point clouds across
shots. Gathers are simulated here (official FlatFault-A gathers are not on disk): 4th-order FD on a 5 m grid, 0.5 ms,
15 Hz Ricker, free surface, sources and receivers at 10 m, five shots at the OpenFWI positions. Model: FlatFault-A
train model 0 (2253 m/s over 3252 m/s, interface at 390 m on the left, up-thrown to 310 m across a fault on the right).

## What works

**The generic decoder.** With the surface velocity v1 from the direct wave and a homogeneous field at that velocity,
every oriented point (shot, receiver, time, slope) maps to the subsurface point where T_s + T_r = t and the
receiver-side horizontal slowness S_r = p, with the normal from grad(T_s + T_r). The migrated points from all five
shots fall on the true interface at 385 m across the whole model and show the up-thrown block at 300 to 320 m on the
right, i.e. the fault, with no flat-layer assumption anywhere (figure `general_flatfault-a_0_homogeneous_search.png`,
bottom-middle panel, and the x0.85 / x1.15 figures). The batched eikonal solver gives all 75 tables in 0.7 s.

**The consistency objective.** The normalized kernel overlap between the per-shot migrated point clouds
(`overlap_consistency`) has its minimum at the true velocity: sweeping a homogeneous velocity over 0.8 to 1.2 of the
truth gives 1 - overlap = 0.59, 0.53, 0.46, **0.45**, 0.49, 0.58, 0.77. It is smooth and invariant to depth
compression. This is bundle adjustment's photometric consistency, written on point clouds.

## What did not work, and why

- **Flat-layer initialisation on a faulted model.** The vote image splits into several velocities at one time, Dix
  explodes. Fixed by merging same-time picks, but a faulted model still gives a poor layered start (MSE/mean 0.78).
- **Joint Gauss-Newton on element positions + slowness, times + slopes.** Diverges: the classic depth-velocity
  instability. Ray-traced sensitivities were verified exact against finite differences, so the instability is in the
  formulation, not the derivatives.
- **Slowness-only tomography with elements fixed at cluster centroids.** Stuck by construction: with elements placed
  under the wrong velocity, the true velocity makes the residual worse (29 ms -> 105 ms). Any alternation of this kind
  fails.
- **Depth-spread consistency.** A 400 m/s layer compresses every point into one band and wins. Not scale-invariant.
- **Time-residual consistency at cluster centroids.** Flat in velocity (277 vs 276 for a 15 % wrong start vs truth),
  dominated by how many points cluster at all.
- **Simulation artefacts that masqueraded as physics:** a missing centre term in the x stencil (blow-up), numerical
  dispersion at 10 m cells (ringing that migrates as ghost layers 75 m apart), and the top absorbing pad next to the
  10 m deep line (offset-dependent wavelet tail). Free surface + 5 m grid fixed all three.

## Refinement result

Nelder-Mead on the overlap objective, velocity in 6 depth bands (v(z), one lateral band), weak prior toward the
start, from two deliberately wrong homogeneous starts (0.85 and 1.15 of the direct-wave velocity, i.e. 1915 and 2591
m/s); about 200 evaluations, 5 minutes each.

| depth band (m) | true | start 0.85 -> recovered | start 1.15 -> recovered |
|---|---|---|---|
| 0-117 | 2253 | 1915 -> 2218 | 2591 -> 2233 |
| 117-233 | 2253 | 1915 -> 2385 | 2591 -> 2103 |
| 233-350 | 2262 | 1915 -> 1993 | 2591 -> 2363 |
| 350-467 (reflector at 385) | 2935 | 1915 -> 1721 | 2591 -> 1773 |
| 467-583 | 3251 | 1915 -> 1607 | 2591 -> 3350 |
| 583-700 | 3252 | 1915 -> 1873 | 2591 -> 3712 |

The overburden the data illuminate (the top three bands) is recovered from both sides, to 1-2 % at the surface and
4-12 % in the band just above the reflector. The three bands below the deepest reflector are a null space of the
kinematics, exactly as in the flat case, and wander; the prior was too weak to hold them. The MSE/mean of the smooth
field therefore does not improve (1.78 -> 2.30 and 0.54 -> 0.68), because that number is dominated by the null-space
bands and by the staircase the 6-band field cannot represent (floor 0.07 for the banded truth). The overlap reached
0.44-0.45 from both starts, the same value the true homogeneous overburden scores, so the search did converge on the
objective; the objective is simply blind below the reflector.

A 12-parameter run (6 x 2 bands, no prior) from the slow start gave the same picture: top two bands 2277-2291 m/s
versus 2253, drift below.

## Gradient version (`--mode gradient`)

Finite-difference gradient (central, 3 % in log v) of the overlap objective over a six-knot log-v(z) spline (knots every
140 m), steepest descent with backtracking, two priors: an illumination-weighted pull toward the start (weight
0.2 / (fraction of migrated points at or below the knot + 0.05), so the bottom knot, which nothing illuminates, is held
20x harder than the top ones) and a smoothness term 1.0 x sum (log v_k+1 - log v_k)^2. 7 iterations, 98 evaluations,
150 s per run.

| knot depth (m) | true | from 0.85 (1939) | from 1.15 (2624) |
|---|---|---|---|
| 0 | 2253 | 2240 | 2249 |
| 140 | 2253 | 2269 | 2243 |
| 280 | 2253 | 2165 | 2272 |
| 420 (reflector at 385) | 3250 | 1987 | 2365 |
| 560 | 3250 | 1969 | 2467 |
| 700 | 3252 | 1940 | 2602 |
| RMS overburden velocity to 385 m | 2269 | 2195 | 2267 |

The illuminated overburden is recovered to 1-4 % per knot from both sides and its RMS velocity to 0.1-3 %. Without
the smoothness prior the same runs converge to the same objective value with the knots trading slow against fast
(2037 / 2364 / 2217 from the slow start): one reflector only fixes the RMS velocity of its overburden, and smoothness
picks the flattest profile, which is the truth here. Below the reflector the knots stay near the start, as designed:
that is the null space, and the headline MSE/mean (1.78 -> 1.59 and 0.54 -> 0.60) is dominated by it and by the
staircase a spline cannot represent. Nelder-Mead needed 200 to 400 evaluations for a worse answer.

## 2D spline on a model with strong lateral change (model 475)

`NKX=3` gives a 6 x 3 bilinear log-velocity spline with the same illumination-weighted and smoothness priors, both
directions. FlatFault-A model 475 has the strongest lateral change of the family in the top 400 m: a 1625 m/s top layer
over 3441 / 4307 m/s, the fast interface at 360 m on the left and 270 m on the right. A second case imposes a +-10 %
lateral gradient on the top layer (1463 to 1787 m/s) to test lateral velocity recovery.

Two things were found, one fixed and one not.

- **First-break calibration** (`direct_wave_fit`) picked the post-critical reflection instead of the direct wave beyond
  170 m, because the reflection off the 4307 m/s layer is four times stronger than the direct wave and the threshold was
  30 % of the trace maximum. The threshold is now taken inside the first-arrival window (t < h/1300 + 0.12 s). Surface
  velocity on this model: 4666 -> 1635 (true 1625). The five FlatVel-A models are unchanged.
- **The overlap objective is biased on this model.** From the correct start, gradient descent moves every knot slower
  (top row 1443-1458 versus 1625) and the lateral gradient is not recovered (1545-1567 versus 1463 / 1623 / 1787).
  A sweep of a homogeneous velocity shows why: the objective decreases monotonically toward slower velocities,
  0.66 at 0.85x, 0.84 at 1.0x, 0.93 at 1.15x, with or without stricter migration-cost or offset filters. On model 0 the
  same sweep had its minimum at 1.0x. The difference is the contrast: with 1625 over 4307 m/s the critical angle is 22
  degrees and the critical offset about 220 m, so most of the oriented points are post-critical reflections (phase
  rotated, mislocated by the migration) and head waves (a linear event whose points migrate to a smeared, velocity-
  dependent locus). Those points dominate the clouds and the overlap no longer measures the velocity.

So the general version is validated for moderate contrasts (model 0: 2253 over 3252, critical angle 44 degrees) and
fails for strong contrasts at this offset range. The fix is not in the optimizer: either the head waves get their own
curve family (linear events, which also carry the refractor velocity) and post-critical points are down-weighted by
their phase, or the objective is restricted to pre-critical offsets predicted from the current model. Both are
modelling work, not tuning.

## 2D spline on a moderate-contrast model with an imposed lateral gradient (model 351)

Model selection needs the *effective* contrast: cell-to-cell jumps miss one-cell-thick steps (model 21 looked like
2522 over 3525 but is 2522 over 4171, and failed exactly like 475). Measuring the velocity 40 m below the first
interface, model 351 has a 50 degree critical angle (2748 over 3600, interface at 280 m left, 390 m right), so the
whole offset range is pre-critical. The lateral variant imposes +-10 % across the top layer: 2473 to 3023 m/s.

The homogeneous sweep now bottoms at the true velocity (1 - overlap 0.60, 0.54, **0.53**, 0.61, 0.71 for 0.85 to
1.15 x v1), as on model 0. The first 2D run with lateral smoothness 1.0 recovered the sign of the gradient but only
40 % of its size: a 20 % lateral jump costs 0.04 under that prior, more than the objective's whole improvement.
With lateral smoothness 0.1 (vertical still 1.0), 6 x 3 knots, homogeneous start at v1 = 2790, 10 iterations, 409 s:

| row (depth m) | true left / middle / right | recovered |
|---|---|---|
| 0 | 2473 / 2744 / 3023 | 2447 / 2898 / 2946 |
| 138 | 2473 / 2744 / 3023 | 2459 / 2891 / 2907 |
| 276 (interface at 280 on the left) | 2913 / 2913 / 3023 | 2579 / 2771 / 2845 |
| 414 and below | 3600 | 2631 to 2791, held near the start |

The imposed lateral gradient is recovered in the illuminated top layer: 2447 to 2946 against 2473 to 3023, the ends
within 1 to 2.5 %, the middle knot 6 % high. RMS velocity to 385 m: 2606 / 2838 / 2871 against 2799 / 2953 / 2951,
low on the left where the fast layer starts at 280 m and the spline cannot follow the staircase. Below the deepest
reflector: null space, held by the prior, as before.

Two engineering notes from this step: running two of these in parallel built two 3846 x 3846 kernel matrices with
temporaries and pushed the machine into swapping (40 s per evaluation instead of 1 s); the overlap is now computed
per shot pair. And the run prints the homogeneous sweep every time, which is the cheapest check that the objective's
minimum is where it should be before any descent.

## Where this leaves the general version

The decoder, the objective and the optimizer are generic and validated on two faulted models of moderate contrast
with simulated gathers, including lateral velocity recovery from a homogeneous start. The method fails on strong
contrasts (critical angle below about 40 degrees at this offset range) because post-critical reflections and head
waves dominate the oriented points; that needs head-wave modelling or a pre-critical restriction, not tuning. Still
missing: that, the official FlatFault-A gathers (700 MB download), a batch over many models, and a way to turn the
migrated elements plus the smooth velocity into a blocky model for the benchmark metric. The layer below the
deepest reflector remains a null space here as in the flat case; only a prior or amplitudes can fill it.
