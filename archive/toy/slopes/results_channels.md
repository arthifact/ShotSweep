# More physics per pixel: does it help a network learn FWI? (2026-10-23)

The idea, borrowed from learned green-screen keying: instead of handing a network raw wiggles and asking it to learn
wave physics, kinematics and geometry at once, compute what the physics already knows and give it to the network as
per-pixel channels aligned with the output. Everything below runs on the **official OpenFWI gathers** (FlatVel-A
`data1.npy`, FlatFault-A `seis2_1_0.npy`, 500 models each).

## The channels (`features.py`, training-free, 7 s per model)

| channel | what it is | green-screen analogue |
|---|---|---|
| v_geo | geometric start: local slopes, closed-form hyperbola votes, multiples rejected, Dix | alpha hint |
| reflect | every oriented point (shot, receiver, time, slope) migrated to the subsurface point where its time and slope match, on traveltime tables of v_geo | depth / edges |
| dip | lateral component of the migrated reflector normal | surface normals |
| agree | fraction of the 5 shots that put a reflector at this pixel | multi-frame consistency |
| illum | share of reflector weight at or below the pixel: 0 = null space | visibility |
| grad | FWI adjoint gradient at v_geo (exact discrete adjoint, source wavelet estimated) | where the data disagree with the start |

`feat_test/channels_452.png` shows them for one model of each family.

## The test (`train_channels.py`, `run_experiments.sh`, `analyze_training.py`)

Same small network for every input: a 6-layer dilated conv refiner (receptive field 41 cells). **raw**: gathers ->
PCA 128 -> MLP -> 70 x 70 map -> refiner. **phys**: the six channels -> refiner, predicting a correction to v_geo.
**both**: raw map + channels. L1 loss, 2000 steps of batch 32 on the Apple GPU (90 s per run), flip augmentation.
Training pool = models 0-451 of each file, test = models 452-499. Metrics: OpenFWI MAE / RMSE / SSIM on velocity
scaled to [-1, 1]. Two seeds per configuration.

## Results (MAE, lower is better; mean over 2 seeds)

| family | training models | raw | phys | both | geometric start, no training |
|---|---|---|---|---|---|
| FlatVel-A | 25 | 0.280 | **0.096** | 0.137 | 0.113 |
| FlatVel-A | 100 | 0.233 | **0.063** | 0.138 | |
| FlatVel-A | 452 | 0.113 | **0.051** | 0.066 | |
| FlatFault-A | 25 | 0.357 | **0.169** | 0.237 | 0.293 |
| FlatFault-A | 100 | 0.283 | **0.169** | 0.214 | |
| FlatFault-A | 452 | 0.115 | 0.154 | **0.111** | |

Cross-family, 452 training models (MAE):

| train -> test | raw | phys | both |
|---|---|---|---|
| FlatFault-A -> FlatVel-A | 0.224 | **0.151** | 0.171 |
| FlatVel-A -> FlatFault-A | 0.198 | 0.221 | **0.180** |

Figures: `data_efficiency.png` (MAE vs training-set size, now with the refraction variants), `predictions_n100.png` (truth, geometric start,
reflectivity channel, raw and physics predictions at 100 training models).

## What it says

- **Data efficiency.** On FlatVel-A, 25 models with physics channels (0.096) beat 452 models with raw gathers
  (0.113): an 18x reduction in training data for better accuracy. At 452 models physics is still 2.2x better.
- **Interfaces land in the right place.** The raw network smears and shifts interfaces; the physics network places
  them where the migrated reflectivity says, and on FlatFault-A 472 it recovers the fault step that raw misses.
- **Where physics stops, raw data still help.** On FlatFault-A the physics-only network plateaus near 0.15-0.17 and
  raw overtakes it at 452 models; raw + physics is best. The plateau is the failure map found earlier: below strong
  contrasts (critical angle under about 40 degrees) the oriented points are dominated by post-critical reflections and
  head waves, which the channels do not model, and the geometric start fails there (0.29). The raw gathers carry
  that information and a network can learn it given enough examples.
- **Transfer.** Trained on faults, tested on flat layers, physics transfers best (0.151 vs 0.224).
- **"both" at small data** is worse than physics alone: the raw branch overfits and adds noise. A dropout or a
  smaller raw branch would likely fix this; not tried.

## Refraction channels and a post-critical mask (FlatFault-A rerun)

**Why head waves could not be picked.** An eikonal solve on true FlatFault-A models shows the direct wave is the first
arrival at every receiver: with interfaces 150-400 m deep and a 700 m spread the crossover distance is beyond the line.
The head wave runs within one wavelet of the post-critical reflection over the whole spread
(`refr_test/headwaves.png`), so a first-break or linear-event picker sees the reflection. Amplitude-versus-slope and
phase-versus-slope critical-angle estimates were tried and were too noisy (`refr_test/phase_vs_p.png`).

**What was built instead** (`refraction.py`, `features_refr.py`, about 10 s per model on the Apple GPU): a two-layer
waveform fit at low frequency (v1 from the direct wave above, v_r below an interface with depth z0 and lateral tilt dz),
all candidates simulated in one batch, so head waves and post-critical reflections are modelled rather than picked.
On 8 test models the interface depth comes out within about 20-50 m, but the best v_r is biased low on strong
contrasts, because thin transition layers and the fault make two layers too crude. So the channels carry the fit's
likelihood, not a point estimate:

| channel | what it is |
|---|---|
| refr_like | per pixel, likelihood that the first strong refractor passes through it |
| refr_v | soft two-layer velocity: v1 above each candidate refractor, the velocity the data prefer below it |
| postcrit | fraction of the migrated reflector weight at the pixel carried by rays too flat to have passed below the refractor (abs p >= 1/v_r) |

Two calibration facts were found on the way and matter for anyone simulating OpenFWI: the source is injected with
weight v^2 at the source cell (best-fit scale / v_top^2 constant to 0.3 % over 8 models), and with that one fixed,
calibrated source the 10 m simulator reproduces every official gather to 4-6 %.

**Results, FlatFault-A** (MAE, mean over 2 seeds; physR / bothR = phys / both plus the three new channels):

| training models | raw | phys | both | physR | bothR |
|---|---|---|---|---|---|
| 25 | 0.357 | 0.169 | 0.237 | **0.151** | 0.205 |
| 100 | 0.283 | 0.169 | 0.214 | **0.137** | 0.156 |
| 452 | 0.115 | 0.154 | 0.111 | 0.115 | **0.078** |

At 452 models, raw gathers plus all nine physics channels reach MAE 0.078 (RMSE 0.129, SSIM 0.826): 30 % below the
best previous input (0.111) and 32 % below raw gathers alone (0.115). At 25-100 models the refraction channels improve
physics-only by 11-19 %, and physics-only with refraction stays the best input at small data. The no-training
two-layer channel by itself scores 0.293, the same as the geometric start: it is a hint the network reads, not an answer.

**Ablation** (one seed): the refraction channels carry the whole gain (refractor likelihood + two-layer velocity only:
0.137 at 100 models, 0.115 at 452), the post-critical mask adds nothing (0.170 and 0.153, the same as phys). The mask
marks where the migrated reflectivity is unreliable, but the network already sees that from the reflectivity itself.

**Transfer** (trained on FlatFault-A, tested on FlatVel-A, 452 models): physR 0.130, best of all inputs (phys 0.151,
raw 0.224). bothR transfers badly (0.250): the raw branch learns family-specific gather patterns.

## Honest scale

These are small networks (about 130k refiner weights) trained on at most 452 models from one file per family,
tested on 48 models of the same file. The OpenFWI paper's InversionNet, trained on 24k (FlatVel-A) or 48k
(FlatFault-A) models, reports MAE 0.011-0.013 and 0.017. The numbers above are a controlled comparison of inputs at
equal network and data, not a leaderboard entry.

## Supporting engineering from the same day

- Forward model calibrated to the official gathers: a 60-cell damped-wave absorbing layer reproduces the true-model
  data to 2 % relative residual (the first sponge left 73 % unexplained, which made the first FWI fit boundary
  reflections). Exact discrete adjoint, checked against finite differences to 0.3 %, 2 s per gradient (`fwi.py`).
- Layered FWI over interface depth knots and layer velocities (`fwi.py`, `fwi_layers`): on FlatFault-A 452 it took the
  geometric start from MAE 0.38 to 0.05 and recovered the fault as an interface dipping from 279 to 199 m (true 270 to
  220). Not tuned further.
