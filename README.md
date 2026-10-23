# Learned seismic velocity inversion with per-shot wave-equation images

Goal: estimate subsurface velocity from surface seismic recordings (the OpenFWI 2D benchmark: 70 x 70 models,
5 shots, 70 receivers) with a network that receives, besides the raw recordings, what the wave physics can already
see: for every shot, an image of the subsurface computed by back-propagating its recording through three trial
background velocities. Everything is general: no family-specific code, no flat-layer assumptions.

* **Method, in plain words:** [openfwi/METHOD.md](openfwi/METHOD.md)
* **Literature review and design rationale:** [reports/Physics informed inputs for learned FWI.md](reports/Physics%20informed%20inputs%20for%20learned%20FWI.md)
  (sources in [research_notes/](research_notes/))
* **How to run:** [openfwi/README.md](openfwi/README.md)

## Status

| part | state |
|---|---|
| Data: one official file pair per family, all ten 2D families (500 models each) | downloaded and checked on the development machine; `1_data.py --download` fetches them (about 7 GB) |
| Simulator: matches the official gathers | 4-7.5 % relative residual on held-out models of all ten families with one calibrated source wavelet |
| Processing: per-shot images at three backgrounds | implemented, about 1 s per model on an Apple M4 Pro GPU; checked visually ([figure](openfwi/results/figures/imaging_steps.png)) |
| Training: our network and the InversionNet baseline | implemented and smoke-tested end to end; **full training not run yet** (moving to a lab workstation) |
| Testing: per-family OpenFWI metrics, tables, figures | implemented and smoke-tested |
| Test-time physics check and refinement (described in METHOD.md) | **not implemented yet** |

There are no results for this architecture yet. Earlier experiments with a different, flat-layer version of the physics
inputs (in [archive/](archive/)) gave the motivation: on FlatVel-A, a network with physics inputs trained on 25
models beat the same network on raw recordings trained on 452 (MAE 0.096 vs 0.113); those inputs failed on folded,
faulted and textured families, which is why they were replaced by the wave-equation images used here.

## Architecture

```
5 shot gathers (5 x 1000 x 70) ──► InversionNet encoder ──────────────► 512-d code ──────────┐
                                                                                             ▼
per-shot images at 3 backgrounds ──► encoder shared by the 5 shots ──► mean and spread ──► U-Net at 70 x 70 ──► velocity
(5 shots x 3 backgrounds x 70 x 70)    (the 3 backgrounds as channels)  over the shots       (+ medium background velocity)
```

* **Images:** for each shot and each background (surface velocity + 0.6 / 1.0 / 1.4 x the mean depth trend), the
  reverse-time-migration image of what that background does not explain: direct wave muted, divided by the shot's
  illumination, Laplacian filtered. Where the background is right, the five shot images agree; where it is wrong they
  move apart, which tells the network about velocity at that pixel.
* **Fusion:** one encoder shared by all shots; the mean over shots and the standard deviation over shots
  (disagreement) feed a U-Net, which also receives the medium background velocity and, at its 9 x 9 bottleneck, the
  InversionNet code of the raw gathers.
* **Training:** all ten families pooled, models 0-451 of each file for training and 452-499 for testing; loss
  L1 + 0.5 (1 - multi-scale SSIM); left-right mirroring; the image branch switched off for 30 % of samples (modality
  dropout); AdamW, cosine schedule, weight averaging.
* **Baseline:** InversionNet (the OpenFWI reference network) trained on exactly the same data, steps and loss, plus its
  original L1-only recipe.
* **Ablations:** one background instead of three (`ours_k1`); images without the raw gathers (`ours_noraw`).

Points we would like validated: whether three backgrounds of the form "surface velocity + scaled mean trend" are a
sound way to expose velocity information; the fusion by mean and spread over shots; the comparison protocol
(single official file per family, fixed split, matched training budget).

## Repository layout

```
openfwi/      the current pipeline: 1_data.py, 2_process.py, 3_train.py, 4_test.py, physics/, tools/, METHOD.md
reports/      literature review and design report
research_notes/  sourced notes behind the report, with small measurement scripts
archive/      earlier experiments (flat-layer slope channels, Gaussian scenes, refraction fits), kept for the record
```

Large files (the OpenFWI data, processed images, trained networks, predictions) are not in the repository; the
scripts regenerate them.
