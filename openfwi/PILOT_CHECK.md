# Pilot verification — 2026-10-23

The corrected pipeline passed a CPU software pilot on a small official OpenFWI subset.
The target machine's GPU, driver, memory and throughput have not been verified. Run the explicit-device check
in [README.md](README.md) there before starting its pilot. This is not a benchmark or evidence of generalization.

## Checks completed

- Regression suite: 57 tests covering SSIM/reference parity and gradient direction, decoder behavior, geometry,
  exact adjoint finite differences, refraction handling, split separation, cache/data integrity, evaluation grouping,
  atomic writes, EMA calibration, and deterministic interruption/resume.
- All five model variants passed a real-data forward/backward/optimizer-step check with finite values.
- Corrected preprocessing completed for all eight selected FlatVel-A examples. The short-offset surface estimates
  differed from known surface velocities by approximately −2.0% to +2.25% in these examples.
- ShotSweep, reference InversionNet and the same-decoder raw-only control each completed 40 training steps.
  ShotSweep stopped at step 10 and resumed through step 40 from its recovery checkpoint.
- Validation evaluation loaded all three completed models, checked their provenance, and wrote separate scores,
  tables, predictions and a visually inspected example figure.
- Python syntax, whitespace and the dependency lock were checked. CPU synchronization used the frozen lock.

## Rehearsal data and recipe

The public FlatVel-A `model1.npy` and `data1.npy` files were partially downloaded from their official OpenFWI Drive
folder. Source indices 0–3 were training, 400–401 validation, and 452–453 reserved for testing. The compact manifest
recorded original indices and whole-file SHA-256 hashes. Calibration used training indices 0–1; the depth trend and
normalization used the four training examples. Test predictions/scores were not produced.

Environment: Python 3.12.13, Torch 2.14.1, NumPy 2.5.4, CPU, two threads. Recipe: seed 0, batch 2, 40 steps,
learning rate 0.0005, L1 plus mean multiresolution SSIM, EMA decay 0.9, evaluation batch 2, validation/save every
10 steps. The full experiment's defaults remain 8,000 steps and EMA decay 0.999.

| Model | Validation MAE | Validation RMSE | Validation SSIM |
|---|---:|---:|---:|
| InversionNet | 0.343451 | 0.426943 | 0.681662 |
| ShotSweep | 0.284870 | 0.362515 | 0.629813 |
| Same-decoder raw-only control | 0.292893 | 0.351915 | 0.644959 |

MAE/RMSE are normalized to [-1,1]; SSIM uses the official [0,1] convention. With only two validation examples
and 40 optimization steps, these numbers confirm that scoring works; they do not establish an accuracy ranking.

## Remaining pilot gate

Obtain the target machine's OS, GPU model/VRAM and driver version, select the matching locked environment, and run
the device check at batch 2. Inspect preprocessing diagnostics on the intended pilot families. Very shallow
refractors/lateral variation can cause explicit rejection; the confidence heuristic is not calibrated uncertainty.
Use independent files and multiple seeds before larger scientific claims. Test-time physics refinement is not
implemented. Old preprocessing caches and pre-fix checkpoints must be regenerated/retrained.
