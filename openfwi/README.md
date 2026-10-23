# ShotSweep pipeline: learned velocity inversion with per-shot wave-equation images

Method: see [METHOD.md](METHOD.md). Four steps, each one script, run from the repository root.

```
data/        gathers and velocity models, one official file pair per OpenFWI family (500 models each)
physics/     library: constants, GPU simulator and imaging (gpu.py), background velocities, OpenFWI metrics
tools/       show_images.py: draw the network inputs for one model
work/        intermediate files: calibrated source wavelet, depth trend, per-model images, trained networks
results/     scores, tables, predictions, figures
logs/        run logs
```

| step | script | what it does | time on an M4 Pro |
|---|---|---|---|
| 1 data | `uv run python -I openfwi/1_data.py` | checks the ten file pairs (`--download` fetches missing model and gather files from the OpenFWI Drive), writes `data/manifest.json` | seconds |
| 2 processing | `uv run python -I openfwi/2_process.py` | calibrates one source wavelet, the mean depth trend, then for every model three background velocities and the 15 per-shot reverse-time-migration images (`work/inputs/<family>.npz`) | about 1 s per model, 90 min for 5,000 (resumable per family) |
| 3 training | `uv run python -I openfwi/3_train.py --model ours` | trains one network on all ten families pooled (models 0-451 of each), 8,000 steps of 64; `--model inversionnet` is the OpenFWI baseline on the same data and recipe; `--model ours_k1` / `ours_noraw` are the ablations; `--loss l1` gives InversionNet's original loss | about 0.4-1 s per step on an M4 Pro GPU |
| 4 testing | `uv run python -I openfwi/4_test.py` | scores every trained network on the held-out models 452-499 of each family (OpenFWI MAE / RMSE / SSIM), writes `results/table.md` and `results/figures/examples.png` | minutes |

Runs on Apple GPUs (MPS), CUDA GPUs or CPU, chosen automatically. On a CUDA machine install a CUDA build of PyTorch
(for example `uv pip install torch --index-url https://download.pytorch.org/whl/cu124`) before step 2.

Suggested full run: `1_data.py --download`, `2_process.py`, then `3_train.py` for `inversionnet` and `ours` with
`--seed 0` and `--seed 1`, `ours_k1`, `ours_noraw` and `inversionnet --loss l1`, then `4_test.py`.

Split: for every family, models 0-451 of the official file are the training pool and 452-499 the test set; the same
split for every network. Nothing in the code depends on the family: the same processing, network and recipe apply to
all ten.

Earlier experiments (flat-layer slope channels, Gaussian scenes, refraction fits) are in `../archive/`.
