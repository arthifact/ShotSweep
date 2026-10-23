# ShotSweep: setup and small pilot

Run commands from the repository root. Method: [METHOD.md](METHOD.md).
Use Python 3.12 for the tested environment. The lockfile includes explicit CPU and CUDA 13.0 Torch variants.

## Environment and regression tests

On a Mac, `uv sync --frozen` installs the native Torch wheel (MPS if available).
For a CPU environment on Linux/Windows:

```sh
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python -m unittest discover -s tests -v
```

For an NVIDIA machine with a CUDA 13.0 compatible driver:

```sh
uv sync --frozen --extra cuda
uv run --frozen --extra cuda python -m unittest discover -s tests -v
```

Keep the same extra on subsequent `uv run` commands; alternatively use `.venv/bin/python` after syncing.
Do not manually install a different Torch build and then run automatic synchronization: it will restore the lock.
CUDA 12.4 wheels do not satisfy the current Torch requirement. If the target needs a different CUDA version,
change the project source and regenerate the lock deliberately, then run the tests on that environment.
[uv's PyTorch configuration guide](https://docs.astral.sh/uv/guides/integration/pytorch/) describes this setup.
An explicit `--device cuda` or `--device mps` fails when unavailable; the pilot should never silently switch to CPU.
On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python` below.

## Data and isolated pilot

`1_data.py` downloads/checks official file pairs. Checks include exact shape, all-sample finite values,
velocity bounds, nonempty recordings and file hashes. A failed check preserves the existing manifest.
The default manifest uses train indices **0–399**, validation **400–451**, test **452–499** per family.
This is a within-file pilot protocol, not the official full-data benchmark.

```sh
.venv/bin/python -I openfwi/1_data.py --download --families flatvel-a
.venv/bin/python -I openfwi/tools/prepare_pilot.py \
  --source-root openfwi --families flatvel-a \
  --output openfwi/pilots/flatvel-a --n-train 8 --n-val 2 --n-test 2
```

The output must be a new directory. The preparation tool preserves source split membership and records original
sample indices. It copies only the requested samples, leaving the source data and caches untouched. Increase
families/sample counts after the first run succeeds. Downloading one full official gather file is about 700 MB.
To rehearse without downloads, replace `--source-root ... --families ...` with `--synthetic`; these generated
layered models test software only and must never be reported as OpenFWI accuracy.

## Process and check the target device

The commands below explicitly use CPU. On the target GPU machine replace `cpu` with `cuda` (or `mps` on a Mac).

```sh
.venv/bin/python -I openfwi/2_process.py \
  --data-root openfwi/pilots/flatvel-a --device cpu --batch 1
.venv/bin/python -I openfwi/tools/check_pilot.py \
  --data-root openfwi/pilots/flatvel-a --device cpu --model ours --batch 2
```

Processing fits the source wavelet and mean depth trend using training labels only. Surface velocity is fitted
at short offsets, with fit diagnostics and rejection of inconsistent arrivals. A rejected example stops with its
family/index; inspect it rather than bypassing the check. The confidence score is heuristic. A very shallow
refractor or laterally varying surface can still defeat this estimator; broader-family validation remains necessary.

The adjoint includes padded-velocity and source-amplitude derivatives. Wavefield checkpointing bounds memory;
`--max-memory-gb` limits estimated tensor storage, with runtime/allocator overhead additional. Start at batch 1
and increase only after measuring the target GPU. Physics/data/split fingerprints invalidate old preprocessing
caches. Interrupted writes never replace a complete cache; existing unversioned inputs must be regenerated.

The device check performs a forward pass, backward pass and optimizer update on a training batch, verifies finite
values/output dimensions and disjoint validation membership, and prints the selected device and Torch runtime.
It does not establish model quality.

## Train, interrupt and resume

```sh
.venv/bin/python -I openfwi/3_train.py --model ours \
  --data-root openfwi/pilots/flatvel-a --device cpu \
  --steps 40 --batch 2 --eval-batch 2 --ema-decay 0.9 \
  --val-every 10 --save-every 10 --experiment pilot40
```

For a direct comparison, repeat with `--model inversionnet` and `--model ours_raw`, retaining the same recipe,
experiment, split and seed. `ours_raw` keeps the same decoder/parameter count but zeroes all images and background
inputs internally. Other ablations are `ours_k1` and `ours_noraw`. Add additional seeds after the first checks pass.
Forty steps and EMA decay 0.9 are a software pilot recipe, not the full experiment's default 8,000 steps/0.999 EMA.
No naive left/right mirroring is used because the middle source is not reflection symmetric.

Periodic recovery snapshots live in `work/checkpoints/`. Completed inference models live in `work/models/`.
Both are written atomically. Recovery includes live/EMA weights, optimizer, scheduler, RNG, data fingerprints
and planned step horizon. Use `--stop-after 10` to rehearse interruption without changing that horizon:

```sh
.venv/bin/python -I openfwi/3_train.py \
  --data-root openfwi/pilots/flatvel-a --device cpu \
  --resume openfwi/pilots/flatvel-a/work/checkpoints/ours_l1ssim_s0_pilot40.pt
```

Resume restores the recipe; incompatible overrides/data/cache changes are rejected. Validation and final inference
use EMA BatchNorm statistics recalculated from training inputs with modality dropout off. Never use test samples
for calibration, normalization, loss selection or stopping decisions. No automatic best-test checkpoint exists.

## Evaluate the pilot on validation

```sh
.venv/bin/python -I openfwi/4_test.py \
  --data-root openfwi/pilots/flatvel-a --device cpu --batch 2 \
  --experiment pilot40 --split val
```

Scores, tables, predictions and example figures are under `results/val/`. After the experimental recipe is fixed,
`--split test` writes separate `results/test/` outputs. Evaluation accepts only completed, compatible runs with
unique model/loss/seed identities. It rejects mixed budgets/populations and does not average smoke runs into final
results. Legacy checkpoints lack required provenance and need retraining after these corrections.

MAE/RMSE use velocities normalized to [-1,1] (multiply by 1500 for m/s). SSIM follows official OpenFWI:
[0,1] inputs, 11×11 Gaussian window and padded convolutions. Training's `l1ssim` adds the mean of SSIM at three
resolutions; this arithmetic average is not conventional product MS-SSIM.

## Gate before a larger run

Require passing tests, valid preprocessing/quality diagnostics, a successful device check, finite training and
validation histories, and a successful interruption/resume rehearsal. Inspect predictions and try to overfit a tiny
training subset. Then test harder geology and independent files before claiming robustness or data efficiency.
Test-time physics refinement remains unimplemented. No software test establishes that the three backgrounds bracket
the true geology, or that shot agreement uniquely determines velocity.
