# Critical read of the "unrolled painter on a Gaussian scene" proposal

Short verdict: the loop will work, the state is the gamble, and the proposal is missing
its most important controls. Details below; the toy in `toy/` tests the parts that can be tested on a laptop.

## What is solid

- Unrolled, adjoint-conditioned updates are established (learned primal-dual, RIMs, and several
  "learned FWI" / "unrolled FWI" papers in geophysics already use exactly this loop). Expect it to beat
  both plain gradient descent at equal adjoint count and a one-shot network at equal data. That part
  is low risk and also not novel; cite it.
- Treating the adjoint as a non-differentiated input is the standard trick and keeps training cheap.
- Cost estimate (adjoints dominate, ~1 day for 10k models) is realistic.

## Where the proposal is wrong or hand-waving

1. **"Sample the gradient image at each slot" throws away the information.** The right per-slot input is
   the exact chain-rule gradient dJ/d(centre, width, amplitude), which is one autograd call through the
   renderer. That makes the painter a learned preconditioner on exact parameter-space gradients, which is
   much more "anchored to the signal" than sampling a pixel under the centre. The toy feeds both.
2. **Dormant slots cannot be woken by their own gradient.** With amplitude zero, the gradient w.r.t. a
   slot's centre and width is exactly zero (it is proportional to the amplitude). So "a split is a dormant
   slot activating next to a parent" has no physics signal driving *where* it activates. The painter needs
   a global view of the gradient image (patch tokens / cross-attention) to place new slots. The proposal
   says "let the painter do it"; this is the piece most likely not to work as sketched.
3. **Missing control: is the physics doing the work?** The data-efficiency claim ("thousands, not hundreds
   of thousands") needs two comparisons at *equal training-set size*: (a) a one-shot network from raw
   gathers, (b) the dataset-mean model (what a painter that ignores the adjoint would converge to on a
   narrow dataset). Without (b) a good number on FlatVel-A proves nothing; those models are nearly 1D.
4. **Gaussians vs pixels will most likely lose on accuracy.** FlatFault has sharp offsets; a smooth
   background plus a dozen blobs smears them. The honest framing is "compact and interpretable at a
   small accuracy cost", and the pixel-state painter is the baseline that decides it.
5. **Cycle skipping is the real risk and the toy cannot settle it.** Frequency continuation across steps
   is the right idea; whether a painter learns to distrust a cycle-skipped first adjoint depends on the
   B families being in the training set. Build that evaluation in from day one (split by family, report
   the hard families separately).
6. **Evaluation should include steps beyond the training unroll** (T+2, T+4). A painter that only works at
   exactly T steps has learned a fixed trajectory, not an update rule.

## What the toy tests

Self-contained PyTorch: 32x40 acoustic grid, 3 shots, exact adjoint by autograd (checked against finite
differences to 7 digits), layered models with sharp ellipses or faults (deliberately not Gaussian).
Runs: Gaussian painter, pixel painter, each at 1000 and 100 training models; one-shot gather network;
plain Adam on pixels and on the Gaussian scene with the same adjoint (4 and 30 steps); dataset mean.
Metric: fraction of the initial velocity error remaining on 96 held-out models (1.0 = no better than the
1D start). Results in `toy/results.md`, examples in `toy/results.png`.

## Results (2026-10-22, 96 held-out models, err = fraction of initial velocity error left)

| method | err | train time |
|---|---|---|
| fast pixel U-Net, gathers + adjoint, 2 stages, 3000 steps | 0.318 | 5 min + 5 min |
| fast pixel U-Net, gathers only, 3000 steps | 0.341 | 6.5 min |
| one-shot CNN on gathers (4000 steps) | 0.420 | 46 s |
| fast Gaussian hypernet head, gathers + adjoint, 2 stages | 0.502 | 8 min |
| fast pixel U-Net, adjoint only, 2 stages, 3000 steps | 0.537 | 9 min |
| unrolled Gaussian painter, 1000 models, T=4 (end-to-end) | 0.672 | 19 min |
| unrolled pixel painter, 1000 models, T=4 (end-to-end) | 0.808 | 20 min |
| plain Adam on Gaussian scene, 30 adjoints, no learning | 0.809 | 0 |
| dataset mean | 0.918 | 0 |
| plain Adam on pixels, 4 or 30 adjoints, no learning | 1.00 | 0 |

Full table in `toy/results.md`, examples in `toy/results.png`, live page `toy/live/index.html`.

### What the toy says
1. **The raw adjoint image is a worse network input than the raw gathers** at 1000 models. Even after a source
   mute and per-depth normalisation (without which the painters were blind: the source row is 1e3-1e4x the
   interior), adjoint-only reaches 0.54 vs 0.34 for gathers. Adding the adjoint to the gathers helps a little
   (0.32). The adjoint is a lossy, badly conditioned projection of the data; the proposal's premise that it
   makes the learning problem "easy" did not hold here.
2. **End-to-end unrolled training was both slow and bad** (20 min, 0.67-0.81) and got worse when run for extra
   steps, i.e. it learned a trajectory, not an update rule. Stage-wise training on cached adjoints reached the
   same quality in 1 minute and better quality in 5.
3. **Gaussians lose to pixels on accuracy** (0.50 vs 0.32) and the second stage made their data misfit worse:
   twelve blobs plus a coarse background cannot fit faults. Their case is compactness/interpretability only.
4. **Second stages improve velocity error but worsen data misfit** (misfit ratio 0.03 -> 0.63 for pixels). The
   refinement is leaning on the prior rather than the physics. Treat with suspicion.
5. Caveats: 32x40 grid, 3 shots, single band (no frequency continuation), fixed 1D start, inverse crime
   (same solver for data and inversion), small nets, 1000 models. Cycle skipping was not tested.

The design for a method that could reach strong results in minutes is in `DESIGN.md`.
