# Factored physics: a design for learning wave inversion in minutes

Principle: never run a wave simulation inside the learning loop. Everything the learner touches is a
precomputed linear operator, a back-substitution, or a cached image. The physics manufactures its own
training pairs; the model zoo is only needed for the prior.

Motivation from the 2026-10-22 toy: a solver in the loop costs ~1 s/step; raw adjoint images hide the signal
under the source row; a one-shot net on raw gathers beat the adjoint-input net at equal data because the
adjoint is a lossy, badly conditioned projection.

## Pieces

1. **Factor the physics.** Frequency-domain Helmholtz at ~6 frequencies (3-15 Hz). Sparse LU per model is
   milliseconds in 2D; every shot, adjoint and Born perturbation is then a ~1 ms back-substitution. One
   factorization for the shared start model serves the whole dataset. Frequency continuation is free.
2. **Explicit Born operator at the start.** With 6 freq x 5 shots x 70 receivers a datum is ~2k complex
   numbers; the Born matrix (model -> datum) is ~10M entries. Compute once (point perturbation per pixel).
   Gauss-Newton Hessian (4900x4900) is explicit; the exact regularised GN step is a ms linear solve; the
   migration image is one matvec. Illumination/source-row problems vanish by construction.
3. **Self-supervised inverse Hessian.** Sample random perturbations from a cheap structural prior (layers,
   blobs, faults), apply the explicit Hessian (instant), train a small net to invert. Millions of pairs in
   minutes, no velocity dataset. This net is the painter.
4. **Stage-wise nonlinear refinement.** Refactorise Helmholtz at each predicted model (ms/model), recompute
   migration image + GN step, train the next stage on cached inputs; raise top frequency each stage.
   Deep unfolding with no solver in the loop.
5. **Gaussian primitives as a Born dictionary.** Lattice of Gaussian atoms (z, x, sigma) pushed through the
   Born operator once -> dictionary of data signatures. Explain the residual by sparse coding on it with
   LISTA (unrolled ISTA, learned thresholds; trains in seconds). Active coefficients = slots; densification =
   coefficient turning on; merge = overlapping atoms collapsing under the L1 penalty. A tiny hypernet maps
   coefficients to continuous slot parameters. Dormant slots are not a problem: an inactive atom's gradient
   is its correlation with the residual, never zero.

## Cost at OpenFWI scale (24k models, 70x70, 5 shots)
One factorisation + explicit Born matrix (seconds); self-supervised Hessian net (~5 min); per stage a few
minutes of back-substitutions across the dataset + ~1 min network training. Three stages in ~20 min on an
M4 Pro. Compare at equal wall time against the one-shot OpenFWI baselines.

## Risks
- Six frequencies lose resolution: verify; always evaluate with the time-domain solver on full-band data.
- Born step invalid far from the model: start at 3 Hz; add a stage 0 eikonal (fast-marching) tomography net
  for a cycle-skip-free background if needed.
- The self-supervised Hessian net is only as good as the perturbation prior: mix in a small model zoo.
- 2D design. In 3D the explicit matrix is replaced by the factorisation applied on the fly.

Novel parts: physics-manufactured training pairs for the inverse Hessian; Gaussian primitives as a Born
dictionary with unrolled sparse coding. Factorisation and stage-wise unrolling are known tools; cite them.
