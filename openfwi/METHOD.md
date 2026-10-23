# ShotSweep: inputs, model and validation

ShotSweep estimates a 70×70 velocity model from five surface seismic recordings. It combines raw gathers with
per-shot wave-equation images computed under three trial backgrounds. This is a research hypothesis: the physics
images may improve data efficiency, but their agreement does not uniquely determine local velocity.

## Physics-derived inputs

The mean depth trend is fitted from training velocity labels only. A source wavelet is calibrated using two training
models per family. Each example gets a surface-velocity anchor fitted to thresholded arrivals at offsets 10–60 m.
Fit curvature, residual and shot disagreement are checked; inconsistent picks are rejected. Longer offsets diagnose
possible refraction. These diagnostics are heuristic, and very shallow refractors/lateral variation remain limitations.

Three laterally constant backgrounds are the surface anchor plus 0.6, 1.0 and 1.4 times the training depth trend,
clipped to 1500–4500 m/s. They need not bracket the true model, particularly for velocity inversions or complex geology.
For every shot/background, the discrete adjoint produces a negative data-misfit gradient, including replicated
boundary-velocity and velocity-dependent source-injection derivatives. A direct-wave mute is held fixed during
differentiation. Illumination normalization and a Laplacian filter emphasize spatial structure. The first two image
rows are zeroed. Each example's 15 images share a 99th-percentile scale, preserving relative amplitudes.

Wavefield checkpointing recomputes short time segments during the adjoint to bound memory. Regression tests compare
derivatives with finite differences in the interior, at boundaries and at sources. Data, code, calibration and split
fingerprints prevent silently reusing old processing outputs.

## Architecture

An InversionNet encoder turns the raw gathers into a 512-dimensional code. A second encoder, shared by all five
shots, reads the three background images as channels. The mean and standard deviation of its features across shots,
plus the medium background velocity, feed a U-Net. The raw code joins at the 9×9 bottleneck, and the final output
uses tanh to stay within the velocity range.

Mean/spread fusion is a simple permutation-invariant summary. It discards explicit source identity and may discard
useful directional information; its adequacy requires experiments. The raw branch retains information lost by
migration, scaling or shot pooling. Image disagreement can reflect aperture, illumination, background errors or
geology, so it is not a calibrated confidence measure.

Controls are the upstream InversionNet decoder, the same U-Net using raw gathers only (`ours_raw`), one background
(`ours_k1`), and no raw branch (`ours_noraw`). The raw-only control zeros both images and background internally.

## Training and scoring

The default within-file split is 400 training, 52 validation and 48 test models per family. Source/trend calibration,
normalization and EMA BatchNorm recalibration use training data only. The objective is L1 on [-1,1] velocity plus
0.5 times one minus mean SSIM over three resolutions on [0,1] inputs. This is not conventional product MS-SSIM.
AdamW, cosine decay, EMA and image-modality dropout are used. Naive mirror augmentation is disabled because the
central source at column 34 reflects to column 35 in the 70-column grid.

Atomic checkpoints preserve optimizer, scheduler, RNG and both live/EMA states for recovery. Before scoring/export,
EMA normalization statistics are recalculated from training examples. Validation guides pilot decisions; test results
are kept separate. Scores use official OpenFWI SSIM and unrounded errors; mixed experiments and duplicate seeds
cannot silently enter the same average.

The pilot establishes software behavior and permits preliminary within-file comparison. Independent files, multiple
seeds, harder families and matched learning curves are needed to establish generalization or data efficiency.
Test-time forward-physics checking and refinement are future work; the current pipeline does not perform them.
