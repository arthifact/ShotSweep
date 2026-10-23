# Physics-derived input representations for deep-learning velocity model building and data-driven FWI

Scope: every physics-derived input that has been fed to neural networks for velocity estimation, either replacing raw shot gathers or added to them. For each one: what it measurably contributed, and what it cost. The user's target setting is OpenFWI-like: 2D acoustic, 5 shots, 70 receivers, 0.7 x 0.7 km, 10 m grid, 15 Hz Ricker, 1 s records.

Reference point for that setting. OpenFWI's raw-gather baselines (InversionNet, VelocityGAN, UPFWI) use exactly this geometry: 5 sources 175 m apart, 70 receivers 10 m apart, 70x70 maps, 15 Hz Ricker, 1,001 steps of 1 ms. Training sets are 24K samples (FlatVel/CurveVel), 48K (Fault families) and 60K (Style) ([OpenFWI, Deng et al. NeurIPS 2022](https://ar5iv.labs.arxiv.org/html/2111.02926)). Baseline numbers are listed under Q7.

---

## Q1. Migrated images (Kirchhoff, RTM, LSRTM) and/or initial/background velocity as network input

### Takeaway
Feeding a migrated image (almost always RTM; sometimes Kirchhoff depth-migrated gathers) together with the background velocity used to migrate it, and predicting a velocity update iteratively, is an established pattern. Multiple independent groups use it: Brazil/CBPF, Xi'an Jiaotong, KAUST, TGS/PGS industry. Most papers report only qualitative improvements. The few hard numbers come from Deep-tomography (Marmousi MAE 185 to 235 m/s, SSIM 0.59 to 0.68) and from KAUST's 2026 diffusion work (in-distribution MAE 33.9 m/s, SSIM 0.942; Marmousi OOD MAE 147 m/s, SSIM 0.68 before FWI). No paper found uses LSRTM images as velocity-network input.

### Cited Findings
**Deep-tomography (Muller, Costa, Bom, Klatt, Faria, de Albuquerque et al., arXiv 2022; GJI)** ([arXiv/ar5iv](https://ar5iv.labs.arxiv.org/html/2204.10813), [PDF](https://arxiv.org/pdf/2204.10813))
- Input: 14 common-offset subsurface-offset panels from RTM with an extended (cross-correlation) imaging condition, horizontal offsets only, clipped at the 96th percentile and normalized to [0, 1].
- The migration velocity is a Gaussian-smoothed true model (sigma = 100 grid points), with the water layer restored.
- Output: the velocity update (target minus migration model).
- Three iterations, each with its own independently trained U-Net. Later iterations use less smoothing (sigma 50, then 20).
- Data: 1,000 synthetic 2D marine models, 3.5 x 9 km, 225 shots and 701 receivers, 20 Hz Ricker. About 720 train, 180 validation, 100 test.
- Training cost: 15 to 110 s/epoch on one V100, 100 epochs per network.
- Marmousi, never seen in training: MAE 235 m/s and SSIM 0.59 with training set 1; MAE 185 m/s and SSIM 0.68 with training set 2, which added low-velocity layers and more realistic faults. The in-distribution test set is reported only qualitatively.
- One-step (non-iterative) prediction is described as poor, without numbers. Using the result as the FWI starting model improved FWI qualitatively, stabilizing after about 50 iterations.
- No raw-gather network was tested. The authors only argue that offset panels are more compact than raw shots or angle gathers.

**Deep-tomography extensions** (SBGf 2023 expanded abstracts): a 3D version ([3D tomo](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/18th_CISBGf/735b90b4568125ed6c3f678819b6e058tomo_3d.pdf)), and well-log velocity added as an extra input channel to the Deep-tomography U-Net ([well info](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/18th_CISBGf/4c56ff4ce4aaf9573aa5dff913df997adeeptomography_wellinfo_AbstractExpanded.pdf)). Both are known only from search snippets; no numbers were seen.

**Zhang & Gao, "Deep-Learning Full-Waveform Inversion Using Seismic Migration Images" (IEEE TGRS, 2021)** ([OpenAlex record/abstract](https://api.openalex.org/works/doi:10.1109/TGRS.2021.3062688))
- Stage 1: RTM images are fed to a CNN that estimates the background velocity.
- Stage 2: an iterative network refines it to high resolution. Its inputs are the reconstructed background velocity, the RTM image, and a gradient-regularization term.
- Claims better accuracy, noise robustness and generalization than data-driven DLFWI on layered and fault synthetics. The abstract gives no numbers, and the paper is closed access.

**Zhang & Gao, common-source migration image gathers (IEEE GRSL, 2022)** ([XJTU record](https://scholar.xjtu.edu.cn/en/publications/high-resolution-velocity-model-building-based-on-common-source-mi/))
- Input: RTM common-source migration image gathers (CSMIGs) plus the migration velocity.
- Claims better reconstruction, noise robustness and easier generalization than raw prestack-data input. Qualitative only.

**Ma & Alkhalifah's 2025 neural-operator paper**: its introduction attributes the following to prior work ([arXiv 2509.20238 HTML](https://arxiv.org/html/2509.20238v1)).
- "Zhang et al. (2020)" fed both the RTM image and its background velocity to a network, which alleviated FWI cycle skipping.
- "Yang et al. (2023)" fused well logs with RTM images in a multimodal network and reported accurate results on out-of-distribution test sets.

**Ma & Alkhalifah, Convolutional Neural Operator (CNO) VMB from RTM images (arXiv, Sep 2025)** ([HTML](https://arxiv.org/html/2509.20238v1), [abs](https://arxiv.org/abs/2509.20238))
- The CNO learns the forward map (initial plus true velocity) to RTM image. Inversion then runs autodiff through the CNO to inject high wavenumbers into a background velocity.
- Data: 8,000 training and 2,000 validation layered synthetic models. Field test on CGG NW Australia data, using 116 of 1,824 shots over a 12 x 3.3 km model.
- About 15 s, against about 2.5 h for multiscale FWI (3 to 16 Hz), with "comparable or higher resolution". This comparison is visual only.
- No MAE/SSIM/RMSE reported. Shallow velocities still need improvement.

**Ma, Wang & Alkhalifah, neural operator with time-shift imaging condition (arXiv, Oct 2025)** ([abs](https://arxiv.org/abs/2510.01735))
- Supervised map from an initial velocity plus time-lag RTM extended images to a high-resolution velocity. A search snippet says three time-lag images are used.
- Includes OOD salt bodies, offshore Australia field data and a well-log check. Prediction takes seconds.
- The abstract gives no metrics.

**Cheng, Harsuko & Alkhalifah, "Shallow-to-deep VMB via diffusion models, Part II" (arXiv, Sep 2026)** ([HTML](https://arxiv.org/html/2609.26482))
- Conditioning inputs: shallow velocity prior, migration velocity (Gaussian-smoothed true model, sigma = 25 points), an RTM image computed with that smooth background, and well logs. Conditions are dropped randomly during training.
- Synthetic training: 300 modified-Marmousi models (600 after flipping), 106 shots, 120 receivers, 20 Hz.
- In-distribution: MAE 33.91 m/s and SSIM 0.942, against GenAIB (49.27 m/s, 0.872) and WDIB (100.67 m/s, 0.767).
- Marmousi OOD, initial models: MAE 147.33 m/s and SSIM 0.681. After FWI: MAE 110.23 m/s and SSIM 0.803 (GenAIB 134.86 m/s, 0.701).
- Training took about 27 h on one A100.
- Conditioning ablation (background only, then plus image, then plus well) is qualitative: each added condition lowers error and uncertainty.
- A synthetic-vs-field migration-image domain shift was blamed for field well mismatch.

**TGS / PGS industrial ML tomography**
- Korsmo, Crawley & Chemingui (IMAGE 2025) ([PDF](https://www.tgs.com/hubfs/2025%20IMAGE%20Preliminary/FNO_MVA_Korsmo_final_pdf.pdf)):
  - Architecture: FNO + CNN + MLP. Inputs: depth-migrated gathers plus the corresponding velocity model. Output: a velocity update.
  - Trained on about 50,000 synthetic 2D models with varied velocity errors. Used iteratively (remigrate after each pass); 2 to 5 iterations usually resolve the macro model.
  - Field updates reached up to 1 km/s, far larger than conventional gradient updates. No RMO picks, horizons or salt masks are needed.
  - Norwegian Sea and Agung (Bali) field examples are qualitative only.
  - Earlier versions mapped velocity errors from data to image domain (Huang et al., 2023), then from image to image (Crawley et al., 2024).
- PGS Crawley et al. (IMAGE 2023): FNO estimates velocity corrections from migrated images and iterates ([PDF](https://tgs.com/hubfs/Technical%20Library/Technical%20Library%20Files/pgs-crawley_etal_image23_tomography-fno.pdf)). Known from a search snippet only.
- Roberts et al. (EAGE 2026) ([PDF](https://www.tgs.com/hubfs/Technical%20Library/EAGE%202026/2026Roberts_ML_VMB_EAGE_v1.pdf)):
  - Hybrid CNN-transformer (RAPUnet-based). Inputs: the velocity model plus Kirchhoff depth-migrated offset gathers, 81 input channels, 512x512 context.
  - Training models were derived from petrophysical analysis of more than 60,000 offshore wells.
  - Synthetic pretraining alone did not generalize to real data.
  - Self-supervised fine-tuning on the target data was essential. Its loss used residual moveout terms (differential semblance, stack power, TV/smoothness).
- Roberts et al. (IMAGE 2026) "MLTomo" abstract: physics-based regularization, no details ([TGS](https://www.tgs.com/technical-library/accelerating-tomographic-updates-with-physics-based-deep-learning-and-realistic-earth-models)).

**Lewis & Vigh (SEG 2017)** used deep learning to interpret migrated images and build prior velocity models for FWI. This is cited in Fabien-Ouellet & Sarkar's reference list ([PDF](https://publications.polymtl.ca/6302/1/2020_Fabien-Ouellet_Seismic_velocity_estimation_deep_recurrent.pdf)).

**Salt from inaccurate migrated subsurface-offset gathers (Muller et al., Geophysics 2022)**
- Input: subsurface-offset gathers migrated with a sediment-only velocity. Output: salt masks.
- Accuracy described as "high", no numbers in the abstract ([arXiv](https://arxiv.org/abs/2204.09710)).

### Inferences
- The dominant design is image plus background velocity, predicting an update, iterated with remigration. It turns the network into a learned tomographic operator rather than a global data-to-model map. The image carries high wavenumbers, and its kinematic errors (moveout, defocusing) carry the low-wavenumber error signal.
- Every successful field deployment found (TGS, KAUST) relies on either self-supervised physics fine-tuning or iterative remigration. That suggests image-domain inputs reduce, but do not remove, the synthetic-to-field gap.
- Cost in the OpenFWI setting is my estimate, not sourced. One RTM image per sample is about 5 forward plus 5 adjoint solves, plus imaging-condition work. That is roughly 2x the cost of simulating the 5 shot gathers, so tractable for 10^4 samples.
- With only 5 shots, RTM images will carry strong acquisition-footprint and low-fold artifacts. None of the above papers tested such sparse source sampling; the closest is WISE on OpenFWI CurveFault-A (Q2).

### Gaps
- No paper found that feeds LSRTM images (as opposed to RTM or Kirchhoff) to a velocity-estimation network.
- Zhang & Gao (2021, 2022) report their accuracy and generalization advantage over raw-data input only qualitatively in the abstracts. The full texts (closed access) were not read.
- The exact references for "Zhang et al. 2020" and "Yang et al. 2023", as cited by Ma & Alkhalifah, could not be confirmed.

---

## Q2. Extended images (subsurface-offset CIGs, angle-domain CIGs, ray-parameter CIGs, RMO panels) as network input

### Takeaway
Extended images are the best-documented physics input. One paper gives a controlled comparison: WISE (Geophysics 2024) showed that subsurface-offset CIGs computed with a poor 1D background are a more informative conditioning input than RTM images at equal network and data budget. Its numbers include the user's OpenFWI geometry: SSIM 0.87 vs 0.85 on CurveFault-A, and 0.63 vs 0.52 on Compass. Ray-parameter CIGs migrated at a constant 2 km/s reach SSIM about 0.95 in-distribution (Geng et al. 2022). Automated MVA from RMO in angle gathers is established, but reported only qualitatively.

### Cited Findings
**WISE: full-Waveform variational Inference via Subsurface Extensions (Yin, Orozco, Louboutin & Herrmann, Geophysics 2024)** ([arXiv HTML](https://arxiv.org/html/2401.06230), [SLIM](https://slim.gatech.edu/content/wise-full-waveform-variational-inference-subsurface-extensions))
- A conditional normalizing flow is conditioned on CIGs, defined as the adjoint of extended Born modeling applied to the data. The baseline conditions on RTM, the adjoint of non-extended Born modeling.
- OpenFWI CurveFault-A setup: 101 subsurface offsets (-250 to +250 m); background is a 1D model from horizontally averaging the true model. 2,800 train, 150 validation, 50 test.
  - Mean SSIM 0.87 with CIGs vs 0.85 with RTM.
- Compass setup: 51 offsets (-500 to +500 m); background is the arithmetic mean of all models (1D). 800 train, 190 validation, 50 test; 200 epochs.
  - Mean SSIM 0.63 with CIGs vs 0.52 with RTM. On an unseen slice, 0.56 vs 0.48.
- Conclusion: CIGs preserve more information than RTM when the background is poor. Posterior standard deviation correlates with error.
- SSIM is the only metric. No CIG compute cost is quantified; inference needs a single set of CIGs.
- Follow-ups:
  - WISER adds non-amortized physics refinements to close the amortization gap ([SLIM](https://slim.gatech.edu/node/7338)).
  - Zeng, Gahlot, Deng & Herrmann (IMAGE 2026) compress 3D CIG volumes with a multiscale self-attention summary network feeding a flow-matching posterior. They claim better accuracy, lower uncertainty and more robustness to background mismatch than conditioning on raw CIGs, with no numbers in the abstract ([SLIM](https://slim.gatech.edu/node/7437)).
  - Rex, Yin & Herrmann (IMAGE 2024) train an FNO to "velocity-continue" CIGs from one migration velocity to another without remigrating ([SLIM](https://slim.gatech.edu/node/7333)).

**Geng, Zhao, Shi, Wu, Fomel & Sen, "Deep learning for velocity model building with common-image gather volumes" (GJI 228(2), 2022)** ([OUP](https://academic.oup.com/gji/article/228/2/1054/6372247))
- Input: ray-parameter CIGs (128 depth x 512 x 161 ray parameters, 20 m) from frequency-domain double plane-wave RTM, migrated with a constant 2 km/s reference velocity.
- Network: ResNet-50 encoder (ImageNet-pretrained) with up-projection decoder; the ray-parameter axis is used as channels.
- Data: 200 2D sections from 20 synthetic 3D models, split 140/30/30. 256 shots per section. Noise at SNR 1 to 10 added to the CIGs.
- Test results by loss (units not stated, probably km/s):

  | Loss | RMSE | MAE | SSIM | MS-SSIM |
  |---|---|---|---|---|
  | L1 + MS-SSIM | 0.075 | 0.044 | 0.946 | 0.988 |
  | L1 | 0.089 | 0.036 | 0.920 | — |
  | L2 | 0.095 | 0.062 | 0.888 | — |

- 3D and hybrid networks were not better (no numbers given). No field test and no shot-gather baseline. The authors say training and test samples are partly similar.

**Ding & Ma, "Automatic migration velocity analysis via deep learning" (Geophysics 87(4), 2022)** ([record](https://hit.globalimpact.cn/en/publications/automatic-migration-velocity-analysis-via-deep-learning/))
- Input: CIGs containing residual moveout, after an RMO normalization step that maps dipping-reflector RMO to the flat case.
- A CNN classifies frown and smile patterns and converts them to velocity updates, replacing velocity-spectrum picking.
- Two synthetic examples and one field example. Qualitative only (gathers flattened).

**Sun, Zhao & Ma, "Deep-learning-based MVA with multi-type input data" (Geophysical Prospecting 74(5), 2026)** ([record](https://hit.globalimpact.cn/en/publications/deep-learning-based-migration-velocity-analysis-with-multi-type-i/))
- Inputs: ADCIGs, migration velocity, envelope of the Radon transform of the ADCIGs, and relative geologic time (RGT), taken from several neighboring locations. GAN keyword.
- Recovers low and middle wavenumbers like conventional MVA, and possibly high wavenumbers at interfaces. Qualitative only.

**Time-lag (time-shift) extended RTM images (KAUST)**
- Ma, Wang & Alkhalifah 2025: initial model plus time-lag RTM images as neural-operator input ([abs](https://arxiv.org/abs/2510.01735)).
- Ma, Taufik & Alkhalifah (Dec 2025): a neural operator is a surrogate for modeling plus migration, producing time-lag images from (true, migration) velocity. The migration velocity is updated by autodiff to match observed time-lag images, with a diffusion prior to suppress artifacts. Time-lag information improves mid and high wavenumbers; the operator alone leaves noticeable artifacts. No numbers in the abstract ([abs](https://arxiv.org/abs/2512.23375)).

**Deep-tomography** uses subsurface-offset panels (details in Q1). The salt paper of Muller et al. (2022) also uses subsurface-offset gathers ([arXiv](https://arxiv.org/abs/2204.09710)).

**CIG-to-CIG correction**: a 2025 IEEE GRSL paper uses a CNN, optionally velocity-constrained, to correct CIGs distorted by velocity errors ([USTC page](https://faculty.ustc.edu.cn/jiaxiaofeng/en/lwcg/997473/content/63086.htm)). Known from a search snippet only.

### Inferences
- In the user's geometry, WISE's CIG-over-RTM gain on CurveFault-A is small (+0.02 SSIM) compared with Compass (+0.11). A plausible reason: with a 0.7 km domain and a near-true 1D background, RTM is already kinematically close. The extension helps most when the background is far off. This is my interpretation, not the paper's.
- Subsurface-offset CIGs with about 100 lags increase input size by about 100x over a single image, which is why SLIM moved to learned summary networks. For a 70x70 grid, 101 lag channels are still small (about 0.5M values). The cross-correlation cost per lag is about one imaging-condition pass.
- Ray-parameter or plane-wave CIGs migrated at a constant velocity (Geng) are notable because they need no background-velocity estimate, so they work as a fixed physics preprocessing of the raw data.

### Gaps
- No paper found reporting CIG-input vs raw-gather-input numbers under the same network and data budget. WISE compares CIG vs RTM only.
- Ding & Ma (2022) and Sun, Zhao & Ma (2026) give no quantitative metrics in the accessible abstracts.

---

## Q3. Velocity spectra / semblance, NMO-corrected CMPs, tau-p, picked traveltimes as input

### Takeaway
Semblance cubes were the first physics-derived input for 2D gridded velocity (Araya-Polo et al., TLE 2018: SSIM about 0.87 to 0.89 on layered models, 0.81 with salt). Semblance and patch inputs are standard for automated stacking-velocity picking. An influential counterpoint (Fabien-Ouellet & Sarkar, Geophysics 2020) deliberately avoided semblance as lossy, using raw CMPs with a semblance-mimicking architecture (Vrms RMSE 44 m/s, Vint RMSE 323 m/s). Tau-p and traveltime inputs exist only in abstracts with no numbers.

### Cited Findings
**Araya-Polo, Jennings, Adler & Dahlke, "Deep-learning tomography" (The Leading Edge 37(1), 2018)** ([PDF](https://sep.stanford.edu/sep/joseph29/pubs/Araya-2018-dltomo.pdf))
- Input: a semblance cube (CMP location x zero-offset time x trial NMO velocity). Raw data are reduced to this feature, which the authors say cuts memory and focuses training.
- Training data: tens of thousands of random 2D models with 3 to 8 layers, up to 4 faults, 2,000 to 4,000 m/s, 140x180 grid; thousands of test models; acoustic modeling.
- Results:
  - Average SSIM 0.8717 with k-means segmentation into the correct number of layers, 0.8603 with 8 segments.
  - Layered-only regression: R² 0.8124, SSIM 0.8939.
  - With salt bodies: R² 0.5536, SSIM 0.8101.
- Training takes a few hours for 2D on K80 GPUs; inference takes seconds. No comparison with raw-gather input.

**Adler, Araya-Polo & Poggio (EAGE 2019)**: recurrent (RNN, LSTM, GRU) VMB architectures. GRU and LSTM predict salt more accurately than non-recurrent networks. The input representation and numbers are not in the abstract ([arXiv](https://arxiv.org/abs/1908.07824)).

**Fabien-Ouellet & Sarkar, "Seismic velocity estimation: A deep recurrent neural-network approach" (Geophysics 85(1), U21-U29, 2020)** ([PDF](https://publications.polymtl.ca/6302/1/2020_Fabien-Ouellet_Seismic_velocity_estimation_deep_recurrent.pdf))
- Input: raw CMP gathers. The authors explicitly reject semblance as a lossy, non-invertible transform that removes amplitude and phase.
- Architecture mirrors semblance analysis:
  - CNN encoder, standing in for the NMO scan (32 feature maps instead of trial velocities).
  - Recursive CNN that collapses the offset axis, standing in for semblance summation.
  - LSTM decoder, standing in for picking.
- Trained on 40,000 random 1D layered models, 16-network ensemble.
- Results:
  - Test on 1,600 1D models: Vrms RMSE 44 m/s, Vint RMSE 323 m/s. Ensemble standard deviation is 36 m/s (Vrms) and 91 m/s (Vint).
  - Gently dipping 2D models (100 of them, at most 6 degrees): Vint RMSE 322 to 424 m/s, Vrms 42 to 44 m/s.
  - Primary-reflection detection: 7.4% true positives, 89.2% true negatives, 2.4% false negatives, 1.0% false positives.
- Training took about 24 h on one GPU; 2,000 field CMPs took under 5 min.
- USGS marine line: a coherent stack, but free-surface multiples (absent from training) corrupted the estimates.

**Park (supervised by Sacchi), U. Alberta MSc thesis 2019** ([record](https://ualberta.scholaris.ca/items/2d0d70af-023c-4154-979f-6d3820c4ab9f))
- Input: the full semblance panel as a guide image plus a small semblance patch at one time step. Labels are RMS velocities.
- The CNN failed when target data differed from synthetics. Transfer learning on a small portion of target data fixed it, validated on Marmousi and a Gulf of Mexico marine line.
- The 2020 Geophysics journal version (Park & Sacchi) was not located; numbers were not seen.

**Ferreira et al. (IEEE TGRS 2020)**: hybrid CNN regression for automatic velocity analysis on semblance. Title only ([IBM](https://researcher.watson.ibm.com/publications/automatic-velocity-analysis-using-a-hybrid-regression-approach-with-convolutional-neural-networks)).

**Self-supervised Vrms (TGS, Roberts et al. EAGE 2026)**: a CNN predicts Vrms with no training data. Physics enters through the loss (differential semblance plus stack power plus TV), not the input. Applied to NMO-corrected CMPs on 20,000 km² offshore Brazil ([PDF](https://www.tgs.com/hubfs/Technical%20Library/EAGE%202026/2026Roberts_ML_VMB_EAGE_v1.pdf)).

**Kazei, Ovcharenko, Plotnitskii, Peter, Zhang & Alkhalifah (Geophysics 2021)**
- Input: multiple neighboring raw CMP gathers (not physics-transformed). Output: a 1D vertical velocity log, from an ensemble of CNNs.
- Better at larger dips than single-CMP input. Trained on Marmousi augmentations, tested on Marmousi II and Overthrust.
- Training data generation plus training costs more than FWI, but inference is far cheaper ([abstract](https://sindex.sdl.edu.sa/esploro/outputs/journalArticle/Mapping-full-seismic-waveforms-to-vertical/9943720808331), [EAGE 2020](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202011980)).

**Tau-p**: Gullikson, Dhara, Tuvi & Sen (SEG 2024) transform shot gathers to tau-p, predict time-domain velocity with a U-Net, then convert to depth by integration, as an FWI starting model ([record](https://cris.bgu.ac.il/en/publications/automated-velocity-modeling-with-domain-transformations-2/)). Known from a search snippet only, no numbers. No other deep-learning tau-p velocity paper was found.

**Picked first-arrival traveltimes**
- Jo & Ha (EGU 2023): traveltimes only, from towed streamers, mapped to velocity. Input is much smaller, cutting GPU memory and training time, but resolution drops; results usable as FWI starting models. No numbers ([abstract](https://meetingorganizer.copernicus.org/EGU23/EGU23-1953.html)).
- A GJI 2023 paper (vol. 235(1), p. 326) feeds first-arrival traveltime-tomography inverted models, not raw data, to a network. It claims a small training set suffices and shows western-China land data ([GJI mirror](https://dbpia.nl.go.kr/gji/article/235/1/326/7181483)). Title, authors and numbers were not retrieved (HTTP 429).
- PINNtomo uses traveltimes in the loss, not as input ([arXiv](https://arxiv.org/pdf/2104.01588)).

### Inferences
- Semblance and NMO-type inputs assume hyperbolic moveout and near-flat layering. Fabien-Ouellet's 2D tests show this degrading with dip (Vint RMSE up to 424 m/s at only 6 degrees).
- For OpenFWI's 0.7 km aperture with 5 shots, CMP fold is very low. Semblance panels from 5 shots x 70 receivers would be sparse and noisy. That is my inference; no paper tests it.
- Fabien-Ouellet's design is instructive. It builds the physics prior (NMO scan, offset collapse, temporal recurrence) into the architecture rather than the input, keeping amplitude and phase.

### Gaps
- No quantitative results found for tau-p or picked-traveltime inputs.
- No direct raw-vs-semblance comparison at equal budget was found. Araya-Polo argues for semblance, and Fabien-Ouellet argues against it, without experiments.

---

## Q4. FWI gradients, adjoint images, Hessian approximations, illumination maps; learned iterative / unrolled FWI

### Takeaway
Gradient and adjoint images as network inputs appear in two forms.
- Iterative learned-update schemes: Zhang et al. AD-DLFWI (TGRS 2021), where the input is the FWI gradient unfolded in the common-source domain, and Sun & Alkhalifah's "ML-descent", a learned gradient conditioner.
- Amortized Bayesian inference: SLIM's adjoint and RTM summary statistics, and ASPIRE's iterated gradient summary.

Learned inverse-Hessian preconditioners (gradient to deblurred gradient) are an active KAUST line. All of these report mostly qualitative results. No seismic recurrent inference machine (RIM) or learned primal-dual for velocity was found. No paper feeding illumination maps as an input channel was found.

### Cited Findings
**Zhang, Gao, Gao & Chen, "Adjoint-Driven Deep-Learning Seismic FWI" (AD-DLFWI, IEEE TGRS 59(10), 2021)** ([XJTU record](https://scholar.xjtu.edu.cn/en/publications/adjoint-driven-deep-learning-seismic-full-waveform-inversion/))
- Input: the image formed by the adjoint of the scattering wave equation, equivalent to the FWI gradient. It is unfolded into the common-source domain to keep velocity-error information, as in WEMVA.
- Two-step iterative scheme with learned FCN update operators, trained sequentially: low wavenumbers first, then high.
- Reliable on reflection-dominant synthetics and insensitive to cycle skipping. No numbers in the abstract.

**Zhang & Gao (TGRS 2021)**: the iterative refinement net takes the RTM image, the current background and a gradient-regularization term (see Q1) ([OpenAlex](https://api.openalex.org/works/doi:10.1109/TGRS.2021.3062688)).

**ML-descent (Sun & Alkhalifah; KAUST)**: a neural network trained as an optimal conditioner of the FWI gradient, one of three components alongside ML-misfit and ML-prior ([KAUST SWAG](https://swagroup.kaust.edu.sa/research/detail/the-shift-to-ml-solutions-for-full-waveform-inversion-in-three-diagrams)). Architecture and numbers for the 2020 Geophysics paper were not verified in this session.

**Deep Hessian deblurring (Alfarhan, Ravasi, Chen & Alkhalifah, arXiv 2024)** ([abs](https://arxiv.org/abs/2403.17518))
- The network maps a Hessian-blurred (doubly migrated) gradient to the gradient, then is applied to the gradient as an approximate inverse Hessian. Weights are carried to the next FWI iteration.
- Tested on two synthetic datasets and one field dataset; no numbers in the abstract.
- A Volve OBC field application trained once at the first FWI iteration, then fine-tuned cheaply ([AAPG abstract](https://imageevent.aapg.org/portals/26/abstracts/2024/4091975.pdf)).
- An EAGE 2025 abstract uses a UNet inverse-Hessian in Gauss-Newton FWI on Marmousi-2 at 5, 10 and 25 Hz and reports faster convergence ([EarthDoc](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202576026)). Snippet only.
- Lineage: CNN inverse Hessian for least-squares migration (DUG), building on GANs (Kaur, Pham & Fomel 2020) and non-stationary matching filters ([DUG PDF](https://dug.com/download/12/technical-papers/9193/a-deep-learning-inverse-hessian-for-least-squares-migration.pdf)).
- A KAUST abstract cautions that networks can reduce crosstalk in encoded gradients, but better-looking gradients do not necessarily improve the inversion ([KAUST event](https://pse.kaust.edu.sa/events/detail/2026/07/15/default-calendar/deep-learning-assisted-full-waveform-inversion)). Snippet only.

**Adjoint and gradient summary statistics (SLIM, Georgia Tech)**
- Orozco, Siahkoohi, Rizzuti, van Leeuwen & Herrmann (SPIE Medical Imaging 2023): preprocessing data with the adjoint (A^T y) before conditional normalizing-flow inference. It partially undoes the physics, giving faster convergence and robustness to acquisition changes. Details in Q7 ([paper HTML](https://slim.gatech.edu/Publications/Public/Conferences/SPIE/2023/orozco2022SPIEadjoint/SPIE_2022_adjoint.html), [SPIE](https://doi.org/10.1117/12.2651691)).
- ASPIRE (Orozco, Siahkoohi, Louboutin & Herrmann, arXiv 2024): alternates between training the flow and recomputing the physics-based summary (gradient or adjoint) at the flow's current estimate. This is an "unrolled in outer loop" scheme, shown on transcranial ultrasound ([arXiv](https://www.arxiv.org/pdf/2405.05398), [MUST 2024](https://indico.kit.edu/event/3830/contributions/16647/contribution.pdf)).
- Siahkoohi, Rizzuti, Orozco & Herrmann (Geophysics 2023): physics-based latent correction of an amortized flow conditioned on RTM for linearized imaging. Robust to changes in source count, noise and prior, at the cost of about 5 RTMs ([arXiv](https://arxiv.org/abs/2207.11640v2)).

**Deep unfolded FWI with gradient input (medical ultrasound, arXiv 2026)**: each stage's iteration-dependent CNN takes the current estimate plus the data-fidelity gradient and outputs an update. Stages are trained sequentially from a homogeneous start ([arXiv](https://arxiv.org/pdf/2603.04070)). Not seismic, but the closest learned-gradient-descent design found.

**Invertible RIMs (Peng, Vasconcelos & Ravasi)**: migrated images plus a smooth velocity prior plus the modeling operator, inverted to impedance perturbations (image-domain deblurring, not velocity). Robust to noise and limited acquisition on field data ([UU portal](https://research-portal.uu.nl/en/publications/image-domain-seismic-inversion-by-deblurring-with-invertible-recu/)).

**Not input-based**: FWI-as-RNN (Richardson 2018 ([arXiv](https://ar5iv.arxiv.org/html/1801.07232)); Sun et al. SEG 2019 ([CREWES](https://www.crewes.org/ForOurSponsors/ConferenceAbstracts/2019/SEG/Sun_SEG_2019.pdf))) recasts the forward solver as a network with velocity as weights. These are reparameterizations, not physics-derived inputs.

### Inferences
- For a fixed background (smooth or 1D), "gradient at background" and "RTM image at background" are the same physics input up to preconditioning. So Q1 and Q4 results overlap. What differs is whether the network is used once (amortized) or re-fed updated gradients (learned iterative).
- The learned-iterative or unrolled design with gradient input is common in medical imaging and CT (learned primal-dual, RIMs) but rare for seismic velocity. Only AD-DLFWI and ASPIRE-style outer loops were found. This looks like an open niche.

### Gaps
- No seismic paper found that feeds illumination maps (source or receiver illumination, Hessian diagonal) as an explicit network channel for velocity estimation.
- No quantitative AD-DLFWI or ML-descent numbers were retrieved, because the full texts were not accessed.
- No seismic-velocity RIM or learned primal-dual paper was found. If one exists, it is not prominent.

---

## Q5. Seismic attributes (envelope, instantaneous phase/frequency, dip / structure tensor) as extra channels

### Takeaway
Attribute channels for velocity estimation are rare and poorly quantified. The documented cases are:
- Envelope of the Radon-transformed angle gathers plus RGT as extra channels for deep-learning MVA (Sun, Zhao & Ma 2026).
- Envelope as an extra RNN input for low-frequency extrapolation (not velocity).
- Plane-wave-destruction structural slopes guiding a diffusion velocity prior (Brandolin & Alkhalifah 2026).

No paper was found using instantaneous phase or frequency as velocity-network channels.

### Cited Findings
- **Sun, Zhao & Ma (Geophysical Prospecting, 2026)**: ADCIGs plus migration velocity plus envelope of the ADCIGs' Radon transform plus RGT models, from neighboring locations. Recovers low and mid wavenumbers; qualitative only ([record](https://hit.globalimpact.cn/en/publications/deep-learning-based-migration-velocity-analysis-with-multi-type-i/)).
- **Envelope for low-frequency reconstruction**: a Journal of Geophysics and Engineering paper adds envelope data as extra input features to a recurrent network, with good synthetic and Arctic field results. The task is bandwidth extension, not velocity ([OUP JGE](https://academic.oup.com/jge/article/22/1/238/7928131)).
- **Envelope inversion background**: physics-based, not neural. A convolution-based envelope inversion updates low wavenumbers and widens the misfit basin ([Petroleum Science 2020](https://www.cup.edu.cn/petroleumscience/docs/2020-04/fbf23c4937d34ee4806964e353519400.pdf)).
- **Structural slopes (dip)**: Brandolin & Alkhalifah (arXiv, Jul 2026) build a joint velocity-slope diffusion prior. Plane-wave-destruction slopes propagate sparse well information along dip. Tested on a Volve synthetic and Viking Graben field data; no numbers in the abstract ([abs](https://arxiv.org/abs/2607.04982)).
  - A search snippet says the paper finds RTM-derived slopes give richer structural guidance than slopes from the smooth initial model ([PDF](https://arxiv.org/pdf/2607.04982)). The abstract does not mention RTM, so this is unverified.
- **RGT and structure**: RGT appears as a channel in Sun, Zhao & Ma (2026), above.

### Inferences
- Attributes in this literature act as regularizing structural priors (dip, RGT) or as amplitude-insensitive kinematic features (envelope). They are not used as primary velocity carriers.
- For raw gathers in the OpenFWI setting, local slopes (p = dt/dx) of events are the most physics-direct attribute. Event slope maps to horizontal slowness, as in stereotomography. No deep-learning velocity paper feeding gather-domain slopes was found. The user's own slope pipeline appears to be novel relative to this literature.

### Gaps
- No paper found with instantaneous phase or frequency channels, or structure-tensor channels computed on gathers, as velocity-network input.
- No ablation found quantifying the gain from adding attribute channels.

---

## Q6. Multiple images computed with several background velocities (velocity scans, image focusing vs velocity) as network input

### Takeaway
Only one clear example was found. Jennings, Clapp, Araya-Polo & Biondi (arXiv 2022) apply a CNN to prestack residual-migration scans (161 Stolt residual migrations, rho = 0.90 to 1.10). The network scores focusing per patch and picks rho. Under limited aperture it beat semblance: fault-segmentation IoU 0.83 vs 0.72. FNO "velocity continuation" (SLIM) makes such scans cheap. Iterative schemes (Deep-tomography, TGS) see one background per pass, not a scan.

### Cited Findings
**Jennings, Clapp, Araya-Polo & Biondi, "Automatic interpretative image-focusing analysis" (arXiv 2202.03646, 2022)** ([ar5iv](https://ar5iv.labs.arxiv.org/html/2202.03646))
- Input: 3D patches (32 aperture angles x 64 depth x 64 midpoint) of angle-domain CIGs, converted from subsurface-offset gathers, taken from 161 prestack Stolt residual depth migrations (rho 0.90 to 1.10 in steps of 0.00125).
- Output: a 0-to-1 focusing score per patch. For each patch, the rho with the highest score is selected.
- Network: 3D CNN with about 8.67M parameters.
- Training:
  - Synthetic pretraining: 8,192 focused and unfocused patches from 1,000 models.
  - Field fine-tuning: only 56 labeled patches, plus 20 for validation.
- Accuracy: nearly 100% on synthetic validation.
- 2D Gulf of Mexico field data: 148 shots, 3.4 km maximum offset, muted to 1.0 km to simulate limited aperture.
  - The CNN's rho matches the full-aperture semblance reference better than semblance on the muted data.
  - Fault-segmentation IoU on corrected images: 0.58 unfocused, 0.72 semblance, 0.83 CNN.
- Cost, much higher than semblance:
  - Semblance: about 0.9 min on 1 CPU core.
  - CNN: synthetic generation about 15 min on 200 cores, pretraining about 10 min on a V100, fine-tuning 0.5 min, inference 1.1 min.
- Related thesis: SEP-190 (Stanford 2023), "Seismic velocity model building with deep convolutional neural networks" ([SEP](https://sep.sites.stanford.edu/publications/theses/seismic-velocity-model-building-deep-convolutional-neural-networks-sep-190-2023)).

**FNO velocity continuation of CIGs (Rex, Yin & Herrmann, IMAGE 2024)**: maps CIGs at one migration velocity to CIGs at another without remigration. Inference cost is "negligible", enabling rapid scanning of candidate velocity models for MVA and uncertainty quantification. No numbers ([SLIM](https://slim.gatech.edu/node/7333)).

**Iterative image-plus-velocity schemes** each use a single background per network call. Examples: Deep-tomography, 3 passes with decreasing smoothing ([ar5iv](https://ar5iv.labs.arxiv.org/html/2204.10813)); TGS FNO tomography, 2 to 5 passes ([PDF](https://www.tgs.com/hubfs/2025%20IMAGE%20Preliminary/FNO_MVA_Korsmo_final_pdf.pdf)).

### Inferences
- A velocity scan (images at, say, 0.9x to 1.1x of a background, or at a set of constant velocities) as a stacked channel input is close to classical residual-migration, focusing and semblance analysis, but in the image domain. Only Jennings et al. test it with a network, and they use it as a classifier, not as a regressor to a velocity map.
- In the OpenFWI setting, constant-velocity scans are cheap. One RTM per trial velocity per sample is about 10 wave solves; 10 to 20 trial velocities would be about 100 to 200 solves per sample. That is costly at 24K to 60K samples, but feasible on subsets. My estimate, not sourced.

### Gaps
- No paper found feeding a stack of full-wave (RTM) images at multiple background velocities to a network that directly regresses a 2D velocity map.
- No study found comparing a scan input against a single-background image input.

---

## Q7. Systematic comparisons of raw-gather vs physics-derived input at equal network and data budget

### Takeaway
No seismic-velocity study was found that compares raw shot gathers against physics-derived inputs under identical network, data budget and OOD tests. The nearest controlled evidence is:
- WISE: CIG vs RTM, same CNF and data. CIGs win, with SSIM 0.87 vs 0.85 on OpenFWI CurveFault-A and 0.63 vs 0.52 on Compass.
- Orozco et al. 2023: adjoint-preprocessed vs raw data, same CNF backbone, photoacoustic rather than seismic. Adjoint gives faster training and lower error, shown qualitatively.

Claims that image-domain input generalizes better than raw data (Zhang & Gao 2021/2022; Ma & Alkhalifah 2025) are single-group and qualitative. Several groups report that image-domain inputs still suffer synthetic-to-field shift.

### Cited Findings
**Within-physics comparison**: WISE, CIGs vs RTM with the same conditional normalizing flow and the same training sets:
- OpenFWI CurveFault-A (2,800 training samples): SSIM 0.87 vs 0.85.
- Compass (800 training samples): SSIM 0.63 vs 0.52.

([arXiv HTML](https://arxiv.org/html/2401.06230)).

**Raw vs adjoint, controlled (non-seismic)**: Orozco et al. (SPIE 2023), photoacoustic imaging ([HTML](https://slim.gatech.edu/Publications/Public/Conferences/SPIE/2023/orozco2022SPIEadjoint/SPIE_2022_adjoint.html)).
- Same CNF architecture and hyperparameters, but the raw-data version needs an extra learned downsampling layer from (N_t x N_rec) to model size.
- N = 2,048 training pairs.
- The adjoint input converged faster and reached lower conditional-mean MSE in less training time. Shown only as plots, with no epoch counts or speedup factor.
- One adjoint-conditioned network handled 8, 16, 32 and 64 receivers. No raw-data baseline was run for this robustness test.
- The authors claim to prove that conditioning on adjoint data is unbiased for a class of inverse problems ([SPIE](https://doi.org/10.1117/12.2651691)).

**Summary network vs raw CIGs**: Zeng et al. (IMAGE 2026) claim better accuracy, lower uncertainty and robustness to background mismatch when CIGs are compressed by a learned summary network rather than fed raw. No numbers ([SLIM](https://slim.gatech.edu/node/7437)).

**Single-group claims that image inputs beat data inputs**
- Zhang & Gao (TGRS 2021): RTM-image input outperforms data-driven DLFWI in accuracy, noise robustness and generalization ([OpenAlex](https://api.openalex.org/works/doi:10.1109/TGRS.2021.3062688)).
- Zhang & Gao (GRSL 2022): CSMIG input gives better reconstruction, noise robustness and generalization than prestack-data input ([XJTU](https://scholar.xjtu.edu.cn/en/publications/high-resolution-velocity-model-building-based-on-common-source-mi/)).
- Neither abstract gives numbers.

**Counter-evidence and caveats**
- Cheng et al. (2026): a domain shift between synthetic and field migration images reduces accuracy relative to in-distribution synthetics. Building matched training data requires care over geometry, wavelet, migration parameters and background smoothing ([HTML](https://arxiv.org/html/2609.26482)).
- Roberts et al. (TGS, EAGE 2026): even with Kirchhoff gathers plus velocity input and 60,000-well-derived training models, synthetic pretraining did not generalize to real data until physics-based self-supervised RMO fine-tuning was added ([PDF](https://www.tgs.com/hubfs/Technical%20Library/EAGE%202026/2026Roberts_ML_VMB_EAGE_v1.pdf)).
- Fabien-Ouellet & Sarkar (2020) argue the opposite direction for semblance: it discards amplitude and phase, so raw waveforms should be the input. They give the argument without an ablation ([PDF](https://publications.polymtl.ca/6302/1/2020_Fabien-Ouellet_Seismic_velocity_estimation_deep_recurrent.pdf)).
- Neither Deep-tomography nor Geng et al. tested a raw-gather baseline ([ar5iv](https://ar5iv.labs.arxiv.org/html/2204.10813), [OUP](https://academic.oup.com/gji/article/228/2/1054/6372247)).

**Raw-input fragility on OpenFWI**
- InversionNet generalizes poorly when the source frequency shifts slightly ([Inversion/SG-DeepONet](https://arxiv.org/html/2408.08005v1)).
- In OpenFWI's cross-dataset tests, VelocityGAN and UPFWI generalize better than InversionNet, and transfer from harder to easier families works better than the reverse ([OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926)).

**OpenFWI raw-gather baselines (Table 4, L1 loss; MAE / RMSE / SSIM on normalized velocity)** ([OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926)):

| Dataset | InversionNet | VelocityGAN | UPFWI |
|---|---|---|---|
| FlatVel-A | 0.0131 / 0.0211 / 0.9895 | 0.0118 / 0.0178 / 0.9916 | 0.0621 / 0.1233 / 0.9565 |
| CurveVel-A | 0.0685 / 0.1273 / 0.8074 | 0.0482 / 0.1034 / 0.8624 | 0.0805 / 0.1411 / 0.8443 |
| FlatFault-A | 0.0172 / 0.0426 / 0.9766 | — | — |
| CurveFault-A | 0.0260 / 0.0650 / 0.9566 | 0.0258 / 0.0606 / 0.9613 | — |
| Style-A | 0.0625 / 0.1024 / 0.8859 | — | — |

These use 24K to 60K training samples.

**Multi-source conditioning on OpenFWI**: DiffusionVel (Zhang, Li & Huang, arXiv 2024; EAGE 2025) combines a seismic-data-conditioned diffusion model, a well-log diffusion model and a geology prior, and injects background velocity via a low-pass filter. It claims to outperform FWI, InversionNet and VelocityGAN on OpenFWI and the Hess VTI model. No numbers in the abstract, and no per-source ablation reported ([arXiv](https://arxiv.org/abs/2410.21776)).

### Inferences
- The CIG-vs-RTM gap on OpenFWI CurveFault-A is about 0.02 SSIM. WISE's CurveFault-A SSIM of 0.87, from 2,800 samples with a probabilistic posterior mean, is below InversionNet's 0.957 from 48K samples. The protocols differ (sample count, deterministic vs posterior mean, normalization), so this is not evidence that physics inputs are worse. It does show that no published number yet shows a physics-derived input beating raw-gather baselines on OpenFWI at equal data.
- The data-efficiency argument for physics inputs rests mainly on:
  1. small training sets succeeding: Deep-tomography about 720, Geng 140, Jennings 56 field patches, WISE 800 to 2,800;
  2. Orozco's faster-convergence plot.

  It has not been tested by learning-curve comparisons against raw input. A raw-vs-physics learning-curve study on OpenFWI at matched budgets (for example 500, 2K, 8K and 24K samples, plus cross-family OOD) would be a genuine gap-filling contribution.
- Established, meaning multiple independent groups:
  - image plus background velocity, predicting an iterated update;
  - semblance or RMO-driven automated velocity analysis;
  - extended images (CIGs) as inputs.
- Single-paper or single-group:
  - CIG beats RTM (WISE);
  - adjoint beats raw (Orozco, non-seismic);
  - focusing CNN beats semblance under limited aperture (Jennings);
  - image input generalizes better than data input (Zhang & Gao).

### Gaps
- No equal-budget raw vs physics-input study for seismic velocity, in or out of distribution, was found.
- No published learning curves (error vs number of training samples) comparing input representations were found.
- OOD results for physics-input networks on OpenFWI cross-family splits were not found.
