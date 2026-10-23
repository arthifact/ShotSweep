# Architectures, Training Losses and Physics-in-the-Loop Strategies for Learned FWI (OpenFWI focus)

Conventions used throughout these notes:
- Family abbreviations: FVA/FVB = FlatVel-A/B, CVA/CVB = CurveVel-A/B, FFA/FFB = FlatFault-A/B, CFA/CFB = CurveFault-A/B, SA/SB (or STA/STB) = Style-A/B.
- OpenFWI 2D input is 5 shots x 1000 time samples x 70 receivers, output 70x70 velocity, velocity range 1500-4500 m/s, min-max normalized to [-1, 1] — [OpenFWI, arXiv 2111.02926](https://ar5iv.labs.arxiv.org/html/2111.02926).
- "Normalized MAE" below means MAE on the [-1, 1] scale. To convert to m/s for the Vel/Fault/Style families, multiply by 1500 (half of the 3000 m/s range). That conversion is my arithmetic, not a published number.
- Numbers from different papers are often NOT comparable: they differ in normalization, training-set size, re-trained baselines, checkpoint selection and (for physics/diffusion methods) oracle initial models. Each such caveat is flagged where it applies.

## 1. Benchmark numbers on OpenFWI (InversionNet, VelocityGAN, UPFWI, InversionNet3D, later methods)

### Takeaway
The canonical OpenFWI table (Deng et al., NeurIPS 2022 D&B) gives supervised InversionNet and VelocityGAN as the strong baselines. Unsupervised UPFWI is worse on every 2D family in that table. B-variants (CVB, FFB, CFB, SB) remain hard, with SSIM around 0.6-0.75. Later work gives large but hard-to-compare gains: a 1B-parameter transformer with tokenized velocity and generated data reaches aggregate normalized MAE 0.0136, against 0.0432 for BigFWI-L. Neural-operator mixtures report average MAE about 0.026, but under a different normalization. Diffusion samplers are reported with oracle-smoothed initial models. Scaling data across families (BigFWI) gives modest in-distribution gains (+13% MAE) and larger leave-one-out gains (+28.6% MAE).

### Cited Findings

**OpenFWI paper (Deng et al., 2022), Table 4: 2D benchmarks, MAE / RMSE / SSIM** — [OpenFWI, arXiv 2111.02926](https://ar5iv.labs.arxiv.org/html/2111.02926)

| Dataset | Loss | InversionNet | VelocityGAN | UPFWI |
|---|---|---|---|---|
| FVA | l1 | 0.0131 / 0.0211 / 0.9895 | 0.0118 / 0.0178 / 0.9916 | 0.0621 / 0.1233 / 0.9565 |
| FVA | l2 | 0.0111 / 0.0180 / 0.9887 | 0.0605 / 0.0783 / 0.9453 | – |
| FVB | l1 | 0.0351 / 0.0876 / 0.9461 | 0.0329 / 0.0807 / 0.9521 | 0.0677 / 0.1493 / 0.8874 |
| FVB | l2 | 0.0417 / 0.0909 / 0.9402 | 0.0328 / 0.0787 / 0.9556 | – |
| CVA | l1 | 0.0685 / 0.1273 / 0.8074 | 0.0482 / 0.1034 / 0.8624 | 0.0805 / 0.1411 / 0.8443 |
| CVA | l2 | 0.0690 / 0.1202 / 0.8223 | 0.0510 / 0.0976 / 0.8758 | – |
| CVB | l1 | 0.1497 / 0.2891 / 0.6727 | 0.1268 / 0.2618 / 0.7111 | 0.1777 / 0.3179 / 0.6614 |
| CVB | l2 | 0.1624 / 0.2801 / 0.6661 | 0.1428 / 0.2611 / 0.6962 | – |
| FFA | l1 | 0.0172 / 0.0426 / 0.9766 | 0.0868 / 0.1485 / 0.9313 | 0.0876 / 0.2060 / 0.9340 |
| FFA | l2 | 0.0174 / 0.0362 / 0.9798 | 0.0319 / 0.0531 / 0.9798 | – |
| FFB | l1 | 0.1055 / 0.1741 / 0.7208 | 0.0925 / 0.1600 / 0.7476 | 0.1416 / 0.2220 / 0.6937 |
| FFB | l2 | 0.1106 / 0.1723 / 0.7186 | 0.0946 / 0.1553 / 0.7552 | – |
| CFA | l1 | 0.0260 / 0.0650 / 0.9566 | 0.0258 / 0.0606 / 0.9613 | 0.0500 / 0.0966 / 0.9495 |
| CFA | l2 | 0.0280 / 0.0602 / 0.9592 | 0.0216 / 0.0505 / 0.9687 | – |
| CFB | l1 | 0.1646 / 0.2477 / 0.6163 | 0.1571 / 0.2427 / 0.5996 | 0.3452 / 0.5010 / 0.3941 |
| CFB | l2 | 0.1669 / 0.2412 / 0.6053 | 0.1583 / 0.2336 / 0.6033 | – |
| SA | l1 | 0.0625 / 0.1024 / 0.8859 | 0.0612 / 0.1000 / 0.8883 | 0.1429 / 0.2342 / 0.7846 |
| SA | l2 | 0.0610 / 0.0989 / 0.8910 | 0.0645 / 0.1025 / 0.8882 | – |
| SB | l1 | 0.0689 / 0.1614 / 0.6314 | 0.0697 / 0.1108 / 0.6953 | 0.1702 / 0.2609 / 0.6102 |
| SB | l2 | 0.0586 / 0.0893 / 0.7599 | 0.0649 / 0.0979 / 0.7249 | – |
| Kimberlina-CO2 | l1 | 0.0061 / 0.0374 / 0.9872 | 0.0122 / 0.0574 / 0.9716 | not evaluated |
| Kimberlina-CO2 | l2 | 0.0098 / 0.0400 / 0.9798 | 0.0119 / 0.0387 / 0.9527 | not evaluated |

- **OpenFWI training-set sizes (Table 2):** Vel family 24K train / 6K test. Fault family 48K / 6K. Style family 60K / 7K. Kimberlina-CO2 15K / 4430. 3D Kimberlina-V1 1664 / 163 — [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926)
- **Forward-modeling setup:** 10 m grid, 15 Hz Ricker wavelet, 5 sources at 175 m spacing, 70 receivers at 10 m spacing, dt = 1 ms, 1000 samples (Table 7 says 1,001 steps), 120 absorbing boundary cells — [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926)
- **Baseline losses and training:**
  - InversionNet: l1 or l2 pixel loss, batch 256, 120 epochs, AdamW, lr 1e-4.
  - VelocityGAN: l1/l2 pixel loss plus a Wasserstein adversarial loss with gradient penalty, batch 64, 480 epochs, 3 critic steps per generator step.
  - UPFWI: l1 + l2 + two VGG-16 perceptual terms on seismic data, each weighted 1.
  - Source: [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926)
- **InversionNet3D on 3D Kimberlina-V1 (Table 6):**
  - l1 loss with 8 of 25 sources selected: MAE 0.0105-0.0108, RMSE 0.0276-0.0286, SSIM 0.9835-0.9838 across three different source subsets.
  - l2 loss: SSIM 0.9427-0.9482.
  - The shallowest InversionNet3D has 14.42M parameters and uses 9.93 GB at batch 1.
  - Source: [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926)
- **Training time (OpenFWI Table 5):**
  - InversionNet: 2 h (Vel), 4 h (Fault), 5.5 h (Style).
  - VelocityGAN: 8.6 h, 16 h, 30 h.
  - UPFWI: 30 h, 60 h, 60 h.
  - The caption says UPFWI and InversionNet3D used 32 GPUs and the others used 1 GPU.
  - Source: [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926)
- **Metric scale:** The OpenFWI text I retrieved does not explicitly say whether MAE/RMSE are on normalized or m/s scale — [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926). Two later papers from the same group give the convention: MAE/MSE on velocity normalized to [-1, 1], SSIM on velocity rescaled to [0, 1] — [Auto-Linear, arXiv 2305.13314](https://arxiv.org/html/2305.13314v3); [BigFWI, arXiv 2307.15388](https://ar5iv.labs.arxiv.org/html/2307.15388). Auto-Linear's InversionNet row reproduces OpenFWI Table 4 values (FVA MAE 0.0131, CVB 0.1497, and so on).

**BigFWI / "An Empirical Study of Large-Scale Data-Driven FWI" (Jin et al., arXiv July 2023; Scientific Reports 2024)** — [arXiv 2307.15388](https://ar5iv.labs.arxiv.org/html/2307.15388); [Sci. Rep.](https://www.nature.com/articles/s41598-024-68573-7)
- **Architecture:** InversionNet-style 2D conv encoder-decoder (BatchNorm, LeakyReLU, Tanh output) in three sizes. BigFWI-B is the base. BigFWI-M adds layers. BigFWI-L is deeper and wider, with a 1024-length latent vector. No ViT or Transformer is compared.
- **Training data:**
  - OpenFWI-204K: 12K from each Vel dataset, 24K from each Fault dataset, 30K from each Style dataset.
  - OpenFWI-408K: the full training set across the 10 2D datasets.
  - Compute: 4x V100, 170 epochs, batch 256.
- **Headline averages (combined vs. per-dataset training):** MAE +13.03%, RMSE (written "MSE" in the abstract) +7.19%, SSIM +1.87%. FlatVel-A/B are exceptions where combined training did not help.
- **Leave-one-out generalization:** MAE +28.60%, RMSE +21.55%, SSIM +8.22%.
- **Model size:** BigFWI-L vs. BigFWI-B gives MAE +20.06%, RMSE +13.39%, SSIM +0.72%. On SB, BigFWI-B beats BigFWI-L.
- **Per-family tables:** These are only in Supplementary Tables S2-S12, which I did not retrieve.
- **Loss:** "L1 + L2" pixel loss. As typeset, the second term reduces to |v − v̂|, so the formula appears to have a typo.

**"Improving Full Waveform Inversion in Large Model Era" (Feng, Jin, Guo, Chen, Lin; arXiv 2603.00377, v2 May 2026)** — [arXiv 2603.00377](https://arxiv.org/html/2603.00377v2)
- **Model:** 1.03B-parameter transformer predicting velocity tokens with non-causal parallel decoding. Tokenizer is a ViT-VQGAN (25x25x196 latent).
- **Data:** A latent diffusion model generates extra velocity maps, which are simulated to grow training pairs from about 408K to about 5M.
- **Training pipeline:** Supervised pretraining, then GRPO-style RL with velocity rewards, then latent-space gradient-descent refinement at inference.
- **OpenFWI test MAE (normalized, aggregate over 10 subsets, Table 1):**

| Model | MAE | Params |
|---|---|---|
| BigFWI-M | 0.0437 | 28M |
| BigFWI-L | 0.0432 | 87M |
| Causal baseline | 0.0770 | 1.09B |
| + Data x2 | 0.0463 | 1.09B |
| + Data x10 | 0.0368 | 1.09B |
| Non-causal, Data x2 | 0.0330 | – |
| + ViT-VQGAN | 0.0167 | – |
| + RL | 0.0159 | – |
| + Latent GD | 0.0136 | 1.03B |

  - No per-family breakdown is given.
- **Cost:** The transformer trained on 4x L40, about 6 h per epoch, 100 epochs. RL ran 700K iterations.

**Auto-Linear (Feng et al., ICML 2024)** — [arXiv 2305.13314](https://arxiv.org/html/2305.13314v3); [PMLR v235](https://proceedings.mlr.press/v235/feng24a.html)
- **Architecture:** Separate masked autoencoders for seismic data and velocity, each self-supervised and then frozen. Only a linear converter (two linear layers, 128-dimensional bottleneck, l1 loss) is trained on paired data. Model size is 12.3M parameters, vs. 24.4M for InversionNet.
- **Full-data MAE (normalized):**

| Model | FVA | FVB | CVA | CVB | FFA | FFB | CFA | CFB | SA | SB |
|---|---|---|---|---|---|---|---|---|---|---|
| Auto-Linear | 0.0081 | 0.0467 | 0.0738 | 0.1820 | 0.0164 | 0.1208 | 0.0277 | 0.1791 | 0.0719 | 0.0638 |
| InversionNet | 0.0131 | 0.0351 | 0.0685 | 0.1497 | 0.0172 | 0.1055 | 0.0260 | 0.1646 | 0.0625 | 0.0689 |
| InvLINT | 0.0532 | 0.1621 | 0.0981 | 0.2462 | 0.0729 | 0.1522 | 0.0853 | 0.1955 | 0.1002 | 0.0835 |

- **Full-data SSIM:**

| Model | FVA | FVB | CVA | CVB | FFA | FFB | CFA | CFB | SA | SB |
|---|---|---|---|---|---|---|---|---|---|---|
| Auto-Linear | 0.9888 | 0.9044 | 0.8057 | 0.6169 | 0.9701 | 0.6868 | 0.9426 | 0.5672 | 0.8423 | 0.7275 |
| InversionNet | 0.9895 | 0.9461 | 0.8074 | 0.6727 | 0.9766 | 0.7208 | 0.9566 | 0.6136 | 0.8859 | 0.6314 |

- **Interpretation:** With full data, Auto-Linear is not better than InversionNet on most families. It wins with little paired data. At 1/10 of the paired data, FVA MAE is 0.0361 vs. 0.0590 for InversionNet, and FFA is 0.0423 vs. 0.0544.

**Neural operators — SA-EMO (Wang et al., arXiv 2511.11627, Nov 2025)** — [arXiv 2511.11627](https://arxiv.org/html/2511.11627)
- **Architecture:** DINOv3 encoder maps 5x1000x70 to a 70x70 latent, followed by a mixture of operator experts (FNO, WNO, MNO, LNO) with top-2 routing. Loss is a hybrid of L1-L2, gradient and Fourier-magnitude terms.
- **MAE, "Encoder + best neural operator per type" (Table 4):**

| FVA | FVB | CVA | CVB | FFA | FFB | CFA | CFB | SA | SB | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0013 | 0.0071 | 0.0182 | 0.0534 | 0.0034 | 0.0280 | 0.0078 | 0.0815 | 0.0294 | 0.0338 | 0.0264 |

  - InversionNet average in the same table: 0.070.
  - The paper uses per-subset standardization, not OpenFWI's [-1, 1] min-max, so these MAEs are probably not on the same scale as OpenFWI Table 4.
  - Its Table 5 average (MAE 0.085) is inconsistent with the Table 4 average (0.0264).
- **Encoder ablation on CVB with an FNO backbone:** no encoder with raw input gives MAE/RMSE/SSIM 0.118 / 0.161 / 0.891. Adding the ViT-DINOv3 encoder gives 0.0598 / 0.1649 / 0.8637.

**Other later methods**
- **Fourier-DeepONet (Zhu, Feng, Lin, Lu; arXiv 2305.17289, 2023):** FNO used as the DeepONet decoder, with source parameters as inputs. It is trained on OpenFWI-derived datasets FWI-F, FWI-L and FWI-FL (varying source frequency, location, or both). A benchmark listing reports CurveVel-A FWI-FL MAE 0.0717 — [arXiv 2305.17289](https://arxiv.org/abs/2305.17289v2) (search-snippet level).
- **Inversion-DeepONet (Guo et al., arXiv 2408.08005, Aug 2024):** CNN branch net, source location and frequency in the trunk net, CNN decoder, on varied-source OpenFWI-derived data — [arXiv 2408.08005](https://arxiv.org/html/2408.08005v1) (abstract level).
- **SPAMoE (arXiv 2604.07421, 2026):** Spectrum-aware hybrid operator. It trains one model per subset, compares with InversionNet, VelocityGAN, UPFWI and FNO, and reports MAE/RMSE/PSNR — [arXiv 2604.07421](https://arxiv.org/pdf/2604.07421) (snippet only, numbers not retrieved).
- **DiffusionVel (Zhang, Li, Huang; arXiv 2410.21776, Oct 2024), Table 2 (MAE / "MSE" / SSIM):**

| Method | FVB | CVB | FFB | CFB |
|---|---|---|---|---|
| FWI | 0.0846 / 0.0262 / 0.7580 | 0.1760 / 0.0662 / 0.6019 | 0.0567 / 0.0094 / 0.8162 | 0.0800 / 0.0170 / 0.7689 |
| InversionNet | 0.0437 / 0.0092 / 0.9343 | 0.1695 / 0.0888 / 0.6604 | 0.2528 / 0.1172 / 0.5890 | 0.2424 / 0.1087 / 0.5918 |
| VelocityGAN | 0.0700 / 0.0147 / 0.8713 | 0.1651 / 0.0759 / 0.6659 | 0.2370 / 0.1009 / 0.5248 | 0.2252 / 0.0977 / 0.5611 |
| DiffusionVel | 0.0232 / 0.0044 / 0.9738 | 0.0983 / 0.0393 / 0.8331 | 0.0714 / 0.0172 / 0.8448 | 0.1280 / 0.0420 / 0.7094 |

  - Training sizes: FVB 23.5k, CVB 21.5k, FFB 24k, CFB 22.5k.
  - The re-trained InversionNet on FFB (MAE 0.2528) is far worse than OpenFWI's own 0.1055, so the baselines are not comparable to OpenFWI Table 4.
  - Source: [arXiv 2410.21776](https://arxiv.org/html/2410.21776v1)
- **Kaggle "Yale/UNC-CH Geophysical Waveform Inversion" competition (2025):**
  - Top solutions used large generated datasets: about 15x the official size (5th place), and about 13 TB generated (tascj repo).
  - 4th place used a ~100M-parameter custom ViT + 2D-conv model trained for about three weeks. 20th place used a CAFormer.
  - Sources: [5th place](https://www.kaggle.com/samson8); [4th place](https://www.kaggle.com/competitions/waveform-inversion/writeups/greysnow-4th-place-solution-for-the-gwi-competitio); [20th place](https://www.kaggle.com/competitions/waveform-inversion/writeups/cpmp-20th-solution-caformer-data-generation); [tascj repo](https://github.com/tascj/kaggle-waveform-inversion).
  - These details come from search snippets. The Kaggle pages did not render for fetching.

### Inferences
- On the OpenFWI scale, InversionNet's FVA MAE of 0.0131 is about 20 m/s and its CVB MAE of 0.1497 is about 225 m/s (×1500, assuming normalized metrics). The "hard" B families are where the headroom is.
- Swapping l1 for l2 changes results by a few percent with no consistent winner. The adversarial term (VelocityGAN) mostly improves CVA/CVB/FFB SSIM, but it destabilizes some families: VelocityGAN-l1 FFA MAE is 0.0868 vs. 0.0172 for InversionNet.
- The biggest documented jumps beyond the OpenFWI baselines come from four sources:
  - more data, either generated or combined across families;
  - discrete/tokenized or pretrained output representations (ViT-VQGAN, frozen masked autoencoders);
  - pretrained input encoders (DINOv3);
  - inference-time refinement (latent GD).
  - Loss-function changes alone account for less.
- Cross-paper comparisons should be restricted to papers that state the [-1, 1] normalized convention: OpenFWI, BigFWI, Auto-Linear, and the Large-Model-Era paper.

### Gaps
- BigFWI per-family MAE/RMSE/SSIM tables (Supplementary S2-S12) were not retrieved.
- I found no verified per-family OpenFWI table for a transformer (ViT/Swin) inversion network trained supervised from scratch. The only CNN-vs-ViT number found is UPFWI's ablation (Section 3).
- The Fourier-DeepONet, Inversion-DeepONet and SPAMoE per-family tables were not retrieved.
- The 1st-place Kaggle GWI write-up and final leaderboard MAEs were not retrieved. The Kaggle pages did not render.

## 2. Multi-view / multi-shot architectures (per-shot shared encoders, cross-shot fusion, shot-order invariance, reciprocity/flip tricks)

### Takeaway
I found no OpenFWI paper that encodes each of the five shots with a shared encoder and fuses them with attention or a permutation-invariant set operator. Every OpenFWI baseline stacks the shots as input channels. InversionNet3D's grouped convolution over sources is the closest published design. Its 3D results are almost insensitive to which 8 of 25 sources are used. Practical evidence on flip equivariance comes from the 2025 Kaggle competition: exact flipped training samples beat naive flip augmentation, because the OpenFWI-style source layout is symmetric except for the central shot. Background velocity as an extra "view" helps strongly in DiffusionVel, but that background is an oracle low-pass of the true model.

### Cited Findings
- **Shot stacking as channels:** OpenFWI 2D seismic tensors are shaped (#source x time x receivers) = 5x1000x1x70, and the baselines consume them as a 5-channel input — [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926). SeisBind (2026) likewise "treats five shot channels as one tensor" rather than encoding shots separately — [SeisBind, arXiv 2601.14607](https://arxiv.org/html/2601.14607) (search-snippet level).
- **InversionNet3D (Zeng, Feng, Wohlberg, Lin; IEEE TGRS):** Treats the 25 per-source records as input channels. It uses group convolution in the encoder to learn from multiple sources with fewer parameters and operations, and invertible layers to cut activation memory. The supplement includes an illumination analysis for the selected-source strategy — [arXiv 2103.14158](https://arxiv.org/abs/2103.14158v3).
- **InversionNet3D source-subset results:** Three different 8-of-25 source subsets give l1 MAE 0.0108 / 0.0105 / 0.0107 and SSIM 0.9838 / 0.9838 / 0.9835 — [OpenFWI Table 6](https://ar5iv.labs.arxiv.org/html/2111.02926).
- **Kaggle GWI 2025, flip augmentation:**
  - Flip augmentation was "the simplest" and appeared in most public notebooks [5th place](https://www.kaggle.com/samson8).
  - 4th place (greySnow) started with horizontal flip plus TTA. They found exact flipped samples could be generated cheaply because "only the central seismic data changes", and training on exact flips beat flip augmentation. They then ensembled a no-flip model with an all-flipped model [4th place](https://www.kaggle.com/competitions/waveform-inversion/writeups/greysnow-4th-place-solution-for-the-gwi-competitio).
  - A community baseline found horizontal flipping the most harmful augmentation. It attributed this to fixed, asymmetric source positions [fancyleo baseline](https://github.com/fancyleo/Geophysical-Waveform-Inversion).
  - All three points are from search snippets.
- **Kaggle GWI 2025, other augmentations:** 20th place combined flipping with velocity scaling (random alpha in [0.8, 1.2]) and pseudo-labelled data generated from test predictions [20th place](https://www.kaggle.com/competitions/waveform-inversion/writeups/cpmp-20th-solution-caformer-data-generation). 5th place used per-pixel learning rates proportional to predicted velocity, rescaled seismic amplitudes to counter vanishing gradients, and a family classifier with per-family ensembles [5th place](https://www.kaggle.com/samson8). Both from search snippets.
- **Source parameters as conditioning:** Fourier-DeepONet takes source parameters as an input so one network handles varying source frequency and location (FWI-F, FWI-L, FWI-FL datasets derived from OpenFWI). The authors report more accurate predictions across source parameters and better robustness to noise and missing traces than earlier data-driven FWI — [arXiv 2305.17289](https://arxiv.org/abs/2305.17289v2). Inversion-DeepONet puts source location and frequency in the trunk net, noting that OpenFWI sources are at fixed locations with identical frequencies — [arXiv 2408.08005](https://arxiv.org/html/2408.08005v1).
- **Multi-shot designs outside OpenFWI:**
  - MFNSR (IEEE, 2023) adds a "shot-record attention" module to recover medium-to-low-frequency information — [IEEE 10308616](https://ieeexplore.ieee.org/document/10308616/).
  - VIF-Net maps multi-shot gathers to velocity and interface maps, then fuses them — [VIF-Net](https://www.sciencedirect.com/science/article/abs/pii/S0098300424003170).
  - SeismoLabV3+ (DeepLabV3+ with ResNeXt50) on the ThinkOnward 5-channel shot-gather data reached MAPE 0.031246 on the hidden test set — [arXiv 2509.21331](https://arxiv.org/abs/2509.21331).
  - None of these is a permutation-invariant per-shot encoder.
- **Multi-information fusion at sampling time (DiffusionVel):** Separate diffusion models for seismic data (conditional), well logs (conditional) and geology (unconditional) are combined by weighted summation during sampling. Background velocity enters through a low-pass substitution, m_seis + m_back − m_low — [DiffusionVel](https://arxiv.org/html/2410.21776v1).
- **DiffusionVel background-kernel ablation (MAE / MSE / SSIM):**

| Dataset | Kernel 49 | Kernel 19 | Kernel 9 |
|---|---|---|---|
| FFB | 0.0481 / 0.0122 / 0.8461 | 0.0426 / 0.0106 / 0.8537 | 0.0335 / 0.0076 / 0.8820 |
| CFB | 0.0987 / 0.0300 / 0.7060 | 0.0925 / 0.0288 / 0.7027 | 0.0643 / 0.0167 / 0.7876 |

  - A smaller Gaussian kernel means a more detailed background.
  - The background is a Gaussian-filtered true model, i.e. an oracle.
  - Source: [DiffusionVel Table 3](https://arxiv.org/html/2410.21776v1)
- **Other augmentation precedents:** A 2023 Sensors paper on learning-based velocity inversion used horizontal flipping, random bandpass filtering and a 2D wave-propagation correction, but not reciprocity — [MDPI Sensors 23(19) 8277](https://www.mdpi.com/1424-8220/23/19/8277). A 2025 GJI paper reports that noise-augmented training gives more stable field-data outputs across neighbouring positions — [GJI ggaf429](https://academic.oup.com/gji/article/244/1/ggaf429/8304015). Both from search snippets.

### Inferences
- **Mirror symmetry is an available equivariance.** Mirroring the velocity map maps shot i to shot (4−i) with receiver order reversed. If the five source columns are not exactly mirror-symmetric on the 70-cell grid (as the Kaggle 4th-place note implies), the mirror is exact for four shots and needs one re-simulated central shot. A model that treats shots as a set with per-shot position embeddings can build this symmetry in rather than learn it from augmentation.
- **Per-shot processing is cheap and appears to cost no accuracy.** The InversionNet3D subset invariance (Table 6) suggests a per-shot shared encoder with pooled or attention fusion and source-position embeddings would lose little. It would also allow shot dropout and subset training as augmentation. No 2D OpenFWI ablation of this exists.
- **Reciprocity cannot be used directly on OpenFWI.** OpenFWI has 5 sources and 70 receivers, so reciprocity only gives sparse "virtual source" gathers at receiver positions. It is not an exact augmentation without re-simulation.
- **Background velocity is a powerful conditioning view** (DiffusionVel kernel ablation). For a fair test, backgrounds should come from a non-oracle procedure: a network prediction, a smoothed FWI result, or several perturbed or smoothed candidates. Multiple backgrounds as views would then also give a natural ensemble.

### Gaps
- No published OpenFWI result was found for:
  - a per-shot shared encoder with cross-shot attention or a DeepSets/Set-Transformer fusion;
  - shot-order permutation invariance;
  - shot-dropout / variable-shot-count training;
  - reciprocity-based augmentation.
- The 2023 Geophysics "Seismic velocity inversion transformer" (J. Yang) appeared only in a reference list. Its tokenization (per-shot or not) is unverified.
- Quantitative gains from exact flips in the Kaggle competition were not retrieved (snippets only).

## 3. Physics-consistency losses (UPFWI, physics-guided/hybrid, cycle-consistency): benefit and cost

### Takeaway
UPFWI (Jin et al., ICLR 2022) closes the CNN-PDE loop: CNN → velocity → differentiable FD solver → seismic, with pixel + VGG perceptual loss on the data. In its own paper, using 48K unlabeled samples, it matched or beat supervised InversionNet on FlatFault and CurvedFault. In the OpenFWI benchmark, however, UPFWI is clearly worse than InversionNet on every family and costs about 480x the GPU-hours. Its cross-family SSIM averages better than InversionNet's. The perceptual loss on seismic data was the single biggest ablation gain. No formally named cycle-consistency loss was found on OpenFWI. Newer physics use moves the PDE out of training and into inference (Sections 5-6).

### Cited Findings

**UPFWI (Jin et al., ICLR 2022), Table 1 (MAE / MSE / SSIM)** — [UPFWI, arXiv 2110.07584](https://arxiv.org/html/2110.07584v2)

| Method | FlatFault | CurvedFault |
|---|---|---|
| InversionNet | 15.83 / 2156.00 / 0.9832 | 23.77 / 5285.38 / 0.9681 |
| VelocityGAN | 16.15 / 1770.31 / 0.9857 | 25.83 / 5076.79 / 0.9699 |
| H-PGNN+ | 12.91 / 1565.02 / 0.9874 | 24.19 / 5139.60 / 0.9685 |
| UPFWI-24K | 16.27 / 1705.35 / 0.9866 | 29.59 / 5712.25 / 0.9652 |
| UPFWI-48K | 14.60 / 1146.09 / 0.9895 | 23.56 / 3639.96 / 0.9756 |

- **Setup notes:**
  - MAE and MSE are computed after denormalizing to the original velocity scale. SSIM is computed on normalized maps.
  - The dataset is an earlier version than final OpenFWI: 15 m grid, 25 Hz Ricker, 70x70, 5 sources, 70 receivers, 1000 steps at 1 ms.
  - Splits: 24K labeled training samples for the supervised baselines, and 24K or 48K unlabeled samples for UPFWI.
- **Forward operator:** Finite differences, 2nd order in time and 4th order in space, with absorbing boundaries. It is fully differentiable and inside the training loop. Training used 8x V100 — [UPFWI](https://arxiv.org/html/2110.07584v2).
- **Loss ablation (FlatFault, 24K unlabeled; velocity MAE / MSE / SSIM)** — [UPFWI Table 2](https://arxiv.org/html/2110.07584v2)

| Loss on seismic data | MAE / MSE / SSIM |
|---|---|
| Pixel l2 only | 32.61 / 10014.47 / 0.9735 |
| Pixel l1 + l2 | 21.71 / 2999.55 / 0.9775 |
| Pixel l1 + l2 + perceptual (VGG-16 conv5, ImageNet-pretrained) | 16.27 / 1705.35 / 0.9866 |

  - All loss weights were set to 1.
- **Robustness:**
  - Gaussian noise: MSE rises from 1343.21 (61.60 dB) to 4010.78 (58.70 dB) to 7592.57 (51.58 dB).
  - Missing traces: 5% gives MSE 1772.05, 10% gives 3504.25, 25% gives 16731.69.
  - On Marmousi, UPFWI MAE is 221.93 vs. 149.67 for InversionNet. On Salt (120 training samples), it is 150.34 vs. 25.98.
  - Source: [UPFWI](https://arxiv.org/html/2110.07584v2)
- **Stated limitations:** High compute and memory cost, because FD states must be stored for backpropagation. Weaker results on maps where adjacent layers have very close velocities — [UPFWI](https://arxiv.org/html/2110.07584v2).
- **OpenFWI benchmark contradiction:** In OpenFWI Table 4, UPFWI is worse than InversionNet-l1 on every 2D family. For example, CFB is 0.3452 / 0.5010 / 0.3941 vs. 0.1646 / 0.2477 / 0.6163, and SA is 0.1429 vs. 0.0625 MAE. UPFWI was not run on Kimberlina-CO2 "because of its high computational cost". It also takes 30-60 h on 32 GPUs vs. 2-5.5 h on 1 GPU for InversionNet — [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926).
- **OpenFWI Appendix L** compares physics-driven and data-driven FWI. Its content was in the truncated part of the page and was not retrieved — [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926).
- **Related physics-loss designs (search-level only):**
  - FWIGAN (Yang & Ma, 2021) couples the acoustic wave equation with a WGAN-GP critic that compares observed and simulated data.
  - The physics-guided Encoder-Solver CNN (arXiv 2405.17696; Inverse Problems 2024) is another hybrid — [arXiv 2405.17696](https://arxiv.org/html/2405.17696v1).
  - A physics-guided deep autoencoder pretrains a UNet with simultaneous sources and uses a time-domain 2D acoustic wave equation as a self-supervised constraint.
  - Numbers for these were not retrieved.
- **Auto-Linear as a "cycle":** The same frozen seismic and velocity autoencoders, linked by linear maps, give joint forward and inverse models. Forward-modeling results on the Fault family (MAE) — [Auto-Linear Table 4](https://arxiv.org/html/2305.13314v3):

| Dataset | Auto-Linear | InversionNet-style |
|---|---|---|
| FFA | 0.0099 | 0.0332 |
| FFB | 0.0193 | 0.0397 |
| CFA | 0.0132 | 0.0373 |
| CFB | 0.0253 | 0.0594 |

### Inferences
- **UPFWI cost:** About 960 GPU-hours (30 h x 32 GPUs) vs. 2 GPU-hours for InversionNet on the Vel family, roughly 480x. This is my arithmetic from OpenFWI Table 5.
- **Where the benefit comes from:** The physics loss mainly helps by allowing unlabeled data to be used (24K → 48K gave the gain). It does not by itself beat supervision at equal data.
- **Perceptual loss on the data side:** The VGG perceptual loss on seismic data cut velocity MSE by about 43% (2999.55 → 1705.35). For any data-misfit term in a hybrid loss, a feature-space misfit is much better behaved than raw l2. That is consistent with the cycle-skipping-mitigating feature losses in Section 5.
- **Cheaper alternative:** A learned surrogate forward model gives a cheap "cycle" term without the FD cost. Examples are the Auto-Linear forward converter, or the user's own fast forward model if available. No measured OpenFWI benefit for a surrogate-cycle loss was found.

### Gaps
- No OpenFWI result was found for a supervised + physics hybrid loss (label loss plus differentiable-simulation data loss) on the final OpenFWI datasets, compared against pure supervision at matched data.
- No explicit cycle-consistency (data → model → data) loss with a learned forward network was benchmarked on OpenFWI.
- The H-PGNN+ method details were not retrieved.

## 4. Structural / perceptual / adversarial losses on velocity maps

### Takeaway
Evidence is thin and mostly indirect. Adversarial loss (VelocityGAN, WGAN-GP) gives small in-distribution SSIM gains on several families. It also gives the clearest measured generalization benefit among the OpenFWI baselines: average off-diagonal cross-family SSIM 0.583 vs. 0.404 for InversionNet. It costs about 4-6x training time and is occasionally unstable (FFA with l1). Edge, gradient and frequency losses (TU-Net, DCNet, SA-EMO hybrid spatial-spectral loss) report better boundaries, but no clean OpenFWI ablation numbers were retrieved. No OpenFWI paper training directly with an SSIM or LPIPS loss on velocity was found.

### Cited Findings
- **VelocityGAN vs. InversionNet, l1 SSIM:** VelocityGAN is higher on FVA, FVB, CVA (0.8624 vs. 0.8074), CVB (0.7111 vs. 0.6727), FFB, CFA, SA and SB (0.6953 vs. 0.6314). It is lower on FFA (0.9313 vs. 0.9766) and CFB (0.5996 vs. 0.6163) — [OpenFWI Table 4](https://ar5iv.labs.arxiv.org/html/2111.02926).
- **VelocityGAN cost:** Training time is 8.6 / 16 / 30 h vs. 2 / 4 / 5.5 h for InversionNet (Vel / Fault / Style) — [OpenFWI Table 5](https://ar5iv.labs.arxiv.org/html/2111.02926).
- **Cross-family generalization (OpenFWI Tables 9-11):** Entries are SSIM on unseen families. Examples:
  - FVB→FVA (row → column): VelocityGAN 0.94, InversionNet 0.42.
  - CFA row: InversionNet values are 0.05-0.15 except STA/STB, while VelocityGAN is 0.34-0.92.
  - Source: [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926)
- **Interfaces and UPFWI:** The OpenFWI supplement notes that UPFWI's data-difference loss makes it sensitive to layer boundaries, the causes of reflections — [OpenFWI NeurIPS supplement](https://papers.nips.cc/paper_files/paper/2022/file/27d3ef263c7cb8d542c4f9815a49b69b-Supplemental-Datasets_and_Benchmarks.pdf) (search snippet).
- **TU-Net (Computers & Geosciences, 2025):** Uses a mixed pixel + edge loss (MAE + MSE + edge term) and texture-warping modules in U-Net skip connections. It reports better MSE, MAE, LPIPS, PSNR, UIQ and SSIM than four baselines on OpenFWI, SEG salt and Marmousi II — [ScienceDirect S0098300425001785](https://www.sciencedirect.com/science/article/pii/S0098300425001785) (abstract level, numbers not retrieved).
- **DCNet (Journal of Applied Geophysics, 2025):** Difference convolution plus a focal frequency loss. It reports clearer boundaries on OpenFWI, SEG and Marmousi II — [ScienceDirect S0926985125001430](https://www.sciencedirect.com/science/article/abs/pii/S0926985125001430) (abstract level).
- **SA-EMO loss:** Hybrid spatial-spectral loss of L1-L2, gradient and Fourier-magnitude terms. The paper claims about 10.308% better boundary resolution, but the page does not show how this was computed — [SA-EMO](https://arxiv.org/html/2511.11627).
- **Kaggle 4th place:** Loss was MAE plus a confidence term, confidence loss = MAE(confidence, MAE(targets, preds)). The network learns to predict its own per-pixel error — [4th place](https://www.kaggle.com/competitions/waveform-inversion/writeups/greysnow-4th-place-solution-for-the-gwi-competitio) (snippet).
- **Loss choice in BigFWI:** l1 + l2 pixel loss only. The paper reports SSIM is computed on [0, 1] rescaled maps — [BigFWI](https://ar5iv.labs.arxiv.org/html/2307.15388).

### Inferences
- **Generalization averages:** I averaged all 90 off-diagonal entries of OpenFWI Tables 9-11 (my computation). Mean cross-family SSIM is 0.404 for InversionNet, 0.583 for VelocityGAN and 0.518 for UPFWI. The average is independent of the row/column convention. Adding a distribution-level loss (adversarial) or a data-space physics loss both improved out-of-family structure relative to pure pixel regression. The adversarial gain was larger and about 30x cheaper than UPFWI in GPU-hours.
- **Interface sharpness:** For sharp faults and interfaces, gradient or Fourier terms on velocity, adversarial terms, and edge losses are the tried options. No paper isolates their effect on OpenFWI B-families with matched data, so this would need the user's own ablation.

### Gaps
- No OpenFWI ablation was found of SSIM-loss, LPIPS/VGG-perceptual-loss or gradient-loss on velocity vs. pure l1 under the same architecture and data.
- TU-Net and DCNet per-family numbers were not retrieved.
- No quantitative "interface sharpness" metric (e.g., boundary F-score) appears in the OpenFWI-standard evaluation. Most papers rely on SSIM plus visuals.

## 5. Test-time physics refinement (network → FWI, unrolled/learned iterations, deep image prior / reparameterized FWI)

### Takeaway
Physics at inference consistently improves accuracy when the starting point is good. It fails or stalls when the starting point is cycle-skipped. Documented recipes include:
- latent-space gradient descent on a pretrained generator (Large-Model-Era paper, normalized MAE 0.0159 → 0.0136);
- network-reparameterized FWI (NNFWI, DR-FWI: Marmousi2 MAPE 6.71% → 4.17%);
- feature-space misfits that fix the background before l2 (foundation-model-assisted FWI: 3D Overthrust relative error 0.266 → 0.139 → 0.056, where plain l2 stalls at 0.240).
I found no OpenFWI-standard table of "InversionNet prediction → conventional FWI".

### Cited Findings
- **Latent-space gradient descent at inference (Large-Model-Era paper):**
  - OpenFWI aggregate normalized MAE goes from 0.0159 to 0.0136.
  - Zero-shot ensemble MAE goes from 148.35 to 139.81 across 26 OOD maps.
  - The authors note latent GD changes visual appearance little; its benefit is mainly numerical.
  - Source: [arXiv 2603.00377](https://arxiv.org/html/2603.00377v2)
- **NNFWI (Zhu, Xu, Darve, Biondi, Beroza; arXiv 2012.11149, 2020-2021):** Represents velocity with a generative network and feeds it to the standard PDE solver, with autodiff through both. Network spatial correlations act as regularization that suppresses gradient noise and helps avoid local minima. It reports better results on Marmousi and 2004 BP, especially high-contrast salt, and under noise. MC dropout gives uncertainty. Only qualitative claims are in the abstract — [arXiv 2012.11149](https://arxiv.org/abs/2012.11149).
- **DR-FWI benchmarking (Liu, Li, Su, Huang, Bai; arXiv 2504.17375; Petroleum Science 2026):**

| Model | Conventional FWI (MAPE % / SSIM) | DR-FWI CNN (MAPE % / SSIM) |
|---|---|---|
| Marmousi2 | 6.71 / 0.55 | 4.17 / 0.70 |
| Overthrust | 3.91 / 0.66 | 2.38 / 0.81 |
| Foothill | 1.63 / 0.84 | 1.37 / 0.87 |

  - Warm-up pretraining of the network to reproduce the initial model beat additive "perturbation learning" for every architecture.
  - Costs: GPU dependence, and the memory of autodiff FWI for 3D.
  - Source: [arXiv 2504.17375](https://arxiv.org/html/2504.17375v2)
- **Self-reinforced DIP-FWI (arXiv 2512.08284, Dec 2025):** Updates both the network input and the weights to address artifacts and instability of fixed-input DIP-FWI — [arXiv 2512.08284](https://arxiv.org/abs/2512.08284v1) (abstract level).
- **Foundation-model-assisted FWI (Alfarhan, Ravasi, Chen, Turkiyyah, Keyes; arXiv 2608.05763, Aug 2026):**
  - **Method:** The misfit is the squared distance between frozen SeisLM (Wav2Vec2-style) features of modeled and observed traces, used inside adjoint-state FWI.
  - **3D Overthrust:** l2-only goes 0.266 → 0.240 relative error and stalls. A SeisLM stage-1 (100 iterations) reaches 0.139, then l2 refinement (300 iterations) reaches 0.056.
  - **Failure mode:** Adding l2 too early (hybrid) re-introduces cycle-skipping sensitivity; on 2D Overthrust the hybrid did worse than feature-only.
  - **Cost:** About 78 s per 2D Marmousi gradient (bf16), of which 14 s is the feature loss.
  - **Scope:** OpenFWI is not used.
  - Source: [arXiv 2608.05763](https://arxiv.org/html/2608.05763)
- **PreAdaptFWI (Dong et al., arXiv 2502.11913, Feb 2025):** Network pretrained on a simple initial model, then updated by FWI gradients with an adaptive residual module. It claims robustness to missing low frequencies, noise and different initial models on two benchmarks. No numbers in the abstract — [arXiv 2502.11913](https://arxiv.org/abs/2502.11913).
- **DiffusionVel's FWI baseline:** Conventional FWI was started from the true model Gaussian-smoothed with kernel 25. With that oracle start, FWI beat all learned methods on FFB and CFB MAE (FFB 0.0567 vs. DiffusionVel 0.0714; CFB 0.0800 vs. 0.1280). It lost badly on FVB and CVB — [DiffusionVel](https://arxiv.org/html/2410.21776v1).
- **Diffusion-prior physics refinement (PSG, Min, Jiang, Ma, Yan; arXiv 2609.12899, Sep 2026):**
  - **Setup:** 100 test samples per family. Initial model is the oracle (truth Gaussian-smoothed, sigma = 10). Metrics in m/s.

| Method | FVB MAE / RMSE / SSIM | FFB | CVB | CFB |
|---|---|---|---|---|
| FWI (300 iterations) | 214.9 / 338.7 / 0.681 | 175.0 / 269.0 / 0.696 | 350.6 / 531.5 / 0.522 | 263.7 / 400.8 / 0.603 |
| TV-FWI | 198.3 / 320.7 / 0.741 | 156.7 / 247.7 / 0.783 | 328.3 / 507.0 / 0.580 | 241.3 / 372.3 / 0.669 |
| DPS | 671.2 / 895.7 / 0.556 | 469.8 / 649.2 / 0.536 | 718.9 / 976.8 / 0.387 | 495.5 / 698.8 / 0.421 |
| RED-DiffEq | 167.0 / 280.4 / 0.797 | 135.3 / 216.7 / 0.841 | 263.7 / 418.9 / 0.673 | 174.1 / 276.2 / 0.774 |
| DiffusionFWI | 144.3 / 249.3 / 0.830 | 141.2 / 243.1 / 0.798 | 300.0 / 490.5 / 0.634 | 200.1 / 332.0 / 0.715 |
| PSG | 110.2 / 209.6 / 0.880 | 84.9 / 166.4 / 0.904 | 188.1 / 357.3 / 0.789 | 112.0 / 219.1 / 0.860 |

  - **Cost:** PSG needs 1000 reverse steps with one physics update each, about 547% of FWI's core iteration time, and about 4.4 GiB peak memory on an RTX 5090.
  - Source: [arXiv 2609.12899](https://arxiv.org/html/2609.12899)
- **Robust physics-guided diffusion (Peng, Jiang, Ma, Yan; arXiv 2603.16393, Mar 2026):** Uses a 1D-W2 optimal-transport data misfit plus preconditioned guidance (OT-WE-PDPS). Relative l2 error, proposed vs. DPS:
  - CVA 3.50% vs. 23.15%
  - CVB 2.04% vs. 15.70%
  - FFA 2.91% vs. 16.68%
  - FFB 6.13% vs. 13.54%
  - CFA 2.41% vs. 10.72%
  - CFB 5.32% vs. 21.82%
  - SSIM: CVB 0.9517, CFB 0.7956.
  - Initial models and solve counts were not visible in the retrieved text.
  - Source: [arXiv 2603.16393](https://arxiv.org/html/2603.16393)
- **Unrolled / learned iterative FWI:** No learned gradient descent or RIM-style unrolled network with OpenFWI numbers was retrieved in this pass. See Gaps.

### Inferences
- **Recurring failure mode:** Physics refinement is only as good as its starting model. Plain l2 stalls from linear or smooth starts. Unguided DPS is worse than plain FWI in the PSG table. The PSG and DiffusionVel FWI baselines rely on oracle-smoothed truth. A network prediction is the natural non-oracle replacement for those starts, but no OpenFWI paper reports that end-to-end pipeline with standard metrics.
- **Cycle-skip-robust misfits:** OT/W2 data misfit, feature-space misfit, or a perceptual misfit (UPFWI) used first, followed by l2, is the documented way to reduce cycle-skip risk during refinement.
- **Cost to budget:** Test-time refinement on 70x70 OpenFWI costs about 300-1000 PDE-coupled iterations per sample, against about 1 ms for a feed-forward network.

### Gaps
- No OpenFWI-standard table of "InversionNet/VelocityGAN prediction as FWI initial model vs. smooth initial model" was found.
- No learned-gradient-descent, RIM or unrolled-FWI results on OpenFWI were retrieved.
- The PSG baselines on OOD models (Marmousi, Overthrust, Salt) appear only in figures.
- NNFWI quantitative results were not retrieved (abstract only).

## 6. Uncertainty-aware learning (heteroscedastic outputs, diffusion posterior sampling, ensembles)

### Takeaway
Uncertainty work on OpenFWI is mostly at the level of "the spread is large where the error is large". The documented methods are:
- MC dropout in OpenFWI: uncertainty-to-edge-gradient correlation 0.5462;
- ensemble or stochastic diffusion sampling (PSG: variability concentrates at faults and interfaces and correlates with error);
- conditional normalizing flows (WISE: standard deviation correlates with error);
- a learned per-pixel confidence head (Kaggle 4th place).
No OpenFWI paper reports formal calibration metrics (coverage, NLL, ECE).

### Cited Findings
- **OpenFWI MC dropout (p = 0.2, InversionNet, CurveVel-A):** Mean variance rises with test noise, from 1.3903e-3 at sigma 0 to 1.5621e-3 at sigma 5e-4. Applying the CVA-trained model to other families raises mean variance up to 4.2857e-3 (FlatFault-A). Correlation between uncertainty and edge gradient magnitude is 0.5462 (Pearson) — [OpenFWI Tables 12-13](https://ar5iv.labs.arxiv.org/html/2111.02926).
- **PSG ensembles:** Repeated stochastic runs keep the dominant structures, and ensemble means stay close to the truth. Variability concentrates near faults and interfaces and correlates positively with local error — [PSG, arXiv 2609.12899](https://arxiv.org/html/2609.12899).
- **Taufik & Alkhalifah (KAUST, arXiv 2512.12797):** Diffusion posterior sampling with simultaneous-source FWI and Langevin refinement under a wave-equation likelihood. It reports lower model error and better data fit than an SVGD baseline at much lower cost. Tested on Overthrust, Salt, SEAM Arid and field data, not OpenFWI — [arXiv 2512.12797](https://arxiv.org/abs/2512.12797); related [GJI 2026](https://academic.oup.com/gji/article/245/2/ggag066/8487641).
- **WISE (arXiv 2401.06230):** Conditional normalizing flows for variational inference with subsurface extensions. The authors report standard deviations that meaningfully correlate with errors — [arXiv 2401.06230](https://arxiv.org/html/2401.06230v1) (search-snippet level).
- **Robust physics-guided diffusion:** The authors focus on reconstruction rather than full UQ — [arXiv 2603.16393](https://arxiv.org/html/2603.16393).
- **DiffusionVel:** Despite being generative, it does not evaluate multiple samples or uncertainty — [DiffusionVel](https://arxiv.org/html/2410.21776v1).
- **Kaggle 4th place:** A learned confidence output trained to regress the per-pixel absolute error, MAE(confidence, |target − pred|) — [4th place](https://www.kaggle.com/competitions/waveform-inversion/writeups/greysnow-4th-place-solution-for-the-gwi-competitio) (snippet).
- **NNFWI:** MC dropout inside a reparameterized FWI gives uncertainty maps — [arXiv 2012.11149](https://arxiv.org/abs/2012.11149).
- **Large-Model-Era paper:** Uses an ensemble for zero-shot evaluation (MAE 148.35 without latent GD) — [arXiv 2603.00377](https://arxiv.org/html/2603.00377v2).

### Inferences
- The consistent signal across MC dropout, diffusion ensembles and the Kaggle confidence head is that predicted uncertainty localizes at interfaces and faults. That is where the B-family errors are.
- A multi-view model gives a cheap uncertainty source by construction. Disagreement across per-shot or per-background predictions before fusion could be evaluated against error in the same way (Pearson correlation, coverage).

### Gaps
- No OpenFWI paper was found with heteroscedastic Gaussian/Laplace NLL training of an InversionNet-type network and calibration metrics.
- No deep-ensemble study on OpenFWI with reported calibration was found.
- The DiffusionInv (arXiv 2505.03138) uncertainty results were not retrieved.

## 7. Generalization across families (cross-family transfer and what improves OOD)

### Takeaway
Single-family training generalizes poorly. Cross-family SSIM is often 0.2-0.6 for InversionNet. Documented improvements, roughly in order of measured effect:
- Train on all families (BigFWI leave-one-out: +28.6% MAE, +8.22% SSIM).
- Scale data with generated velocity maps and use tokenized/pretrained representations. The 1B transformer reaches zero-shot SSIM 0.7669 vs. 0.5579 for BigFWI-L on 26 classic models.
- Adversarial or physics losses (mean off-diagonal SSIM 0.583 VelocityGAN, 0.518 UPFWI, 0.404 InversionNet).
- Self-supervised pretrained encoders (Auto-Linear: better with 1/10-1/20 of paired data and better noise robustness).
- Conditioning on acquisition parameters (Fourier-DeepONet, for source-parameter shifts).
Physics refinement at test time also helps OOD, at much higher cost (PSG on Marmousi, Overthrust, Salt).

### Cited Findings
- **OpenFWI cross-family SSIM (Tables 9-11):** Some InversionNet transfers collapse, e.g. the CFA row is 0.05-0.15 for most targets. The authors note lower-triangle entries are generally larger than upper ones, with a few InversionNet outliers — [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926).
- **BigFWI:**
  - Leave-one-out (train on 9 families, test on the held-out one) beats every single-dataset InversionNet on the held-out target: average MAE +28.60%, RMSE +21.55%, SSIM +8.22%.
  - On Marmousi and Overthrust, BigFWI generally beats InversionNet, but the authors call performance still insufficient for real use.
  - Source: [BigFWI](https://ar5iv.labs.arxiv.org/html/2307.15388)
- **Large-Model-Era zero-shot results (26 maps from 2004 BP, Marmousi, SEAM I, Overthrust, Salt, Sigsbee):**

| Model | Overall MAE | Overall RMSE | Overall SSIM |
|---|---|---|---|
| Proposed | 139.81 | 241.78 | 0.7669 |
| BigFWI-M | 199.20 | 312.15 | 0.5844 |
| BigFWI-L | 210.99 | 327.79 | 0.5579 |

  - MAE/RMSE appear to be in m/s.
  - BigFWI-M is still better on SEAM I (185.02 vs. 199.44 MAE) and Salt (150.13 vs. 155.22).
  - Source: [arXiv 2603.00377](https://arxiv.org/html/2603.00377v2)
- **Auto-Linear generalization and robustness:**
  - Fault-family encoders/decoders reused on other families give MAE FVA 0.0073 vs. InversionNet 0.0131, SB 0.0646 vs. 0.0689. They are worse on FVB (0.0570 vs. 0.0351) and CVB (0.1804 vs. 0.1497).
  - Noise robustness on CFA (test-only noise): at sigma² = 1e-4, MAE is 0.0630 vs. 0.1439 for InversionNet. At 5e-4, it is 0.1093 vs. 0.4496.
  - Source: [Auto-Linear](https://arxiv.org/html/2305.13314v3)
- **OpenFWI noise training (Table 15, partial):** Noise-trained InversionNet does better on noisy input. VelocityGAN and UPFWI drop slightly when test noise is lower than training noise — [OpenFWI](https://ar5iv.labs.arxiv.org/html/2111.02926).
- **Fourier-DeepONet:** Targets the field-relevant shift in which the survey source function differs from training. Source parameters are inputs, and the authors report better accuracy across a wide range of source parameters plus robustness to noise and missing traces — [arXiv 2305.17289](https://arxiv.org/abs/2305.17289v2).
- **SA-EMO:** Claims OOD generalization (Marmousi2) from a pretrained DINOv3 encoder and an operator mixture. Quantitative OOD numbers were not visible — [SA-EMO](https://arxiv.org/html/2511.11627).
- **PSG OOD (oracle-smoothed initial model, sigma_init = 20):**
  - Marmousi RMSE 331.6 m/s, SSIM 0.745.
  - Overthrust RMSE 216.2 m/s, SSIM 0.825.
  - BP2004 Salt MAE 43.9, RMSE 123.1, SSIM 0.956.
  - Salt degrades to RMSE 404.7 and SSIM 0.770 with a smoother start (sigma_init 32).
  - Source: [PSG](https://arxiv.org/html/2609.12899)
- **Kaggle 2025:** Top teams generated many times the official data, used family classifiers with per-family ensembles, and pseudo-labelled the test set — [5th place](https://www.kaggle.com/samson8); [20th place](https://www.kaggle.com/competitions/waveform-inversion/writeups/cpmp-20th-solution-caformer-data-generation) (snippets).

### Inferences
- In the published record, data diversity dominates: combined families, generated maps, about 5M pairs. Architecture or loss changes alone are second-order in comparison.
- Among loss choices, adversarial loss gave the largest measured OOD gain on the OpenFWI cross-family tables, at moderate cost.
- **A multi-view design fits this evidence.** It combines:
  - per-shot shared encoders, which can be trained on variable shot subsets and positions;
  - background-velocity views drawn from a non-oracle distribution;
  - exact mirror augmentation;
  - optional test-time feature-space or OT physics refinement.
  Each component addresses a failure the literature documents: acquisition shift, background dependence, flip asymmetry, and the cycle-skipping of refinement.
- **Suggested evaluation protocol:** OpenFWI Table 4 metrics on the [-1, 1] normalized scale, OpenFWI Tables 9-11-style cross-family SSIM, BigFWI leave-one-out, and the 26-map zero-shot set from the Large-Model-Era paper. This would make results comparable with the strongest published baselines.

### Gaps
- No domain-adaptation (e.g., adversarial feature alignment) study with OpenFWI cross-family numbers was retrieved.
- BigFWI per-family leave-one-out tables (S8-S9) and Marmousi/Overthrust tables (S10-S12) were not retrieved.
- No study isolates the OOD effect of physics-derived input channels (e.g., migration images, background models, first-arrival picks) vs. raw gathers on OpenFWI. This is presumably covered by the parallel "physics-informed inputs" research thread.
