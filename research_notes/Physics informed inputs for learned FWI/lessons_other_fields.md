# Lessons from other fields: auxiliary physics/geometry per-pixel inputs, multi-view fusion, and losses in learned inverse problems (2016-2026)

Notation used below:
- **[single paper]** marks a claim that rests on one paper and has not been replicated in the sources I checked.
- **[conflict]** marks a point where sources disagree.
- PSNR is in dB. SSIM is unitless (1 = perfect). "Post-processing" means a CNN (usually a U-Net) applied to a classical reconstruction, for example FBPConvNet-style FBP + U-Net. "Unrolled" means a network that calls the forward operator and its adjoint inside its layers.

---

## Q1. Medical and computational imaging: physics-based reconstructions vs raw measurements; unrolled / learned iterative reconstruction

### Takeaway
The strongest results come from giving the network the forward operator and adjoint *inside* its layers (unrolled / learned primal-dual), rather than only a physics image such as FBP as input. Two numbers show this:
- On a CT human phantom, learned primal-dual beat FBP + U-Net post-processing by 2.2 dB, and the authors say the gap grows as data quality falls.
- In photoacoustics, unrolled nets were about 10x less sensitive to a change in sampling pattern.

Feeding an FBP image as an extra initial input to the unrolled net did not improve its final accuracy. Robustness to distribution shift is mixed and contested:
- Mismatches in SNR and preprocessing dominate.
- Hallucinations persist even in top unrolled methods.

### Cited Findings

**Learned primal-dual vs FBP + U-Net post-processing (Adler & Öktem 2018, IEEE TMI) [single paper]** — [arXiv 1707.06474](https://arxiv.org/abs/1707.06474)
- Ellipse phantoms, sparse-view low-dose CT (PSNR / SSIM / parameters):
  - FBP: 19.75 / 0.597
  - TV: 28.06 / 0.929
  - FBP + U-Net: 29.20 / 0.944
  - FBP + residual CNN denoiser: 32.38 / 0.972
  - Learned Gradient: 32.29 / 0.981
  - Learned PDHG: 28.32 / 0.909
  - Learned Primal: 36.97 / 0.986, 1.2·10^5 parameters
  - Learned Primal-Dual: **38.28 / 0.989**, 2.4·10^5 parameters
  - Source: [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)
- Human phantom, 512×512 (PSNR / SSIM):
  - FBP: 33.65 / 0.830
  - TV: 37.48 / 0.946
  - FBP + U-Net: 41.92 / 0.941
  - LPD, linear operator: **44.11 / 0.969**
  - LPD, non-linear operator: 43.91 / 0.969
  - In summary: +10.5 dB over FBP, +6.6 dB over TV, +2.2 dB over the U-Net denoiser. The U-Net improved PSNR over TV but *not* SSIM; LPD improved both.
  - Source: [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)
- Author explanation: "the learned post-processing [is] limited by the information content of the FBP while the Learned Primal-Dual algorithm works directly with data". The advantage over post-processing "increases with decreasing data quality". — [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)
- **The physics image as an extra input did not help the final result.** The authors initialized LPD with the pseudo-inverse (FBP) instead of zeros. This "marginally decreased training time, but did not give better final results", so they report zero initialization only. — [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)
- Cost: learned methods that call the forward operator were about 6x slower than FBP + U-Net. All learned methods were at least 2 orders of magnitude faster than TV. — [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)
- Using a non-linear (pre-log) forward model gave no notable gain over the linearized model. — [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)
- The authors observed over-smoothing, attributed it to the squared-L2 training loss, and suggested perceptual losses. — [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)

**Model-based (unrolled gradient) learning vs U-Net post-processing, limited-view 3D photoacoustic tomography (Hauptmann et al. 2018, IEEE TMI) [single paper]** — [ar5iv 1708.09832](https://ar5iv.labs.arxiv.org/html/1708.09832)
- Network input per iteration: the current image plus the data-fit gradient (an adjoint image), i.e. a physics-derived per-pixel channel recomputed every iteration.
- Sampling-pattern change at test time: relative error worsened by 0.5% for the unrolled DGD network but by 5% for the U-Net. The authors call the U-Net "much more sensitive to variations". — [Hauptmann et al. 2018](https://ar5iv.labs.arxiv.org/html/1708.09832)
- Tumour phantom (structures absent from training), relative error: DGD 0.4925, U-Net 0.6584, TV 0.4749. The U-Net "does not perform well with the new image structures". — [Hauptmann et al. 2018](https://ar5iv.labs.arxiv.org/html/1708.09832)
- Synthetic-to-real gap:
  - Training on clean segmented lung-CT vessels was "not sufficient for real data".
  - Before transfer training, in vivo PSNR / SSIM was 32.93 / 0.723.
  - The fix was to add simulated background to the training data, then fine-tune on just 20 real measurements. After fine-tuning: DGD 41.40 / 0.945 vs U-Net 40.81 / 0.933 (TV 38.05 / 0.912).
  - Source: [Hauptmann et al. 2018](https://ar5iv.labs.arxiv.org/html/1708.09832)
- Cost, 5 iterations: DGD 184 s vs U-Net 19.75 s, about 9x. The claimed "speed-up by 20" over TV is a projection from iteration counts, not a measured wall-clock result. — [Hauptmann et al. 2018](https://ar5iv.labs.arxiv.org/html/1708.09832)

**Raw-measurement-to-image learning (AUTOMAP) and instability**
- AUTOMAP (Zhu et al. 2018, Nature) learns a direct sensor-to-image mapping with the same architecture across tasks. It was trained on ImageNet photos and tested on MR. The authors report "superior immunity to noise" over hand-crafted reconstruction. — [Zhu et al. 2018](https://arxiv.org/pdf/1704.08841)
- Antun et al. 2020 (PNAS) tested six deep reconstruction networks, including AUTOMAP, and found three kinds of instability:
  - tiny perturbations cause severe artefacts;
  - small structures such as tumours can vanish;
  - "more samples may yield poorer performance".
  - Source: [Antun et al. 2020](https://arxiv.org/pdf/1902.05300)
- **[conflict]** Genzel, Macdonald & März (2022, IEEE TPAMI) found standard end-to-end networks "resilient against adversarial perturbations" and statistical noise. They compared against TV on Gaussian compressed sensing, Fourier/Radon recovery and fastMRI. — [arXiv 2011.04268](https://arxiv.org/abs/2011.04268)
- **[conflict]** Darestani, Chaudhari & Heckel (ICML 2021):
  - Trained *and* untrained (non-data-driven) methods are both vulnerable to adversarial perturbations.
  - Under distribution shift, methods tuned to one dataset degrade "in nearly the same way" whether trained or not.
  - State-of-the-art DL reconstruction beats traditional methods without losing robustness.
  - Source: [Darestani et al. 2021](https://arxiv.org/abs/2102.06103)

**Distribution shift in learned MRI**
- Knoll et al. (MRM 2019; online 2018), variational network [single paper]:
  - SNR deviations between training and test caused "substantial decreases" in quality; image contrast mattered less.
  - Training on heterogeneous data generalized across acquisition parameters.
  - Training only on synthetic non-MR images left residual aliasing, which fine-tuning on a small in-vivo subset removed.
  - Test set: 10 in-vivo knee scans from 2 sequences not used in training.
  - Source: [Knoll et al.](https://cris.fau.de/publications/262824564)
- fastMRI 2020 challenge (Muckley et al. 2021, IEEE TMI):
  - Scale: 19 submissions from 8 groups, 7,299 training brain scans, 894 held-out scans.
  - The baseline was the End-to-End Variational Network (unrolled, with data consistency), and finalists were largely unrolled cascades with data-consistency steps.
  - 4X track: the top 4 teams' SSIM fell between 0.955 and 0.965. 8X track: top team 0.952, others < 0.944.
  - The Transfer track (trained on Siemens, tested on GE and Philips) showed "the greatest variation". GE raw data lacked frequency oversampling, which "can decrease SSIMs for models by as much as 0.1-0.4 if no other adjustments are made" and "rendered many models useless in this track without modification".
  - Source: [Muckley et al. 2021](https://arxiv.org/pdf/2012.06318v2)
- fastMRI 2020 hallucinations:
  - Radiologists saw false vessels, artifact-mimics, and abnormalities "morphed … into a more normal brain structure".
  - Several radiologists said none of the 8X or Transfer images would be clinically acceptable.
  - One finalist added "a hard data-consistency mechanism at the network end to minimize any risk of hallucination".
  - Training the E2E-VarNet baseline took 32 GPUs (32 GB each) for about 3.5 days.
  - Source: [Muckley et al. 2021](https://arxiv.org/pdf/2012.06318v2)
- Bhadra et al. (2021, IEEE TMI) split the image estimate into measurement and null-space components and defined a "hallucination map". In a stylized tomography setup, a U-Net produced false structures for an out-of-distribution object; PLS-TV and DIP did not. — [arXiv 2012.00646](https://arxiv.org/abs/2012.00646v1); [Anastasio lab page](https://anastasio.bioengineering.illinois.edu/research/comp-imag-learn/hallucinations-recon/)
- Genzel et al. (ICML 2022) won the AAPM sparse-view CT challenge with an iterative end-to-end network. It reached reconstructions "near numerical precision" on noise-free data, after a data-driven calibration step to estimate the unknown fan-beam geometry. — [PMLR 162](https://proceedings.mlr.press/v162/genzel22a.html)

### Inferences
- The two head-to-head studies (CT, PAT) agree on one hierarchy. Operator inside the loop beats a physics image as input, which beats raw data with no physics. The margin is moderate on in-distribution data (2.2 dB in CT) and larger under shift (10x smaller sensitivity to a sampling change in PAT). Both are single-paper results.
- A physics image such as FBP or an adjoint image is a useful *starting point* (faster training) but caps what a post-processor can recover. The value of physics channels may lie more in robustness and data efficiency than in peak in-distribution accuracy.
- The "AUTOMAP-style" fully learned raw-to-image route is the most fragile and least parameter-efficient option in this literature. That argues against feeding raw shot gathers alone when physics channels are cheap.
- The robustness literature is genuinely contested (Antun vs Genzel vs Darestani). The consistent message is that distribution shift in *acquisition/preprocessing details* (SNR, oversampling, sampling pattern) hurts more than shift in image content.

### Gaps
- I found no study that sweeps training-set size head to head between post-processing and unrolled networks on the same CT/MRI task, so the claim that unrolled nets are "more data-efficient" is not quantified here. A targeted search returned only memory/compute-oriented papers ([arXiv 2601.02141](https://arxiv.org/pdf/2601.02141)).
- I did not retrieve Hammernik et al. 2018 (variational network), LEARN (Chen et al. 2018) or E2E-VarNet (Sriram et al. 2020) numbers, nor the Arridge et al. 2019 and Ongie et al. 2020 survey text.
- Knoll et al. RMSE/SSIM numbers were behind a CAPTCHA, so only qualitative conclusions are reported.

---

## Q2. Auxiliary inputs and multi-task learning in computer vision: when extra channels help, when they are ignored, when they hurt

### Takeaway
Mid-level channels (depth, segmentation, normals, distance) as *inputs* reliably improve out-of-distribution generalization and sample efficiency:
- Robot navigation: about 10x less data than learning from raw pixels.
- Driving: success rate roughly doubled with ground-truth channels.

Three caveats apply. The best channel is task-dependent (rank reversal). Predicted channels give less than ground-truth ones. The advantage disappears when training randomization already covers the test distribution. Naive multi-input fusion can *underperform* the best single input, because the network overfits and the inputs learn at different speeds.

### Cited Findings

**Sax et al. (arXiv 2018/2019), mid-level vision for visuomotor policies**
- Setup: 20 frozen mid-level feature networks (from Taskonomy) feed an RL policy, tested in buildings separate from training. — [arXiv 1812.11971](https://arxiv.org/pdf/1812.11971)
- Agents trained from scratch on raw RGB "overfit completely—rarely doing better than blind agents in the test environment". Several feature-based agents beat scratch on every task (p < 0.002 per task). — [Sax et al.](https://arxiv.org/pdf/1812.11971)
- Autoencoder and VAE features "have strong training curves that belie exceptionally weak test-time performance". Generic learned features are not the same as physically meaningful channels. — [Sax et al.](https://arxiv.org/pdf/1812.11971)
- **Rank reversal / no universal feature:**
  - Object classification was best for navigation; distance estimation was best for exploration.
  - "navigation is characteristically semantic while exploration is geometric".
  - Maximizing performance across tasks "requires giving up 3-4 ranks for each task".
  - Source: [Sax et al.](https://arxiv.org/pdf/1812.11971)
- A max-coverage set of k = 4 features "nearly match[es] or exceed[s] the performance of the best task-specific feature", with no noticeable sample-efficiency penalty from the larger input. — [Sax et al.](https://arxiv.org/pdf/1812.11971)
- The selected feature sets used "at least an order of magnitude less data than learning from scratch" and still reached higher final performance. — [Sax et al.](https://arxiv.org/pdf/1812.11971)
- **Key boundary condition:** "only feature-based agents were able to generalize without texture randomization during training. Once we added in randomized training textures that resembled the test textures … this distinction disappeared." — [Sax et al.](https://arxiv.org/pdf/1812.11971)

**Zhou, Krähenbühl & Koltun (Science Robotics 2019), "Does computer vision matter for action?"** — [arXiv 1905.12887](https://ar5iv.labs.arxiv.org/html/1905.12887)
- Image only vs image + all ground-truth modalities:
  - Urban driving success rate: 0.35 → 0.67
  - Off-road traversal: 0.42 → 0.89
  - VizDoom battle frags: 29.6 → 36.2 (> 20%)
- Single ground-truth modalities:
  - Urban SR: albedo 0.40, flow 0.45, depth 0.48, segmentation 0.54
  - Off-road SR: depth 0.81, segmentation 0.78
  - Optical flow and albedo "appear less helpful".
- With *predicted* modalities instead of ground truth:
  - Urban 0.56, off-road 0.64, battle 34.0: smaller gains, but still significant.
  - On off-road, predicted depth alone (0.65) matched all predicted modalities (0.64).
- All agents were near-perfect on training tracks (urban SR 0.97-1.0), so the gains are generalization gains. The image-only agent needed about an order of magnitude more data to match the predicted-vision agent.

**When more inputs hurt: Wang, Tran & Feiszli (CVPR 2020)** — [ar5iv 1905.12681](https://ar5iv.labs.arxiv.org/html/1905.12681); [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_What_Makes_Training_Multi-Modal_Classification_Networks_Hard_CVPR_2020_paper.html)
- Kinetics top-1 accuracy, single inputs: RGB-only 72.6%, optical flow only 62.1%, audio only 19.7%.
- Naive joint models do *worse* than RGB alone: RGB + audio 71.4%, RGB + flow 71.3%, RGB + flow + audio 70.0%.
- Causes: (1) "multi-modal networks are often prone to overfitting due to their increased capacity"; (2) inputs overfit and generalize at different rates, so a single joint optimization is sub-optimal.
- Fixes (RGB + audio unless noted):
  - Gradient-Blending, which reweights per-input auxiliary losses by an overfitting-to-generalization ratio: 74.7% (RGB+A), 73.1% (RGB+OF), 74.9% (RGB+OF+A), and 75.8% online.
  - Plain dropout: 72.9%
  - Mid-level concatenation: about 72.8%
  - Pretraining: 71.7% (no gain)
  - Equal-weight auxiliary losses: 70.8% (worse)

**Multi-task affinity and consistency**
- Standley et al. (ICML 2020): task objectives "can compete", and multi-task affinity differs from transfer affinity. Grouping cooperating tasks into a few networks beats both one big multi-task net and separate single-task nets on accuracy for a given inference time. — [PMLR 119](https://proceedings.mlr.press/v119/standley20a.html)
- Zamir et al. (CVPR 2020), cross-task consistency:
  - Enforcing consistency between predicted domains (e.g. normals vs depth) gives "more accurate predictions, better generalization to out-of-distribution samples, and improved sample efficiency".
  - The unsupervised "consistency energy" correlates with supervised error at r = 0.67.
  - It detects OOD inputs with AUC 0.99 (CVPR version) vs 0.95 (arXiv version) **[conflict between versions]**.
  - Source: [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zamir_Robust_Learning_Through_Cross-Task_Consistency_CVPR_2020_paper.html)

### Inferences
- Physically meaningful intermediate channels act as an inductive bias that cuts the *train/test gap*, not mainly the training error. To measure their value, the test set must be out of distribution. In an in-distribution test (or with heavy training randomization covering the test domain) the benefit can vanish (Sax).
- The useful channel depends on the downstream objective. Semantic/identity channels help "find the target"; geometric channels help "explore". Try a small diverse set (about 4) rather than hunting for one best channel.
- Imperfect predicted channels still help but deliver roughly half to two thirds of the ground-truth-channel gain (Zhou). The accuracy of the physics-derived channel bounds its value.
- Naive concatenation is a known failure mode. Plan for per-input regularization: dropout, per-branch auxiliary heads with tuned weights (Gradient-Blending), or a learned selector (Q5).

### Gaps
- No quantitative tables were retrieved from Taskonomy (Zamir et al. 2018) or Standley et al. 2020.
- I found no study that measures whether networks literally *ignore* an extra input channel (e.g. via attribution). The evidence is indirect: performance parity, or degradation under a missing input (Q5).

---

## Q3. Multi-view fusion: set/attention-based fusion, permutation invariance, number of views

### Takeaway
Permutation-invariant aggregation lets one network accept any number of views, and accuracy improves with more views, but returns saturate quickly:
- MVS (DTU overall error): N = 2 → 3 gives about 10-14% improvement, 3 → 5 about 2%, 5 → 7 about 0%.
- Video: long-range recurrent propagation beats short sliding windows, and distant frames still carry information.
- Learned matting with neighbouring-frame cues helped noticeably in only a minority of real clips.

### Cited Findings
- **MVSNet (Yao et al., ECCV 2018):**
  - A variance-based cost metric maps N views' features into one cost volume, so the network accepts an arbitrary N.
  - Trained with N = 3; testing with N = 5 gave lower validation loss than N = 3.
  - Source: [arXiv 1804.02505](https://arxiv.org/pdf/1804.02505)
- **Point-MVSNet (2019), DTU overall error in mm (lower is better):** N = 2: 0.533, N = 3: 0.478, N = 5: 0.468. — [arXiv 1908.04422](https://arxiv.org/pdf/1908.04422)
- **VA-MVSNet (2019), trained with N = 3, DTU overall error:** test N = 2: 0.441, N = 3: 0.380, N = 5: 0.371, N = 7: 0.370. — [arXiv 1912.03001](https://arxiv.org/pdf/1912.03001)
- **Region-Aware MVSNet (2023):** performance was *not* monotone in N and peaked at N = 5, attributed to large baselines between DTU images. **[partial conflict with "more views is better"]** — [arXiv 2304.13614](https://arxiv.org/pdf/2304.13614)
- **Deep Sets (Zaheer et al., NeurIPS 2017):** a permutation-invariant function over a set can be written as a sum/pool of per-element embeddings followed by a network. This is the general template behind mean/variance/attention pooling over views. — [arXiv 1703.06114](https://arxiv.org/abs/1703.06114)
- **BasicVSR (Chan et al., CVPR 2021):**
  - Splitting test sequences into independently processed segments lowers PSNR, and most at segment edges. The authors read this as "distant frames carry useful information".
  - Sliding-window methods are limited to a local neighbourhood.
  - Reported REDS4 PSNR: BasicVSR 31.42 dB, IconVSR 31.67 dB, BasicVSR++ 32.39 dB (+0.82 dB over BasicVSR at similar parameter count).
  - Sources: [arXiv 2012.02181](https://arxiv.org/pdf/2012.02181); [arXiv 2508.10453 comparison table](https://arxiv.org/pdf/2508.10453); [BasicVSR++ CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Chan_BasicVSR_Improving_Video_Super-Resolution_With_Enhanced_Propagation_and_Alignment_CVPR_2022_paper.pdf)
- **Background Matting (Sengupta et al., CVPR 2020) [single paper]:**
  - Used the motion cue of 4 neighbouring grayscale frames (I±T, I±2T); for single images the frame is repeated.
  - Its cue-selection block "helps significantly in 9 out of 50 real videos". No numeric ablation of the motion cue is reported.
  - Source: [ar5iv 2004.00626](https://ar5iv.labs.arxiv.org/html/2004.00626)
- **Recurrent Video Masked Autoencoders (RVM; Zoran et al., Google DeepMind, CVPR 2026):** a transformer-based recurrent encoder with "stable feature propagation over long temporal horizons with linear computational cost". It is reported as competitive with VideoMAE / V-JEPA and up to 30x more parameter-efficient. This is the "RVM small" backbone listed for CorridorKey 2.0 (Q6). — [arXiv 2512.13684](https://arxiv.org/html/2512.13684v1)

### Inferences
- A view-count-agnostic aggregator (mean/variance pooling or attention over views, Deep Sets style) lets a model train on few views and test on more with a modest gain. The gains flatten around 5 views in MVS. Extra views are worth most when they add new geometric coverage (baseline or angle), not redundancy (Region-Aware MVSNet).
- For temporal or sequential "views", recurrence or long windows beat short windows. Overlapping windows with tapered blending (as in CorridorKey 2.0) are a practical compromise when memory is bounded.

### Gaps
- Light-field fusion results were not retrieved.
- I found no controlled study comparing *attention* vs *variance/mean* pooling as the number of views varies within one paper.

---

## Q4. Losses: perceptual, SSIM, gradient/edge, physics-consistency, cycle/temporal consistency

### Takeaway
L1, or L1 + MS-SSIM, beats L2 even on L2-type metrics, because L2 trains poorly. Gradient (edge) terms sharpen boundaries. Physics/compositing-consistency losses tie outputs to the measurements and permit self-supervised training. Perceptual/GAN losses buy visual sharpness at a provable distortion cost and raise hallucination risk, which is the main concern in medical reconstruction.

### Cited Findings
- **Zhao, Gallo, Frosio & Kautz (IEEE TCI 2017)**, same network, different loss — [ar5iv 1511.08861](https://ar5iv.labs.arxiv.org/html/1511.08861):
  - Joint denoising + demosaicking, PSNR: L2 33.18, L1 34.42, SSIM5 33.15, MS-SSIM 33.29, **Mix (MS-SSIM + L1) 34.61**.
  - Super-resolution, PSNR: L2 30.66, L1 31.26, Mix 31.34.
  - JPEG deblocking, PSNR: L2 32.73, L1 32.96, Mix 33.25. MS-SSIM alone *collapsed* to 27.66, below the 32.60 input.
  - Networks trained with L1 reached *lower L2 error* than networks trained with L2: 0.43 vs 0.56 (denoising), 1.106 vs 1.241 (SR), 0.603 vs 0.651 (deblocking). The authors attribute this to L2 "get[ting] stuck more easily in a local minimum".
- **Perception-distortion tradeoff (Blau & Michaeli, CVPR 2018):** an unattainable region in the perception-distortion plane exists "regardless of the algorithmic scheme". Lowering mean distortion forces worse perceptual quality, and vice versa. GANs approach the bound. — [CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html); [project page](https://webee.technion.ac.il/people/tomermic/PerceptionDistortion/PD_tradeoff.htm)
- **L2 over-smoothing in learned primal-dual CT:** the authors blame the squared-L2 loss and recommend perceptual losses. — [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)
- **Matting losses (Background Matting, CVPR 2020)** — [ar5iv 2004.00626](https://ar5iv.labs.arxiv.org/html/2004.00626):
  - L1 on alpha
  - L1 on the alpha *gradient* ("to sharpen mattes")
  - 2× L1 on foreground
  - L1 on the compositing residual I − αF − (1−α)B, a physics/forward-model consistency term
  - Real-data fine-tuning adds an LS-GAN loss on re-composites, with the teacher loss weighted by λ = 0.05 and halved every 2 epochs.
  - Ours-Real vs Ours-Adobe user study on handheld video: 16.4% "much better", 35.5% "better", 42.7% "similar", 5.4% "worse".
- **Physics-consistency as the only supervision: SSDU (Yaman et al., MRM 2020) and multi-mask SSDU** — [deepai summary](https://deepai.org/publication/self-supervised-learning-of-physics-based-reconstruction-neural-networks-without-fully-sampled-reference-data); [arXiv 2008.06029](https://arxiv.org/abs/2008.06029v2):
  - Measured k-space is split in two: one part enforces data consistency inside the unrolled net, the other is the training loss.
  - At R = 4 it "performed closely with supervised learning" across 5 knee sequences, beating CS and parallel imaging.
  - The multi-mask version beat single-mask at high acceleration, and a reader study ranked it above supervised DL for SNR and aliasing.
- **Hard data consistency against hallucination:** one fastMRI 2020 finalist placed a hard data-consistency step at the network end "to minimize any risk of hallucination". Hallucinations were still observed across all tracks. — [Muckley et al. 2021](https://arxiv.org/pdf/2012.06318v2)
- **SSIM as a training/selection metric:** in fastMRI 2020, radiologist ranks correlated with SSIM "for large SSIM separations". Among close competitors SSIM did not discriminate clinical quality. — [Muckley et al. 2021](https://arxiv.org/pdf/2012.06318v2)
- **Temporal consistency (Lai et al., ECCV 2018):** a recurrent network trained with *short-term and long-term temporal losses plus a perceptual loss*, balancing stability and fidelity to the per-frame result. It needs no optical flow at test time. — [arXiv 1808.00449](https://alphaxiv.org/abs/1808.00449)
- **Cross-task consistency as a loss:** see Zamir et al. 2020 in Q2 (better OOD generalization and sample efficiency). — [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zamir_Robust_Learning_Through_Cross-Task_Consistency_CVPR_2020_paper.html)

### Inferences
- Default to L1, or L1 + (MS-)SSIM, for image-like targets. Add a gradient/edge term where boundaries matter (alpha edges in matting; velocity interfaces in seismic).
- A forward-model residual term is the most transferable loss. It is the analogue of the compositing loss in matting and of data consistency in MRI. It allows training or fine-tuning on unlabeled field data (SSDU-style) and constrains hallucination, though it does not eliminate it.
- Treat perceptual/GAN losses with caution when the output is a quantitative physical field. The perception-distortion theorem guarantees a fidelity cost, and medical studies document plausible-looking false structures.

### Gaps
- I did not retrieve inverse-problem studies that quantify LPIPS-trained vs L1-trained reconstruction (the LPIPS paper itself, Zhang et al. 2018, was not fetched).
- I found no quantitative ablation of temporal-consistency losses in matting.

---

## Q5. Robustness: modality/channel dropout, input perturbation, cue selection, confidence inputs

### Takeaway
Modality dropout is cheap and very effective against missing or corrupted inputs:
- ModDrop at p = 0.1 lifted accuracy with a missing pose stream from 38% to 93%, at no cost when all inputs were present.
- Per-input loss balancing turns a multi-input loss into a gain.

Networks over-rely on whichever auxiliary cue is "easiest". The fixes are perturbing that cue during training, gating/selection architectures, and explicit trust or confidence inputs.

### Cited Findings
- **ModDrop (Neverova et al., IEEE TPAMI 2016; arXiv 2014)** — [ar5iv 1501.00102](https://ar5iv.labs.arxiv.org/html/1501.00102):
  - Each modality is dropped with probability 10% during gradual fusion; input dropout is 20%.
  - Accuracy, dropout only → dropout + ModDrop:

    | Condition | Dropout only | + ModDrop |
    |---|---|---|
    | All inputs present | 96.77% | 96.81% |
    | Mocap (pose) missing | 38.41% | **92.82%** |
    | Audio missing | 84.10% | 92.59% |
    | Both hands missing | 53.13% | 73.28% |
    | Right hand missing | 81.25% | 85.36% |

  - The method placed 1st of 17 teams at ChaLearn 2014. — [HAL](https://hal.archives-ouvertes.fr/hal-01178733)
- **Gradient-Blending / dropout for multi-input nets:** naive RGB+audio 71.4% → dropout 72.9% → G-Blend 74.7-75.8% (see Q2). — [ar5iv 1905.12681](https://ar5iv.labs.arxiv.org/html/1905.12681)
- **Cue over-reliance and its fixes (Background Matting, CVPR 2020) [single paper]** — [ar5iv 2004.00626](https://ar5iv.labs.arxiv.org/html/2004.00626):
  - Plain concatenation of the inputs (image, background plate, soft segmentation, motion frames) "focused too heavily on the color difference between I and B′", leaving holes where foreground and background colours match.
  - Fix: a Context Switching block in which per-cue "selector" branches pair image features with each cue before a "combinator".
  - The Adobe-trained model also "tended to trust B′ too much". The fix was to perturb training backgrounds (gamma γ ~ N(1, 0.12) plus Gaussian noise) and to degrade the segmentation input (erosion, dilation, blur).
  - Removing the background cue raises SAD from 1.73 to 8.33, so the cue is heavily used. Eroding/dilating the segmentation by 20 steps raises SAD only from 1.73 to 1.76, so the model was made robust to that cue's errors.
  - Simulating alignment errors during training "degraded results overall". Not every perturbation helps.
- **Confidence as an explicit input/output (Eldesokey, Felsberg & Khan, IEEE TPAMI 2020; arXiv 2018):**
  - Normalized-convolution layers propagate a per-pixel confidence alongside sparse depth.
  - The loss minimizes data error while maximizing output confidence.
  - Reported parameter count: about 5% (v1) or 1-5% (later version) of state-of-the-art methods, on KITTI-Depth (and NYU-Depth-v2 in the later version).
  - Source: [arXiv 1811.01791](https://arxiv.org/abs/1811.01791v1)
- **Scalar trust input in a production keyer:** CorridorKey 2.0 exposes `--hint_quality`, a scalar setting "how much the model trusts an explicit hint". — [GitHub CorridorKey2.0](https://github.com/nikopueringer/CorridorKey2.0)
- **Training-distribution coverage as robustness:**
  - Training randomization that covers the test conditions removed the generalization advantage of mid-level features (Sax et al.). — [Sax et al.](https://arxiv.org/pdf/1812.11971)
  - Heterogeneous MRI training data generalized across acquisition parameters (Knoll et al.). — [Knoll et al.](https://cris.fau.de/publications/262824564)
  - Adding simulated background to PAT training data was needed before real data worked (Hauptmann et al.). — [Hauptmann et al.](https://ar5iv.labs.arxiv.org/html/1708.09832)

### Inferences
- For any physics-derived channel that may be missing, noisy or biased at test time, use *both* modality dropout (about 10% per channel group) and realistic corruption of that channel during training. Expect no in-distribution cost and large gains when the channel degrades.
- Pair each physics estimate with a confidence/illumination channel or a scalar trust knob. This lets the network learn *when* to defer to the estimate, and it gives users a control at inference time.
- If you concatenate a strong but brittle cue with raw data, check whether the network has collapsed onto the cue: ablate it at test time and look for "holes" where the cue is ambiguous.

### Gaps
- I did not retrieve HeMIS (Havaei et al. 2016) or later modality-competition papers (e.g. Huang et al. 2022), or the details of OGM-GE.
- I found no study on confidence-map inputs in learned tomographic or MRI reconstruction.

---

## Q6. Public description of CorridorKey 2.0 (Corridor Digital) and similar learned keying

### Takeaway
CorridorKey 2.0 is publicly described as a two-stage temporal model built on three frozen foundation backbones:
- C-RADIO, an agglomerative semantic model;
- MoGe-2, a monocular geometry model;
- RVM, a recurrent video model.

It takes optional matte hints with a trust scalar and is trained entirely on in-house 3D renders. No losses, benchmarks or ablations are published. The user's specific list (semantic identity, depth, normals, motion vectors, perceptual/temporal losses) is only partly verifiable: semantic, geometric and temporal *foundation features* are documented; explicit normal/motion-vector channels and the loss terms are not.

### Cited Findings
- **Inputs:**
  - RGB frames, internally converted to "the model's signed asinh HDR encoding".
  - Optional per-pixel matte hints, either a first-frame matte or a full hint video ("white means subject, black means background").
  - A `--hint_quality` trust scalar.
  - With "Hann carry", the model's own predicted auxiliary matte from overlapping windows is fed back as a hint.
  - Source: [GitHub CorridorKey2.0](https://github.com/nikopueringer/CorridorKey2.0)
- **Backbones:**
  - Frozen C-RADIO-SO400M, MoGe-2 ViT-B and RVM small. "C-RADIO, MoGe, and RVM receive SDR-style RGB proxies", not unclipped HDR, and RVM runs recurrently over the clip. — [GitHub CorridorKey2.0](https://github.com/nikopueringer/CorridorKey2.0)
  - The HF card lists C-RADIOv4-SO400M (NVIDIA), MoGe-2 ViT-B (Microsoft) and RVM small (Google DeepMind), plus a `foundation_stats.safetensors` file with normalization statistics for the foundation features. — [Hugging Face CorridorKey2.0](https://huggingface.co/nikopueringer/CorridorKey2.0)
- **Temporal handling:**
  - "Stage 1 runs on overlapping temporal windows and combines its hidden states with a Hann-style taper" (e.g. 80-frame windows, stride 40).
  - "Stage 2 then streams the final full-resolution predictions from those combined hidden states."
  - The network runs in bfloat16.
  - Source: [GitHub CorridorKey2.0](https://github.com/nikopueringer/CorridorKey2.0)
  - Marketing describes a "time awareness" option that analyses "the entire video, not just one frame" to reduce flicker and chattering. — [corridorkey.com](https://corridorkey.com/)
- **Training data and outputs:**
  - "All training data for CorridorKey 2.0 was generated in-house at Corridor Digital using 3D rendering."
  - Weights file `corridorkey_v2.safetensors` is 2.66 GB.
  - The card reports no losses and no quantitative results.
  - Source: [Hugging Face CorridorKey2.0](https://huggingface.co/nikopueringer/CorridorKey2.0)
  - Outputs are a straight (un-premultiplied) foreground colour plus a linear alpha, i.e. colour unmixing on soft edges. — [beeble.ai summary](https://beeble.ai/models/corridor-key); [corridorkey.com](https://corridorkey.com/)
- **Credits:** model design and training are credited to Zachary Harris "in direct collaboration with Niko and Corridor Digital". — [GitHub CorridorKey2.0](https://github.com/nikopueringer/CorridorKey2.0); [corridorkey.com](https://corridorkey.com/)
- **Product:** the paid plugin is priced at $49.50 for Resolve/Nuke/AE/Premiere/Blender. — [corridorkey.com via search summary](https://corridorkey.com/)
- **CorridorKey 1.0 (third-party description):**
  - Inputs: green-screen plate plus a coarse alpha hint ("a coarse mask that tells the network approximately where the subject is").
  - Processing was frame by frame, and the original needed ≥ 24 GB VRAM.
  - Release dates conflict across third-party sources (February vs early March 2026). **[conflict]**
  - Sources: [EZSCAPE explainer](https://ezscape.space/learn/corridorkey-explained/); [podcastvideos.com](https://www.podcastvideos.com/corridor-digital-corridorkey-ai-green-screen-tool/)
- **Published analogue with ablations: Background Matting (Sengupta et al., CVPR 2020).**
  - Explicit auxiliary per-pixel inputs: captured background plate, soft person segmentation (DeepLabv3+, eroded/dilated/blurred), and 4 neighbouring frames as a motion cue.
  - A selector/combinator block chooses among the cues; losses include alpha-gradient and compositing-consistency terms.
  - Quantitative ablations and robustness tricks are listed in Q3-Q5.
  - Source: [ar5iv 2004.00626](https://ar5iv.labs.arxiv.org/html/2004.00626)

### Inferences
- Architecturally, CorridorKey 2.0 follows the "mid-level priors" recipe from Q2: frozen, pretrained semantic + geometry + video features feed a small task head trained on synthetic renders. Treat this as an inference from the backbone list.
  - MoGe-2 is a monocular geometry (point-map/depth, likely normals) model; I did not verify its exact outputs here.
  - The "semantic identity, depth/normals, motion" decomposition in the user's description matches the backbones' roles. It may come from a video or talk not found in text sources.
- Its design choices map onto lessons from the literature:
  - optional hints with a trust scalar → confidence inputs (Q5);
  - windowed temporal context with tapered overlap → long-range temporal fusion (Q3);
  - purely synthetic 3D-rendered training → large controllable synthetic datasets, which the medical literature says need realistic nuisance modelling or small real fine-tuning sets (Q1).

### Gaps
- I found no technical report, paper, loss description, benchmark or ablation for CorridorKey 1.0 or 2.0.
- The claims about surface normals, motion vectors as explicit channels, perceptual loss and temporal-consistency loss could not be verified from public text. They may come from Corridor Digital videos, which were not searched.
- I did not retrieve Robust Video Matting (Lin et al. 2022), MODNet or BGMv2 numbers.

---

## Q7. Transferable lessons for learned seismic inversion (synthesis of Q1-Q6)

### Takeaway
Six principles carry over:
1. Put the physics operator inside the network where affordable, rather than only feeding its outputs.
2. Use a small, diverse set of physically meaningful channels, each paired with confidence, dropout and corruption during training.
3. Fuse shots/gathers with permutation-invariant pooling.
4. Train with L1(+SSIM/gradient) plus a data-residual term.
5. Evaluate under real distribution shift, because that is where auxiliary inputs pay off.
6. Expect acquisition/preprocessing mismatch, not image content, to be the main failure source.

### Cited Findings
- Operator-in-the-loop beat physics-image-as-input by +2.2 dB (CT human phantom), and its advantage grew as data quality fell. — [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)
- Unrolled PAT was about 10x less sensitive to a sampling-pattern change (0.5% vs 5% error increase), at about 9x the runtime of a U-Net. — [Hauptmann et al. 2018](https://ar5iv.labs.arxiv.org/html/1708.09832)
- Physics initialization sped training but did not improve LPD's final result. — [Adler & Öktem 2018](https://arxiv.org/pdf/1707.06474)
- Mid-level channels cut data needs by about 10x and closed the train/test gap, but not once training randomization covered the test domain. — [Sax et al.](https://arxiv.org/pdf/1812.11971); [Zhou et al. 2019](https://ar5iv.labs.arxiv.org/html/1905.12887)
- Naive multi-input fusion lost about 1-2.6 points vs the best single input; balancing per-input losses gained +2-3 points. — [Wang et al. 2020](https://ar5iv.labs.arxiv.org/html/1905.12681)
- Modality dropout at p = 0.1 protects against a missing input (38% → 93% accuracy). — [Neverova et al. 2016](https://ar5iv.labs.arxiv.org/html/1501.00102)
- More views help with diminishing returns (DTU error 0.441 → 0.380 → 0.371 → 0.370 for 2/3/5/7 views). — [VA-MVSNet](https://arxiv.org/pdf/1912.03001)
- Small preprocessing mismatches (missing frequency oversampling) cut SSIM by 0.1-0.4 and broke many models. — [Muckley et al. 2021](https://arxiv.org/pdf/2012.06318v2)
- SNR mismatch was the dominant failure under shift in learned MRI. — [Knoll et al.](https://cris.fau.de/publications/262824564)
- Synthetic-only training needed real-data fine-tuning (20 real PAT measurements; a small in-vivo MRI subset). — [Hauptmann et al. 2018](https://ar5iv.labs.arxiv.org/html/1708.09832); [Knoll et al.](https://cris.fau.de/publications/262824564)
- Measurement-split self-supervision with a physics-consistency loss came close to supervised MRI training. — [arXiv 2008.06029](https://arxiv.org/abs/2008.06029v2)

### Inferences
These are mappings for an FWI / learned velocity-model project. None is a seismic result; all are extrapolations from the findings above.

1. **Channel hierarchy to test:**
   - (a) raw gathers only
   - (b) raw + cheap physics channels: adjoint/RTM image or FWI gradient, slope- or traveltime-derived v(z), illumination/fold map as a confidence channel
   - (c) an unrolled net that recomputes the gradient channel each iteration with the wave operator

   The literature predicts (b) > (a) mainly out of distribution, and (c) > (b) modestly in distribution and more under acquisition shift, at about an order of magnitude more compute.
2. **Evaluate out of distribution:** for example, train on one OpenFWI family and test on another or on Marmousi-like models, or shift the source wavelet, frequency band or acquisition geometry. In-distribution tests (or heavy randomization) can hide the value of physics channels (Sax et al.).
3. **Expect rank reversal:**
   - "Where is the interface?" (geometric: slopes, gradients, dip fields) and "what is the background trend?" (kinematic: v(z), traveltimes) may favour different channels.
   - A diverse set of about 4 channels is a reasonable default.
4. **Guard against cue collapse:**
   - Use per-channel-group dropout (about 10%) and realistic corruption of each physics channel (wrong wavelet, cycle-skipped gradient, noisy slopes).
   - Add a confidence/illumination channel or trust scalar.
   - Use per-branch auxiliary losses or a gated selector instead of plain concatenation.
   - At test time, ablate each channel to check reliance.
5. **Multi-shot fusion:** treat shots or offsets as a set. Use mean/variance or attention pooling so the shot count can vary between training and test. Expect diminishing returns beyond a handful of well-separated views.
6. **Losses:**
   - Use L1 (+ SSIM or a gradient/edge term for interfaces) instead of L2.
   - Add a forward-modelled data-residual term (the analogue of the matting compositing loss / MRI data consistency). It enables self-supervised fine-tuning on field data and limits hallucinated structure.
   - Avoid GAN/perceptual losses for quantitative velocity unless hallucination is measured, for example with a null-space / hallucination-map style check.
7. **Preprocessing parity:** normalize amplitude, wavelet, sampling and SNR conventions between synthetic training data and target data. The fastMRI transfer track shows that seemingly trivial format differences can invalidate a model.

### Gaps
- The cross-field evidence here is mostly single-paper (LPD, PAT, Sax, Zhou, Background Matting, ModDrop). The magnitudes should be read as indicative.
- I found no quantified data-efficiency curve for unrolled vs post-processing networks.
- The seismic-specific literature is outside this note's scope and should be checked against these principles.
