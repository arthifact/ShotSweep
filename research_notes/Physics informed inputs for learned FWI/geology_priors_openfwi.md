# Geology encoded in OpenFWI and geological priors for velocity estimation (classical and deep learning)

Scope note: the published OpenFWI paper gives the generating equations but **not** the per-family layer counts, thicknesses, fault throws or dips. To fill that gap I measured one 500-model shard per family from the local copy of the official data (`/Users/torus/Desktop/Research/veronicae_betifulae/openfwi/data/<family>/model1.npy` or `vel*_1_0.npy`). The scripts are `stats.py`, `probe.py` and `recount.py` in `/private/tmp/claude-501/-Users-torus-Desktop-Research-veronicae-betifulae/9f9d44d9-6d3c-441e-b58b-9aabf1b50d5c/scratchpad/openfwi_stats/`. These numbers are tagged **[local measurement]**. They describe one shard per family, so treat them as descriptive, not as the generator's specification. OpenFWI metric values are on normalized velocity maps.

---

## Q1. How is each of the ten OpenFWI 2D families generated (layers, velocity ranges, curvature, faults, Style, A vs B)?

### Takeaway
All ten 2D families are 70×70 maps with 10 m cells, velocities between 1500 and 4500 m/s, and data from 5 surface shots × 70 receivers (15 Hz Ricker, 1 ms, 1000 samples). The paper defines each family by an equation:
- **Vel:** random-width flat layers, optionally warped by recursive sine displacements.
- **Fault:** the same, plus straight fault lines below which the map is shifted.
- **Style:** a COCO image restyled to look like Marmousi and blended with a linear 1D background.

For the "A" versions the paper only says "velocity increases with depth" (Vel) or "fewer, smaller discontinuities" (Fault), and the shards confirm this. FlatVel-B and CurveVel-B put layer velocities in random order, so about 50% of interfaces are velocity inversions. Fault-B has more layers (median about 6), sharp interfaces, larger offsets (median 120 m, up to about 370 m), and layer velocities that change across the fault.

### Cited Findings
**Common acquisition and size (OpenFWI paper, Deng et al. 2021/2022, NeurIPS Datasets & Benchmarks 2022)**
- Vel, Fault and Style families:
  - 10 m grid; 15 Hz source; 5 sources about 175 m apart; 70 receivers at 10 m; dt 0.001 s, 1001 steps.
  - Velocity map 70×70; seismic data 5×1000×70; 120-cell absorbing boundary.
  - Train/test sizes: Vel 24K/6K, Fault 48K/6K, Style 60K/7K. Disk sizes: Vel 43 GB, Fault 77 GB, Style 95 GB each.
  - The ar5iv rendering prints the units as "mm", which is a rendering glitch; the domain is 700 m wide. — [OpenFWI arXiv 2111.02926 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2111.02926); [OpenFWI data page](https://openfwi-lanl.github.io/docs/data.html)
- The velocity range for all Vel, Fault and Style datasets is 1500–4500 m/s. For comparison, Kimberlina-CO2 is 947–2545 m/s and 3D Kimberlina-V1 is 1975–3892 m/s. — [OpenFWI ar5iv](https://ar5iv.labs.arxiv.org/html/2111.02926); [NeurIPS supplement](https://proceedings.neurips.cc/paper_files/paper/2022/file/27d3ef263c7cb8d542c4f9815a49b69b-Supplemental-Datasets_and_Benchmarks.pdf)
- The velocity maps come from three priors: mathematical functions (Vel, Fault), natural images transformed by style transfer (Style), and geological reservoir simulation (Kimberlina). — [OpenFWI ar5iv](https://ar5iv.labs.arxiv.org/html/2111.02926)

**Vel family (FlatVel-A/B, CurveVel-A/B)**
- The generator starts from a map c₀ of flat layers with random widths. Curvature comes from recursive warping, cᵢ(x,y) = cᵢ₋₁(x, aᵢ·sin(2πkᵢx)), with a random amplitude aᵢ and frequency kᵢ at each iteration. The paper gives neither the number of iterations nor the parameter ranges. — [OpenFWI ar5iv](https://ar5iv.labs.arxiv.org/html/2111.02926)
- A vs B: "In A, velocity increases gradually with depth; in B, velocity is randomly distributed." — [OpenFWI ar5iv](https://ar5iv.labs.arxiv.org/html/2111.02926)
- The website notes are "Simple situation with flat layers", "Simple situation with curved layers" and "Difficult version of …". — [OpenFWI data page](https://openfwi-lanl.github.io/docs/data.html)
- **[local measurement] FlatVel-A, 500 models:**
  - Layer count (layers ≥3 cells thick in one column): 2–8, median 5. Histogram over 2/3/4/5/6/7/8 layers: 6/64/106/133/109/68/14 models.
  - Velocity order: 0 of 2035 interfaces is a velocity decrease, so every model is monotone in depth.
  - Interface contrast: v_below/v_above has p5/median/p95 of 1.01/1.12/1.56; |Δv| has median 312 m/s (p5 29, p95 1260).
  - No lateral variation at all.
  - Layer thickness in the centre column: p5/median/p95 of 5/12/23 cells (50/120/230 m).
  - Surface velocity median 1820 m/s; bottom velocity median 3920 m/s.
- **[local measurement] FlatVel-B:**
  - Same layer-count distribution as A (2–8, median 5).
  - 48.4% of interfaces are velocity decreases; only 3.6% of models are monotone.
  - Contrast ratio p5/p95 is 0.49/2.06, and the median |Δv| is 856 m/s.
  - Surface velocity median about 2980 m/s, so a fast top layer is common.
  - Still no lateral variation.
- **[local measurement] CurveVel-A:**
  - 2–8 layers, median 5; 100% monotone in depth.
  - The first interface's depth varies across the 700 m width by a median of 20 cells (200 m), with p5 9 cells and p95 51 cells (510 m). These are large folds relative to the model size.
  - Lateral coefficient of variation (mean over rows of std_x/mean_x): median 0.068.
- **[local measurement] CurveVel-B:** 50.5% of interfaces are inversions; lateral CoV median 0.143. Its fold amplitudes match CurveVel-A.

**Fault family (FlatFault-A/B, CurveFault-A/B)**
- The fault family uses the same recursive curvature, plus a fault term. Below a random linear function fᵢ(x), the map takes values from c₀ shifted by random offsets sᵢ and s′ᵢ; pixels that fall outside the boundary take the nearest valid value. The paper gives no fault count, throw or dip values. A vs B: "B has more discontinuities and larger velocity jumps than A." — [OpenFWI ar5iv](https://ar5iv.labs.arxiv.org/html/2111.02926)
- The website describes FlatFault-A and CurveFault-A as "Flat/Curve layers with one fault". — [OpenFWI data page](https://openfwi-lanl.github.io/docs/data.html)
- **Precursor recipe (UPFWI, Jin et al. 2021, arXiv 2110.07584)**, which is not necessarily identical to the released OpenFWI sets:
  - 2–4 layers per map; thickness 15–35 grid cells; 70×70 grid with 15 m spacing.
  - Each layer's velocity drawn uniformly from 3000–6000 m/s and "increase with depth".
  - Curved layers come from a sine function.
  - Faults: fault shift of 10–20 grid cells; the tilting angle renders as "-123° to 123°", which is probably garbled (see Gaps).
  - Acquisition: 25 Hz Ricker source, 5 shots at 255 m spacing, 70 receivers at 15 m. — [UPFWI ar5iv](https://ar5iv.labs.arxiv.org/html/2110.07584)
- **[local measurement] FlatFault-A, shard `vel2_1_0.npy`:**
  - Mostly 2 layers: 436 of 500 models.
  - In 97% of models the interfaces are spread over a 2-cell ramp of intermediate values, for example 2253 → 2306 → 3198 → 3252 m/s. That works like built-in anti-aliasing or smoothing.
  - Only 1.9% of interfaces are inversions.
  - Vertical offset between the leftmost and rightmost columns (best alignment, a proxy for throw): median 6 cells (60 m), p95 10 cells (100 m). In 31.6% of models the offset is zero.
  - Layer velocities are nearly identical across the fault.
- **[local measurement] FlatFault-B, shard `vel6_1_0.npy`:**
  - 3–9 layers, median 6; sharp interfaces.
  - Inversions: 4.9% of interfaces in column 0, or 10.5% of all vertical jumps.
  - Offset proxy: median 12 cells (120 m), p95 37 cells (370 m); only 4% of models have zero offset.
  - Each model has 11–12 distinct velocity values for about 6 layers, because velocities differ across the fault. In model 0, the left column reads 1612/2389/2735/3583/3765/4097 m/s and the right column reads 1864/2503/2799/3379/4019/4381 m/s. So the hanging wall is not the footwall stratigraphy displaced.
  - Lateral CoV median 0.122.
- **[local measurement] CurveFault-A:** like FlatFault-A (mostly 2 layers, ramped interfaces, 2.7% inversions). The offset proxy (median 2 cells, p95 16) is confounded by the folding.
- **[local measurement] CurveFault-B:**
  - Median 6 layers (3–9); 4.8% inversions.
  - Offset proxy: median 13 cells, p95 37 cells.
  - Lateral CoV median 0.157, the highest lateral heterogeneity of all ten families.

**Style family (Style-A/B)**
- **Recipe:** a style-transfer network R(·) is trained with Marmousi as the style image and COCO images as content. The velocity is c = (1−α)·R(n) + α·c_b, where c_b is a 1D background whose velocity increases linearly with depth. OpenFWI does not give α. A vs B: "A is smoothed with a Gaussian filter and its seismic data contain fewer events; B is high-resolution." — [OpenFWI ar5iv](https://ar5iv.labs.arxiv.org/html/2111.02926)
- **Parameters of the source method (Feng, Lin & Wohlberg 2021, arXiv 2103.04007):**
  - About 67,000 COCO content images; Marmousi as the style image; a feed-forward style network trained on VGG-16 losses.
  - The output is converted to grayscale and becomes a velocity perturbation added to a linearly increasing 1D background, with perturbation weight 0.1–0.3.
  - Gaussian smoothing σ = 6–10 cells for low-resolution models and 0–5 for high-resolution models.
  - 2 km × 2 km models on a 10 m grid with a 15 Hz Ricker source.
  - OpenFWI's A (smoothed) and B (high-resolution) split likely mirrors this low/high-resolution split, but that is an inference. — [Feng et al. 2021 ar5iv](https://ar5iv.labs.arxiv.org/html/2103.04007)
- **[local measurement] Style-A:**
  - Continuous fields, about 1950 distinct values per model; velocity range 1535–4420 m/s.
  - The dataset-mean, laterally averaged velocity rises steadily from about 1720 m/s at the surface to about 3560 m/s at 690 m. Yet only 8.4% of models have a laterally averaged profile that is monotone in depth.
  - After removing the depth trend, the autocorrelation at a 5-cell lag is 0.72 vertically and 0.65 laterally, so the texture is nearly isotropic.
- **[local measurement] Style-B:** mean velocity rises from 1726 to 3763 m/s; detrended autocorrelation is 0.34 vertically and 0.35 laterally (rougher and isotropic); no model has a monotone laterally averaged profile.

### Inferences
- **What each family encodes geologically:**
  - FlatVel-A and CurveVel-A encode the textbook "velocity increases with depth" compaction prior with layer parallelism. CurveVel adds open folds whose relief, often 200–500 m in a 700 m model, is large compared with real dips.
  - The B variants deliberately break the monotonic-depth prior: half of their interfaces are inversions. That is far more inversions than typical compacting basins show (see Q3).
  - The Fault families encode one straight fault per model. In B, the velocities also change across the fault, which is not a pure kinematic offset of strata.
  - Style encodes a smooth depth trend plus isotropic, non-stratigraphic texture. It has no layer parallelism, because content comes from natural images and real sediments are laterally continuous.
- **Interfaces differ between families.** FlatFault-A and CurveFault-A have ramped (smoothed) interfaces, while every other family has sharp jumps. A network or prior tuned on one family's interface sharpness may therefore mis-handle another family.
- **Measured contrast ratios are large.** For the A families, the median reflection-coefficient-equivalent contrast is about 1.12, with a tail to about 1.56. In the B families, contrasts reach about 2×. Strong post-critical energy and head waves therefore matter in B.

### Gaps
- OpenFWI does not publish:
  - the number of layers or the layer-width distribution;
  - the ranges of aᵢ and kᵢ or the number of warping iterations;
  - the number of faults or the throw and dip ranges;
  - the Style α or Gaussian σ.
  I found no released generator code. The OpenFWI-Prior-Toolkit repository (github.com/minchen073/OpenFWI-Prior-Toolkit) appeared in search but was not checked.
- The fault shard names (`vel2_1_0`, `vel6_1_0`) may encode a generation parameter, possibly a layer-count group. The single FlatFault-A and CurveFault-A shards measured (mostly 2-layer) may therefore not represent the whole family. Other shards should be checked before the layer statistics are quoted as family-wide.
- The UPFWI fault "tilting angle −123° to 123°" is almost certainly a rendering artifact; it might be ±12°, but that is unverified. I could not measure fault dips robustly from the shards: only 5 FlatFault-A models had a single clean fault line, with dips of 64–88° from horizontal.

---

## Q2. Which families are hardest for current networks, and why?

### Takeaway
The "B" families are consistently the hardest. The order is roughly:
1. CurveFault-B (SSIM about 0.60–0.63)
2. CurveVel-B (about 0.67–0.73)
3. FlatFault-B (about 0.72–0.76)
4. Style-B (about 0.70–0.76)

The easiest are FlatVel-A and FlatFault-A (SSIM about 0.98–0.99). Physics-driven FWI fails on the same B families. Cross-family transfer into CurveVel-B collapses (SSIM 0.42).

### Cited Findings
**OpenFWI Table 4, MAE / RMSE / SSIM (ℓ1 loss)**

| Dataset | InversionNet | VelocityGAN | UPFWI |
|---|---|---|---|
| FlatVel-A | 0.0131 / 0.0211 / 0.9895 | 0.0118 / 0.0178 / 0.9916 | 0.0621 / 0.1233 / 0.9565 |
| FlatVel-B | 0.0351 / 0.0876 / 0.9461 | 0.0329 / 0.0807 / 0.9521 | 0.0677 / 0.1493 / 0.8874 |
| CurveVel-A | 0.0685 / 0.1273 / 0.8074 | 0.0482 / 0.1034 / 0.8624 | 0.0805 / 0.1411 / 0.8443 |
| CurveVel-B | 0.1497 / 0.2891 / 0.6727 | 0.1268 / 0.2618 / 0.7111 | 0.1777 / 0.3179 / 0.6614 |
| FlatFault-A | 0.0172 / 0.0426 / 0.9766 | 0.0868 / 0.1485 / 0.9313 | 0.0876 / 0.2060 / 0.9340 |
| FlatFault-B | 0.1055 / 0.1741 / 0.7208 | 0.0925 / 0.1600 / 0.7476 | 0.1416 / 0.2220 / 0.6937 |
| CurveFault-A | 0.0260 / 0.0650 / 0.9566 | 0.0258 / 0.0606 / 0.9613 | 0.0500 / 0.0966 / 0.9495 |
| CurveFault-B | 0.1646 / 0.2477 / 0.6163 | 0.1571 / 0.2427 / 0.5996 | 0.3452 / 0.5010 / 0.3941 |
| Style-A | 0.0625 / 0.1024 / 0.8859 | 0.0612 / 0.1000 / 0.8883 | 0.1429 / 0.2342 / 0.7846 |
| Style-B | 0.0689 / 0.1614 / 0.6314 | 0.0697 / 0.1108 / 0.6953 | 0.1702 / 0.2609 / 0.6102 |

With ℓ2 loss, InversionNet on Style-B reaches 0.0586 / 0.0893 / 0.7599. — [OpenFWI ar5iv](https://ar5iv.labs.arxiv.org/html/2111.02926)

- **Physics-driven FWI, OpenFWI Table 16:**
  - CurveVel-B from a homogeneous start gives SSIM 0.4157.
  - FlatFault-B is poor from every start: SSIM 0.5547 (homogeneous), 0.5907 (linearly increasing), 0.6596 (smoothed true model).
  - On 3D Kimberlina the inverted SSIM (0.5861) is lower than the starting model's (0.7975). — [OpenFWI ar5iv, appendix](https://ar5iv.labs.arxiv.org/html/2111.02926)
- **Pseudo-real Gulf of Mexico data (OpenFWI appendix N):**
  - For ℓ1 training, the text names FlatVel-B as the best-fitting training set, but the figure caption says FlatFault-B. The paper is internally inconsistent here.
  - For ℓ2 training, Style-A is best.
  - The authors recommend the Style family when no prior is available, because it "yields highly diversified features". — [OpenFWI ar5iv, appendix](https://ar5iv.labs.arxiv.org/html/2111.02926)

**Jin et al. 2023 (arXiv 2307.15388; Scientific Reports 2024), InversionNet → BigFWI, MAE / RMSE / SSIM**

| Dataset | InversionNet | BigFWI |
|---|---|---|
| FlatVel-A | 0.0092 / 0.0163 / 0.9930 | 0.0066 / 0.0137 / 0.9922 |
| CurveVel-B | 0.1344 / 0.2697 / 0.6995 | 0.1155 / 0.2471 / 0.7309 |
| FlatFault-B | 0.1012 / 0.1672 / 0.7267 | 0.0855 / 0.1514 / 0.7573 |
| CurveFault-B | 0.1610 / 0.2419 / 0.5981 | 0.1425 / 0.2251 / 0.6292 |
| Style-B | 0.0609 / 0.0971 / 0.7303 | 0.0600 / 0.0962 / 0.7324 |

The authors attribute the B difficulty to "the greater complexity of the B datasets" and note "the gap is narrower" for CurveFault-B and Style-A. — [Jin et al. 2023 HTML](https://arxiv.org/html/2307.15388v2)

- **Leave-one-out generalization (Jin et al., Table S8), target trained from source, InversionNet → BigFWI SSIM:**

| Target (trained on) | InversionNet SSIM | BigFWI SSIM |
|---|---|---|
| CurveVel-B (FlatFault-B) | 0.4234 | 0.5373 |
| CurveFault-B (FlatFault-B) | 0.5369 | 0.5905 |
| FlatVel-B (CurveVel-B) | 0.7797 | 0.8345 |

  Training on all 10 sets (470K pairs) improves average MAE by 13.03% over per-dataset training, and by 28.60% in the leave-one-out tests. Both figures are abstract-level aggregates. — [Jin et al. 2023](https://arxiv.org/pdf/2307.15388)

### Inferences
- The difficulty ranking lines up with the geology measured in Q1:
  - CurveVel-B combines random velocity order (about 50% inversions) with large folds. Low-velocity layers under fast layers are weakly constrained by reflections and absent from diving waves, a classic FWI blind spot.
  - CurveFault-B has the highest lateral heterogeneity (CoV 0.157), the most layers, and velocity changes across faults.
  - The A families satisfy "velocity increases with depth", so a monotone or compaction prior is almost exactly correct for them.
- **How much can a depth prior help?** A depth prior can substantially help FlatVel-A, CurveVel-A, FlatFault-A, CurveFault-A and Style (mean trend). It is wrong for about half the interfaces in FlatVel-B and CurveVel-B. Any monotone-depth prior must be soft, or it will bias B results.

### Gaps
- No per-family error analysis says *where* errors concentrate (for example below inversions versus at faults). I found no paper that attributes B errors to velocity inversions specifically; that link is my inference.

---

## Q3. Which geological priors are realistic and useful for velocity estimation?

### Takeaway
The robust priors are:
- **Compaction-driven increase of velocity with depth** under hydrostatic pressure. It has known exceptions: overpressured or undercompacted shales, exhumation, salt, carbonates and gas.
- **Stratigraphic continuity**, meaning smoothness along reflectors and not across them.
- **Piecewise-constant or sharp bodies** such as salt (halite about 4500 m/s, often treated as constant) and fault discontinuities.

Each of these maps onto a classical regularizer (Q4) or a learned prior (Q5).

### Cited Findings
- **Normal compaction trend.** Sonic velocity increases with depth in relatively homogeneous, brine-saturated sediments as compaction reduces porosity. Compaction is "normal" when fluid pressure is hydrostatic and exhumation has not thinned the overburden. This is pre-2010 background (Japsen, Mukerji & Mavko 2007, Geophysical Prospecting 55(2)); the attribution comes from a search snippet. — [Geophysical Prospecting 55(2), 2007](https://www.earthdoc.org/content/journals/gpr/55/2)
- **Exceptions: overpressure and undercompaction.**
  - Rapid burial traps pore fluid, lowers effective stress and lowers velocity and impedance relative to the normal trend. Departure from the normal compaction trend is used to flag overpressure.
  - Seismic velocities may under-predict overpressure; one example is the West Baram Delta, where the extra pressure is "inflationary".
  - Seismic prediction is limited by resolution and by uncalibrated velocity models.
  - These are snippet-level findings from multiple regional studies. — [Search & Discovery 2012](https://www.searchanddiscovery.com/abstracts/html/2012/90155ice/abstracts/ib.htm); [Niger Delta, Afam field 2014](https://www.searchanddiscovery.com/abstracts/html/2014/90188geo/abstracts/alao.htm); [NERC NORA 502603](https://nora.nerc.ac.uk/id/eprint/502603/)
- **Salt as a high-velocity, near-constant body.**
  - Pure halite is about 4500 m/s and is "often assumed to be a constant in velocity modelling".
  - Salt bodies can contain anhydrite (about 6500 m/s) or K-Mg salts (as slow as about 3500 m/s).
  - One survey flattened gathers with clean salt at 4350 m/s, while "dirty" salt velocity varied with thickness.
  - Sources: Jones & Davison 2014, Interpretation salt tutorial, and a TGS East Mediterranean velocity-model-building paper (2018); attribution from search snippets. — [Jones & Davison 2014 (TGS PDF)](https://tgs.com/hubfs/ION%20Papers/2014_INT_IJones_salt_tutorial.pdf); [TGS EastMed VMB 2018](https://www.tgs.com/hubfs/Igor/residuals/2018_EastMed_VMB_Th_EM_01.pdf)
- **Salt in inversion practice.** Salt-specific FWI regularizers assume a known salt velocity with an unknown shape (level sets), or blocky, bounded models (TV with bound constraints); see Q4. — [Kadu et al. 2017](https://arxiv.org/pdf/1610.00251); [Esser et al. 2018](https://ar5iv.arxiv.org/html/1608.06159)
- **Background depth trend in practice.** Practitioners build the water and sediment background as V₀ + kZ (velocity increasing linearly with depth), then insert salt. Benchmark generators such as OpenFWI-Style and Feng et al. 2021 likewise use a linearly increasing 1D background. — [TGS EastMed VMB 2018](https://www.tgs.com/hubfs/Igor/residuals/2018_EastMed_VMB_Th_EM_01.pdf); [Feng et al. 2021](https://ar5iv.labs.arxiv.org/html/2103.04007)
- **Structural orientation as a prior.**
  - Dip and azimuth fields and coherence lengths estimated from migrated images are used as "prior geological information" to steer anisotropic smoothing in FWI.
  - Examples: Trinh et al. 2017 (GJI 209(3), 1489–1512); the EGU 2018 follow-up tested on Marmousi and SEAM II. — [EAGE 2017 proceedings listing](https://www.earthdoc.org/content/proceedings/paris2017-annual?page=13); [EGU2018-5440](https://meetingorganizer.copernicus.org/EGU2018/EGU2018-5440.pdf?pdf)

### Inferences
- **Usefulness of each prior for networks:**
  - A *soft* monotone or compaction prior is a strong default for sedimentary basins. It matches every OpenFWI A family exactly.
  - A *hard* monotone prior would be wrong in overpressured shales, under salt or basalt, and in about 50% of FlatVel-B and CurveVel-B interfaces.
  - The most transferable priors are geometric rather than numeric. They are: (a) smoothness along local reflector dip; (b) sharp discontinuities allowed only across interfaces and faults; (c) blocky high-velocity bodies with known velocity.
  - These are exactly what the image-derived dip field encodes, and they map directly onto per-pixel input channels.
- **Fault throw as a prior.** Fault throw in real data is usually a consistent offset of the same stratigraphy. FlatFault-B and CurveFault-B change layer velocities across the fault (Q1), so a "same layers on both sides" prior would be violated in those B families.

### Gaps
- I found no sourced quantitative compaction law with parameters (for example Athy-type porosity decay or Faust or Gardner relations) within this research budget. Classical sources predate 2010.
- I found no sourced numbers on real horizontal-versus-vertical correlation-length ratios, which would quantify how non-geological the isotropic Style textures are.
- Fault throw/length scaling and typical dip ranges for normal and reverse faults were not sourced.

---

## Q4. Classical geology-guided regularization in FWI and tomography, and reported improvements

### Takeaway
The main classical families are:
- Structure-oriented or image-guided smoothing and sparsity, using dip fields from migrated images;
- TV and directional TV;
- Parametric level sets and shape optimization for salt;
- Space-variant anisotropic Tikhonov regularization.

Most reported benefits are qualitative: sharper salt boundaries, preserved interfaces, robustness to noise and sparse data, faster convergence. Few papers give standardized numbers.

### Cited Findings
- **Image-guided FWI (Ma 2012, Colorado School of Mines thesis, advisor D. Hale).** IGFWI uses structures extracted from migrated images to constrain a sparse model, solved by image-guided conjugate gradients with projected BFGS. It reports faster convergence than conventional FWI and big time and memory savings from P-BFGS compared with L-BFGS. The thesis, not a journal paper, was confirmed. — [Mines repository](https://repository.mines.edu/handle/11124/76636)
- **Geologically constrained FWI by preconditioning (Guitton & Ayeni, SBGf 2011).** This is the precursor of the "model reparameterization" approach. I could not confirm the Geophysics 2012 journal version or its directional-Laplacian details. — [SBGf 2011](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/12th_CISBGf/SBGF_2739.pdf)
- **Adaptive structure-oriented smoothing (EAGE 2013).** Local slopes from the migrated image steer directional smoothing, updated during inversion. On the SEG/EAGE salt model, "layer interfaces are faithfully preserved and sharpened"; the result is qualitative. — [EarthDoc 10.3997/2214-4609.20131009](https://www.earthdoc.org/content/papers/10.3997/2214-4609.20131009)
- **Directional TV (EAGE 2017).** An L1 directional-TV operator is built from the local dip field of the seismic image. On Marmousi it outperforms both unregularized FWI and isotropic TV (qualitative, single paper). From search, it is one of these two EarthDoc records, and the mapping is uncertain. — [EarthDoc 201701360](https://www.earthdoc.org/content/papers/10.3997/2214-4609.201701360); [EarthDoc 201701340](https://www.earthdoc.org/content/papers/10.3997/2214-4609.201701340)
- **Structure-oriented constraints for traveltime tomography (EAGE 2017).** Dip fields and discontinuity locations from migrated images feed a structure-oriented filter within TV regularization. — same EarthDoc records as above (search-level)
- **Seislet-regularized FWI (Xue et al. 2017, Geophysics).** This enforces sparsity in the seislet domain, whose basis needs structural dips estimated iteratively from migrated images. It is more robust to crosstalk and noise on Marmousi. — [UT Dallas repository](https://utd-ir.tdl.org/items/67388c5c-3aeb-4664-a225-78f7cc6dfe00)
- **Structure-smoothing Bessel filter (Trinh, Brossier, Métivier, Virieux, Wellington 2017, GJI 209(3)).** A nonstationary anisotropic filter built from Bessel-function PDEs, taking space-varying coherence lengths and 3D orientation as geological priors. It preconditions FWI gradients on spectral-element meshes and was tested on Marmousi and SEAM II. — [EGU2018-5440](https://meetingorganizer.copernicus.org/EGU2018/EGU2018-5440.pdf?pdf); [EAGE 2017 listing](https://www.earthdoc.org/content/proceedings/paris2017-annual?page=13)
- **TV regularization strategies (Esser, Guasch, van Leeuwen, Aravkin, Herrmann 2018, SIAM J. Imaging Sci. 11, 376–406).** Bound and TV constraints on squared slowness sharpen salt delineation on BP 2004. With poor starts they miss the bottom of salt, and a one-sided asymmetric TV constraint fixes that. — [arXiv 1608.06159](https://ar5iv.arxiv.org/html/1608.06159); [SLIM](https://slim.gatech.edu/node/6656)
- **Parametric level set for salt (Kadu, van Leeuwen, Mulder 2017, IEEE TCI 3(2), 305–315).** The model is a background plus a salt body of known velocity and unknown shape. The level set is represented by radial basis functions, which gives few unknowns, and the boundary width is adapted during inversion. It is stable under modest noise and was extended to joint background-plus-salt inversion (SEG 2017). — [arXiv 1610.00251](https://arxiv.org/pdf/1610.00251)
- **Space-variant anisotropic Tikhonov (Gholami & Gazzola 2025, IEEE GRSL 22).** The orientation and anisotropy of directional smoothing are treated as unknowns and estimated jointly with velocity. The authors report "significantly improving" sparse-data results compared with standard Tikhonov (qualitative in the abstract). Related: robust estimation of structural orientation for local anisotropic Tikhonov (arXiv 2409.05754, 2024). — [arXiv 2503.08187](https://arxiv.org/abs/2503.08187); [arXiv 2409.05754](https://arxiv.org/pdf/2409.05754)
- **Other priors listed in a 2023 review:**
  - Tikhonov regularization favours smooth models; TV favours piecewise-smooth ones.
  - Curvelet and seislet sparsity depend on the chosen basis.
  - Dictionary-learning FWI (Zhu et al. 2016/2017; Huang et al. 2019) is sensitive to hyperparameters. — [Wang, Huang, Alkhalifah 2023](https://arxiv.org/html/2306.12776v1)

### Inferences
- Classical work already turns "smooth along reflectors, sharp across them" into a dip field computed from a migrated image. A per-pixel dip, coherence or fault-likelihood channel is therefore the natural network-input analogue of these regularizers, and it is cheap to compute.
- Level sets and asymmetric TV encode "blocky, known-velocity bodies". That prior is irrelevant to OpenFWI except to the extent its layers are piecewise constant (all Vel and Fault-B families).

### Gaps
- I found no paper that benchmarks structure-oriented regularization on OpenFWI, or that reports standardized MAE/SSIM gains for classical geology-guided FWI.
- The EAGE 2013 and 2017 EarthDoc pages returned HTTP 403, so author names and numbers are not confirmed.

---

## Q5. Deep generative priors (GAN, VAE, diffusion) for velocity models, and their results

### Takeaway
Generative priors moved from GANs to diffusion models:
- GAN plus MCMC (Mosser et al. 2018/2020);
- Diffusion-regularized FWI (Wang, Huang, Alkhalifah 2023), with a prior trained on OpenFWI;
- Conditional diffusion with geology-class and well-log conditioning, CD-FWI (2024/2025). On OpenFWI-derived tests it raises SSIM from 0.670 (FWI) to 0.900 (unconditional diffusion) to 0.976 (class + well).

The repeated caveat is that the prior must match the target geology. An OpenFWI prior was not well matched to field data, and well-guided random models worked better there.

### Cited Findings
- **GAN prior (Mosser, Dubrule, Blunt; arXiv 2018, journal 2020).** A GAN prior on geology is combined with an acoustic wave solver. The posterior is sampled with approximate Metropolis-adjusted Langevin MCMC, using adjoint gradients through the latent space. — [arXiv 1806.03720](https://arxiv.org/pdf/1806.03720)
- **Diffusion-regularized FWI (Wang, Huang, Alkhalifah 2023, arXiv 2306.12776).**
  - Prior: trained on 84,000 OpenFWI velocity models cropped to 64×64.
  - Method: about 10 FWI iterations run between reverse-diffusion steps; the denoiser acts as a proximal or MAP Gaussian denoiser, and only the late, high-resolution reverse steps are needed.
  - Results (qualitative): better resolution than FWI and robustness to noise, 3-shot acquisition, and missing frequencies below 5 Hz.
  - Field data (NW Australia, 116 shots, 8 Hz): the OpenFWI-trained prior did not match the target area well. A prior trained on 2000 well- and geology-guided random models (Kazei et al. 2021 and Ovcharenko et al. 2022 generators) improved lateral continuity and illumination artifacts. — [arXiv 2306.12776 HTML](https://arxiv.org/html/2306.12776v1)
- **CD-FWI: geological and well prior via conditional diffusion (Wang, Huang, Alkhalifah; arXiv 2412.06959, v2 2025).**
  - Method: a classifier-free-guided DDPM conditioned on geological class labels (eight OpenFWI families) and on well logs (a vertical profile plus positional encoding).

  | Test | Conventional FWI | Diffusion FWI | CD-FWI (class + well) |
  |---|---|---|---|
  | Synthetic: PSNR / SSIM / rel. L2 | 28.10 / 0.670 / 0.0489 | 36.49 / 0.900 / 0.0183 | 41.51 / 0.976 / 0.0105 |
  | Limited shots: SSIM | 0.468 | 0.543 | 0.623 |
  | Well at 0.40 km: SSIM | – | 0.786 | 0.810 |

  - Field CGG data (NW Australia): the well-conditioned result was biased low near 3 km, attributed to too few low-velocity training examples at that depth and to anisotropy.
  - These are single-paper results. — [arXiv 2412.06959v2](https://arxiv.org/html/2412.06959v2)
- **DiffusionVel (Zhang, Li, Huang; arXiv 2410.21776, Oct 2024).**
  - Method: two conditional diffusion models (one on seismic, one on well logs) plus an unconditional "geology-oriented" model. A low-pass background velocity is added. Scores are combined by weighted summation during sampling.
  - Data: OpenFWI and Hess VTI. Compared against FWI, InversionNet and VelocityGAN.
  - The abstract gives no numbers. — [arXiv 2410.21776](https://arxiv.org/abs/2410.21776)
- **Later diffusion-prior FWI, mostly 2025–2026 preprints (search level):**
  - RED-DiffEq uses pretrained diffusion as a regularizer for PDE inverse problems, including FWI (arXiv 2509.21659).
  - Diffusion-guided optimization with a prior trained on GeoFWI patches, applied patchwise to Marmousi, follows the true profile better than ℓ2 FWI or TV; TV "underestimates velocity contrasts" (arXiv 2607.21987).
  - Decoupled Latent Optimization transfers an OpenFWI 70×70 prior to Marmousi and Overthrust (arXiv 2606.14139; secondary summary).
  - A conditional diffusion regularizer handles velocity–density coupling (arXiv 2603.22307, Mar 2026).
  - Diffusion regularization with model-wavenumber continuation for elastic FWI (GJI vol. 240, issue 2). — [arXiv 2509.21659](https://arxiv.org/html/2509.21659v1); [arXiv 2607.21987](https://arxiv.org/pdf/2607.21987); [EmergentMind 2606.14139](https://www.emergentmind.com/papers/2606.14139); [alphaXiv 2603.22307](https://www.alphaxiv.org/abs/2603.22307); [GJI 240(2)](https://dbpia.nl.go.kr/gji/article/240/2/1215/7924249)
- **Generative transformer prior.** A pre-trained velocity-model generative transformer propagates the prior from shallow to deep (arXiv 2408.09767, 2024). Search result only; not read. — [arXiv 2408.09767](https://arxiv.org/pdf/2408.09767)

### Inferences
- **Practical lesson for networks.** Generative priors give the largest gains when they are *conditioned* on geology class and wells, as in CD-FWI. A cheap explicit geological channel, such as a dip field, fault probability or depth trend, plays a similar conditioning role without training a diffusion model.
- **OpenFWI as a prior has known limits.** Its priors transfer poorly to field data: Wang et al. 2023 had to regenerate training models guided by wells.

### Gaps
- I found no VAE-prior FWI paper with numbers in this budget.
- I found no head-to-head comparison of GAN versus diffusion priors on the same benchmark.
- The 2026 diffusion-FWI preprints were seen only through search summaries; their metrics are unverified.

---

## Q6. Geological knowledge as training-data design (synthetic generators, domain randomization, large and foundation models) and its effect on field generalization

### Takeaway
Rule-based generators are the most widely used route for injecting geology into training data:
- Folded and faulted layered models (Wu et al. 2019/2020) let networks trained only on synthetics work on field seismic, at least for interpretation tasks.
- Style transfer from natural images (Ovcharenko 2019; Feng et al. 2021; OpenFWI-Style) adds texture diversity but generalizes only weakly out of distribution.
- Scaling data diversity and model size (Jin et al. 2024; Feng et al. 2026) gives the clearest measured gains on unseen benchmarks, for example SSIM 0.584 → 0.767 across six classic models.

### Cited Findings
- **FaultSeg3D (Wu, Liang, Shi, Fomel 2019, Geophysics 84(3) IM35–IM45).**
  - About 200 automatically generated realistic folded and faulted structure models with synthetic seismic and fault labels; a class-balanced cross-entropy loss.
  - Training takes about 2 hours on a TITAN Xp, and the network predicts faults in a 450×1950×1200 volume in under 3 minutes.
  - It is claimed to outperform conventional methods on multiple field datasets.
  - Generalization is contested: later work reports degradation on field data for models trained on small synthetic sets. — [BEG seminar abstract](https://www.beg.utexas.edu/node/4342); [Wu SEG wiki](https://wiki.seg.org/wiki/Xinming_Wu); [review arXiv 2606.09094](https://arxiv.org/pdf/2606.09094)
- **Wu, Geng, Shi, Pham, Fomel, Caumon 2020 (Geophysics 85(4) WA27–WA39).** Structural features are simulated in 3D models from parameters drawn randomly within predefined ranges. The output is many diverse structure models, synthetic seismic images and their structural labels. The trained CNNs "performed well on multiple field seismic images". The parameter ranges could not be retrieved because the HAL PDF was blocked. — [HAL hal-02570636](https://hal.univ-lorraine.fr/hal-02570636)
- **Style transfer for texture.**
  - Ovcharenko, Kazei, Peter, Alkhalifah (SEG 2019) generate realistically textured subsurface models by style transfer.
  - Kazei, Ovcharenko, Alkhalifah, Simons (EAGE 2019) shuffle wavelet-packet coefficients of a guiding model to replicate realistic textures, used for low-frequency extrapolation on SEAM. — [EAGE 2019 PDF](https://geoweb.princeton.edu/people/simons/PDF/reprints/EAGE-2019.pdf)
- **Feng, Lin, Wohlberg 2021 (arXiv 2103.04007).**
  - Out of distribution, the low-resolution network reaches MSE/SSIM of 0.1065/0.5266 and 0.0720/0.6397 on Marmousi and 0.0343/0.6374 and 0.0200/0.7377 on Overthrust.
  - The high-resolution network was *worse than its initial model* on both: Marmousi SSIM 0.1693 → 0.1522; Overthrust 0.3586 → 0.2895.
  - On Gulf of Mexico field data the results were qualitatively consistent with traveltime tomography in the shallow section and better focused at depth, but the high-resolution network created a spurious low-velocity zone. — [ar5iv 2103.04007](https://ar5iv.labs.arxiv.org/html/2103.04007)
- **Jin et al. 2023/2024.** Training on all 10 OpenFWI 2D subsets (470K pairs) improves MAE by an average of 13.03% over single-set training and by 28.60% in leave-one-out generalization. Model capacity must scale with data. — [arXiv 2307.15388](https://arxiv.org/pdf/2307.15388)
- **Feng, Jin, Guo, Chen, Lin 2026 ("Improving FWI in Large Model Era", arXiv 2603.00377).**
  - A billion-parameter model trained "entirely on simulated and relatively simple data", scaled along model capacity, data diversity and training strategy.
  - It generalizes to Marmousi, SEG/EAGE Salt, Overthrust, BP 2004, Sigsbee and SEAM I, with SSIM improving from 0.5844 to 0.7669 (aggregate) and state of the art on OpenFWI. Single paper. — [arXiv 2603.00377](https://arxiv.org/abs/2603.00377)
- **Prior-guided random generators for field priors.** Wang et al. 2023 generated 2000 random velocity models "guided by well information and geological expectations" with the Kazei et al. 2021 and Ovcharenko et al. 2022 implementations. These worked better on field data than an OpenFWI-trained prior. — [arXiv 2306.12776](https://arxiv.org/html/2306.12776v1)
- **New realistic datasets (2026).** SubsurfaceGen (May 2026, secondary summary) argues that Marmousi, SEAM and OpenFWI lack spatial and temporal extent, geological diversity and physical realism, and releases field-scale paired data from 3D models. GeoFWI is a small PyPI velocity-model benchmark with no documentation found. — [EmergentMind 2605.30541](https://www.emergentmind.com/papers/2605.30541); [PyPI geofwi](https://pypi.org/project/geofwi/)

### Inferences
- **Measured benefit comes mostly from diversity and conditioning, not from realism alone.** Style-transfer textures help in distribution but degraded results out of distribution in Feng et al. 2021.
- **Geometric rules transfer better.** Rule-based structural generators (folds, faults, unconformities) carry geometric rules that transfer across surveys. This supports encoding geology as geometry: dips, faults and layer parallelism.

### Gaps
- I found no study that isolates the effect of a single geological rule in training data on velocity-inversion accuracy. Examples of such ablations would be monotone versus random layer order, or fault on versus off.
- The exact parameter ranges of the Wu 2020 generator (fold amplitudes, fault dips and throws) were not retrieved.

---

## Q7. Work that uses well logs or geological constraints as extra network inputs

### Takeaway
Well logs enter networks in two ways:
- As a sparse 2D "well channel": a mask that is zero except at well locations, optionally alongside a migrated image and an initial model.
- As a conditioning vector with positional encoding in diffusion models.

The reported gains are moderate:
- **CD-FWI:** SSIM 0.900 → 0.976 with class plus well, and 0.786 → 0.810 from one well.
- **Others:** "improved data fit and structural similarity", stated only qualitatively.

### Cited Findings
- **CD-FWI.** The well log is a vertical velocity profile with positional encoding of its location, and geology class labels are embedded through an MLP. Gains are listed in Q5. — [arXiv 2412.06959v2](https://arxiv.org/html/2412.06959v2)
- **DiffusionVel.** A dedicated well-log-conditional diffusion model is fused with seismic-conditional and unconditional geology models, plus a low-pass background velocity. — [arXiv 2410.21776](https://arxiv.org/abs/2410.21776)
- **Zhang, Cheng, Chen, Mao 2025 (arXiv 2503.02815).** A generative-diffusion "well interpolation" produces FWI starting models with uncertainty. Per the search snippet, the input has four channels: noisy velocity, initial model, a sparse well mask, and a migrated image used as a structural constraint. The abstract gives no numbers. — [arXiv 2503.02815](https://arxiv.org/abs/2503.02815); [PDF](https://arxiv.org/pdf/2503.02815)
- **Deep tomography with well information (SBGf, 18th CISBGf).** Well-log velocity is added as a sparse extra input channel to a U-Net that predicts velocity updates from subsurface-offset panels. The preliminary result is improved data fit and structural similarity. — [SBGf PDF](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/18th_CISBGf/4c56ff4ce4aaf9573aa5dff913df997adeeptomography_wellinfo_AbstractExpanded.pdf)
- **Kazei et al. (KAUST, Geophysics 2021).** CNN ensembles map gathers of neighbouring CMPs to 1D velocity logs, which "allows us to integrate well-log data into the inversion". — [KAUST SWAG](https://swagroup.kaust.edu.sa/research/detail/an-alternative-to-fwi); [Kazei profile](https://swagroup.kaust.edu.sa/people/detail/vladimir-kazei)
- **EAGE 2026.** A depth-progressive 3D diffusion model for velocity building conditions on shallower depth priors, well logs and images (search snippet). — [EarthDoc 202610063](https://www.earthdoc.org/content/papers/10.3997/2214-4609.202610063)
- **Text-guided generation.** Well-log-constrained realistic subsurface model generation via Stable Diffusion (presentation; details not retrieved). — [SlideShare](https://www.slideshare.net/slideshow/text-guided-well-log-constrained-realistic-subsurface-model-generation-via-stable-diffusion/270634335)
- **Pre-deep-learning facies and well priors** (Singh et al. 2018; Zhang et al. 2018) were used as inversion constraints, and Li et al. 2021 and Zhang & Alkhalifah 2022 learned links between inverted models and well facies. These come from the CD-FWI paper's review. — [arXiv 2412.06959v2](https://arxiv.org/html/2412.06959v2)

### Inferences
- The well-as-sparse-channel design is the direct precedent for feeding "physics or geology per-pixel channels" to an OpenFWI network. Precedents exist for each candidate channel:
  - **Depth trend:** a low-pass background, as in DiffusionVel.
  - **Structural constraint:** a migrated image, as in 2503.02815.
  - **Well profile.**
- **Proposed ablation:** none of these papers isolates the benefit of each channel on OpenFWI families. A per-channel ablation (raw data versus data plus a depth-trend channel versus plus a dip channel versus plus a well channel), broken down by A/B family, would be new.

### Gaps
- The quantitative per-channel gains for 2503.02815 and the SBGf deep-tomography paper were not retrieved; only abstracts and snippets were read.
- I found no foundation-model paper (for example a seismic foundation model) that explicitly takes geological constraints as input. The 2026 billion-parameter FWI model (Q6) is data-scaled, not geology-conditioned, according to its abstract.
