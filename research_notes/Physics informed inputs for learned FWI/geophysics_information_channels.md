# Information channels in surface shot gathers and classical methods that map them to per-pixel quantities

Target setting used for all numbers below: 2D acoustic, constant density, 5 surface shots, 70 receivers at 10 m (700 m spread), 15 Hz Ricker, 1 s record, depths to 700 m, v = 1500-4500 m/s, absorbing boundaries (no free-surface multiples; internal multiples between strong contrasts are still in the data), noise-free. Where shot positions matter I assume they are spread evenly along the line (about 0, 170, 345, 520, 690 m), so the end shots reach about 690 m offset and the centre shot about 350 m.

Convention: "Cited Findings" are sourced statements. Numbers specific to this geometry are in "Inferences". I computed them from the cited formulas with a short script (scratchpad `geo_numbers.py` / `geo_numbers2.py`), and they are labelled as derived.

Reference numbers for this geometry (derived, used throughout):

| v (m/s) | lambda at 15 Hz | lambda/2 | lambda/4 (tuning) | lambda at about 37.5 Hz (upper useful band) |
|---|---|---|---|---|
| 1500 | 100 m | 50 m | 25 m | 40 m |
| 2000 | 133 m | 67 m | 33 m | 53 m |
| 3000 | 200 m | 100 m | 50 m | 80 m |
| 4500 | 300 m | 150 m | 75 m | 120 m |

Ricker amplitude spectrum relative to the 15 Hz peak, computed from A(f) proportional to f^2 exp(-f^2/fp^2): 1 Hz -38 dB, 2 Hz -26.5 dB, 3 Hz -19.6 dB, 5 Hz -11.4 dB, 10 Hz -2.2 dB, 30 Hz -14 dB, 40 Hz -36 dB, 50 Hz -67 dB. The usable band in noise-free data is therefore roughly 2-40 Hz.

---

## Q1. Kinematic information: first arrivals and diving waves, reflection traveltimes (NMO/semblance, Dix, stereotomography, reflection tomography). What each constrains and where it is blind in this geometry

### Takeaway
In a 700 m spread, first arrivals (direct, diving and head waves) only sample the top tens of metres to about 150 m. Most of the 700 m section is constrained kinematically only by reflection traveltimes and their slopes. Those have small moveout at depth (tens of ms) and a velocity-depth trade-off, and they constrain only the overburden above the deepest reflector. NMO and Dix assume flat layers. Stereotomography and reflection tomography do not, but they need picks of locally coherent events and regularization.

### Cited Findings
**Diving waves and first-arrival tomography**
- Longer offsets give deeper penetration of refracted (diving) energy. Data with offsets three or more times the target depth are called long-offset data. — [US patent 10353092 (low-authority source)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10353092)
- Worked example: with a surface velocity of 2 km/s rising 1 km/s per km of depth, a 4.47 km offset reaches about 1 km depth and a 6.93 km offset reaches 2 km. That is an offset-to-depth ratio of about 3.5-4.5. — [Harlan, SEP "Flexible tomography"](https://sepwww.stanford.edu/data/media/public/docs/sep89/harlan/paper_html/node1.html)
- For v(z) linear in depth there is a closed-form maximum penetration depth (Geldart and Sheriff problem set) that depends on surface velocity, gradient and offset. — [SEG Wiki, Diving waves](https://wiki.seg.org/wiki/Diving_waves) (search snippet; the page returned HTTP 403 to direct fetch)
- Diving-wave tomography includes vertical velocity gradients, which handles nonlinear first-arrival moveout and allows a wider offset range than head-wave methods. The inversion is nonlinear and sensitive to the initial model. — [CSEG Recorder, Refraction tomography: a practical overview](https://csegrecorder.com/articles/view/refraction-tomography-a-practical-overview-of-emerging-technologies)
- Resolution of first-arrival tomography: in a homogeneous medium the first-Fresnel-zone width scales as sqrt(lambda L) (Williamson 1991), so longer paths give coarser tomograms. The midpoint half-width for diving-wave transmission is sqrt(L lambda / 4). — [Y. Huang thesis, KAUST, resolution limits](https://csim.kaust.edu.sa/files/Theses/Yunsong.Huang_2013/out/node31.htm); [same, diving-wave transmission](https://csim.kaust.edu.sa/files/Theses/Yunsong.Huang_2013/out/node39.htm)
- Counterpoint: other authors argue that the Fresnel width is not a hard resolution limit, and that resolution near a scatterer approaches about lambda. — [Huang thesis node42](https://csim.kaust.edu.sa/files/Theses/Yunsong.Huang_2013/out/node42.htm); [Li and Duric, Rochester](https://labsites.rochester.edu/ultrasound-tomography-center/wp-content/uploads/2022/11/Resolution-limitation-of-travel-time-tomography_Beyond-the-first-Fresnel-zone.pdf)

**Head waves, critical distance, hidden layers**
- A head wave forms when a ray hits a faster layer at the critical angle, sin(theta_c) = V1/V2. It exists beyond the critical distance, but until the crossover distance it arrives after the direct wave and is often obscured by it. — [SEG Wiki, Head waves and effect of hidden layer](https://wiki.seg.org/wiki/Head_waves_(refractions)_and_effect_of_hidden_layer)
- There are two kinds of hidden layer: a layer too thin for its head wave ever to arrive first, and a low-velocity layer beneath a faster one, which produces no head wave at all. Missing a hidden layer makes the deeper refractor look too deep. — [SEG Wiki, Effect of a hidden layer](https://wiki.seg.org/wiki/Effect_of_a_hidden_layer); [LMU Munich lecture notes, sedi_ray](https://www.geophysik.uni-muenchen.de/~igel/Lectures/Sedi/sedi_ray.pdf)

**Reflection traveltimes: NMO, semblance, Dix**
- The hyperbolic NMO equation is exact only for a single homogeneous isotropic layer. With several interfaces it is an approximation that degrades at large offsets. — [Search and Discovery, Best fitting of NMO equations](https://www.searchanddiscovery.com/abstracts/html/2015/90217ice/abstracts/2151992.html)
- "There is no single velocity function" that corrects NMO at all offsets of a shot record. Moveout is non-hyperbolic for anything beyond one layer. — [First Break 2021](https://www.earthdoc.org/content/journals/10.3997/1365-2397.fb2021022)
- The RMS hyperbola strictly applies only at small offsets. Dip is usually absorbed into a "stacking velocity", which is then no longer a true RMS velocity. The Dix framework assumes plane horizontal layers. — [CREWES report 1997-29](https://crewes.org/Documents/ResearchReports/1997/1997-29.pdf)

**Reflection tomography and velocity-depth ambiguity**
- With limited offset and lateral velocity change, reflection tomography cannot always tell a traveltime error caused by velocity from one caused by reflector depth (Bickel 1990; Lines 1993). Interfaces and velocity cannot be uniquely determined without a-priori depth information. — [Soupios et al., Balkan Geophysical Congress](https://balkangeophysoc.gr/menu/congresses/oral/10-11-O3-3-soupios-pantelis.PDF)
- A 2026 statistical study finds that dozens of depth-velocity combinations produce reflection hyperbolas overlapping a base model's. — [arXiv 2601.04328](https://arxiv.org/html/2601.04328v1)
- Combining transmission and reflection traveltimes in one inversion (Mao and Stuart 1997) reduces the depth-velocity ambiguity substantially. — [OSTI record](https://www.osti.gov/biblio/514751) (attribution taken from the search summary)

**Slope-based tomography (stereotomography)**
- Stereotomography (Billette and Lambaré 1998) estimates a velocity macro-model from reflection traveltimes plus local slopes of locally coherent events in the prestack cube. It avoids interpretive picking of continuous horizons. — [Geophysical Prospecting, 10.1111/1365-2478.13181](https://www.earthdoc.org/content/journals/10.1111/1365-2478.13181)
- It is a generalized slope tomography that extended CDR tomography. Its large Fréchet matrix can make it infeasible for big surveys, which motivated adjoint stereotomography. — [CREWES 2018 report CRR201835](https://www.crewes.org/Documents/ResearchReports/2018/CRR201835.pdf)
- Data incompleteness makes regularization necessary for a stable slope tomography. — [Costa et al., WIT 2007](https://www.wit.uni-hamburg.de/import/documents/reports/2007/wit2007-costa.pdf)

### Inferences
- **Diving-wave depth in this geometry (derived).** For v = v0 + g z, rays are circular arcs and the turning depth for offset x is z_t = sqrt((x/2)^2 + (v0/g)^2) - v0/g. This reproduces Harlan's SEP numbers (999 m and 2001 m). With v0 = 1500 m/s:
  - x = 700 m: z_t = 20 m (g = 0.5/s), 40 m (g = 1/s), 78 m (g = 2/s), 145 m (g = 4.3/s, i.e. 1500 to 4500 m/s linearly over 700 m).
  - x = 350 m (centre shot): z_t = 10-41 m.
  - So diving waves sample only the top 3-20 % of the model. That matches the "offset at least 3-5 times depth" rule (700 m offset gives at most about 140-230 m).
- **Head waves in this geometry (derived).** Critical and crossover distances are x_c = 2h tan(theta_c) and x_cross = 2h sqrt((v2+v1)/(v2-v1)).
  - h = 200 m, 2000 over 3000 m/s: theta_c = 41.8 deg, x_c = 358 m, x_cross = 894 m. Beyond the spread, so the head wave is never a first arrival.
  - h = 200 m, 1500 over 2500 m/s: x_cross = 800 m.
  - h = 100 m, 1500 over 4500 m/s: x_cross = 283 m.
  - Head waves are first arrivals only for shallow (about 150 m or less), strong-contrast interfaces. Otherwise they arrive later, inside or near the post-critical reflection. This matches the user's earlier OpenFWI finding that head waves are never first arrivals there (project memory).
- **Lateral resolution of first-arrival tomography (derived).** sqrt(L lambda / 4) gives a midpoint half-width of about 94 m (L = 350 m, v = 1500 m/s) to 153 m (L = 700 m, v = 2000 m/s). That is a 10-15 cell blur on the 10 m grid, so first-arrival tomography yields a smooth near-surface v(x, z) map only in the top about 50-150 m.
- **Reflection moveout sensitivity (derived, far offset, 10 % velocity error).**

  | Reflector depth, velocity | Moveout at 700 m offset | Change from 10 % velocity error |
  |---|---|---|
  | z = 200 m, v = 2000 | 203 ms | 30 ms (0.46 periods at 15 Hz) |
  | z = 400 m, v = 2500 | 105 ms | 18 ms |
  | z = 700 m, v = 3000 | 55 ms | 10 ms (0.16 periods) |
  | z = 700 m, v = 3000, centre shot (350 m) | 14 ms | 3 ms |

  - Interval-velocity resolution from moveout therefore collapses with depth. Dix differentiation amplifies these small errors for thin intervals.
  - A 700 m reflector under slow (1500 m/s) cover has t0 = 0.93 s and t(700 m) = 1.04 s, which is cut off by the 1 s record.
- **Maximum reflection incidence angle with a 700 m offset (flat reflector, homogeneous overburden; derived).** About 74 deg at 100 m, 60 deg at 200 m, 35 deg at 500 m and 27 deg at 700 m. The centre shot gives roughly half these.
- **Methods ranked for this setting (inference).**
  - NMO, semblance and Dix: valid only for flat-layer families. They give a v_rms(t0) or v_int(z) column per CMP. CMP fold is 1-5, so semblance is poorly sampled.
  - Stereotomography and kinematic/slope tomography: general (dip and lateral variation). A natural per-location output is a smooth v(x, z) plus a ray-density map.
  - A 70-receiver dense line is ideal for receiver-side slopes. With only 5 shots, source-side slopes are not measurable (no source-side spatial derivative), so only receiver-side slope p_r plus traveltime is available per event. That is half the stereotomographic data vector. This is my reasoning, not a sourced statement.
- **Velocity-depth ambiguity in practice.** With one reflector, only the RMS (average) overburden velocity is constrained. This matches the user's earlier toy result (project memory: one reflector constrains only RMS overburden velocity; smoothness priors resolve the profile).

### Gaps
- Could not fetch the SEG Wiki closed-form diving-wave formula (HTTP 403). The z_t formula above is derived from circular-ray geometry and checked against Harlan's numbers, not quoted.
- No primary source fetched for semblance (Taner and Koehler 1969) or for Dix's 1955 derivation; covered here via secondary reports.
- No quantitative published resolution figure for stereotomography at 15 Hz on short spreads was found.
- Lambaré's 2008 review ("Stereotomography", Geophysics) was not located in open access.

---

## Q2. Image-domain velocity information: MVA, common-image gathers (offset, angle, subsurface offset / extended images), residual moveout, focusing and defocusing, velocity continuation, differential semblance. How velocity errors appear in images

### Takeaway
Velocity errors show up in the image domain in three ways: (1) residual moveout, meaning non-flat events in offset or angle CIGs; (2) energy leaking away from zero subsurface offset or zero time lag in extended images; (3) shift and defocusing of diffractions and reflectors. These measurements are per-pixel by construction. Shot-profile wave-equation migration with a subsurface-offset imaging condition works with sparse shots, which suits a 5-shot survey. Ray-based RMO analysis and depth focusing analysis assume small velocity error or flat dips. DSO-type objectives are robust to large errors but suffer from acquisition-edge artefacts.

### Cited Findings
- Waveform inversion has spurious local minima and needs accurate long-scale velocity. MVA "is capable of correcting substantially erroneous initial estimates" of long-scale velocity, and extended modelling links the two. — [Symes 2008, Geophysical Prospecting 56(6) abstract](https://www.earthdoc.org/content/journals/10.1111/j.1365-2478.2008.00704.x)
- **Biondi and Symes on wavefield-continuation CIGs.**
  - In an angle-domain CIG cube, the image point lies on the normal to the apparent reflector dip, through the point where the source and receiver rays intersect. Its position along that normal depends on the velocity error. With correct velocity the image point is the ray intersection. They derive analytical residual-moveout functions in ADCIGs from this.
  - The normal shift caused by incomplete focusing at zero offset is Delta n_tot = Delta n_h0 (1 + tan^2 gamma), where gamma is the aperture angle.
  - Horizontal-offset CIGs degenerate for steep dips ("their focusing around zero offset blurs"). Vertical-offset CIGs are needed for steep events.
  - Source for all three points: [Biondi and Symes, Angle-domain CIGs for MVA by wavefield-continuation imaging (Geophysics 2004)](https://sepwww.stanford.edu/sep/biondo/PDF/Journals/Geophysics/adcig-geo-online.pdf)
- Time-shift gathers (Sava and Fomel 2006 space- and time-shift imaging conditions): departures from zero shift indicate velocity errors or illumination problems. Differential semblance can be applied to these gathers. — [Guerra and Biondi, SEP / SEG 2009](https://sep.stanford.edu/sep/claudio/Research/SEG/2009/claudio2/paper.pdf)
- Extended-Born modelling enlarges the search space in the model domain with subsurface offsets or time lags (Symes 2008; Biondi and Almomin 2014). — [Operto et al. 2023, FWI beyond the Born approximation](https://arxiv.org/pdf/2212.10141)
- Migration smiles on subsurface-offset gathers, caused by finite acquisition, can stop the differential semblance functional from reaching its minimum at the expected value. A data-domain weighting has been proposed to fix this. — [HAL hal-01143751](https://hal-mines-paristech.archives-ouvertes.fr/hal-01143751v1)
- Velocity continuation (Fomel) is an MVA tool. It improves on residual NMO by accounting for lateral movement of migrated events as migration velocity changes. It can be implemented efficiently in the Fourier domain after transforming time to squared time. — [Fomel, SEP-100 velocity continuation](https://sepwww.stanford.edu/data/media/public/docs/sep100/sergey2/paper_html/index.html)
- Diffraction-based depth-domain velocity updating uses the moveout of over- or under-migrated diffractions and "velocity rays" from the image-wave equation. It is cheap once diffractions are picked, and was tested on synthetics only. — [Coimbra et al., WIT 2011](https://www.wit.uni-hamburg.de/import/documents/reports/2011/wit2011-coimbra.pdf)
- Depth focusing analysis (Doherty and Claerbout 1974; MacKay and Abma 1992) estimates velocity from the depth of best focusing. It assumes flat horizons, small velocity errors and small dip and offset. In complex areas focusing energy is mispositioned. — [SEP-94, Hermes](https://sep.stanford.edu/data/media/public/docs/sep94/hermes/paper_html/node1.html)

### Inferences
- **Per-pixel channels this question provides (inference).**
  1. Shot-to-shot image consistency: the 5 single-shot migrated images disagree in depth and position when velocity is wrong. Their per-pixel variance or normalized overlap is a crude common-shot CIG. This is the user's existing "cross-shot agreement" channel.
  2. Subsurface-offset energy ratio, e.g. 1 - |I(x, h=0)|^2 / sum_h |I(x, h)|^2. This is 0 where focused and grows with velocity error. It is computed per shot during the same RTM run as the image, so sparse shots are not a limitation: each shot yields a full subsurface-offset gather.
  3. A residual-curvature or RMO parameter map from angle gathers.
  4. Focusing-depth error delta = (Zf - Zm)/2 from depth-focusing panels (flat dips only).
- **Sampling in this geometry (derived).** With shots at about 0/170/345/520/690 m, CMP fold is 5 only near x = 345 m and drops to 2 at x = 100-150 and 600 m, and to 1 for x < about 85 m or x > about 605 m. Offset-domain CIGs and semblance are therefore very sparse at the edges. Angle gathers from shot-profile migration are better filled because receivers are dense.
- **How large errors look (derived from cited RMO theory and the moveout numbers in Q1).** At 15 Hz a 10 % velocity error causes RMO of about 0.5 period at 200 m depth but only about 0.15 period at 700 m. Deep RMO is therefore subtle, while defocusing of the zero-subsurface-offset image is more visible because of the (1 + tan^2 gamma) amplification of normal shift.
- **Faults and curved layers (inference).** Diffractions from fault tips and terminations carry velocity information independent of reflector dip (velocity continuation and diffraction focusing). These are general methods with no flat-layer assumption.
- **Flat-layer assumptions (summary).** Depth focusing analysis and NMO-type RMO assume small dips. Velocity continuation in time migration assumes laterally slowly varying velocity. Wave-equation subsurface-offset and angle CIGs and DSO are general but need a smooth background velocity (scale separation).

### Gaps
- Did not retrieve primary texts of Symes and Carazzone (1991) DSO, Shen and Symes (2008), Sava and Fomel (2003, 2006), or Rickett and Sava (2002); their content is taken via secondary citations.
- No published study of extended-image MVA with only 5 shots on a 700 m spread was found.

---

## Q3. Wave-equation imaging: RTM, LSRTM, illumination compensation, artefact-suppressing imaging conditions, PSFs and resolution analysis. Costs

### Takeaway
RTM in a smooth background maps reflectivity (velocity contrast, since density is constant) to every pixel for 2 simulations per shot. Raw cross-correlation images carry strong low-wavenumber backscatter, plus diving-wave and source-proximity artefacts, along with illumination imprint. A Laplacian filter, or the inverse-scattering imaging condition (ISIC, an asymptotic inverse of Born), plus division by source or receiver illumination, gives an approximately true-amplitude map at similar cost. LSRTM costs about 2 extra simulations per shot per iteration. Hessian diagonals, illumination maps and point-spread functions quantify resolution and illumination pixel by pixel.

### Cited Findings
- RTM correlates the forward source field with the back-propagated receiver field. In the ray-Born view, the medium is a smooth background plus a perturbation containing only wavelength-scale wavenumbers. Stolk, de Hoop and Op 't Root adapt RTM so "its resolution only reflects the illumination". They also describe the source-normalized single-shot variant, which divides by the incident field. — [Stolk, de Hoop, Op 't Root, SEG 2009](https://staff.fnwi.uva.nl/c.c.stolk/Papers/rtmseg09.pdf)
- **Laplacian filter and ISIC.**
  - A Laplacian filter is commonly applied to RTM images to attenuate backscattering noise. A source-illumination term P(x; xs) compensates amplitude loss with depth ("CLI" or "CLIIC"). CLIIC amplitudes do not correspond to true reflection coefficients.
  - The ISIC (Kiyashchenko et al. 2007; Op 't Root et al. 2012) is an asymptotic inverse of Born. It gives more reliable reflectivity and strongly reduces backscattering artefacts.
  - The Laplacian-based ISIC form costs about the same as CLI. The time-derivative form costs more.
  - Source for all three points: [Albano et al., Revista Brasileira de Geofísica: LSRTM preconditioned with ISIC](https://sbgf.org.br/revista/index.php/rbgf/article/view/2330); [PDF](https://www.sbgf.org.br/revista/index.php/rbgf/article/download/2330/1971)
- ISIC extended to separated-wavefield imaging removes low-wavenumber artefacts from backscattered energy. — [Crawley et al., SEG 2018 (PGS/TGS)](https://tgs.com/hubfs/Technical%20Library/Technical%20Library%20Files/seg2018_crawley_etal_rtm-swim.pdf)
- **Illumination normalization.**
  - Source-normalized cross-correlation (Kaelin and Guitton 2006) divides by source illumination intensity and requires storing the source wavefield. — [Frontiers in Earth Science 2025](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1605436/epub)
  - For large impedance contrasts, cross-correlation produces low-frequency artefacts and illumination effects. Dividing by receiver illumination removes illumination effects efficiently. — [Kaelin, Guitton and Ortigosa, EAGE 2014](https://www.earthdoc.org/content/papers/10.3997/2214-4609.201401930)
  - The deconvolution imaging condition (Claerbout 1971) gives better wavelet removal and illumination compensation but is unstable and artefact-prone. — [CGG abstract](https://viridiengroup.com/sites/default/files/2020-11/cggv_0000025672.pdf)
- **Hessian, PSFs and illumination.**
  - Each row of the Hessian is a point-spread function that encodes bandwidth, geometry and illumination. The Hessian diagonal is an illumination measure, and the inverse diagonal (Duquet et al. 1998) has been used as an illumination measure in Kirchhoff imaging. — [NSF PAR: PSF deconvolution imaging method](https://par.nsf.gov/biblio/10578977-efficient-stable-highresolution-seismic-imaging-method-pointspread-function-deconvolution); [SEP, wave-equation Hessian by phase encoding](https://sep.stanford.edu/sep/claudio/Research/Geophysics/wave-equation_hessian_by_phase_encoding/revision0/paper.pdf) (attribution of individual sentences between these two is from the search summary)
  - The diagonal of the resolution matrix is a rough direct estimate of model uncertainty and can be estimated iteratively (Berryman and Fomel 1996). — [Fomel, SEP illumination report](https://sepwww.stanford.edu/sep/sergey/sepsergey/illum.pdf)
  - Lecomte (2008) simulates PSDM images by filtering reflectivity with ray-computed PSFs ("Resolution and illumination analyses in PSDM", The Leading Edge 27(5)). Jensen et al. (2021) validated the method. — [NORSAR record](https://www.norsar.no/r-d/publications/key-publications_2/resolution-and-illumination-analyses-in-psdm-a-ray-based-approach); [EAGE abstract citing it](https://www.earthdoc.org/content/papers/10.3997/2214-4609.201601374)
- **LSRTM cost.** A data-domain LSRTM gradient needs one Born modelling and one migration per iteration, about 20 migration-equivalents over 10 iterations. Image-domain LSM roughly doubles that. — [SBGF, Least-squares RTM: theory and applications](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/15th_CISBGf/session/RTM,%20LEAST%20SQUARES%20AND%20KIRCHHOFF%20METHODS/Least-squares%20RTM-Theory%20and%20applications.pdf); [arXiv 1911.06027](https://ar5iv.arxiv.org/html/1911.06027)
- Mora (1989), "Inversion = Migration + Tomography": waveform inversion is equivalent to doing migration (high wavenumbers) and reflection tomography (background) simultaneously. — [Mora 1989, Geophysics 54(12)](https://pure.kfupm.edu.sa/en/publications/inversion-migration-tomography/)
- **Resolution conventions.** The tuning thickness (about lambda/4) is the practical vertical resolution. The SEG Wiki gives lambda/8 as the detectability limit, so the two sources differ in criterion. Unmigrated lateral resolution is the first Fresnel zone; migration shrinks it towards the trace spacing. — [AAPG Wiki](https://wiki.aapg.org/Vertical_and_lateral_seismic_resolution_and_attenuation); [SEG Wiki, Seismic resolution](https://wiki.seg.org/wiki/Seismic_Resolution:_Vertical_and_Horizontal)

### Inferences
- **Resolution numbers for this geometry (derived from k = (4 pi f / c) cos(theta/2), cited in Q5).**
  - Vertical: lambda/4 at 15 Hz is 25 m (1500 m/s) to 75 m (4500 m/s). At the upper band (about 37.5 Hz), lambda/2 is 20-60 m. So about 2-7 grid cells.
  - Lateral (post-migration, aperture-limited): of order lambda / (4 sin alpha_max), with alpha_max = atan(half-aperture / z). That is about 40-60 m at z = 200 m (v = 2000-3000 m/s) and about 90 m at z = 500 m (v = 3000 m/s). This formula is my derivation from the wavenumber relation, not a cited formula.
  - Unmigrated Fresnel radius sqrt(z lambda / 2) at v = 2500 m/s: 91 m (100 m), 158 m (300 m), 242 m (700 m).
- **Artefacts specific to this setting (inference).**
  - Strong contrasts (R up to 0.5) and the direct and diving waves create large backscatter artefacts in plain cross-correlation RTM.
  - The source row dominates the adjoint image. The user's toy measured 1e3-1e4 times the interior (project memory), so a source-zone mute and per-depth normalization or illumination division are needed before any network sees the map.
  - The Laplacian or ISIC route is cheap: the same simulations plus spatial derivatives.
- **Per-pixel channels (inference).**
  - (a) Reflectivity image with ISIC or Laplacian filtering.
  - (b) Source illumination S(x) = sum_s sum_t u_s(x, t)^2. This is free from the forward runs.
  - (c) Receiver illumination. It needs back-propagation from all 70 receivers: 70 impulse simulations, or 1-2 simulations with random-phase or simultaneous-source encoding.
  - (d) Normalized image (a)/(b) or (a)/sqrt(b c).
  - (e) PSF grid. Put sparse point scatterers on a coarse lattice (for example every 10-15 cells, wider than the PSF), then Born-model and migrate them, for about 2 simulations per shot per lattice. The output gives per-location resolution length and orientation. The PSF diagonal approximates the Hessian diagonal.
- **Costs for 5 shots on a 70 x 70 grid with 1000 steps (derived).**
  - One simulation is about 4.9e6 cell updates, about 1e7 with absorbing pads.
  - RTM: 10 simulations. Source-wavefield storage is 70 x 70 x 1000 float32 = 19.6 MB per shot (98 MB for 5 shots), so no checkpointing is needed.
  - LSRTM: 10-20 iterations at about 2 simulations per shot per iteration, so about 100-200 simulations.
  - PSF lattice: about 10 simulations per lattice offset.

### Gaps
- No primary-source fetch of Youn and Zhou (2001) or Zhang and Sun (2009) on Laplacian filtering; it is sourced via Albano et al.
- No numbers were found for PSF widths specific to a 700 m spread. The lateral-resolution formula above is my derivation.
- The Fichtner and van Leeuwen (2015) "resolution analysis by random probing" and Fichtner and Trampert (2011) Hessian kernels were not retrieved.

---

## Q4. Dynamic information: amplitudes, AVO/AVA, impedance contrasts, critical angles, post-critical reflections and head waves, phase. What constant-density acoustic data can and cannot reveal

### Takeaway
With constant density, impedance contrast equals velocity contrast. Reflection amplitude and polarity therefore measure the velocity ratio across each interface directly, with no density trade-off. The AVO gradient is fully determined by the same ratio, so it is redundant with the intercept but gives a self-check. The critical angle where amplitude saturates gives v2 = v1 / sin(theta_c) directly. Polarity reveals velocity inversions that first-arrival methods cannot see. Without data-domain amplitude calibration (wavelet, spreading, transmission losses), amplitudes give only relative information. Density, shear and anisotropy are absent by construction.

### Cited Findings
- **Shuey form of the Zoeppritz equations.**
  - R(theta) = R(0) + G sin^2 theta + F (tan^2 theta - sin^2 theta), with:
    - R(0) = 1/2 (dVp/Vp + d rho/rho);
    - G = 1/2 dVp/Vp - 2 (Vs^2/Vp^2)(d rho/rho + 2 dVs/Vs);
    - F = 1/2 dVp/Vp, which governs large angles near critical.
  - The two-term form (dropping F) holds for incidence below about 30 deg.
  - Source: [Wikipedia, Zoeppritz equations](https://en.wikipedia.org/wiki/Zoeppritz_equations)
- Critical angle sin(theta_c) = V1/V2. Head waves exist beyond the critical distance. — [SEG Wiki, Head waves](https://wiki.seg.org/wiki/Head_waves_(refractions)_and_effect_of_hidden_layer)
- A low-velocity layer under a faster one produces no head wave. — [LMU lecture notes](https://www.geophysik.uni-muenchen.de/~igel/Lectures/Sedi/sedi_ray.pdf)
- Widess tuning: amplitude builds up as thickness approaches lambda/4. The model assumes equal and opposite reflection coefficients. — [AAPG Wiki](https://wiki.aapg.org/Vertical_and_lateral_seismic_resolution_and_attenuation); [SBGF abstract](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/12th_CISBGf/SBGF_3121.pdf)
- The Ricker spectrum is proportional to omega^2 exp(-sigma^2 omega^2 / 2), with peak at omega_p = sqrt(2)/sigma. — [Wikipedia, Ricker wavelet](https://en.wikipedia.org/wiki/Ricker_wavelet). The maximum frequency of a Ricker exceeds its peak frequency. — [Agile Scientific](https://agilescientific.com/blog/2018/3/27/real-and-apparent-frequency)
- W2 (optimal-transport) misfit is sensitive to small amplitudes and has a low-frequency bias. It can extract information from below the lowest reflecting interface, where no waves return, using the amplitude and kinematics of the reflection. — [Engquist and Yang, OT-based seismic inversion: beyond cycle skipping](https://arxiv.org/pdf/2002.00031)
- If the monochromatic wavefield were known everywhere, m could be recovered by pointwise division, m = y(u) / L(u). Because data are recorded only near the surface, the problem is underdetermined. — [Operto et al. 2023](https://arxiv.org/pdf/2212.10141)

### Inferences
- **Acoustic, constant-density specialization (derived from Shuey with d rho = 0, Vs = 0).** R(0) = 1/2 dv/v and G = F = 1/2 dv/v, so R(theta) is about 1/2 (dv/v) sec^2 theta. The exact constant-density form follows from the impedance formula: R = (v2 cos theta1 - v1 cos theta2) / (v2 cos theta1 + v1 cos theta2). Consequences:
  - Polarity gives the sign of the velocity jump at every reflector. A negative polarity marks a velocity inversion, which is invisible to first-arrival methods (hidden layer).
  - |R(0)| gives v2/v1 = (1 + R)/(1 - R). Example normal-incidence values:

    | Interface (m/s) | R(0) |
    |---|---|
    | 1500 over 4500 | 0.50 |
    | 2500 over 3000 | 0.091 |
    | 3000 over 2500 | -0.091 |
    | 2000 over 2200 | 0.048 |

  - Amplitude rises with angle as sec^2 theta, then reaches |R| = 1 at theta_c. Critical angles: 19.5 deg (1500 over 4500), 56 deg (2500 over 3000), 65 deg (2000 over 2200). The critical offset x_c = 2 z_eff tan(theta_c) is within 700 m only for strong contrasts or shallow interfaces. When observed, theta_c gives the lower-layer velocity directly, which the user's AVO fit already exploits (project memory).
- **Recursive "impedance chain" (inference).** Anchor v1 from the direct-wave slope, then v_{n+1} = v_n (1 + R_n)/(1 - R_n) along each image column. This turns a true-amplitude (ISIC/LSRTM) reflectivity map into an absolute-velocity map. Errors accumulate with depth and are biased by transmission loss (two-way transmission 1 - R^2 = 0.75 for R = 0.5), tuning (beds under about lambda/4 = 25-75 m) and wavelet scaling. A calibrated source wavelet is needed: the user's calibrated OpenFWI source fits gathers to 4-6 % (project memory). This is classical recursive or trace-integration inversion; no primary source was fetched (see Gaps).
- **Post-critical reflections and head waves (inference).** Beyond theta_c the reflected phase rotates with angle and amplitude is total, so post-critical events are strong. They bias kinematic picks: the user's toy found the overlap objective biased slow on a 1625/4307 model. Head waves are weak and arrive tangent to the post-critical reflection. In this geometry they are seldom first arrivals (crossover numbers in Q1).
- **What the data cannot reveal (inference).**
  - Density, Vs and anisotropy, by construction.
  - Velocity below the deepest reflector, except the single ratio in the last reflection amplitude.
  - Velocity structure in beds thinner than about lambda/8-lambda/4, only their average and net reflectivity.
  - Absolute amplitude-derived velocity without wavelet calibration.
- **Phase and frequency (inference).** In noise-free synthetics the Ricker has energy at 2-5 Hz (-26 to -11 dB relative to peak). Low-frequency phase and envelope content are usable signals that field data usually lack.

### Gaps
- No primary-source fetch of the exact acoustic reflection coefficient formula or of the 2D head-wave amplitude decay law; the formula above is derived.
- Recursive impedance inversion (e.g., Lindseth 1979) was not sourced in this session.
- No published study quantifying how accurately AVO or critical-angle velocity can be estimated from 5-shot, 700 m data was found.

---

## Q5. FWI-related information: gradients, sensitivity kernels, low- vs high-wavenumber split, frequency continuation, cycle-skipping criteria at 15 Hz, objectives robust to cycle skipping

### Takeaway
The FWI gradient is an RTM-like per-pixel map that costs 2 simulations per shot. Its wavenumber content follows k = (4 pi f / c) cos(theta/2). Wide-angle transmission (diving waves, only the top 40-150 m here) and low frequencies fill low wavenumbers. Narrow-angle reflections (everything deeper) fill only a band near 2f/c. The deep low-wavenumber background must therefore come from reflection-tomography components ("rabbit ears"), RWI/JFWI, MVA, or frequencies below about 5 Hz. The half-period rule at 15 Hz allows only about 33 ms of traveltime error (about 3-17 % velocity error depending on arrival time). Noise-free data with usable 2-5 Hz content relax this 3-5 fold. OT, AWI, envelope and extended-source/WRI objectives are the established cycle-skip-robust alternatives.

### Cited Findings
- **Adjoint state.**
  - The adjoint-state method computes the gradient without Fréchet derivatives by solving one extra linear (adjoint) system. — [Plessix 2006, GJI 167(2)](https://academic.oup.com/gji/article/167/2/495/559970)
  - In FWI "one only needs to solve two wave equations numerically" (forward plus adjoint/back-propagation) per source to get the gradient. — [Engquist and Yang](https://arxiv.org/pdf/2002.00031)
  - The cost does not depend on the number of parameters. — [Wikipedia, Adjoint state method](https://en.wikipedia.org/wiki/Adjoint_state_method)
- **Wavenumber relation.**
  - k = (4 pi f / c0) cos(theta/2) = (4 pi / lambda) cos(theta/2), where theta is the scattering angle. "Low wavenumbers can be obtained thanks to low frequencies data and wide aperture angles." For deep targets "the lack of large scattering angles prevents the reconstruction of low wavenumbers." — [Brossier et al. 2017, SEISCOPE review](https://html.rhhz.net/SYWT/HTML/1701-02.htm)
  - The same relation, with the statement that limited-offset deep targets are sampled only by narrow-angle reflections. — [CREWES 2016, CRR201605](https://crewes.org/Documents/ResearchReports/2016/CRR201605.pdf) (search snippet; the server refused the direct fetch)
  - Gradient resolution is controlled by the angle between interacting wavefields and their local wavelengths. Where transmissions illuminate, local low wavenumbers are resolvable everywhere. — [Kazei et al., WIT 2015](https://www.wit.uni-hamburg.de/import/documents/reports/2015/wit2015-kazei.pdf)
- **Kernels.**
  - Woodward's band-limited wavepaths are hollow "bananas" (a doughnut in cross-section). — [SEP-103](https://sep.stanford.edu/data/media/public/docs/sep103/james3.pdf)
  - "Rabbit ears" are the low-wavenumber (tomographic) part of a reflection-generated gradient, most useful deeper than bananas penetrate. The migration "smile" (isochron) sits at their base. — [US patent 10353092 (industry source)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10353092)
  - Impedance kernels isolate high wavenumbers and velocity kernels the low. Dynamic weighting removes specular migration isochrons and enhances low-wavenumber gradient content. — [Ramos-Martinez et al., PGS 2016](https://www.tgs.com/hubfs/Technical%20Library/Technical%20Library%20Files/2016may_pgs_ramos-martinez_etal_fwi.pdf)
- **Reflection waveform inversion.** RWI (Xu et al. 2012) uses reflections in their tomographic regime under scale separation, alternating updates of perturbation and background. JFWI (Zhou et al. 2015) splits the misfit into diving and reflection terms, gives smooth background updates, and can start classical FWI. In JFWI, zero-offset reflections are free of cycle skipping; long-offset reflections and diving waves still need a reasonable start. — [Brossier et al. 2017](https://html.rhhz.net/SYWT/HTML/1701-02.htm); [Zhou et al., KFUPM record](https://pure.kfupm.edu.sa/en/publications/combining-diving-and-reflected-waves-for-velocity-model-building-/)
- **Cycle skipping.**
  - FWI is trapped "as soon as the simulated data don't predict the recorded counterpart with an error lower than half a period" (Virieux and Operto 2009). This is necessary but not sufficient, because the wavefield must match everywhere, not only at receivers. Example: 10 wavelengths propagated through a 20 % contrast inclusion give a 0.1875 s traveltime mismatch, about twice the half period at 5 Hz. — [Operto et al. 2023](https://arxiv.org/pdf/2212.10141)
  - Local descent needs the start to predict arrival times "to within a small multiple of a dominant wavelength". — [Symes et al., FWI by source extension](https://arxiv.org/pdf/2003.12538)
- **Frequency continuation.** Bunks et al. (1995) invert bandpassed data from low to high frequency, recovering low wavenumbers first. Long-wavelength data greatly reduce the number of local minima. — [Boonyasiriwat et al. 2009 (citing Bunks)](https://csim.kaust.edu.sa/web/FWI&LSM_papers/FWI/Geophysics2009Boonyasiriwat.pdf). Sirgue and Pratt (2004) showed that a limited set of frequencies can recover the wavenumber spectrum. — [SEG Wiki, R. Gerhard Pratt](https://wiki.seg.org/wiki/R._Gerhard_Pratt)
- **Optimal transport.**
  - W2 is convex with respect to translation and dilation (Theorem 3.1). Signed seismic traces need normalization into probability measures (linear, exponential, softplus). Adding a constant gives a "Huber-type" metric and enhances low-frequency gradient content. — [Engquist and Yang](https://arxiv.org/pdf/2002.00031)
  - Métivier et al. use the Kantorovich-Rubinstein dual on whole shot-gather panels, solved with a proximal splitting algorithm. On Marmousi from a smooth start, "only OT-FWI is able to converge toward meaningful result". — [Brossier et al. 2017](https://html.rhhz.net/SYWT/HTML/1701-02.htm)
- **AWI.** AWI designs least-squares convolutional (Wiener) filters that match predicted to observed traces and updates the model to drive the filters toward a zero-lag delta. The authors report it "appears to be immune" to cycle skipping. Its per-iteration cost is similar to FWI. — [Warner and Guasch 2016, Imperial repository](https://spiral.imperial.ac.uk/handle/10044/1/42416); [IPAM abstract](https://www.ipam.ucla.edu/abstract/?tid=14114). I found no independent verification of the immunity claim.
- **Envelope inversion.** Envelope misfits (Bozdağ et al. 2011; Wu, Luo and Wu 2014) extract ultra-low-frequency information to build a smooth starting model. Direct-envelope variants invert strong-scattering large-scale models without low-frequency data. — [Wu and Chen, arXiv 1808.05275](https://arxiv.org/pdf/1808.05275); [Frontiers 2023](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2023.1259710/pdf)
- **WRI and extended-source FWI.**
  - WRI minimizes ||P u - d||^2 + lambda^2 ||A(m) u - q||^2. The wavefield comes from an overdetermined "augmented" system that fits the data and the wave equation. No adjoint wavefields are needed, and the model update is a diagonal (Newton-like) solve. As lambda tends to infinity the formulation recovers FWI. The examples use a 2D constant-density Helmholtz code. — [van Leeuwen and Herrmann 2013, GJI](https://slim.gatech.edu/Publications/Public/Journals/GeophysicalJournalInternational/2013/vanLeeuwen2013GJImlm/vanLeeuwen2013mlm.pdf)
  - On a (v0 = 2000 m/s, gradient 0.65 /s) family, the FWI cost function shows several local minima where extended-source FWI does not. Extended-source FWI reconstructed an inclusion even when the traveltime mismatch was about 2x the half period. — [Operto et al. 2023](https://arxiv.org/pdf/2212.10141)

### Inferences
- **Transmission vs reflection split in this geometry (derived).**
  - Diving-wave (wide-angle, theta near 180 deg, cos(theta/2) near 0) low-wavenumber updates are confined to the top 20-150 m (Q1).
  - Below that, reflection opening angles are at most 120 deg at 200 m, at most 70 deg at 500 m and at most 53 deg at 700 m. So cos(theta/2) >= 0.5, 0.82 and 0.89 respectively.
  - At 15 Hz and v = 3000 m/s, 500 m-deep reflections fill vertical wavelengths of only about 100-122 m. At 5 Hz they fill 300-366 m, and at 3 Hz 500-610 m.
  - The deep long-wavelength background (scales above 300 m) is therefore reachable by waveform gradients only with frequencies of 5 Hz or less, or through the reflection-tomographic ("rabbit-ear") component.
- **Cycle-skipping budget (derived; relative velocity error tolerated is about 1/(2N), N = f t periods).**
  - T/2 is 33 ms at 15 Hz, 100 ms at 5 Hz and 167 ms at 3 Hz.

    | Arrival time | 15 Hz | 5 Hz | 3 Hz |
    |---|---|---|---|
    | 0.2 s | dv/v < 17 % | 50 % | 83 % |
    | 0.6 s | 5.6 % | 17 % | 28 % |
    | 1.0 s | 3.3 % | 10 % | 17 % |

  - A typical 1D/linear start, often 10-30 % off at depth, is cycle-skipped at 15 Hz but generally not at 3 Hz.
  - Noise-free synthetics carry 3 Hz at -20 dB, so a Bunks-style schedule from 2-3 Hz is feasible here, unlike most field data. This is a key difference between benchmark and field settings.
- **Gradient as a channel (inference).** A first gradient from a smooth start, after source mute, pseudo-Hessian (source illumination) preconditioning and per-depth normalization, is a reflectivity-like map plus near-surface transmission updates. Its sign gives the "push faster or slower" direction only where data are not cycle-skipped. Low-pass or ISIC-like decomposition separates the tomographic (low-k) and migration (high-k) parts, following Mora's split.
- **Robust-misfit costs for 5 shots on a 70 x 70 grid (inference from the cited per-iteration statements).**
  - L2-FWI: 10 simulations per gradient, plus about 5 for a line search, so about 15 per iteration. A multiscale run of 50-200 iterations needs about 750-3000 simulations.
  - OT: same simulation count. The 1D trace-wise W2 via cumulative distribution functions is O(nt) per trace, which is negligible. Panel-wise KR via proximal splitting is more expensive but still small at 70 x 1000 samples.
  - AWI: same simulations plus one small Toeplitz solve per trace.
  - Envelope: same simulations, with the adjoint source changed only.
  - WRI/ES-FWI: natural in the frequency domain. A 4900-unknown 2D Helmholtz system (about 1e4 with absorbing pads) can be factorized directly, so a few frequencies times 5 shots is cheap.

### Gaps
- Could not fetch Virieux and Operto (2009) itself (dead mirror URLs). The half-period criterion is cited via Operto et al. (2023) and Brossier et al. (2017). The "1/(2N_lambda)" relative-velocity form is my derivation from it, not a quoted formula.
- Sirgue and Pratt (2004) frequency-selection formula (offset-dependent wavenumber coverage, Delta f) was not retrieved verbatim.
- The OT paper by Métivier et al. 2016 was not fetched (URL 404); it is covered through Brossier et al. (2017).
- No independent benchmark comparing OT, AWI, envelope and WRI on short-spread (700 m) data was found.

---

## Q6. Null space and uncertainty: what is unconstrained in this geometry, and how illumination and resolution maps quantify it

### Takeaway
Six regions are poorly constrained in this geometry:
1. Everything below the deepest reflector (except one amplitude ratio).
2. The long-wavelength background between the about 150 m diving-wave limit and depth, which is constrained only by small reflection moveouts and carries the velocity-depth trade-off.
3. The lateral edges, where CMP fold drops to 1 within about 85-100 m of the ends.
4. Zones under strong positive contrasts, where post-critical energy dominates and transmission is reduced.
5. Velocity inversions, which are invisible to first arrivals.
6. Detail finer than about lambda/4 (25-75 m).

Hessian-diagonal or illumination maps, PSF grids and resolution-matrix diagonals are the established per-pixel quantifications.

### Cited Findings
- Below the lowest reflecting interface "no seismic waves return to the surface". L2-FWI with reflections only is problematic without an accurate start. W2 can partially recover velocity below reflectors. — [Engquist and Yang](https://arxiv.org/pdf/2002.00031)
- Data recorded only near the surface make FWI underdetermined. Matching at receivers is not sufficient. — [Operto et al. 2023](https://arxiv.org/pdf/2212.10141)
- Lack of large scattering angles at depth prevents low-wavenumber reconstruction. — [Brossier et al. 2017](https://html.rhhz.net/SYWT/HTML/1701-02.htm)
- Reflection tomography cannot uniquely separate velocity and interface depth without a-priori information (Bickel 1990; Lines 1993). — [Soupios et al.](https://balkangeophysoc.gr/menu/congresses/oral/10-11-O3-3-soupios-pantelis.PDF)
- Low-velocity layers produce no head waves, and missing hidden layers biases depth. — [SEG Wiki, Effect of a hidden layer](https://wiki.seg.org/wiki/Effect_of_a_hidden_layer)
- The Hessian diagonal measures illumination and its rows are PSFs. The resolution-matrix diagonal is a rough uncertainty estimate. — [NSF PAR PSF paper](https://par.nsf.gov/biblio/10578977-efficient-stable-highresolution-seismic-imaging-method-pointspread-function-deconvolution); [Fomel, SEP illumination](https://sepwww.stanford.edu/sep/sergey/sepsergey/illum.pdf)
- Ray-based PSFs (Lecomte 2008) give resolution and illumination per target location. — [NORSAR](https://www.norsar.no/r-d/publications/key-publications_2/resolution-and-illumination-analyses-in-psdm-a-ray-based-approach)

### Inferences
- **Concrete null-space map for this geometry (derived).**
  1. **Below the deepest reflector.** No kinematic constraint. In constant-density acoustics, only R at the last interface (giving v_below / v_above) and, if within the spread, the critical angle constrain it. This is what the user's AVO fallback exploits.
  2. **Mid-depth background.** Between about 40-150 m (diving limit) and the deepest reflector, long wavelengths (above about 300 m) are constrained only by moveouts of 10-30 ms per 10 % error. With one reflector, only the RMS overburden velocity is constrained.
  3. **Edges.** With shots at about 0/170/345/520/690 m, CMP fold is 5 at mid-spread and 1 for x < about 85 m and x > about 605 m. Steeply dipping reflectors near the edges have specular rays that leave the spread entirely. Image amplitude and kinematic constraint taper toward the edges and with depth, roughly as a trapezoid.
  4. **Beneath fast layers.**
     - With 1500 over 4500 m/s, theta_c = 19.5 deg. At 100 m depth the critical offset is about 70 m, so most of the gather from that interface is post-critical and dominates both picks and gradients.
     - Two-way transmission is 1 - R^2 = 0.75 for R = 0.5.
     - Internal multiples between strong contrasts remain (only free-surface multiples are removed) and can be mistaken for deeper primaries.
  5. **Velocity inversions.** Low-velocity layers under fast layers are invisible to first-arrival tomography. Reflection polarity is the main clue.
  6. **Thin beds.** Layers under about lambda/4 (25-75 m) are seen only as composite reflectivity.
- **Quantifying the null space as per-pixel channels (inference).**
  - (a) Source illumination sum_s sum_t u_s^2: free.
  - (b) Receiver illumination or the pseudo-Hessian diagonal.
  - (c) Ray or wavepath density from eikonal tables (number of specular reflection points or diving paths per pixel).
  - (d) PSF width and orientation on a coarse lattice.
  - (e) Cross-shot agreement or extended-image focusing as an uncertainty proxy.
  - (f) A "below deepest confident reflector" mask from the reflectivity image.
  - These directly mark where a learned model must rely on priors. The user's earlier result, that channels are "blind below strong contrasts" and plateau on FlatFault-A (project memory), is consistent with items 1 and 4.

### Gaps
- No published resolution or uncertainty study for exactly this geometry (5 shots, 700 m, 15 Hz) was found.
- Bayesian or ensemble FWI uncertainty methods (e.g. Hessian-based posterior covariance, random-probing resolution) were not retrieved in this session.

---

## Q7. Cost per information channel for a 70 x 70 grid with five shots (order of magnitude, number of wave simulations)

### Takeaway
Everything kinematic, from eikonal tables (first-arrival and reflection tomography, kinematic migration, stereotomography), costs zero wave simulations: O(N log N) or O(N) eikonal solves on 4900 cells, so milliseconds. Every wave-equation image or gradient channel costs about 10 simulations (5 forward plus 5 adjoint). Iterative methods (LSRTM, FWI, robust-misfit FWI, MVA/DSO loops) cost 100-3000 simulations. At 1e7 cell updates per simulation on this grid, a single simulation is milliseconds on a GPU or compiled code, so even the iterative channels are seconds to minutes per model.

### Cited Findings
- Fast marching solves the eikonal equation in O(N log N) for N grid points. — [Sethian, Fast marching overview](https://math.berkeley.edu/~sethian/2006/History/hold.chart.overview.fastmarching.html). Fast sweeping is O(N), with a fixed number of Gauss-Seidel sweeps. — [Zhang, Zhao and Qian](https://users.math.msu.edu/users/qian/papers/ZhangZhaoQian.pdf)
- The FWI or RTM gradient needs two wave solves per source. — [Engquist and Yang](https://arxiv.org/pdf/2002.00031); [Plessix 2006](https://academic.oup.com/gji/article/167/2/495/559970)
- LSRTM needs one Born modelling and one migration per iteration (about 20 migrations for 10 iterations, data domain). — [SBGF LSRTM](https://sbgf.org.br/mysbgf/eventos/expanded_abstracts/15th_CISBGf/session/RTM,%20LEAST%20SQUARES%20AND%20KIRCHHOFF%20METHODS/Least-squares%20RTM-Theory%20and%20applications.pdf)
- AWI costs about the same per iteration as FWI. — [IPAM abstract](https://www.ipam.ucla.edu/abstract/?tid=14114)
- WRI needs no adjoint wavefields but solves an augmented least-squares system per source. — [van Leeuwen and Herrmann 2013](https://slim.gatech.edu/Publications/Public/Journals/GeophysicalJournalInternational/2013/vanLeeuwen2013GJImlm/vanLeeuwen2013mlm.pdf)
- Source-normalized imaging conditions need the stored source wavefield. — [Frontiers 2025](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1605436/epub)
- Classical stereotomography builds a large Fréchet matrix. — [CREWES CRR201835](https://www.crewes.org/Documents/ResearchReports/2018/CRR201835.pdf)

### Inferences
- **Simulation size (derived).** 70 x 70 x 1000 steps is 4.9e6 cell updates, about 1e7 with 20-cell absorbing pads.
  - The Courant number at 10 m, 1 ms and 4500 m/s is 0.45 (stable for 2nd-order 2D).
  - At 40 Hz and 1500 m/s there are only about 3.8 points per wavelength, so higher-order spatial stencils or a 5 m grid (about 4e7 updates) are needed for dispersion-free simulation of the upper band.
  - Order of magnitude: about 1-10 ms per simulation on GPU or compiled code; about 0.1-1 s in eager PyTorch on CPU. The user's other engine reports a 52 ms native adjoint, and the current feature pipeline takes about 7 s per model for six channels (project memory).
- **Per-pixel channel catalogue with cost (inference; physical content from Q1-Q6).**

| Channel (per pixel) | Physical information | Classical method | Wave sims (5 shots) | Valid where / assumptions | Main failure modes |
|---|---|---|---|---|---|
| Near-surface v(x, z) | first-arrival times (direct, diving) | first-arrival traveltime tomography (eikonal) | 0 (eikonal, ms) | top about 20-150 m; smooth v | no turning in sharp-layer models; hidden LVLs; Fresnel blur about 100-150 m |
| Head-wave refractor velocity and depth | critical refraction | refraction (delay-time / GRM-like) or fitted two-layer model | 0 | shallow strong contrasts only (crossover < 700 m) | rarely first arrival; overlaps post-critical reflection |
| v_rms(t0), v_int(z) column | reflection moveout | NMO, semblance, Dix | 0 (data-domain scans) | flat layers, small offset | dip, laterally varying, non-hyperbolic; fold 1-5 |
| Smooth v(x, z) plus ray density | reflection traveltimes plus receiver slopes | reflection, slope or stereotomography | 0 (eikonal / ray tracing) | general; needs picks | velocity-depth ambiguity; picking errors; no source-side slopes with 5 shots |
| Reflectivity (velocity contrast) image | reflection amplitudes and phase | RTM with Laplacian or ISIC | 10 | smooth background known to within moveout tolerance | backscatter, source-row artefacts; misplaced if v wrong |
| True-amplitude reflectivity | same, deblurred | LSRTM / ISIC-preconditioned | 100-400 | Born, scale separation | slow convergence; strong contrasts violate Born |
| Velocity-contrast polarity / ratio, critical angle | AVO / AVA, sign of R | angle gathers + Shuey (acoustic: R about 1/2 dv/v sec^2 theta) | 10 (from RTM angle gathers) | angles < theta_c for linear fits | spreading, transmission, wavelet scaling; tuning |
| Source and receiver illumination | energy reaching each pixel | sum_t u^2 (source), receiver back-propagation | 0 extra (source); 1-70 (receiver) | none | ignores aperture direction |
| PSF width and orientation | local resolution | Born-model plus migrate a lattice of point scatterers | about 10 per lattice offset | Born | lattice too dense causes PSF overlap |
| Cross-shot agreement / CIG flatness | velocity error from redundancy | common-shot images, ADCIG RMO | 10 | smooth background | sparse fold at edges; dip artefacts |
| Subsurface-offset focusing | velocity error | extended imaging (Sava-Fomel / Rickett-Sava) | 10 plus lag correlations | scale separation | edge smiles bias DSO |
| Velocity-error update direction | all of the above | DSO / WEMVA gradient | about 20 per iteration | smooth background | artefacts; slow |
| FWI gradient (low-k plus high-k) | full waveform | adjoint state | 10 per gradient | not cycle-skipped (about 33 ms at 15 Hz) | cycle skipping; source-row dominance |
| FWI-refined v | full waveform | L2 / OT / AWI / envelope / WRI with frequency continuation from about 2-3 Hz | about 750-3000 | robust-misfit or extended objectives widen the basin | below-reflector null space; edges |

- **Cost tiers (inference).** "Free" (eikonal and data-domain) channels run in milliseconds. Single-migration channels need about 10 simulations, i.e. 0.01-1 s. Iterative channels need 1e2-1e3 simulations, i.e. seconds to minutes per model. For dataset-scale feature extraction (tens of thousands of models), channels in the first two tiers are practical; iterative FWI per model is not, without GPU batching.

### Gaps
- No benchmarked wall-clock figures for these operations on a 70 x 70 grid were found in the literature. The per-simulation times above are order-of-magnitude estimates plus the user's own project measurements.
- Cost of practical stereotomography (picking plus tomography) at this scale was not quantified in any found source.
