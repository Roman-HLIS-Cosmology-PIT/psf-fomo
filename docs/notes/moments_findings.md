# Phase 3 findings: PSF moments on native (undersampled) Roman images

Runs in `notebooks/phase3_moments.ipynb` (project folder), 2026-11-02/03. Setup unless stated:
H158, SCA 1, simple WCS (world axes = pixel axes), mag 18, Gaussian noise 1 e⁻ (very high
S/N), seed 1, Piff `pupil_bin = 4`. Numbers are from single runs of 20 or 100 stars; quoted
± are star-to-star standard deviations unless marked "error on the mean".

## Definitions

- **Truth / injected error**: HSM adaptive moments of the 5× oversampled, pixel-convolved
  effective PSF of each star (`*_hsm_os`). Chosen as the most accurate single-exposure
  measurement and the convention that matters downstream.
- **ratio** = native measured error / injected error, per star (accuracy: mean ≈ 1; stability:
  small std). **self_ratio** = measured / the same algorithm on the oversampled image: tests
  robustness to undersampling only. The two differ because algorithms define "size"
  differently (a fixed narrow weight sees only part of a size change).
- T = Mxx + Myy [native pix²], e = distortion.

## Physics of the injected errors

- The pixel response is not dilated, so a 1 % dilation of optics + MTF gives an effective
  dT/T ≈ 1.40 % (measured 0.01396). Unweighted-moment estimate:
  dT/T ≈ 2ε (T_eff − T_pix)/T_eff with T_pix = 1/6 pix², T_eff ≈ 0.96 pix².
- Size errors produce a small real shape change: isotropic parts (pixel, MTF) add to T but not
  to Mxx − Myy, and the PSF has e1 ≈ −0.015. Dilation: de/e ≈ +0.2 δ (|e| grows; δ = effective
  dT/T); MTF broadening: de/e ≈ −δ (|e| shrinks). Observed ~5e-5 for δ ≈ 3 %.
- A shear of the optics is diluted by MTF and pixel: de_eff ≈ de_opt · T_opt/T_eff ≈ 0.6 de_opt.
- Dilation/shear (coordinate transforms) differ from convolution with a Gaussian kernel beyond
  second moments; a positive kernel can only broaden (cannot mimic a shear without growing).

## Algorithm behaviour

| algorithm | dil (dT/T) | de1 | de2 | notes |
|---|---|---|---|---|
| hsm (≈ am) | 0.37 ± 0.22 (6/20 stars) | — | — | fails on most stamps |
| gauss | 0.85 ± 0.21 | 0.86 ± 0.18 | 1.00 ± 0.08 | accurate-ish on average, phase-driven scatter |
| wmom 0.4″ | 0.59 ± 0.02 | 0.83 ± 0.02 | 0.83 ± 0.003 | stable, responds only partly |

(100 stars, noisy; ratio vs HSM-oversampled truth.)

- **hsm/am**: GalSim adaptive moments fail (flag) on all pixel-centred stars and most
  random-phase ones at H158 sampling (σ_eff ≈ 0.7 pix). Where they work (pixel corner) T is
  ~33 % too large and only ~15 % of a size error is recovered. At random phase the surviving
  stars are a phase-selected sample. `am` was identical to `hsm` everywhere.
- **gauss** can be flagged on an exactly noiseless image although the fit is right: an analytic
  Gaussian (σ = 3 pix, g = (0.1, −0.05), `no_pixel`) gave ngmix flag 512 after 50 evaluations
  with exact parameters, independent of the weight scale; Piff reports it as flag 1. Meaning of
  bit 512 not checked. No robust-test star was flagged.
- **gauss**: ratio 1.13 at pixel centre, 0.27 at pixel corner (20 stars) → per-star scatter
  0.1–0.26 from the sub-pixel phase; averages over phases to ~0.85 (size, e1) and ~1.00 (e2).
- **wmom** (weight fixed at the catalog position, not deweighted): self_ratio ≈ 1 for FWHM ≳ 0.3″
  at any phase (0.94–1.07 between centre and corner). Ratio vs truth for 1 % dilation vs FWHM
  0.11/0.2/0.3/0.4/0.5/1.0″: 0.31±0.30, 0.36±0.19, 0.44±0.07, 0.58±0.02, 0.67±0.006,
  0.63±0.004. A wider weight responds more and scatters less; the drop above 0.5″ is not
  understood.

## Centroid sensitivity of wmom

wmom has no centroid fit. Random offsets of the measurement position (0.1 pix rms per axis,
100 stars):
- **same offset** in both measurements of a star: unchanged (dil 0.58 ± 0.03, de 0.82 ± 0.02–0.03).
  This is the data-vs-model case in Phase 4 *if both are measured about one common position*.
- **independent offsets**: std 1.0–1.7 — useless per star. Raw T bias from the offset 1.1 %.
- gauss (fits its own centre) is unaffected.

## Empirical calibration of wmom with gauss

k = ⟨gauss⟩/⟨wmom⟩ (or the slope of gauss on wmom through the origin) from half the stars,
applied to wmom on the other half (noisy, 50 + 50 stars):
- dil: 0.83 ± 0.03 (k = 1.41 ± 0.04) — gauss's mean with wmom's scatter; still ~15 % low
  because gauss is.
- de2: 1.011 ± 0.004 (k = 1.22 ± 0.01).
- de1 with its own k: 0.84 ± 0.02; **with the de2 k: 1.015 ± 0.029** — wmom responds equally to
  e1 and e2, and gauss is unbiased on e2.
- Ratio-of-means and slope estimates agree. k's precision is set by gauss's per-star scatter
  (≈ 0.2/√N).

## Pixel-grid anisotropy (e1 vs e2)

A square grid passes frequencies up to a square limit, √2 further along the diagonals: e2
(diagonal) suffers less aliasing than e1 (x vs y). Round analytic Gaussian, 200 random phases:

| pre-pixel σ [pix] | gauss e1 | gauss e2 | wmom e1 | wmom e2 |
|---|---|---|---|---|
| 0.63 (≈ H158) | 1.013 ± 0.010 | 0.999 ± 0.008 | 0.848 ± 0.026 | 0.838 ± 0.003 |
| 0.43 | 1.157 ± 0.039 | 0.999 ± 0.054 | 1.006 ± 0.276 | 0.922 ± 0.033 |

- wmom's ~8× larger e1 scatter on the Roman PSF is reproduced by the round Gaussian: pure grid.
- gauss's poor e1 on Roman (0.86 ± 0.18) is *not* reproduced at σ = 0.63: it comes from the
  Roman profile (core, rings, baseline e1) combined with the sampling.
- At σ = 0.43 wmom's e1 and e2 mean responses differ, so the e2→e1 calibration holds at H158
  but needs checking for bluer filters.

## Practical conclusions

- Do not use hsm/am on native H158 stamps.
- For per-star PSF-error measurements on native images: wmom (0.4–0.5″) for precision, with an
  empirical scale factor from gauss (shapes: from e2), measured about a common position for
  data and model.
- e1 is intrinsically noisier than e2 per star (grid geometry); no calibration removes that.
- Phase-averaged results can hide large per-star errors (gauss) or come from a phase-selected
  subsample (hsm): always report the star-to-star std and the number flagged.

## Open checks

1. Analytic test with the Roman baseline ellipticity (e1 = −0.015) and a σ scan; locate each
   Roman filter on that scale; robust test in F087/F062.
2. Calibration transfer to other error types (MTF, Zernike-driven size/shape, jitter); other
   SCAs and field positions; fainter stars (mag 18–22).
3. Realistic estimated centroids instead of random offsets; scaling of the independent-offset
   scatter with the rms (offset² vs linear).
4. wmom response at 1.0″ below 0.5″.
5. Less biased size reference for the calibration: pixel-integrated Gaussian fit, centre fixed.
6. Roman WCS: pixel-frame shapes for the e2-based calibration (world ≠ pixel axes).
7. Pre-pixel oversampled truth (`include_pixel=False`): the de of a pure dilation should vanish.
8. Error budget per error type (nominal → pre-pixel → effective → native), response plot
   (measured/injected with a line at 1).
9. Gaussian-kernel error type (Mxx, Mxy, Myy) in the simulation — planned, not implemented.
10. Higher-order moments (Piff `calculate_moments`) — deferred to Phase 4.
