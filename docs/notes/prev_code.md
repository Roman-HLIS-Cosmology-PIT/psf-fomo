# `prev_code/` — Phase 0 notes

Contents of `prev_code/` (all dated 2026-11-02 11:14):

| file | lines | what it is |
|---|---|---|
| `simulate.py` | 646 | `SimConfig` dataclass + truth rendering (`simulate_dataset`) |
| `compare.py` | 747 | Piff config builder, fit, Zernike and moment comparisons, SCA painting |
| `run_first_test.py` | 153 | `main(**knobs)` closed-loop driver used by the notebook |
| `kernel_moments.py` | 232 | `GaussianMoments` Piff model (positive / signed) + self-check |
| `pixelgrid_residual.py` | 219 | `PixelGridResidual` (zero-flux, zero-centroid grid, ridge) |
| `check_piff_PRs.ipynb` | 129 cells, 8 MB | experiments; **also contains, as code cells, the modules that are not on disk**: `test_pg_manystars.py` (cell 52, incl. `preflux()`, `run_case`, `run_all`), `pixelgrid_viz.py` (53), `reference_model.py` (54, generic joint optimum), `reference_optimum.py` (103, optics+PixelGrid joint optimum) |

Not present on disk although CLAUDE.md §2 lists them: `reference_model.py` (only as notebook
cell 54), the **patched `roman_psf.py`** (not anywhere — only its traces in the code below),
`io_utils.py` (referenced in `compare.py` docstrings), `PixelGridOversampled` (used in notebook
cells 91–101, source not found). `.ipynb_checkpoints/` holds older copies.

---

## simulate.py

- Truth PSF is produced by **Piff's own `RomanOpticalModel.getProfile`** (4-corner bilinear
  interpolation of sample-point PSFs, `aberration_interp` constant/linear/global) — not
  `galsim.roman.getPSF` evaluated at each star.
- Error chain (`apply_model_error`): `dilate(psf_dilation)` → `shear(psf_g1, psf_g2)` →
  `Convolve(MTF)` if `apply_mtf`; then shift by `fit.center + center_offset`, × flux,
  `drawImage(method='auto')`. MTF = 3 Gaussians (§5.3 constants) with
  σ = 0.3279 · psf_dilation · mtf_scale · pixel_scale.
- `extra_aberration_sigma`: per-star Gaussian scatter on the aberration vector (recorded as
  `gen_params`).
- `truth_chromatic`: flat SED in photons (`SED('1', 'nm', 'fphotons')` normalised to flux 1
  in the bandpass), drawn through the real bandpass.
- Noise: optional Poisson on `sky_level` (then sky subtracted, weight = 1/(image+sky)) and/or
  Gaussian `noise_sigma` (weight = 1/σ²; overrides the Poisson weight if both set).
  No detector effects.
- Positions: `grid` (√N × √N linspace, truncated to N) or `random` uniform, inside
  `edge_margin` = 10 % of the SCA on each side. Seeds: `seed + 1000·sca` (positions),
  `+2000·sca` (fluxes), `+3000·sca` (noise / scatter).
- Flux in counts (default 1e5), constant or uniform in a range — no magnitudes.
- WCS: `simple` (PixelScale 0.11) or `roman` (`galsim.roman.getWCS`, default pointing
  RA 61.24°, Dec −48.42°).
- Stamps only: stars are drawn on 25×25 postage stamps, not full SCA images
  (`compare.paint_sca_images` pastes them onto a 4088² frame afterwards).
- Defaults: 9 stars, grid, SCA 1, H158, max_zernike 12, prior σ 1e6 (effectively none),
  `chisq_thresh=1e-8, min_iter=5, max_iter=60`, pupil_bin 4.
- Internal inconsistencies: `aberration_amplitude` described as "radians, roughly" (notebook
  cell 84 says waves); `linear` truth docstring says shape (4, n) corners but code uses
  (3, n) = (a, b, c); `pupil_bin` field is documented as a Roman constructor arg but is never
  passed anywhere.

## compare.py / run_first_test.py

- `build_psf_config`: `RomanOptics` (+ `apply_mtf` key, only valid with the patched model)
  → optional `Convolve` with a `Gaussian` kernel (`Polynomial` interp) → optional `Sum` with
  `PixelGrid` (`BasisPolynomial`).
- `make_truth_psf`: sets `interp.set_sca_solution(truth)` to render with truth without fitting.
- Moments: HSM adaptive on a 5× (or 8×) oversampled image. For `'auto'` profiles the
  **native** pixel is convolved first, then drawn `no_pixel` on the fine grid; `'no_pixel'`
  profiles are drawn directly (documented double-pixel pitfall, ~+29 % in T).
  dT/T uses σ = HSM `moments_sigma` (determinant radius), T := σ² — matches CLAUDE.md §5.5.
- `compare_moments_native`: uses `piff.HSMCatalogStats().measureShapes` — truth side measured
  on the noisy native stamps, model side on noiseless model draws.
- `run_first_test.main` prints "CLOSED-LOOP RECOVERY TEST PASSED" unconditionally (the
  assertions are commented out) and defaults to `fit_apply_mtf=True`, which only runs with the
  patched `roman_psf.py`.

## kernel_moments.py — `GaussianMoments`

- Params (Mxx, Mxy, Myy) in native pix²; `_method='auto'`, `_centered=True`; config type
  `GaussianMoments`; `signed`, `init`, `m0=0.0025`, `pixel_scale`, `floor_arcsec`,
  `diff_step`, `max_nfev=30`.
- `signed=False`: Gaussian with PSD-clipped covariance, trf with bounds Mxx, Myy ≥ 0, start
  (0.01, 0, 0.01). `signed=True`: (1−Σw)G(M0) + Σ w_j G(M0 + h E_j), h = m0/2, linear in M,
  start 0.
- Per-star `scipy.optimize.least_squares(method='trf')`, flux/centre fixed; **does not check
  whether `max_nfev` was hit** (CLAUDE.md §12 requires reporting that).
- Registers itself twice (auto via `_type_name`, then manually if missing).
- Self-check draws on a fine grid and compares unweighted moments with M.

## pixelgrid_residual.py — `PixelGridResidual` (out of initial scope)

- Forces the parent to `centered=False` (because upstream centred PixelGrid now works in an
  (n²−2)-dim reduced space) and imposes Σp = Σu p = Σv p = 0 itself via an augmented
  (A, b) in `chisq()` (Basis interp) or a null-space solve in `fit()`; optional ridge
  λ = ridge · max diag(AᵀA).
- Docstring notes for new Piff: reflux no longer has the model-centroid term; the
  `_model_can_be_offset` switch "no longer does anything" — but upstream `GaussianMixture`
  still sets `_model_can_be_offset = False` (harmless attribute).

## Notebook-only modules (cells 52, 54, 103)

- **`preflux()`** (cell 52): context manager that patches `SumPSF.initialize_params` to run
  `self.reflux_stars(stars, logger)` once after the original init, restored on exit.
  Upstream now does the same reflux inside `PSF.fit` for every PSF type (see piff.md §1).
- **`reference_model.py`** (cell 54): generic joint optimum walking Sum/Convolve leaves
  (RomanOptics per-SCA vector, Simple+Polynomial/BasisPolynomial as polynomials in position),
  fluxes in closed form, centres at truth, linear-component derivative caching. Matches
  CLAUDE.md §8. Uses `piff.StarFit` with fallback to `piff.star_fit.StarFit` (that module does
  not exist upstream; `StarFit` lives in `piff/star.py`).
- **`reference_optimum.py`** (cell 103): earlier, optics + PixelGrid only (bilinear LM).
- Notebook cells 108–115 read HSM catalogs from `data/*hsm*.fits` (not in the project folder).

## What to carry over (per CLAUDE.md §7) vs. what changed

- Keep: the rendering-chain idea and §5.3 MTF constants (now identical to Piff's test), the
  oversampled-moment convention, GaussianMoments (signed/positive), the reference solver
  design, seeds per (sca, purpose).
- Replace: patched `apply_mtf` → `Convolve(RomanOptics, Simple(GaussianMixture, Fixed))`;
  `preflux()` → upstream reflux-after-init (to be confirmed by the §6.2 test); Piff-model-based
  truth rendering → `galsim.roman.getPSF` exact / Legendre (per CLAUDE.md §3.2).
