# Piff `main` — Phase 0 notes

Studied: local clone `Piff/` (project folder, next to `psf-fomo/`), branch `main`,
commit **`6ada1330bd7b1cb1cf23f50dcb3bdcbc2db69800`** (from `Piff/.git/refs/heads/main`;
git was not run). `piff/_version.py` says `1.8`; `CHANGELOG.rst` for 1.7→1.8 is still empty.
The clone has a compiled `_piff*.so` in `piff/`, i.e. it was built in place.

All statements below come from reading the source/tests; file:line references are to this
commit.

---

## 1. Fitting loop (`PSF.fit`, `piff/psf.py:395`)

Order of operations:
1. `initialize_flux_center` — flux = sum of pixels with w≠0; centre = data centroid if
   `fit_center` and not `trust_pos` (psf.py:172).
2. `initialize_params` (component inits; see §3).
3. **Reflux immediately after initialisation** (psf.py:423–432). Applies to *every* PSF type
   (Simple, Sum, Convolve, RomanOptics), not only Sum. The comment states the purpose: account
   for model flux outside the stamp so that a PixelGrid in a Sum does not absorb the flux
   deficit. This is exactly our `preflux()` (notebook `test_pg_manystars`, a patched
   `SumPSF.initialize_params` that runs one `reflux_stars` pass) — see §6.2 in `conflicts.md`.
4. Loop `max_iter` times: `single_iteration` → update Poisson weights from the model
   (`addPoisson`, only if a `gain` property exists, star.py:1328) → `reflux_stars` →
   outliers → convergence test.
5. Convergence: stop when no stars were removed, `oldchisq − chisq < chisq_thresh · dof`,
   and `iteration+1 ≥ min_iter`. No `abs`, so a χ² *increase* also stops the loop. If the loop
   runs out, only a `logger.warning("PSF fit did not converge...")` is emitted
   (psf.py:486). **There is no convergence flag attribute**: `psf.niter == max_iter` is
   ambiguous (it also happens when convergence occurred exactly on the last iteration). We will
   need our own check (re-evaluate the criterion from `chisq`, `last_delta_chisq`, `dof`,
   `niter`, or capture the warning).
6. Defaults (Simple/Sum/Convolve/RomanOptics): `chisq_thresh=0.1, min_iter=2, max_iter=30`.

`reflux` (psf.py:247):
- Linear flux solve, then (if `fit_center` and not `trust_pos`) one Gauss–Newton step in
  (flux, du, dv) with finite-difference centroid derivatives (step 1e-5 arcsec).
- Renders in double precision even for float32 data (psf.py:273).
- Reported χ² is now recomputed from a fresh draw at the new flux/centre (psf.py:339–343),
  not the linearised value.
- dof = N(w≠0) − 3 with centre free, − 1 otherwise.
- `reflux_stars` swallows exceptions and flags the star (psf.py:358–368).
- `fit_center` of a composite = `any(component.fit_center)`.

`set_context(wcs, pointing, bandpass)` is the documented way to attach context; passing
`wcs`/`pointing` to `fit()` still works but logs an error-level "WARNING" (psf.py:414).
`fit(stars, wcs=None, pointing=None, logger=None, convert_funcs=None, draw_method=None)`.

## 2. Composite PSFs

**SumPSF** (`sumpsf.py`)
- `initialize_params` passes `default_init='zero'` to all components after the first
  (sumpsf.py:182). This is a *default*: an explicit `init` on the component wins.
- `single_iteration`: for each component k, subtract the current images of the others from the
  data, fit k, redraw k (block-coordinate descent; unchanged in substance).
- `_getRawProfile`: if components mix `'auto'` and `'no_pixel'`, the `'auto'` ones are
  convolved with the local pixel and the sum is `'no_pixel'` (sumpsf.py:290–302).
- `_num` bookkeeping via `set_num` and `num_components` (recursive), unchanged.

**ConvolvePSF** (`convolvepsf.py`)
- Components after the first get `default_init='delta'` (convolvepsf.py:202).
- Each component is fitted with a `convert_func` that convolves the others
  (`_ConvolveProfile`).
- Method: `'no_pixel'` if any component is `'no_pixel'`, else `'auto'` (convolvepsf.py:298).
  So `Convolve(RomanOptics, GaussianMixture)` is `'auto'` → pixel applied once at draw.

## 3. Models

**PixelGrid** (`pixelgrid.py`)
- `init`: `gauss` (default unless a composite supplies a default), `zero`, `delta`; `hsm` is a
  deprecated alias of `gauss`. `init='zero'` = Gaussian × 1e-10 and **forces `fit_flux=True`**
  (pixelgrid.py:192–202). So "Sum forces init='zero', fit_flux=True" holds only when no
  explicit `init` is given.
- `centered=True`: centroid constraint Σu·p = Σv·p = 0 imposed through a
  `HouseholderTransform` (util.py:861). `A` is restricted to the null space →
  **n² − 2 columns**; params are transformed/reconstructed via `transform_params` /
  `reconstruct_params`, which `SimplePSF` passes to the interpolator.
- `fit_flux=False` → `normalize()` divides by the sum (drawing not linear in params).
- `_method = 'no_pixel'`. Grid scale in arcsec.
- `initialize` raises if the stamp is smaller than `size·scale/pixel_scale`.
- Uses `star.measure_size_shape()` (the global shape method, §5) for the initial Gaussian.

**GaussianMixture** (`gaussian_mixture_model.py`) — this is the "MixtureGaussian" in CLAUDE.md
- Config type name **`GaussianMixture`** (class `piff.GaussianMixture`).
- Params: `weights` (normalised to sum 1, must be ≥ 0), `sigmas` (**arcsec**, > 0), optional
  per-component `g1, g2` (reduced shear, |g|<1), `du, dv` (arcsec).
- `_method='auto'` (no pixel), `_centered=True`.
- **No fitting**: `fit()` raises `NotImplementedError`; must be used with `interp: Fixed`.
  Nothing (width, weights) can be fitted — an MTF scale error cannot be absorbed by this
  component.
- Test `tests/test_roman.py::test_roman_diffusion` (line 1427) uses exactly our §5.3 kernel:
  weights (0.17519, 0.53146, 0.29335), rel. sigmas (0.4522, 0.8050, 1.4329), base σ = 0.3279
  native pix, converted to arcsec with `galsim.roman.pixel_scale`, in
  `Convolve[Simple(Gaussian, init=delta, Mean), RomanOptics, Simple(GaussianMixture, Fixed)]`.
  The test only checks image agreement to 0.1 % of peak (noiseless, 3 stars), not parameters.

**Fixed interp** (`fixed_interp.py`): subclass of `Mean`; `fit_parameters=False` makes
`SimplePSF.single_iteration` skip model fitting entirely (simplepsf.py:179). Requires identical
initial params for all stars.

**GSObject / Gaussian** (`gsobject_model.py`)
- Params (scale, g1, g2) [+ flux if `fit_flux`, + du,dv if not centred].
- In `_resid`, `scale ≤ 0` or `|g| ≥ 1` returns 1e300 residuals (gsobject_model.py:224) —
  no longer a crash, but still a hard wall near zero size.
- `least_squares_fit` resets a starting `scale² < pixel area` up to one pixel
  (gsobject_model.py:304) — so a `delta`-initialised kernel restarts at ~1 pix size every fit.
- `fastfit` is disabled when `shape_method == 'wmom'`.

**Model registration**: `Model.__init_subclass__` registers `_type_name` and raises
`ValueError` if the name already exists (model.py:84). Re-executing a class definition
(notebook re-run / autoreload) will raise; our `models/` must guard this.

## 4. Roman optics (`piff/roman/roman_psf.py`)

Config `type: RomanOptics` → `RomanOpticsPSF(model=RomanOpticalModel, interp=RomanSCAInterp)`.
- Options: `filter_name` (F062…F213 aliases or GalSim names), `chromatic` (default **True**),
  `max_zernike` (4..22, default 22), `aberration_interp` (`constant` default / `global` /
  `linear` = a + b x̂ + c ŷ per SCA, x̂ = x/4088), `nominal_interp` (`bilinear` default = 4
  corners / `five_point` = corners + centre with basis [1, x, y, xy, x²+y²], plus a centre
  correction to undo GalSim's own bilinear nominal-aberration interpolation),
  `aberration_prior_sigma` (default 0.05, scalar or per-Zernike; Gaussian prior toward **0**),
  `nproc`, `chisq_thresh/min_iter/max_iter`, `outliers`.
- **No `apply_mtf`, no `pupil_bin` argument.** `pupil_bin` is a module global
  `piff.roman.roman_psf.pupil_bin = 4` (tests set 16 via a context manager).
- Params: Z4..Z_max passed as `extra_aberrations` (index = Noll number, 0..3 zero) to
  `galsim.roman.getPSF(sca, filter, SCA_pos, pupil_bin, wcs=star.image.wcs,
  extra_aberrations, wavelength)`; `wavelength = bandpass.effective_wavelength` if
  not chromatic. Units of `extra_aberrations` = GalSim's (verify: waves).
- PSF at a star = weighted sum of the 4 (or 5) sample-point **profiles** (interpolation in
  PSF space, not in aberration space).
- Fit (`_fit_sca_group`): one SCA-level Gauss–Newton step per Piff iteration, finite-difference
  Jacobian (step 0.05·max(|p|, 0.02)), ridge 1e-6 + prior, `lstsq`. Per-star dof = N(w≠0) − 3.
- SCA from star property `sca`, else `chipnum`.
- Chromatic needs per-star `sed_eff` (from Piff's SED input machinery); drawing multiplies the
  profile by `sed_eff` and draws through a *flat* version of the bandpass (`make_flat`).
- `_method='auto'`, `_centered=True`.
- `devel/roman/piff.yaml`: production-like config — `Sum[RomanOptics(chromatic False,
  max_zernike 12, prior 3e-3, constant, bilinear), Simple(PixelGrid 0.11″ × 5,
  BasisPolynomial 2)]`, outliers Chisq + Centroid, stamp 25, one image file with
  `image_hdu: [1..18]` and `properties: {sca: '$@image_num + 1'}`.

## 5. Moment / shape measurement (new)

`piff.set_shape_method(method, weight_fwhm=None)` (star.py:30) sets a **process-wide global**
used everywhere `measure_size_shape()` is called: stats, PixelGrid `gauss`/`zero` init,
GSObject init and `fastfit`, Select (flag, `size_reject`, `max_pixel_cut`), size–mag
selection. (`HSMCatalogStats` always uses HSM; `calculate_moments` takes an optional
per-call `shape_method`.) Methods, all returning `(flux, cenu, cenv, sigma, g1, g2, flag)` in local
**world** coordinates (arcsec), `sigma` = determinant radius det(M)^¼, g = reduced shear:

| method | implementation | notes |
|---|---|---|
| `hsm` (default) | `FindAdaptiveMom(strict=False)`, weight used only as a mask | σ, shape converted through the local Jacobian; flag bits 1 status, 2 flux<0, 4 centroid>1″ |
| `gauss` | `ngmix.fitting.Fitter(model='gauss')` | maximum-likelihood elliptical Gaussian fit; σ from T, g |
| `am` | `ngmix.admom.AdmomFitter` | flux reported ×2 to match HSM's convention |
| `wmom` | `ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)` | **fixed circular Gaussian weight, FWHM in arcsec**, about the catalog position; raw weighted moments, *no deweighting*; e→g converted |

- ngmix Jacobian anchored at the catalog position (star.py:282).
- Config: top-level `shape_method`, `shape_weight_fwhm` keys. **`piff.process` / `piffify` /
  `plotify` call `set_shape_method(config.get('shape_method','hsm'), ...)` at start**
  (config.py:147, 226) — running Piff silently resets a method chosen earlier in the session.
- Worker processes receive the method via `Pool(initializer=set_shape_method)` (util.py:238).
- `calculate_moments` (util.py:312) builds its weight from hsm/gauss/am; `wmom` falls back to
  `gauss` with a warning. Error estimates only verified for `hsm`.
- Tests (`tests/test_moments.py:303`) check `gauss` and `am` on noiseless analytic
  Gaussians drawn with `no_pixel`; nothing in Piff tests undersampled behaviour or
  native-vs-oversampled bias. That is our Phase 3 job.
- Because the method is global, choosing e.g. `wmom` also changes fitting initialisation and
  star selection (list above). For our moments module we should call the
  `Star.measure_*` methods (or ngmix directly) rather than flip the global.

## 6. Inputs, reserve stars, stats

**Files input** (`input.py:172`): one image per `image_num`; `image_file_name`,
`cat_file_name` per image (string, list, glob, or GalSim config value), `image_hdu`
(can be a list), `cat_hdu`, `x_col/y_col` or `ra_col/dec_col`, `weight_hdu` or `noise`
(variance), `sky`, `gain`, `satur`, `property_cols`, `properties` (e.g. `sca`),
`stamp_size` (default 32), `trust_pos`, `bandpass` (GalSim config dict), SED options
(`sed_col`, `sed_file_name` + `sed_wave_type`, `sed_flux_type`; FITS-table SEDs; reduction via
`sed_tol`, `sed_max_samples`). Parquet catalogs are supported.
- **The catalog is not filtered by chip**: every row of the catalog paired with an image is
  treated as a star on that image (with `ra/dec` columns, off-image positions are dropped
  through the WCS). The WCS dict is keyed by `chipnum` (default `image_num`).

**Reserve** (`select.py:242`): `select.reserve_frac` (default 0), `select.seed`; applied per
chipnum, `int(frac·N)` rounded down, sets property `is_reserve`. Reserve stars are skipped in
fitting (`SimplePSF`, `RomanOpticsPSF`) but are interpolated, refluxed and kept in `psf.stars`.
In memory we can set `star.data.properties['is_reserve']` ourselves.

**Stats** (`stats.py`, `star_stats.py`, `twod_stats.py`, `size_mag.py`): types include
`Rho`, `ShapeHist`, `ShapeCatalog`, `HSMCatalog`, `StarImages`, 2-D field plots, size–mag.
- `measureShapes` returns T = 2σ²(1+g²)/(1−g²) (trace convention, arcsec²), g1, g2, flag;
  optional 4th-order and raw moments.
- **RhoStats** (`stats.py:510`): ρ1–ρ5 via TreeCorr `GGCorrelation`, using reduced shear g
  (not e), dT = T_data − T_model, positions from star properties `u, v` (arcsec);
  defaults `min_sep=0.5`, `max_sep=300` arcmin, `bin_size=0.1`; extra kwargs passed to
  TreeCorr. Stars with flag≠0 on either side are dropped. Uses the global shape method.
- **No τ statistics** in Piff (they need a galaxy shear catalogue); we must implement τ
  with TreeCorr ourselves.

## 7. Things to remember when building against this commit

- Pin `6ada1330`; Piff's version string does not distinguish it from the release.
- `pupil_bin` for fitting is the module global; for speed in tests copy Piff's
  `fast_pupil_bin` context-manager pattern (test_roman.py:41).
- Per-star exceptions are swallowed in init, fit and reflux (they flag the star). Reproduce
  outside Piff's try/except to debug (CLAUDE.md §10 still applies).
- `SumPSF.initialize_params(stars, logger)` still needs a real logger (it calls
  `logger.debug` directly); `RomanOpticsPSF.initialize_params` too.
