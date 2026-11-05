# Piff validation module (`psf_fomo.piffval`)

Fits simulated stars with Piff under chosen model options and reports diagnostics.
**Status (Phase 4, step D):** model builder, in-memory and file-based fits of one SCA, flux /
centroid / χ² diagnostics, convergence flag, moment residuals (several algorithms), component
summaries, plots, save/load, reference optimum, ρ statistics.

## Quick start

```python
from psf_fomo.sim import RomanSimulation
from psf_fomo.piffval import FitConfig, PiffValidation, load_fit

s = RomanSimulation(n_stars=30, seed=1); s.add_kernel(0.03, 0.0, 0.02); sim = s.run()

cfg = FitConfig()
cfg.set_optics("pinned")                 # or "free", max_zernike=11, prior_sigma=0.05
cfg.add_kernel("signed", order=0)        # GaussianMoments; "positive", or Piff "gaussian"
# cfg.add_pixelgrid(size=5, scale=0.11, order=1)
# cfg.set_outliers(chisq_nsigma=4, max_centroid=0.5)
# cfg.set_piff_shape_method("gauss")     # Piff's internal method; default "hsm"

pv = PiffValidation(sim, cfg, sca=1)     # one SCA per fit; observations pooled
pv.set_reserve(0.2, seed=3)
pv.set_diagnostics(native=("wmom", "gauss"), wmom_fwhm=(0.4, 0.5), oversampled=("hsm",))
res = pv.run(save_to="out/fit01")

res.fit_summary          # chisq, dof, chisq_dof, niter, max_iter, converged, ...
res.summary()            # flux / centroid / χ²: mean, std, err, n, n_flagged; fit vs reserve
res.table                # one row per star
res.moments              # one row per star × comparison × algorithm
res.moment_summary()     # dT/T, de1, de2 per (set, comparison, algorithm)
res.components()         # {'optics': Zernikes vs truth, 'kernel': M vs truth, 'pixelgrid': ...}
res.plot_mean_images(oversampled=True); res.plot_chi2_radius()
res.plot_moment_residuals("truth_os"); res.plot_field("dT_T", "truth_os", "hsm")
res.plot_components("kernel"); res.plot_zernikes()
same = load_fit("out/fit01", sim=sim)   # sim needed for images / new moments / components
```

## Model (`FitConfig`)

Generated Piff structure (pieces left out when not requested):
`Sum[ Convolve[ RomanOptics, kernel, MTF ], Simple(PixelGrid, BasisPolynomial) ]`.

| method | what | default |
|---|---|---|
| `set_optics(mode, max_zernike, prior_sigma, aberration_interp, nominal_interp)` | `'free'`: Z4…Z_max fitted per SCA, Gaussian prior toward 0 of width `prior_sigma` waves (scalar, one per Zernike, or None = no prior). `'pinned'`: held at 0 (nominal optics) by σ = 1e-10 on Z4 only | free, Z4–Z22, 0.05, constant, bilinear (Piff defaults) |
| `set_mtf(on)` | fixed `GaussianMixture` with the §5.3 constants (sigmas in arcsec), `interp: Fixed`; cannot be fitted | on |
| `add_kernel(kind, order, init, max_nfev, interp)` | `'signed'` / `'positive'`: `GaussianMoments` (M in pix², flux and centre fixed in its fit); `'gaussian'`: Piff `Gaussian` (its fit also moves flux and centre); `'blur'`: Piff `GaussianBlur` (Piff branch `gaussian_blur` only, see below); interpolator of `order`, `interp='Polynomial'` or (blur only) `'BasisPolynomial'` | none |
| `add_pixelgrid(size, scale, order, interp, centered, init)` | `PixelGrid` in a Sum, `BasisPolynomial` | none |
| `set_outliers(chisq_nsigma, max_remove, max_centroid)` | Piff `Chisq` / `Centroid` (arcsec) | none |
| `set_iterations(max_iter, min_iter, chisq_thresh)` | Piff fit loop | 30, 2, 0.1 |
| `set_piff_shape_method(method, weight_fwhm)` | Piff's global shape method during the fit (PixelGrid / GSObject init, Piff stats); restored afterwards | `'hsm'` |
| `FitConfig.from_piff_config(dict)` | use a hand-written Piff `psf` dict as is | — |

`chromatic` is always False (chromatic fits deferred). `piff_config(filter_name)` returns the
Piff dict; `to_yaml` / `from_yaml` round-trip the settings.

### `GaussianMoments` (`psf_fomo.models`)
Convolution kernel with parameters (Mxx, Mxy, Myy) in native pix² along world axes.
`signed=False` is a PSD-clipped Gaussian (only broadens; starts at (0.01, 0, 0.01));
`signed=True` is a sum of four tiny Gaussians with moments exactly M of any sign (linear in M,
first-order accurate, starts at 0). Stars whose per-star fit stops at `max_nfev` are counted
(`fit_summary["kernel_max_nfev"]`). Importing `psf_fomo.models` registers it with Piff;
re-importing is safe.

### `GaussianBlur` (Piff PR branch `gaussian_blur`, under test)
Piff's Gaussian blur kernel with parameters `[ixx, iyy, ixy]` in arcsec² (world axes; note the
order), drawn as the PSD projection of the parameters (broadens only, same profile family as
`positive`), exact at zero blur, starts at 0. Not on our pinned Piff commit: the tests skip and
`add_kernel("blur")` fails at `piff.PSF.process` without the branch. `init` is given in our pix²
(Mxx, Mxy, Myy); `psf_fomo.piffval.config.blur_params` / `blur_moments` convert. Components
report M in pix². `interp='BasisPolynomial'` uses the branch's joint solve with a χ² line
search. The reference solver uses a 1e-3 pix² step and starts PSD-clipped kernels at 0.01 pix²
per axis. Study notes: `docs/notes/piff_gaussianblur.md`; tests: notebook
`notebooks/piff_gaussianblur.ipynb`.

## Runs (`PiffValidation`)
- One SCA per fit; the selected observations are pooled.
- `set_reserve(frac, seed)`: `int(frac × N)` reserve stars (Piff's rule), not fitted but
  refluxed and diagnosed.
- `set_verbose(level)`: Piff log output, 0 (errors only) … 3.
- **Convergence**: Piff has no convergence attribute; `converged` is False when Piff logged
  "PSF fit did not converge" (it reached `max_iter`). A fit is never reported as converged
  otherwise.

## Star table columns
`obs`, `sca`, `star_id`, `x`, `y`, `phase_x`, `phase_y`, `mag`, `reserve`, `flag`
(1 = flagged by Piff), `flux` (fitted, ADU), `flux_true` (catalog e⁻ / gain), `flux_r`,
`model_flux` (total flux of the unit model), `flux_tot_r` = `flux_r` × `model_flux`,
`du`, `dv` (fitted centre, arcsec), `dx`, `dy` (pix), `chisq`, `dof` (Piff: pixels − 3),
`chisq_dof`.

## Moment residuals (`res.moments`, `psf_fomo.piffval.residuals`)
Set with `set_diagnostics(moments, native, wmom_fwhm, oversampled, oversample, include_pixel,
comparisons)`; defaults: native wmom 0.4″ + gauss, oversampled hsm, ×5, effective PSF, all
three comparisons. Any of Piff's `hsm`, `am`, `gauss`, `wmom` (several FWHMs) can be listed;
each gives its own rows. Conventions as `psf_fomo.moments` (T = Mxx + Myy in pix², distortion
e, world axes).

| comparison | reference | model |
|---|---|---|
| `data` | noisy data stamp | model at the fitted flux and centre, same stamp; both measured about the catalog position (as wmom needs) |
| `truth_native` | noiseless truth stamp (`render_stamp`) | model at the true position and flux; isolates model error from noise |
| `truth_os` | oversampled truth (`truth_psf`, effective PSF) | model on the same fine grid: `'auto'` profiles convolved with the native pixel once, `'no_pixel'` ones (PixelGrid) as they are |

Columns: star identifiers, `reserve`, `comparison`, `algorithm`, `<key>_ref` / `<key>_model`
for T, e1, e2, g1, g2, sigma, du, dv, flux; `dT_T = (T_model − T_ref)/T_ref`, `de1`, `de2`;
`flag_ref`, `flag_model`, `flag` (OR of both, + 16 if Piff flagged the star). Summaries use
`flag == 0` and report `n_flagged`. The raw `*_ref` / `*_model` moments are kept so that
calibrations (e.g. wmom scaled by gauss) can be done from the table.

## Components (`res.components()`)
- `optics`: per Noll index, fitted extra Zernike at the stars vs catalog truth (waves): mean,
  std, `diff_mean`, `diff_std`, Piff's σ (`fit_err`), prior σ.
- `kernel` (GaussianMoments or Piff Gaussian): per star M = (mxx, mxy, myy), T in pix², vs the
  truth kernel (`kernel_*` catalog columns, 0 if none). A positive GaussianMoments reports the
  PSD-clipped M it draws (`clipped`). Piff's Gaussian (σ, g) is converted to M.
- `pixelgrid`: per star the grid's net flux in the unit model and its largest |value|.

## Plots
`plot_mean_images(which, oversampled)` (mean data / model / residual; oversampled truth /
model / residual), `plot_chi2_radius`, `plot_moment_residuals(comparison, algorithms)`
(histograms, fit vs reserve), `plot_field(column, comparison, algorithm)` (values across the
SCA), `plot_components(kind)` (a component at 3 × 3 SCA positions), `plot_zernikes`.
`which` = `'fit'` (default), `'reserve'`, `'all'`; Piff-flagged stars are left out.

## File-based runs (`run(mode="files")`)
`PiffValidation(sim_or_folder, cfg, sca=5, obs=0).run(save_to="out/fit02", mode="files")`
writes a Piff config (`save_to/piff.yaml`) and runs Piff's own input → select → fit
(`piff.config.process`) on the saved simulation (saved to `save_to/simulation` first if it
was given in memory). One observation per run. The YAML also runs with Piff's `piffify`
command (it imports `psf_fomo.models`).

- Input: image HDU of the SCA; catalog per-SCA HDU with `x`/`y` (simple WCS) or the full
  table with `ra`/`dec` (Roman WCS; pointing from the simulation). `chipnum` = SCA,
  properties `sca`, `obs`, `star_id`.
- Weight 1 / `BKGVAR` (the SCA's median background variance; in memory each star uses its own
  — identical for Gaussian noise). `gain` only when star shot noise is simulated.
- `select.max_snr_weight: 0`: Piff's default (100) would down-weight every star with
  S/N > 100; it is switched off so both modes fit the same χ². Reserve via Piff's
  `reserve_frac` + `seed` (a different random choice than in memory).
- Stars are matched to the catalog by `star_id`; the result is a normal `FitResult`.

## ρ statistics (`res.rho()`, `psf_fomo.piffval.rho`)
`res.rho(algorithms=("wmom", "gauss"), wmom_fwhm=(0.4,), which="all", min_sep=0.1,
max_sep=10, bin_size=0.5)` → table of ρ1 … ρ5 (ξ+, error, ξ−, npairs, `n_used`) per
algorithm; `res.plot_rho(table)`. Piff's `RhoStats` with the per-star measurement replaced by
the chosen algorithm (the global shape method is not changed; Piff's default `hsm` fails on
native stamps). Data vs model at the fitted flux and centre, reduced shear g, T in arcsec²,
separations in arcmin (one SCA ≈ 7.5′). Needs `treecorr`. τ statistics are not included.

## Reference optimum (`psf_fomo.reference`)
The joint χ² optimum of the *same* Piff model (every image drawn by Piff's `psf.drawStar`),
to tell optimiser limitations (Piff above the optimum of its own model) from model
limitations (the optimum itself biased or with excess χ²). CLAUDE.md §8.

```python
out = res.reference(starts=("given", "zero"), max_nfev=50)   # from a Piff fit
from psf_fomo.reference import reference_from_config, comparison_table
out0 = reference_from_config(pv, starts=("zero",))          # no Piff fit, only Piff's init
comparison_table({"Piff": res, "Piff, true centres": out["given"],
                  **{f"optimum ({k})": v for k, v in out["optimum"].items()}})
```

- Parameters: RomanOptics one vector per SCA (with its prior), Simple components as
  polynomials with the interpolator's own terms (`Polynomial`, `BasisPolynomial`; `Mean` =
  constant). Frozen: `Fixed` components, pinned optics, and kinds listed in `freeze`
  (`'optics'`, `'kernel'`, `'pixelgrid'`).
- Fluxes profiled in closed form, centres held at the truth: the reference χ² is an upper
  bound on the true optimum of Piff's problem. A centred PixelGrid is optimised over the full
  grid. Only fit stars (not reserve, not flagged) enter χ²; reserve stars are evaluated too.
- dof = weighted pixels of fit stars − fit stars − free coefficients (Piff: pixels − 3 per
  star). Compare Piff with the optimum through `out["given"]` (Piff's parameters with true
  centres and profiled fluxes, same dof).
- Starts: `'given'` (the stars' parameters) and `'zero'` (kernels and grids at zero; optics
  and Piff Gaussians at their given values). The same χ² from both is evidence of the global
  optimum. `converged` is False when scipy stopped at `max_nfev`.
- Each solution is a `ReferenceResult` with `stars`, `table`, `summary` (χ², dof, per-leaf
  info incl. `linear` and `proj_err` = how well the basis reproduces the given parameters),
  `moments` (with the fit's diagnostics settings, `moments='same'`) and `components`.
- Cost: one draw per star per parameter for each Jacobian; components that enter linearly
  (PixelGrid in a Sum) are differentiated once. Free optics cost one set of `getPSF` calls per
  parameter per Jacobian.

## Saved folder
`psf.fits` (Piff's own file, incl. the fitted stars), `stars.fits` (the table), `moments.fits`
(if computed), `config.yaml` (run settings incl. the Piff dict and diagnostics settings, fit
summary, provenance). `load_fit(path, sim=None)` reloads; the simulation is reloaded when it
was given as a folder, or can be passed. Piff's file has no star images, so images are re-cut
from the simulation and the saved fits attached (`fitted_stars()`).

## Limitations
- Piff's convergence test is Δχ² < `chisq_thresh` × dof. With very high S/N or noiseless
  stars χ² has no noise scale, the default 0.1 is met at once, and Piff stops after `min_iter`
  and reports convergence. For such runs use `set_iterations(chisq_thresh=1e-6)` (or raise
  `min_iter`) and check `niter`.
- Achromatic fits only.
- Observations are pooled assuming they share the WCS (true for the current simulations).
- In-memory reserve choice uses its own random generator, so it differs from Piff's
  `select.reserve_frac` for the same seed.
- In-memory runs apply no Piff selection (no `max_snr_weight` down-weighting, no S/N or size
  cuts); file-based runs apply Piff's selection with `max_snr_weight: 0`.
