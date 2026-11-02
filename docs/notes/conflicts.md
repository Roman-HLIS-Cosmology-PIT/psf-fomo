# Conflicts and corrections to CLAUDE.md (Phase 0)

Sources: Piff `main` @ `6ada1330` (see `piff.md`) and `prev_code/` (see `prev_code.md`).
GalSim / ngmix / TreeCorr not yet studied (see `README.md`). Severity: **A** = changes what we
build, **B** = wording/convention fix, **C** = process.

## Conflicts

| # | CLAUDE.md | Finding | Sev. |
|---|---|---|---|
| 1 | §2, §3.4, §6.1: "`MixtureGaussian` model for the MTF" | It is **`GaussianMixture`** (`piff/gaussian_mixture_model.py`). It **cannot be fitted** (`fit` raises; it must be used with `interp: Fixed`). Sigmas are in **arcsec**, weights renormalised. So an MTF width error (`mtf_scale ≠ 1`) can only be absorbed by other components (kernel, PixelGrid), never by the MTF component itself. §6.1 "free/fixed parameters" → fixed only. | A |
| 2 | §5.2 "current implementation: optical profile from `galsim.roman.getPSF` (exact or Legendre)" | `prev_code/simulate.py` draws truth with **Piff's own `RomanOpticalModel.getProfile`**, i.e. the 4-corner bilinear interpolation that Piff fits with. All §9 numbers and the "old no-error results" in §6.1 come from truth = fit approximation. Moving truth to exact `getPSF` (or Legendre-9) adds an interpolation model error, so the old results will not be reproduced exactly. To compare, we need an extra "Piff-bilinear" truth mode (or we accept the change). | A |
| 3 | §3.2 `legendre5` "corners + centre (cf. Piff's five-point scheme)" | Piff's default `nominal_interp` is `bilinear` (4 corners). `five_point` uses basis [1, x, y, xy, x²+y²] at GalSim's aberration sample positions (from `_read_aberrations`) with a centre correction. It is not a Legendre basis. Also, Piff's comment says GalSim's `getPSF` itself interpolates the nominal aberrations bilinearly, so "exact" means exact getPSF with GalSim's aberration interpolation (verify in GalSim). | B |
| 4 | §5.4 "Inside a Sum, Piff forces the PixelGrid to `init='zero'`, `fit_flux=True`" | It is a *default*: `default_init='zero'` for components after the first, and an explicit `init` wins. `fit_flux=True` is forced only when the effective init is `zero`. Convolve components after the first default to `init='delta'`. | B |
| 5 | §5.1 "kernel parameters and PSF-level moments in native pixel²" | Piff works in **arcsec** throughout: GaussianMixture sigmas, PixelGrid scale, shape σ (arcsec), Piff stats T (arcsec², trace convention), `wmom` FWHM (arcsec). We need explicit converters at the Piff boundary. | B |
| 6 | §5.5 dT/T with σ = HSM `moments_sigma` (T := σ², determinant radius) | Piff stats and ngmix use T = trace = 2σ²(1+g²)/(1−g²); ρ-stats use **g** (reduced shear), not e. The two dT/T definitions differ at O(e²). §3.3 already asks to reconcile; this is the concrete mismatch. | B |
| 7 | §3.3 algorithm list "HSM, fixed-size weighted moments, Piff's new methods, ngmix" | Piff's new methods are `set_shape_method('hsm'\|'gauss'\|'am'\|'wmom', weight_fwhm)`. Three of the four are thin **ngmix wrappers** (Gaussian fit, admom, GaussMom). `wmom` is the fixed-size weighted moment: circular weight, FWHM in arcsec, centred on the catalog position, no deweighting. The list overlaps, so we should decide whether "ngmix" means calling ngmix directly in addition to these. | A |
| 8 | §3.3 / §3.4 moments as a diagnostic | The shape method is a **process-wide global**. It also changes PixelGrid/GSObject initialisation and Select cuts, and `piff.process`/`piffify` reset it from the config (default `hsm`). Our moments module should call `Star.measure_*` / ngmix directly and never flip the global as a side effect. | A |
| 9 | §3.2 storage: one FITS per observation (HDU per SCA) + one catalog per observation with an `sca` column; §3.4 file-based mode via Piff's input | Piff `Files` input pairs **one catalog with each image and does not filter by chip**. With x/y columns every star would be placed on every SCA. Workable options: per-SCA catalog HDUs (`cat_hdu` list), per-(obs, SCA) catalog files, or ra/dec columns with a real WCS (fails for WCS mode "none"). Also, `chipnum` must be unique per image in one Piff run, so a multi-observation fit needs chipnum ≠ SCA and an explicit `sca` property. (to confirm) | A |
| 10 | §3.4 "ρ and τ statistics via TreeCorr" | Piff has `RhoStats` (ρ1–ρ5) only; **no τ statistics** (they need a galaxy catalogue). τ must be ours. ρ uses star properties `u, v` in arcsec and default `min_sep=0.5`, `max_sep=300` arcmin. One SCA spans ~7.5′, so the binning needs care. | B |
| 11 | §6.4 "retire local `roman_psf.py` patch"; §2 lists `reference_model.py`, patched `roman_psf.py` in `prev_code/` | The patched `roman_psf.py` is **not in `prev_code/`**. `reference_model.py`, `reference_optimum.py`, `test_pg_manystars.py` (with `preflux`) and `pixelgrid_viz.py` exist **only as notebook cells** (52–54, 103). `PixelGridOversampled` and `io_utils.py` are referenced but absent. `compare.py`/`run_first_test.py` pass `apply_mtf`, so they will not run against upstream Piff. | C |
| 12 | §6.2 reflux after initialisation; §9.1 | Upstream `PSF.fit` now runs one `reflux_stars` pass after `initialize_params` for **every** PSF type. This is the same operation as `preflux()` (which patched only Sum). Differences: reflux χ² is now recomputed from a real redraw, and reflux renders in double precision. The §9.1 "PG default" baseline (χ² 39542, flux_r 0.916) cannot be reproduced on upstream without patching the step out. | B |
| 13 | §10 "Piff's `Gaussian` (scale, g) fails near zero size (g unconstrained, \|g\|>1 crash)" | Now `scale ≤ 0` or \|g\| ≥ 1 returns 1e300 residuals (no crash). A start with scale² < pixel area is reset to one pixel each fit. The hard wall near zero size remains, so the case for the signed kernel stands, but the wording should change. | B |
| 14 | §4 "our Model subclasses register on import" | `Model.__init_subclass__` **raises** if `_type_name` is already registered, e.g. on a notebook re-run or reload. Needs a guard, or definitions kept in an importable module only (never in notebook cells). | B |
| 15 | §3.2 "expose `pupil_bin`" | In Piff, `pupil_bin` is a **module global** (`piff.roman.roman_psf.pupil_bin = 4`), not a model parameter. Matching truth and fit `pupil_bin` therefore needs setting that global. §4 allows monkeypatches "for diagnostics only", so we need to decide whether a context-managed global setting is acceptable for runs. (to confirm) | A |
| 16 | §3.4 "optics free/pinned" | RomanOptics' prior pulls toward **zero** (σ default 0.05). "Pinned" via a tiny prior σ pins at 0, not at truth. That is fine for zero-aberration truth, but pinning at a nonzero truth needs a different mechanism (e.g. `set_sca_solution` without fitting, as `prev_code` `make_truth_psf` does). `chromatic` defaults to **True** in RomanOptics; our configs must set it explicitly. | B |
| 17 | §5.5 Piff χ²/dof "pixels − 3 per star" | True for reflux with a free centre and for RomanOptics. With `trust_pos` (or all components non-centred) reflux uses N − 1. PixelGrid's own per-star `chisq` dof is N. `psf.dof` is the sum of final-reflux dofs over non-reserve, unflagged stars. | B |
| 18 | §12 "never present a fit as converged when it hit max_iter" | Piff has **no convergence flag**: only a log warning, and `niter == max_iter` is ambiguous. We need our own check. Also, `prev_code` `GaussianMoments` ignores `max_nfev` hits, and `run_first_test` prints "PASSED" unconditionally. | A |
| 19 | §12 "Verify GalSim/ngmix/TreeCorr internals by reading source"; Phase 0 study list (galsim.roman source) | Contradicts the "Scope" section: only read inside the project folder, and use your knowledge or online docs for external packages. Needs a rule: e.g. allow reading the installed GalSim/ngmix/TreeCorr sources read-only, or rely on docs plus tests that pin each assumption. | C |
| 20 | "Shell commands": use Read/Grep tools | No Grep tool is available in this session; searches were done with `grep` via Bash (relative paths, no `cd` after the first command). The file is named `CLAUDE.MD` (upper-case extension) in the project root, outside the `psf-fomo` repo. On a case-sensitive filesystem it may not be auto-loaded as `CLAUDE.md`. | C |
| 21 | Units of Zernike errors (§3.2, §9.5 "per wave") | `prev_code/simulate.py` documents `aberration_amplitude` as "radians, roughly", while a notebook cell says waves. Piff passes params straight to `getPSF(extra_aberrations=...)` (GalSim units; verify: waves). Docs must state waves once verified. | B |

## Resolutions (user, 2026-11-02)

| # | Decision |
|---|---|
| 1 | Fixed `GaussianMixture` MTF is fine. |
| 2, 3 | Keep rendering truth with Piff's `RomanOpticalModel` as `prev_code/simulate.py` did. An exact/Legendre-9 mode may be added later. |
| 7, 8 | The moments module may call Piff's existing methods directly; don't reimplement them. |
| 9 | One catalog file per observation (HDU 1: full table with `sca` column). File-based Piff runs: with the Roman WCS use `ra_col`/`dec_col` (as `Piff/devel/roman/piff.yaml`); with the simple WCS also write per-SCA subset table HDUs in the same file and set `cat_hdu: [...]`. In-memory runs build stars per SCA directly. |
| 10 | ρ via Piff's `RhoStats`; τ and TreeCorr are deferred (not urgent). |
| 11 | Ignore the old patched `roman_psf.py`; use upstream. Read reference model / preflux from notebook cells. |
| 12 | Use Piff's built-in reflux after initialisation; no changes. |
| 15 | Use Piff's default `pupil_bin` (module global, 4). |
| 18 | Use Piff's convergence behaviour as is (no Piff changes). |
| 19 | No need to read GalSim/ngmix/TreeCorr sources; use Piff's usage, prior knowledge, online docs. |
| 20 | `CLAUDE.MD` renamed to `CLAUDE.md`. |

Additional useful facts (no conflict):
- Piff's `test_roman_diffusion` uses jitter as `Simple(Gaussian, init=delta, interp=Mean)` inside
  the Convolve: an upstream precedent for a jitter-like kernel component.
- Reserve stars: `select.reserve_frac` (+`seed`), applied per chip, rounded down; in memory set
  `properties['is_reserve']`. They are refluxed and kept in `psf.stars` (separate them in
  diagnostics).
- Poisson weight updates happen only if stars carry a `gain` property.

## §6 checklist — status after reading (nothing executed yet)

| # | Item | Status from source | Still to do (Phase 2/4 tests) |
|---|---|---|---|
| 1 | MTF consistency | Piff's own test uses our exact 3-Gaussian constants (arcsec = rel σ · 0.3279 · 0.11). Method `'auto'`, so the pixel is applied once by Convolve. Fixed only (conflict 1). | Profile/moment agreement truth-kernel vs `GaussianMixture`. Closed loop `mtf_scale=1` vs old results (affected by conflict 2). `mtf_scale=1.1` → +0.022 pix²/axis must be absorbed by another component. |
| 2 | Reflux after init | Present upstream for all PSF types (psf.py:423). Equivalent to `preflux()` for Sum. | Re-run stamp-17 flux test without the patch, and compare with preflux numbers in §9.1. |
| 3 | Moments | Piff methods = hsm / ngmix gauss / admom / gaussmom(wmom). Not tested upstream on undersampled images. | Compare on identical images with ours and ngmix direct (Phase 3). |
| 4 | Retire `roman_psf.py` patch | Patch not present on disk; upstream has no `apply_mtf`. | Port configs to `Convolve(RomanOptics, Simple(GaussianMixture, Fixed))` once (1) passes. |
| 5 | API drift | `set_context(wcs, pointing=None, bandpass=None)` ✔. `fit(stars, wcs=None, pointing=None, logger=None, convert_funcs=None, draw_method=None)` with wcs/pointing deprecated ✔. `RomanSCAInterp._transformed_params` exists (class default False, set in `solve`) ✔. Centred PixelGrid n²−2 columns via `HouseholderTransform` ✔. `StarFit.newParams(params, params_var=None, num=None, **kwargs)` ✔ (`piff.StarFit`, no `piff.star_fit` module). Sum/Convolve `_num` via recursive `set_num`/`num_components` ✔. | One test per item pinning these. |
