# Changelog

## 0.1.0 (unreleased)

First version shared with others.

- Simulation module (`psf_fomo.sim`): Roman star images with controlled PSF errors, Gaussian
  or Roman noise, simple or Roman WCS, save/load, YAML configuration.
- Moments module (`psf_fomo.moments`): Piff's moment algorithms on native, noiseless and
  oversampled images; comparisons between simulations.
- Piff validation (`psf_fomo.piffval`): model builder (`FitConfig`), in-memory and
  file-based fits, diagnostics, ρ statistics, plots.
- Reference solver (`psf_fomo.reference`): joint χ² optimum of the fitted model.
- Real-data fits (`psf_fomo.datafit`): L2 images, detection catalogs, star selection, Piff
  fits, reference optimum; synthetic L2 files from simulations
  (`psf_fomo.datafit.synthetic`).
- Piff pinned to `main` at `280064f`.
- Kernel models: signed `GaussianMoments` (ours) and Piff's `GaussianBlur` (kind `'blur'`).
  The former broadening-only `GaussianMoments(signed=False)` (kind `'positive'`) was removed;
  use `'blur'`.
- Star selection cuts for simulation fits (`PiffValidation.set_cuts`), the same as for real
  data and built by one function (`piffval.fitting.make_cuts`). Default `min_sep` = 2.5″ for
  both (simulations: per observation); rejected stars in `FitResult.rejected`.
- Chromatic by default: simulations draw chromatic PSFs with a flat SED
  (`set_chromatic('flat' | file | galsim.SED | None)`); `FitConfig` leaves RomanOptics'
  `chromatic` at Piff's default (True) unless set (`set_optics(chromatic=...)`); real-data
  fits assume a flat SED for every star (`DataFit.set_sed`). Saved chromatic simulations
  include `sed.fits` for Piff's file input.
- Truth optical PSF modes, `set_psf_model(spatial_interpolation=...)`: `'fast'` (default;
  Piff's corner blending with each corner summed over the filter once per SCA and
  observation), `'piff'` (Piff's model, drawn as Piff draws it) and `'exact'`
  (`galsim.roman.getPSF` at each star). The sum over the filter in `'fast'` and `'exact'`
  uses `n_waves` wavelengths (10 by default; None = every 5 nm). Configurations
  saved without `spatial_interpolation` load as `'piff'`.
- `DataFit.set_attitude` is off by default (experimental; needs unreleased GalSim/Piff
  changes, see `attitude_support()`).
