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
- Roman noise kinds `set_noise('roman-L1')` (every `galsim.roman` effect, uncorrected) and
  `set_noise('roman-L2')` (without nonlinearity, IPC and quantization, which L1 → L2
  processing removes); the earlier `'roman'` is refused for new settings and loads as
  `'roman-L1'` from saved configurations.
- Chromatic by default: simulations draw chromatic PSFs with a flat SED
  (`set_chromatic('flat' | file | galsim.SED | None)`); `FitConfig` fits a monochromatic
  RomanOptics by default (`set_optics(chromatic=False)`); `chromatic=True` is allowed only
  for RomanOptics alone (Piff draws Convolve and Sum models without the stars' SEDs) and
  `piff_config` refuses it otherwise; real-data fits with a chromatic model assume a flat SED
  for every star (`DataFit.set_sed`). Saved chromatic simulations
  include `sed.fits` for Piff's file input.
- Truth optical PSF modes, `set_psf_model(spatial_interpolation=...)`: `'fast'` (default;
  Piff's corner blending with each corner summed over the filter once per SCA and
  observation), `'piff'` (Piff's model, drawn as Piff draws it) and `'exact'`
  (`galsim.roman.getPSF` at each star). The sum over the filter in `'fast'` and `'exact'`
  uses `n_waves` wavelengths (10 by default; None = every 5 nm). Configurations
  saved without `spatial_interpolation` load as `'piff'`.
- Measured sizes are in arcsec: `T` (and every `T_*` column) in arcsec², `sigma` in arcsec,
  in the moments, Piff-validation and real-data tables (earlier: native pix² and pix).
  Moment tables saved before are converted when loaded. Kernel parameters stay in pix².
- Component images include the native pixel response by default
  (`component_image(..., include_pixel=True)`, also `plot_components`); new
  `plot_component_images()` on fit results: every model component alone at one position,
  native and oversampled.
- Roman science frame everywhere: needs a GalSim with its science-frame update (GalSim
  `main`), whose `galsim.roman` WCS and PSFs are in the frame of L2 files. `L2Image` has no
  `frame` option and fits files as they are (a working copy only to clean non-finite pixels,
  `image_cleaned.fits`); removed `flip_axis`, `flip_wcs_header`, `compare_frames`,
  `to_science_frame`, `simulation_to_l2(frame=...)` and the `x_l2`, `y_l2` columns. Fits
  saved on the earlier flipped working copy are refused by `load_data_fit`.
- Plot colors are named matplotlib colors; American spelling throughout.
- `DataFit.set_attitude` is off by default (experimental; needs unreleased GalSim/Piff
  changes, see `attitude_support()`).
