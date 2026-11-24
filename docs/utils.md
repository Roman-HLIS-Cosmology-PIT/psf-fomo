# Helpers for real data (`psf_fomo.utils`)

Preparing real Roman exposures for `psf_fomo.piff_datafit`: convert the calibrated ASDF files
to FITS, detect the sources, and look at the result. The ASDF converter is adapted from a
script by Mike Jarvis.

Install the extra dependencies (asdf, asdf-astropy, roman_datamodels, gwcs, sep, astroquery,
fsspec, aiohttp, s3fs) with

```
pip install ".[utils]"
```

They are imported only when a function needs them, so `psf_fomo` works without them.

## Quick start

```python
from pathlib import Path
from psf_fomo.utils import convert_to_dir, detect_sources, plot_detections, plot_detection_sizes
from psf_fomo.piff_datafit import L2Image, StarCatalog

asdf_file = "r0001_wfi01_f158_cal.asdf"
fits_file = convert_to_dir(asdf_file, "fits_out", uniform_weights=True)

catalog = detect_sources(fits_file, metadata=asdf_file)            # every detection
stars = detect_sources(fits_file, gaia=True, metadata=asdf_file)   # Gaia matches only
plot_detections(fits_file, catalog, stars)
plot_detection_sizes(catalog, stars)

image = L2Image(fits_file)
cat = StarCatalog(stars)            # or StarCatalog(catalog)
```

Several files:

```python
for path in sorted(Path("asdf").glob("*_cal.asdf")):
    convert_to_dir(path, "fits_out", uniform_weights=True)
```

## ASDF → FITS: `convert_asdf`, `convert_to_dir`

`convert_asdf(source, output, ...)` writes one file; `convert_to_dir(source, output_dir,
name=None, ...)` names it after the input (`*_cal.asdf` → `*_cal.fits`). `source` is a
local path, a URL (`http://`, `https://`, `s3://`), an open `asdf.AsdfFile` or an open
`roman_datamodels` model.

| option | default | what |
|---|---|---|
| `include_wcs` | True | TAN-SIP FITS WCS from the ASDF GWCS, checked to 0.01 pixel (`WCSERR`) |
| `include_gain` | False | `GAIN` HDU (on with `background_weights`) |
| `background_weights` | False | `IVAR` without the sources' Poisson noise, `SKYBKG` (needs sep; on with `uniform_weights`) |
| `uniform_weights` | False | `MASK_UNIF`, `IVAR_UNIF`, `GAIN_UNIF` for `SCI_UNIF` (needs `DUMO`) |
| `mesh_size` | 128 | tile size [pixels] of the uniform noise calibration |
| `overwrite` | False | replace an existing output |
| `download_dir` | None | keep downloads here (reused later); None = temporary folder, deleted |
| `verbose` | True | print the WCS residual and noise summaries |

`L2Image`'s default variant (`'unif'`) needs `uniform_weights=True`.

Remote files: set `MAST_API_TOKEN` for proprietary MAST data; `s3://` access is anonymous
unless AWS credentials are configured.

### Output HDUs

| HDU | content |
|---|---|
| `PRIMARY` | `TELESCOP`, `INSTRUME`, `DETECTOR`, `FILTER`, `EXPTIME`, `EFFEXPTM`, `ORIGFILE`; pointing `RA_V1`, `DEC_V1`, `PA_V3` (+ `TARG_RA`, `TARG_DEC`, `PA_APER`), degrees |
| `SCI` | science image (ASDF `data`, optimal ramp slope, DN/s) with the WCS |
| `DQ` | data-quality flags |
| `ERR` | σ of `SCI` (read + Poisson of sources and background) |
| `MASK` | 1 = masked: DQ bits other than JUMP_DET (4) and UNRELIABLE_SLOPE (16777216), invalid values |
| `IVAR` | 1/`ERR`² by default; 1/(read variance + `SKYBKG`/`GAIN`) with `background_weights` |
| `VAR_POISSON`, `VAR_RNOISE`, `VAR_FLAT`, `CHISQ`, `DUMO` | copied when present |
| `GAIN` | `SCI`/`VAR_POISSON`, effective gain per pixel |
| `SCI_UNIF` | `SCI` + `DUMO`, the uniform-weight slope (when `DUMO` exists) |
| `MASK_UNIF` | `MASK` + isolated JUMP_DET pixels with a large `DUMO` (most JUMP_DET pixels, e.g. star cores under poor guiding, stay unmasked) |
| `IVAR_UNIF` | 1/(uniform-ramp read variance + `SKYBKG`/`GAIN_UNIF`), background only |
| `GAIN_UNIF` | effective gain of `SCI_UNIF` |

`SKYBKG` (with `background_weights`) is one SEP background level per detector in `SCI` units,
written to every header. The uniform read variance and gain are inferred from the optimal
fit's variance planes on clean background pixels, as medians per `mesh_size` tile,
interpolated across the image (details in the docstring of
`psf_fomo.utils.convert._estimate_uniform_noise`).

## Detection: `detect_sources`

`detect_sources(fits_file, gaia=False, gaia_catalog=None, metadata=None, obs_time=None,
match_radius_arcsec=0.11, detection_config=None, background_config=None, kron_factor=2.5,
min_radius=1.75, gaia_margin_arcsec=10.0, verbose=True)`

- Runs SEP on `SCI_UNIF` − `SKYBKG`, with `MASK_UNIF` and the noise 1/`IVAR_UNIF`
  (`DEFAULT_DETECTION_CONFIG`: threshold 3σ, `minarea` 3, `deblend_nthresh` 64,
  `deblend_cont` 0.001). The sky is re-derived from `SCI` as a check (warning only).
- Measures the masked AUTO (Kron) flux, `kron_factor` × Kron radius, with a circular
  aperture of `min_radius` pixels for small or invalid Kron apertures, and the half-light
  radius `r50`.
- With `metadata` (the original ASDF file or model) adds `flux_auto_jy`, `mag_auto_ab` and
  the header `ABZP`.
- With `gaia=True` returns only detections matched to Gaia (mutual nearest neighbors within
  `match_radius_arcsec`). The Gaia catalog is `gaia_catalog` (file or Table) or queried from
  the archive over the image footprint; positions are moved with their proper motion to the
  observation time (`obs_time`, else `metadata`, else `DATE-OBS`/`MJD-OBS`).

Main columns: `id`, `x`, `y` (zero-based pixels), `ra_deg`, `dec_deg`, `flux_auto` [DN/s],
`flux_auto_err_bkg` and `snr_auto_bkg` (background and read noise only), `r50` [pix],
`extract_flag` (SEP's flag), `kron_flag`, `auto_flag`, `r50_flag`, `auto_masked_fraction`,
`clean` (every flag 0, no masked pixel in the aperture, valid flux and radius). Gaia columns:
`gaia_match`, `gaia_sep_arcsec`, `gaia_n_candidates`, `gaia_ambiguous`, `gaia_pm_applied`,
`gaia_<column>`, `clean_gaia_star` (`clean`, matched, not ambiguous). A Gaia match is not a
PSF-star selection: no magnitude, saturation or RUWE cuts.

The steps are also available separately: `load_exposure`, `measure_sources`,
`read_metadata`, `add_ab_magnitudes`, `read_gaia`, `query_gaia`, `match_gaia`
(`psf_fomo.utils.detect`).

## Plots

- `plot_detections(fits_file, catalog=None, stars=None, region=None, apertures=None, ax=None,
  title=None, show_mask=True)`: the sky-subtracted `SCI_UNIF` with detections (blue; AUTO
  ellipses when the region is ≤ 1000 pixels wide) and Gaia matches (orange); masked pixels in
  green. `region = (xmin, xmax, ymin, ymax)`, zero-based pixels.
- `plot_detection_sizes(catalog=None, stars=None, ax=None, title=None, ylim=(0, 10))`: `r50`
  vs AUTO magnitude (AB if available, else instrumental), Gaia matches filled when
  `clean_gaia_star`. For moments measured on stamps, use `piff_datafit.plot_size_mag`.
