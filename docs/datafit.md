# Real-data fits (`psf_fomo.datafit`)

Piff fits of real Roman L2 images, kept separate from the simulation validation
(`psf_fomo.piffval`). The two share only truth-free functions: `FitConfig` (the model),
`piffval.fitting` (logger, convergence flag, shape method, reserve, per-star fit columns, fit
summary), `piffval.residuals` (`'data'` and `'model_os'` moments, `model_moments`,
`compare_fits`), and the reference solver.

**Status:** `L2Image`, `StarCatalog`, `DataFit` → `DataFitResult`, reference optimum with
fitted centres. Walk-through: `notebooks/real_data.ipynb` in the project folder (closed loop on
a simulated SCA in the L2 layout, then a real image). Deferred: Gaia cross-match, own
detection, per-star local sky, several exposures per fit.

## Quick start

```python
from psf_fomo.piffval import FitConfig, compare_fits
from psf_fomo.datafit import L2Image, StarCatalog, DataFit, load_data_fit

img = L2Image("exp_wfi05.fits")                  # SCI_UNIF, model Poisson weights, SKYBKG
cat = StarCatalog("exp_wfi05_det.fits")          # SEP catalog, 0-based x, y
cat.select_stars("sizemag")                      # Piff SizeMag (gauss); or "all", "column"
cat.set_cuts(clean_only=True, min_snr=50, min_sep=2.0, max_mask_pixels=0)

cfg = FitConfig(); cfg.set_optics("free"); cfg.add_kernel("signed"); cfg.add_pixelgrid(5)
fit = DataFit(img, cat, cfg, name="signed+PG")
fit.set_reserve(0.2, seed=3)
fit.set_diagnostics(native=("wmom", "gauss"), oversampled=("hsm",))
res = fit.run(save_to="out/real01")

res.fit_summary          # chisq_dof, niter, converged, n_catalog, n_objects, n_candidates, ...
res.checks               # weights_ratio, orientation, nonfinite (total, unmasked)
res.summary()            # flux_cat_r, dx, dy, chisq_dof, sky_resid: fit vs reserve
res.table                # one row per star
res.moments              # 'data' (data and model, native) and 'model_os' rows
res.model_moments(n_grid=5)                      # oversampled model moments across the SCA
res.components()         # fitted Zernikes, kernel M, PixelGrid (no truth columns)
res.plot_size_mag(); res.plot_mean_images(); res.plot_chi2_radius()
res.plot_field("sky_resid"); res.plot_field("T_model", "model_os", "hsm")
res.plot_moment_residuals(); res.plot_components("kernel"); res.plot_zernikes(); res.rho()
compare_fits({"Piff PG": res_pg, "signed+PG": res})  # model − model at the same stars
same = load_data_fit("out/real01")
```

## Input files

L2 images converted from ASDF by the user's converter, one exposure of one SCA per file:

| HDU | content |
|---|---|
| `PRIMARY` | metadata: detector, filter, exposure time; pointing `RA_V1`, `DEC_V1`, `PA_V3` (+ `TARG_RA`, `TARG_DEC`, `PA_APER`), degrees, if the converter copies ASDF `meta.pointing` |
| `SCI` | optimal-weight slope, DN/s, TAN-SIP WCS (+ `WCSERR`) |
| `DQ`, `ERR` | copied from the ASDF |
| `MASK` | 1 = bad |
| `IVAR` | 1/`ERR`² by default; 1/(`VAR_RNOISE` + `SKYBKG`/`GAIN`) with `--background-weights` |
| `VAR_POISSON`, `VAR_RNOISE`, `VAR_FLAT`, `CHISQ`, `DUMO` | if present in the ASDF |
| `GAIN` | `SCI / VAR_POISSON` (electrons per DN/s) |
| `SCI_UNIF` | `SCI` + `DUMO` (uniform-weight slope), whenever `DUMO` exists |
| `MASK_UNIF`, `IVAR_UNIF`, `GAIN_UNIF` | with `--uniform`; `IVAR_UNIF` = 1/(read_unif + `SKYBKG`/`GAIN_UNIF`) |

`SKYBKG` (every header): one sky level per detector, DN/s per pixel, measured with SEP on
`SCI`. The images are not background-subtracted.

## `L2Image`

```python
from psf_fomo.datafit import L2Image

img = L2Image("exp_wfi05.fits")       # variant="unif"; sca / filter_name from the header
img.check_orientation()                # pixel frame vs galsim.roman (parity, pixel scale)
img.check_weights()                    # standard variant + model_poisson only
img.check_finite()                     # NaN pixels: total (cleaned for fits), unmasked (should be 0)
img.set_weights("model_poisson")       # default; or "ivar"
img.set_sky()                          # default: SKYBKG; set_sky(0) subtracts nothing
img.gain_at(x, y, stamp_size=25)       # median gain in the stamp, e⁻ per DN/s
img.piff_input()                       # image part of Piff's input field
img.pointing, img.attitude             # pointing from the header (degrees); attitude or None
```

- **Variant**: `'unif'` (default) uses `SCI_UNIF`, `MASK_UNIF`, `IVAR_UNIF`, `GAIN_UNIF`
  (missing without the converter's `--uniform` → error); `'standard'` uses `SCI`, `MASK`,
  `IVAR`, `GAIN`. The planes of the two are never mixed.
- **Header**: SCA from `DETECTOR` (e.g. `'WFI05'`), else `SCA` / `SCA_NUM`; filter from
  `FILTER`, else `OPT_ELEM`; exposure time from `EXPTIME` / `EFFEXPTM` / `EXPOSURE_TIME`
  (recorded only). Both SCA and filter can be passed instead (`sca=`, `filter_name=`). Roman F
  names are converted to GalSim names (`F158` → `H158`).
- **Weights**: `'model_poisson'` (default): the weight plane holds read + background noise,
  and Piff adds each star's Poisson variance from the model at every iteration (Piff's
  `gain`, one value per star from `gain_at`). `IVAR_UNIF` is always of that form. With the
  standard variant, `check_weights` verifies that `IVAR` is the `--background-weights` form
  on pixels > 10σ above the sky; a data-based `IVAR` (1/`ERR`²) would make Piff count the
  stars' Poisson noise twice. `'ivar'`: the plane as it is, no Poisson term added.
- **Sky**: `SKYBKG` of the image HDU, passed to Piff as a number (Piff's header-key option
  reads only the image HDU header and is deprecated). It is measured on `SCI`; the fit uses
  `SCI_UNIF` by default, so any offset between the two shows up in the per-star sky residual
  (step 2).
- **Orientation**: Piff's `RomanOptics` evaluates GalSim's field-dependent aberrations at the
  star's pixel position, so fits must run in `galsim.roman`'s SCA frame (see *Pixel frames*
  below). `check_orientation` compares the fit frame's WCS Jacobian at the image centre with
  `galsim.roman.getWCS` for the SCA: a mirrored axis (parity) or a pixel scale off by > 5 % is
  an error (`strict=False` warns instead; `DataFit` is strict only for models with
  RomanOptics, the one component that depends on the pixel frame) (the local
  scale varies by a few % across the focal plane through field distortion, e.g. ≈ 0.107″ at a
  corner of SCA 5, so it is a sanity check for a wrong WCS, not a precision test). With
  `pa_deg` it also reports the rotation; a pure 180° rotation of the pixel frame is visible
  only that way.
- **Coordinates**: GalSim / FITS, pixel centres at 1 … 4088 (the detection catalog's 0-based
  x, y are shifted by +1 when the Piff catalog is written).
- **Pointing**: `img.pointing` holds the PRIMARY keys `RA_V1`, `DEC_V1`, `PA_V3`, `TARG_RA`,
  `TARG_DEC`, `PA_APER` (degrees) under the ASDF names (`ra_v1`, ..., `pa_aperture`). For
  files converted without them, pass the ASDF dict: `L2Image(path, pointing=meta_pointing)`
  (its values replace the header's; other keys are ignored). `img.attitude` =
  `{'boresight_ra', 'boresight_dec', 'telescope_pa'}` from `ra_v1`, `dec_v1`, `pa_v3`, or
  None; `DataFit` uses it to orient RomanOptics (*Telescope attitude*, below). The pointing
  is saved with the image config.

## Pixel frames: Roman science frame vs GalSim

Real L2 files are in Roman's **science frame** (RDox "Coordinate Systems": each detector's
readout frame flipped along one axis so that all SCAs look alike). `galsim.roman`, and so
Piff's `RomanOptics`, uses a frame **mirrored** relative to it. Measured on real files by
comparing the file's WCS with `galsim.roman.getWCS` at the exposure's own pointing
(`compare_frames`): SCA 11 needs a y flip (rms 1.0″, extra rotation 0.07°), SCA 3 an x flip
(1.3″, 0.22°); the RDox figure shows the same pattern for all SCAs:

| SCAs | L2 → GalSim |
|---|---|
| 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17 | y flip: y_GalSim = 4089 − y_L2 (measured: 11) |
| 3, 6, 9, 12, 15, 18 | x flip: x_GalSim = 4089 − x_L2 (measured: 3) |

- `L2Image(path, frame="science")` (default, real L2 files): fits run on a **working copy**
  flipped into GalSim's frame (`write_fit_copy`, written by `DataFit.run` as
  `image_galsim_frame.fits` in the fit folder). **The original file is never changed.**
- The working copy's WCS header is transformed **algebraically**, not refitted: on the flipped
  axis CRPIX → n + 1 − CRPIX, that column of the CD (or PC / CDELT) matrix changes sign, and
  each SIP coefficient (A, B, AP, BP) gets the sign the flip implies (`flip_wcs_header`).
  Every flipped pixel keeps its sky position (tested to 1e-6″).
- `frame="galsim"`: the file is already in GalSim's frame (e.g. simulations); nothing is
  flipped.
- **Non-finite pixels** are cleaned in the working copy: where image or weight is NaN/inf,
  image = 0, weight = 0, mask = 1. Piff zero-weights masked pixels, but its sums still multiply
  weight × image and 0 × NaN = NaN: one NaN in a stamp made a star's flux and centre NaN and
  Piff excluded it (first real file: 1036 NaN pixels, all masked). For `frame="galsim"` a
  copy is written only when such pixels exist. `check_finite()` and `res.checks["nonfinite"]`
  give `total` (cleaned) and `unmasked` (should be 0; a warning otherwise).
- **Stars Piff cannot fit** (it catches the error, excludes and flags the star) get NaN `chisq`,
  `dof`, `chisq_dof` in the star table, moment rows with NaN values and flag 8 (+16), and are
  left out of the reference solver; summaries skip them (`flag` 1).
- **Catalog positions** are flipped the same way for the fit; the star and selection tables
  keep `x_l2`, `y_l2` (positions in your file) next to `x`, `y` (fit frame).
- **Measurements** (`StarCatalog.measure`) use the original file in its own frame: moments
  are in world axes, so the frame does not matter there, and x, y are those of your file.
- **Fits without RomanOptics** (Gaussian, PixelGrid, our kernels with no optics component) do
  not depend on the pixel frame; they also run on the working copy, for uniformity.
- **Check a file**: `img.compare_frames(pointing)` with the ASDF `meta.pointing` as a dict
  (`ra_v1`, `dec_v1`, `pa_v3`, `pa_aperture`, `target_ra`, `target_dec`, degrees; default:
  `img.pointing`, which then needs all six) returns a
  table per pointing interpretation × candidate mapping. The GalSim convention that matched
  on SCAs 3 and 11 is `world_pos` = target, `PA` = `pa_v3` (`PA_is_FPA=False`), with SCA
  centres within ≈ 1″. Read it in three steps: the matching interpretation has a small
  `centre_offset`; `rms_rotation_free` (PA-independent) is ≈ 1″ only for the right parity;
  in the matching interpretation the right mapping has `rms` ≈ `rms_rotation_free` and
  `extra_rotation` ≈ 0°. Needs the full 4088 × 4088 image.

## Detection catalog and star selection (`StarCatalog`)

The user's SEP catalog (FITS table, HDU 1): `id` (1…N), `x`, `y` (0-based, numpy convention),
`ra_deg`, `dec_deg`, `flux_auto` [DN/s], `flux_auto_err_bkg`, `snr_auto_bkg`, `r50` [pix],
shape columns, flags, boolean `clean`; header `SKY`, `DETCFG`, `MATCHRAD`, `OBSTIME`,
`NMATCH`, `ABZP` (recorded).

`StarCatalog(catalog, hdu=1, id_col="id", x_col="x", y_col="y", zero_based=True,
flux_col="flux_auto", size_col="r50", clean_col="clean", extra_cols=())`.

- **`catalog`** is a FITS table file (the SEP command-line output) or the astropy `Table`
  itself, e.g. `StarCatalog(detect_sources(...))`. A table is copied; its header values
  (`ABZP`, `SKY`, `OBSTIME`, ...) come from `table.meta` (keys matched in any case). A fit
  from a table saves it as `detections.fits` in the output folder and records that path in
  the run config, so `StarCatalog.from_config(res.config["catalog"])` rebuilds it; before a
  fit, `from_config(config, table=...)` needs the table.

- **All detections go to Piff** as input objects, so `min_sep` sees every neighbour
  (galaxies and unclean objects included). Piff skips objects whose stamp is not fully on the
  image or is fully masked.
- **Star selection** `select_stars(method, shape_method="gauss", ...)`:
  - `"sizemag"` (default): Piff's `SizeMag` (stellar locus in size vs magnitude); options
    `impurity`, `fit_order`, `num_iter`, `initial_select` are passed to Piff. It measures every
    object with Piff's global shape method, `gauss` by default here (HSM fails on most native
    stamps). The fit has its own shape method (`FitConfig.set_piff_shape_method`).
  - `"all"`: every detection.
  - `"column"`: a flag column (`column=`, `use_flag=` / `skip_flag=` as Piff's `Flag`).
- **Choosing `min_sep`**: a neighbour inside the stamp puts light no single-star model can
  fit (χ²/dof rises; CLAUDE.md §10). A square stamp of `stamp_size` pixels reaches
  `stamp_size/√2` pixels from its centre at the corners (25 → 17.7 pix ≈ 1.9″), so use
  `min_sep` ≳ that plus a few PSF widths, ≈ 2.5″ for 25-pixel stamps (bright neighbours need
  more).
- **Cuts** `set_cuts(clean_only=True, min_snr, min_sep [arcsec], max_mask_pixels,
  max_edge_frac, stamp_center_size, max_pixel_cut, reject_where, nstars,
  max_snr_weight=0)`: Piff's rejection step, applied after the selection. `clean_only` adds
  `clean == 0` to Piff's `reject_where`; `reject_where` can use any column listed in
  `extra_cols`. `max_snr_weight` keeps Piff's default (100) unless set: every star with
  S/N > 100 has its variance inflated to S/N = 100, so bright stars do not dominate the fit,
  and χ²/dof falls below 1 for them. `set_cuts(max_snr_weight=0)` turns the cap off (as the
  simulation validation does), so that χ²/dof ≈ 1 means a model at the noise level.
- **Piff catalog** (`piff_table`, written to `catalog.fits`): `x`, `y` (+1, 1-based), `id`,
  `flux_cat`, `size_cat`, `clean` (0/1), `mag` (`ABZP` − 2.5 log10 `flux_cat`), the selection
  column, `extra_cols`, and with model Poisson weights `gain` (median `GAIN_UNIF` in the stamp;
  the image median where undefined).

## Before fitting: moments of every detection (`StarCatalog.measure`)

Size–magnitude diagrams with any moment algorithm, and a preview of Piff's star selection,
without running a fit:

```python
from psf_fomo.datafit import L2Image, StarCatalog, plot_size_mag

img = L2Image("exp_wfi05.fits")
cat = StarCatalog("exp_wfi05_det.fits")
cat.select_stars("sizemag"); cat.set_cuts(min_snr=50, min_sep=2.5)
t = cat.measure(img, algorithms=("gauss", "wmom"), wmom_fwhm=(0.4, 0.6),
                preview_selection=True)            # mag_range=(18, 22), max_objects=5000
plot_size_mag(t, algorithm="gauss")                # size T [pix²] vs catalog magnitude
plot_size_mag(t, algorithm="wmom_0.4", mag="measured", size="sigma")
plot_size_mag(t)                                   # the catalog's own size (r50)
```

- **Stamps** are the ones a fit would use: Piff's Input cuts them (stamp size, sky
  subtraction, weights, bad pixels) from a temporary copy of the Piff catalog. Detections
  without a full stamp on the image (or fully masked) are left out (`t.meta["n_catalog"]`,
  `t.meta["n_measured"]`).
- **Table** (one row per measured detection): `id`, `x`, `y` (1-based), `flux_cat` [DN/s],
  `mag`, `size_cat` [pix], `clean`, and per algorithm label `<a>` (`gauss`, `wmom_0.4`, ...):
  `T_<a>` [pix², = Mxx + Myy], `sigma_<a>` [pix, det(M)^¼], `e1_<a>`, `e2_<a>` (distortion,
  world axes), `flux_<a>` [DN/s], `mag_<a>` (`ABZP` − 2.5 log10 flux; instrumental without
  `ABZP`), `flag_<a>` (1 failure, 2 negative flux, 4 centroid > 1″, 8 exception). Measured
  about the catalog position, conventions of `psf_fomo.moments`.
- **Algorithms on native stamps**: `gauss` (elliptical Gaussian fit) gives a tight stellar
  locus; `wmom` reports raw moments with a fixed round weight, so all sizes are compressed
  toward the weight's (stars and galaxies still separate); `hsm` / `am` fail on most native
  stamps.
- **`preview_selection=True`** runs the catalog's Piff selection (selection type, then the
  cuts; no reserve, no fit) and adds `candidate` and `star`: the same choice a `DataFit` with
  this catalog makes. With `mag_range` / `max_objects` the selection sees only that subset
  (fewer neighbours for `min_sep`, a different locus for SizeMag). Each stage is reported as
  it is: if the cuts reject every candidate, `candidate` still shows them (with a warning;
  a fit would stop there), and `t.meta` holds `n_candidates`, `n_stars`, the selection and
  the cuts. `algorithms=()` runs the preview alone (fast); `verbose=2` prints Piff's log,
  including how many stars each cut rejects.
- **Diagnosing an empty selection**: first `t["candidate"].sum()` (the star finder), then
  the cuts one at a time (`set_cuts(...)` with a single cut, re-run the preview).
- **SizeMag on real L2 data** (first real file, a catalog trimmed to ≈ 700 Gaia stars,
  2026-11-06): it found only 7 candidates, all unclean and with masked pixels, which the
  cuts then removed. How SizeMag works (`Piff/piff/size_mag.py`): a seed from the brightest
  20 % of objects, their smallest 20 %, the tighter half (15 objects here, as designed),
  grown within ≈ ±5 % in T (`max_spread`); then a locus fit across the field and a histogram
  scan from bright to faint that stops at the first "valley" above 1 % of the peak
  (`impurity`), meant to find where galaxies merge with stars. With a star-only catalog of
  undersampled stars this goes wrong: the seed comes from the brightest stars (likely
  saturated: masked, unclean), the ±5 % window is narrower than the native-stamp size
  scatter, and a single star out of place stops the scan. For star-only catalogs use our
  own selection (`select_locus`, below) or `select_stars("all")` with cuts.
- **Cost**: milliseconds per detection and algorithm (gauss is the slowest); use
  `mag_range` or `max_objects` (brightest by catalog flux) for large catalogs.
- `plot_size_mag(table, algorithm=None, size="T"|"sigma", mag="catalog"|"measured")` also
  takes `DataFitResult.selection` (catalog size only); flagged measurements are left out and
  counted in the title.

## Our own stellar-locus selection (`StarCatalog.select_locus`)

For catalogs that are already star lists (e.g. trimmed to Gaia stars), where the question is
which stars are usable rather than which objects are stars. Replaces Piff's star finder only;
Piff's cuts and reserve still apply afterwards.

```python
t = cat.measure(img, algorithms=("wmom",))          # sizes T_wmom_0.4
res = cat.select_locus(t)                           # size="T_wmom_0.4", order=1, k=3
res.summary                                         # counts, saturation limit, width, ...
res.table                                           # per star: size, size_ref, resid, reason
res.plot()                                          # size–mag, residual–mag + scan, map
cat.set_cuts(min_snr=50, min_sep=2.5, max_mask_pixels=5)
fit = DataFit(img, cat, cfg)                        # fits the selected stars
```

Steps; each star ends with one `reason`:

1. `flagged`: the size measurement failed, or size / magnitude not finite.
2. `faint`: fainter than `mag_max` (optional, default none).
3. **Locus centre**: log(size) fitted as a polynomial of order `order` in the normalised SCA
   position (0 constant; **1 plane, default**; 2 quadratic), with clipping at `k` × width
   around the median residual (so a few large outliers cannot drag the fit off a tight locus;
   if clipping leaves fewer stars than polynomial terms, a clear error).
   The PSF size varies across an SCA; a field-dependent centre keeps the band equally fair
   everywhere (a constant centre with a tight band cuts stars where the PSF is larger).
   Residual = size / centre(x, y) − 1; **width** = robust σ (1.4826 × MAD) of the residuals
   of all usable stars.
4. `saturated`: the saturation turn, detected from the data as a *change* of the locus with
   magnitude. Residuals binned in magnitude (`mag_bin` = 0.25); reference level = the median
   of the bin medians (a constant offset between the fitted centre and the bin medians, e.g.
   from a skewed size distribution, cancels). A bin with ≥ `min_bin_count` (3) stars deviates
   when its median differs from the reference by more than `sat_sigma` (3) × its uncertainty
   (1.2533 × width / √n) **and** by at least `min_shift` (1) × the locus width (real
   saturation grows sizes by many widths; this keeps a very tight locus from turning tiny
   trends into a turn; `min_shift=0` turns it off). The limit is the faint edge of the run of
   deviating bins (same sign) that extends to the bright end, at least `sat_bins` (2) long;
   on a flat faint end this is the first departure seen scanning from faint to bright. A turn
   that would mark more than half of the usable stars is **not applied**
   (`bright_limit_source = "unclear"`, a warning; `detected_limit` keeps it). `bright_limit=
   <mag>` sets the limit by hand. The centre is refitted without the saturated stars.
5. `off_locus`: |residual| > `k` × width (**k = 3**: keeps 99.7 % of a Gaussian locus, so only
   clear outliers go: binaries, blends, a galaxy among the matches). `summary
   ["off_locus_fraction"]` should be ≈ 0.3 % plus the real outliers; much more means the
   locus is not described well (look at the residual map; raise `order`).
6. `selected`.

- **Size**: `size="T_wmom_0.4"` (default) has the smallest star-to-star scatter on native
  stamps (CLAUDE.md §9.8); its compression of sizes toward the weight does not matter for a
  selection. Any `T_<alg>`, `sigma_<alg>` or the catalog `size_cat` can be used. Magnitudes:
  `mag="mag"` (catalog, default) or `mag_<alg>`.
- **Result on the catalog**: the selected stars become a 0/1 column `psf_locus` read by
  Piff's `Flag` selection. The settings, summary and selected ids are saved in the catalog's
  config (and so with every fit); `StarCatalog.from_config` rebuilds the same selection.
- `select_locus(table, ...)` (module function) runs the selection without changing a
  catalog.

## Running a fit (`DataFit`)

`DataFit(image, catalog, fit_config, name=None)`; `set_stamp_size(25)`,
`set_reserve(frac, seed)` (Piff's `reserve_frac`), `set_diagnostics(...)` (as the simulation
module, comparisons `'data'` and `'model_os'` only), `set_verbose(level)`,
`set_attitude("auto" | True | False)`; `run(save_to)`.

`run` (one exposure, one SCA):
1. image checks: `check_weights`, `check_orientation` (raises on a mirrored frame),
   `check_finite`, and whether the telescope attitude is used (below);
2. writes `catalog.fits` and `piff.yaml` (full Piff config; also runs with `piffify`, which
   uses one shape method for selection and fit);
3. Piff Input (all detections) → Piff Select under the selection's shape method (selection
   type, cuts, reserve) → Piff fit under the fit's shape method. These are Piff's own steps of
   `piff.config.process`, called one by one;
4. star table, fit summary, moments; saves everything to `save_to`.

### Telescope attitude (RomanOptics orientation)

`galsim.roman.getPSF` builds the optical PSF in arcsec along the SCA axes and must rotate it
onto the sky. Released GalSim applied the whole WCS Jacobian for this, distortion included,
which drawing with the same WCS then undid (the optical PSF came out fixed in pixels: ≈ 2–4 %
small in T on SCA 11). A GalSim change (PR branch, not yet released) applies only the
WCS's rotation and reflection, or, given the telescope attitude (`boresight` = V1 axis,
`telescope_PA` = position angle of V3), the rotation from the attitude itself; the WCS then
only maps the PSF onto pixels, so the distortion is applied once.

With a RomanOptics model, `DataFit` writes the image's attitude (`img.attitude`) into the
RomanOptics block of `piff.yaml` as `boresight_ra`, `boresight_dec`, `telescope_pa`
(degrees):

- `set_attitude("auto")` (default): when the image has an attitude and the installed GalSim
  and Piff support it (`attitude_support()` lists what is missing); otherwise RomanOptics is
  oriented from the WCS (warning when only the software is missing).
- `set_attitude(True)`: required; `run` raises before doing anything if it cannot be used.
- `set_attitude(False)`: never (WCS orientation; for comparisons).

`res.checks["attitude"]` records `requested`, `used` and the `values` passed. The flipped
working copy is still needed: GalSim evaluates the aberrations at `SCA_pos` in its own frame
and, with an attitude, finds the star's sky position from the same position through the WCS.

## Results (`DataFitResult`)

**Star table** (`res.table`): `sca`, `star_id`, `x`, `y` (1-based), `phase_x`, `phase_y`,
`mag`, `flux_cat` [DN/s], `gain`, `reserve`, `flag`, `flux` (fitted, DN/s), `flux_cat_r` =
fitted / catalog flux (a Kron flux misses light: not expected to be 1), `model_flux`, `du`,
`dv` [arcsec], `dx`, `dy` [pix] (fitted centre − catalog position, not an error), `chisq`,
`dof`, `chisq_dof`, `sky_resid` [DN/s] (median unmasked stamp-edge pixel after the sky
subtraction: an offset means a wrong sky level, a trend across the SCA a gradient).

**Fit summary**: as for simulations (χ², dof = pixels − 3 per star, `niter`, `converged`
False when Piff stopped at `max_iter`, ...) plus `n_catalog` (detections), `n_objects` (with a
stamp), `n_candidates` (picked by the selection type).

**Selection** (`res.selection`, `selection.fits`): the Piff catalog with `on_image`,
`candidate`, `star` for every detection; `plot_size_mag()` shows them on the catalog's size
vs magnitude (measured sizes: `StarCatalog.measure`, above).

**Moments**: `'data'` rows (data stamp and model stamp at the fitted flux and centre, both
measured about the catalog position; residuals model − data) and `'model_os'` rows (the
model alone, oversampled; `*_ref` NaN). `moment_summary()` summarises dT/T, de1, de2 (data);
for the model use `moment_summary(columns=("T_model", "e1_model", "e2_model"))`.
`model_moments(...)` and `piffval.compare_fits(...)` as in `docs/piffval.md`.

**Plots**: `plot_size_mag`, `plot_mean_images(which)`, `plot_chi2_radius`,
`plot_moment_residuals`, `plot_field(column, comparison, algorithm)`, `plot_components(kind)`,
`plot_zernikes`, `plot_rho`.

## Reference optimum (`res.reference()`)

The joint χ² optimum of the same Piff model on the same stars (`psf_fomo.reference`,
CLAUDE.md §8), to tell optimiser limits (Piff above the optimum of its own model) from model
limits (the optimum itself leaves χ²/dof above 1, or its moments differ from the data's).

```python
out = res.reference(starts=("given", "zero"), max_nfev=50)   # centers="fitted"
from psf_fomo.reference import comparison_table
comparison_table({"Piff": res, "Piff, profiled": out["given"],
                  **{f"optimum ({k})": v for k, v in out["optimum"].items()}},
                 comparison="data", algorithm="gauss")
```

- **Centres** are held at Piff's fitted centres (`centers="fitted"`, default); real data has
  no true positions. `centers="catalog"` holds them at the detection-catalog positions (a
  measurement, not the truth). Fluxes are profiled. Piff also fits centres, so the reference
  χ² is an upper bound on the optimum of Piff's problem.
- `out["given"]` is Piff's solution with those centres and profiled fluxes, on the
  reference's χ² and dof (weighted pixels of fit stars − fit stars − free coefficients);
  compare Piff with the optimum through it. `out["optimum"][start]` are the optima
  (`converged` False when scipy stopped at `max_nfev`).
- Each solution (`ReferenceResult`) carries the real-data star table, components (no truth)
  and moments (`'data'`, `'model_os'`, with the fit's diagnostics settings by default).
- **Speed.** The optimisation draws only the fit stars. Each reported solution is drawn once
  for all stars (`report="all"`, default) and its moments computed. With a large reserve
  fraction most of the time goes there: `res.reference(starts=("zero",), report="fit",
  given=False, moments=None)` draws only the fit stars, once per evaluation step, and skips
  Piff's solution; compute moments afterwards for the solution you keep.
- Parameters, frozen components, starts and cost as for simulations (`docs/piffval.md`).
  `DataReferenceSolver` (in `psf_fomo.datafit`) is the solver; it differs from the
  simulation one only in the held centres and the truth-free diagnostics.
- No config-only reference (Piff initialisation without a fit) for real data yet.

**Saved folder**: `catalog.fits`, `piff.yaml`, `image_galsim_frame.fits` (the image flipped
into GalSim's frame, for `frame="science"`; the size of your file), `detections.fits` (only
for a catalog given as a table), `psf.fits` (Piff), `stars.fits`,
`selection.fits`, `moments.fits`, `config.yaml` (run settings incl. the Piff config, fit
summary, checks, provenance). `load_data_fit(path)` reloads: the catalog and (for
`frame="science"`) the working copy come from the folder, so it can be moved; a
`frame="galsim"` image must still be at its recorded path. Stamps are re-cut with Piff's Input and the fits attached; Piff's
`max_snr_weight` scaling is re-applied from the saved `snr` / `raw_snr`; with model Poisson
weights the Poisson term is re-added from the final model (Piff's last update preceded its
final reflux, so weights agree to ~1 %, not exactly).

## Limitations
- One exposure of one SCA per fit.
- Achromatic fits only (one effective wavelength; stars' SEDs differ in real data).
- No truth: model quality is judged by reserve-star residuals, χ², model − model comparisons,
  and the reference optimum.
- The sky is one number per detector; gradients are not modelled.
- `getWCS` in `check_orientation` uses a fixed observable pointing (not the image's): parity
  and scale are independent of it, the reported rotation weakly dependent.
