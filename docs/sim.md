# Simulation module (`psf_fomo.sim`)

Simulated Roman WFI star images with known truth and controlled PSF errors.

## Quick start

```python
from psf_fomo.sim import RomanSimulation, load_simulation

sim = RomanSimulation(scas=[1], n_obs=1, n_stars=9, filter="H158", seed=0)
sim.set_placement("grid")                      # default "random"
sim.set_brightness(mag_range=(17, 20))         # default mag=18
sim.add_dilation(1.01, sigma_obs=0.002)
sim.set_noise("roman")                         # default Gaussian, sigma = 1 e⁻
result = sim.run(save_to="out/run01", save_noiseless=True)

stars, rows = result.stars(sca=1)              # Piff Stars + truth rows
psf = result.truth_psf(sca=1, obs=0, star_id=0, oversample=5)
result.plot_sca(); result.plot_stamps()

same = load_simulation("out/run01")            # same Simulation object
sim.to_yaml("run01.yaml"); RomanSimulation.from_yaml("run01.yaml").run()
```

## Settings and defaults

| method | what | default |
|---|---|---|
| `RomanSimulation(scas, n_obs, n_stars, filter, seed, stamp_size, render_size, max_zernike)` | common settings | SCA 1, 1 obs, 1 star, H158, seed 0, stamps 25 px, drawn patch 64 px, Z4–Z22 |
| `set_placement(kind, phase, edge_margin)` | `'random'` / `'grid'`; fixed sub-pixel phase (fx, fy) | random, random phase, margin = render_size/2 + 1 px |
| `set_brightness(mag \| mag_range)` | AB magnitudes | 18 |
| `set_zernikes(values, sigma_obs, sigma_field)` | extra aberrations {Noll: waves} | none |
| `add_dilation(value, sigma_obs, sigma_field)` | size factor of the optical PSF | 1 |
| `add_shear(g1, g2, sigma_obs, sigma_field)` | reduced shear of the optical PSF | 0 |
| `set_mtf(scale, sigma_obs, sigma_field)` / `set_mtf(None)` | charge-diffusion kernel width | on, scale 1 |
| `add_kernel(mxx, mxy, myy, sigma_obs, sigma_field)` / `add_kernel(None)` | Gaussian convolution kernel with second moments M, native pix², world axes, PSD only | off |
| `add_aberration_scatter(sigma)` | per-star Zernike scatter, waves | 0 |
| `add_jitter('random_walk', rms_mas, n_steps)` / `add_jitter('timeseries', t, dx_mas, dy_mas)` | pointing jitter | off |
| `set_chromatic('flat' \| sed_file \| galsim.SED \| None, wave_type, flux_type)` | one SED for all stars; `None` = monochromatic | chromatic, flat SED |
| `set_noise('gaussian', sigma)` / `set_noise('roman', effects)` | noise model | Gaussian, 1 e⁻ |
| `set_wcs('simple' \| 'roman', ra, dec, pa, date)` | WCS | 0.11″ pixel scale |
| `set_psf_model(spatial_interpolation, nominal_interp, n_waves)` | how the optical PSF is evaluated across the SCA and summed over the filter (below) | `'fast'`, `'bilinear'`, 10 |

### Variations of the errors
Every error (Zernikes, dilation, shear, MTF scale, kernel moments) can vary:
- `sigma_obs`: a Gaussian offset drawn once per observation, shared by all SCAs.
- `sigma_field`: a random linear field a + b x̂ + c ŷ per observation and SCA, with
  x̂ = x/4088, ŷ = y/4088 and a, b, c ~ N(0, `sigma_field`). For Zernikes this is Piff's
  `aberration_interp='linear'` form.

### Truth rendering chain
optical PSF (below) → dilate → shear → Gaussian kernel → jitter → MTF (three Gaussians, widths
× dilation × mtf_scale, defined in pixels) → × flux → `drawImage(method='auto')` (pixel
response applied once) → noise.

### Optical PSF: `set_psf_model`
`spatial_interpolation` chooses how the optical PSF of a star is obtained:

| mode | optical PSF of a star | cost |
|---|---|---|
| `'fast'` (default) | Piff's `RomanOpticalModel`: galsim.roman PSFs at the SCA corners, blended with position weights (`nominal_interp`; `pupil_bin` = Piff's 4). Each chromatic corner PSF is summed over the filter (weighted by bandpass × SED) once per SCA and observation, into an achromatic effective PSF; stars blend the effective PSFs. | setup once per SCA and observation, then cheap per star |
| `'piff'` | the same corner blending, drawn exactly as Piff draws its chromatic model (corner PSFs × SED through the bandpass, GalSim's defaults) | every star sums each corner over the filter: slow |
| `'exact'` | `galsim.roman.getPSF` at the star's own position (GalSim interpolates its design aberrations to the star), with the catalog's aberrations at the star, summed over the filter per star | per star |

`'fast'` and `'piff'` describe the same PSF (Piff's spatial model), so a fit with Piff's
`RomanOptics` uses the truth's own spatial approximation. `'exact'` does not use the corner
blending. Monochromatic simulations: `'fast'` and `'piff'` are identical.

**Sum over the filter** (`'fast'`, `'exact'`): the optical PSF is computed at `n_waves`
wavelengths evenly spaced from the filter's blue to red limit (default 10, every filter)
and taken as linear in wavelength between them, so the integral is a weighted
sum of these PSFs, drawn directly on a fine grid (a quarter of the Nyquist scale at the blue
limit, 2 × `render_size` wide). More wavelengths are more accurate and slower to set up;
`n_waves=None` uses one every 5 nm. `'piff'` is always exact in wavelength.

### Chromatic PSFs
By default the optical PSF is wavelength dependent and each star is drawn through the
bandpass, weighted by its SED (Piff's `RomanOptics` with `chromatic=True`). One SED is shared
by all stars:
- `set_chromatic("flat")` (default): constant photons per unit wavelength;
- `set_chromatic("my_sed.dat", wave_type="nm", flux_type="flambda")`: an SED file (copied
  into the folder when the simulation is saved);
- `set_chromatic(galsim.SED(...))`: a spectral `galsim.SED`, stored in the configuration as
  a table sampled every 0.5 nm over the filter's wavelength range (photons per nm; finer
  features are smoothed);
- `set_chromatic(None)`: monochromatic, the PSF at the filter's effective wavelength
  (faster).

`stars()` gives each Piff star its effective SED (SED × bandpass), as Piff's chromatic
model needs. A saved chromatic simulation includes `sed.fits` (or the copied SED file) for
Piff's file input.

### Gaussian kernel
`add_kernel(mxx, mxy, myy)` convolves the PSF with an elliptical Gaussian whose second-moment
matrix is M (native pix² along world u, v; × 0.11″² on the sky). It adds M to the pre-pixel
second moments, is not dilated with the optics, and is the in-family truth for the
`GaussianMoments` fit kernel. M must be positive semi-definite. With `sigma_obs` /
`sigma_field` each element varies; a star whose M leaves the PSD cone is clipped to the nearest
PSD matrix, and the catalog records the M actually used. The kernel has its own random streams,
so adding it does not change any other draw.

### Jitter
- `'random_walk'`: a new random walk per observation (`n_steps` samples), re-centred and
  scaled to exactly `rms_mas` per axis.
- `'timeseries'`: your (t [s], Δu [mas], Δv [mas]) samples, weighted by the time spacing and
  re-centred on their mean; identical in every observation.
- The default `rms_mas` is GalSim's `galsim.roman.jitter_rms` (14 mas, documented by GalSim as
  a worst-case design value). It is a **placeholder for testing**, not a measured Roman value.

### Noise and units
Images are in **ADU** (e⁻ divided by `galsim.roman.gain`), background subtracted.
- Gaussian: σ in e⁻ per pixel.
- Roman: any subset of `sky`, `poisson`, `reciprocity`, `dark`, `nonlinearity`, `ipc`,
  `read`, `quantize`, applied in that order (GalSim demo13 order). Sky = zodiacal level at the
  pointing (or SCA centre) × (1 + stray light) + thermal background. Persistence is not
  available. After the effects, the expected sky and dark levels are subtracted.

Fluxes: AB magnitude → e⁻ with the GalSim Roman AB zeropoint, collecting area and
`galsim.roman.exptime`.

## Outputs

`Simulation` holds `config`, `provenance`, `images[(obs, sca)]`, `catalogs[obs]`
(astropy Tables) and optionally `noiseless[(obs, sca)]`.

Catalog columns: `star_id`, `obs`, `sca`, `x`, `y` (pix, GalSim convention: pixel centres at
integers, SCA spans 1…4088), `phase_x`, `phase_y` (pix), `u`, `v` (arcsec; simple WCS) or
`ra`, `dec` (deg; Roman WCS), `mag`, `flux` (e⁻), `dilation`, `g1`, `g2`, `mtf_scale`
(if MTF on), `kernel_mxx`, `kernel_mxy`, `kernel_myy` (pix², if a kernel is set), `z4`… (waves, value at the star), `sed_id` (chromatic), `bkg_var` (ADU²).

Saved folder:
- `obs_NNN.fits`: HDUs 1… = SCAs in configured order (`EXTNAME` `SCA01`…), keywords `GAIN`,
  `BKGVAR`, `BUNIT`.
- `noiseless_NNN.fits` (optional), same layout.
- `catalog_NNN.fits`: HDU `STARS` = all stars; with the simple WCS also one HDU per SCA
  (Piff's file input does not split a catalog by chip; with the Roman WCS Piff can use
  `ra`/`dec` instead).
- `config.yaml`: configuration + provenance.
- chromatic simulations: `sed.fits` (`WAVE` nm, `FLUX` photons per nm; flat or
  `galsim.SED` input), or a copy of the user's SED file. Piff's file input reads it.

Access:
- `stars(sca, obs)`: Piff `Star`s (stamp_size stamps, weight = 1/`bkg_var`, properties
  `sca`, `chipnum`, `star_id`, `obs`, and `gain` when star shot noise is simulated).
- `truth_psf(sca, obs, star_id, oversample, include_pixel)`: unit-flux truth on a finer grid.
  `include_pixel=True` is the "oversampled image" convention (pixel-convolved effective PSF).
- `render_stamp(star_id, obs)`: the noiseless stamp re-rendered from config + seed.

## Reproducibility
Each random stream is seeded from (seed, obs, sca, purpose), so any star, error value or noise
realisation is regenerated exactly from the configuration. Changing one setting does not
change the draws of the others.

## Limitations
- Each star is drawn on a `render_size` patch; flux beyond it (≈1–3 % for 64 px) is not drawn.
- With per-star aberration scatter each star needs its own optical PSF (4 `getPSF` calls
  and, when chromatic, 4 sums over the filter with `'fast'`): slow for many stars.
  `'exact'` and `'piff'` are slow per star in any case.
- With a Zernike field, the catalog `z` values are the field evaluated at the star; with
  `'fast'` and `'piff'` the image uses Piff's interpolation of corner PSFs evaluated at the
  corner values (with `'exact'`, the values at the star).
- Weights use the expected background variance at the star centre, constant over the stamp.
- Only one SED shared by all stars.
