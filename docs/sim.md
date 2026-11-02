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
| `add_aberration_scatter(sigma)` | per-star Zernike scatter, waves | 0 |
| `add_jitter('random_walk', rms_mas, n_steps)` / `add_jitter('timeseries', t, dx_mas, dy_mas)` | pointing jitter | off |
| `set_chromatic('flat' \| sed_file, wave_type, flux_type)` | one SED for all stars | achromatic |
| `set_noise('gaussian', sigma)` / `set_noise('roman', effects)` | noise model | Gaussian, 1 e⁻ |
| `set_wcs('simple' \| 'roman', ra, dec, pa, date)` | WCS | 0.11″ pixel scale |
| `set_psf_model(nominal_interp)` | Piff optical-model interpolation | `'bilinear'` |

### Variations of the errors
Every error (Zernikes, dilation, shear, MTF scale) can vary:
- `sigma_obs`: a Gaussian offset drawn once per observation, shared by all SCAs.
- `sigma_field`: a random linear field a + b x̂ + c ŷ per observation and SCA, with
  x̂ = x/4088, ŷ = y/4088 and a, b, c ~ N(0, `sigma_field`). For Zernikes this is Piff's
  `aberration_interp='linear'` form.

### Truth rendering chain
optical PSF from Piff's `RomanOpticalModel.getProfile` with the truth Zernikes (galsim.roman
PSFs at the SCA corners, interpolated across the SCA; `pupil_bin` = Piff's default 4) →
dilate → shear → jitter → MTF (three Gaussians, widths × dilation × mtf_scale, defined in
pixels) → × flux → `drawImage(method='auto')` (pixel response applied once) → noise.

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
(if MTF on), `z4`… (waves, value at the star), `sed_id` (chromatic), `bkg_var` (ADU²).

Saved folder:
- `obs_NNN.fits`: HDUs 1… = SCAs in configured order (`EXTNAME` `SCA01`…), keywords `GAIN`,
  `BKGVAR`, `BUNIT`.
- `noiseless_NNN.fits` (optional), same layout.
- `catalog_NNN.fits`: HDU `STARS` = all stars; with the simple WCS also one HDU per SCA
  (Piff's file input does not split a catalog by chip; with the Roman WCS Piff can use
  `ra`/`dec` instead).
- `config.yaml`: configuration + provenance; a user SED file is copied next to it.

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
- With per-star aberration scatter each star needs its own optical PSF (4 `getPSF` calls):
  slow for many stars. Chromatic PSFs are much slower than achromatic ones.
- With a Zernike field, the catalog `z` values are the field evaluated at the star; the image
  uses Piff's interpolation of corner PSFs evaluated at the corner values.
- Weights use the expected background variance at the star centre, constant over the stamp.
- Only one SED shared by all stars.
