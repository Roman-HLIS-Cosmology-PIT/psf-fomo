# Moments validation module (`psf_fomo.moments`)

Measures PSF size and shape with several algorithms on native (undersampled) and oversampled
images of simulated stars, and compares simulations star by star. Needs `ngmix` for `am`,
`gauss` and `wmom` (`pip install -e ".[moments]"`).

## Quick start

```python
from psf_fomo.sim import RomanSimulation
from psf_fomo.moments import MomentsValidation, load_moments

base = RomanSimulation(n_stars=50, seed=1).run()
s = RomanSimulation(n_stars=50, seed=1); s.add_dilation(1.01); dil = s.run()

mv = MomentsValidation({"base": base, "dil": dil}, reference="base")
mv.set_algorithms(["hsm", "am", "gauss", "wmom"], wmom_fwhm=0.4)   # the defaults
mv.set_images(["noisy", "noiseless", "oversampled"], oversample=5) # the defaults
res = mv.run(save_to="out/moments01")

res.table                       # one row per sim × star × image × algorithm
res.summary("moments")          # raw T, sigma, e1, e2
res.summary("bias")             # native vs oversampled, same star and algorithm
res.summary("differences")      # measured vs injected PSF errors
res.plot_bias(); res.plot_recovery(); res.plot_vs_phase(); res.plot_vs_snr()
same = load_moments("out/moments01")
```

## Settings

| method | what | default |
|---|---|---|
| `MomentsValidation(simulations, reference)` | dict name → `Simulation` or saved folder | reference = first |
| `set_algorithms(algorithms, wmom_fwhm)` | subset of `hsm`, `am`, `gauss`, `wmom`; wmom weight FWHM in arcsec (one or several) | all, 0.4″ |
| `set_images(images, oversample, include_pixel)` | subset of `noisy`, `noiseless`, `oversampled` | all, ×5, pixel included |
| `select(scas, obs)` | restrict SCAs / observations | all |
| `to_yaml(path)` / `from_yaml(path)` | configuration; only for simulations given as folders | — |

## Image types
- `noisy`: the stamp cut from the simulated image (`Simulation.stars`), weight 1/`bkg_var`.
- `noiseless`: the same stamp re-rendered without noise (`Simulation.render_stamp`), same
  weight. It has no detector effects (IPC, nonlinearity, ...) even with Roman noise.
- `oversampled`: the unit-flux truth PSF on a grid `oversample` times finer
  (`Simulation.truth_psf`), by default the pixel-convolved effective PSF, weight 1, centered.

## Algorithms (Piff's, called per star)
| name | what | weight |
|---|---|---|
| `hsm` | GalSim adaptive moments | adaptive elliptical Gaussian |
| `am` | ngmix adaptive moments (flux ×2, HSM convention) | adaptive elliptical Gaussian |
| `gauss` | ngmix maximum-likelihood elliptical Gaussian fit | — (model fit) |
| `wmom_<fwhm>` | ngmix moments, fixed circular Gaussian weight of FWHM `<fwhm>` arcsec at the star position, not deweighted | fixed |

Piff's global shape method (`piff.set_shape_method`) is never changed.

## Conventions (all algorithms)
Full definitions in [conventions.md](conventions.md#moments).
- `sigma` [arcsec] = det(M)^¼; `T` [arcsec²] = Mxx + Myy (trace)
  = 2 sigma² (1 + g²)/(1 − g²). Divide T by 0.11² for nominal native pix².
- `e1, e2` = distortion (Mxx − Myy, 2 Mxy)/T; `g1, g2` = reduced shear; world (u, v) axes.
- `du, dv` [arcsec]: centroid relative to the star position (wmom: weighted centroid).
- Sizes are measured on the sky through the image's local WCS. Tables record this in
  `meta['SIZEUNIT'] = 'arcsec'`; results saved by earlier versions (T in pix², sigma in pix)
  are converted when loaded, with a warning.
- `flag`: 0 good; 1 failure, 2 negative flux, 4 centroid > 1″ off, 8 exception raised.
  Failed rows have NaN size/shape. Summaries use only `flag == 0` rows and report
  `n_flagged`.
- `snr` = √(Σ I² / bkg_var) over the noiseless native stamp.
- Converters in `psf_fomo.moments.measure`: `g_to_e`, `e_to_g`, `sigma_g_to_T`,
  `T_e_to_sigma`, `moments_matrix`.

## Comparisons
- `bias()`: native rows vs the oversampled row of the same sim, star and algorithm:
  `dT_T = (T − T_os)/T_os`, `de1 = e1 − e1_os`, `de2 = e2 − e2_os`.
- `differences(reference)`: each other simulation vs the reference, paired on
  (obs, sca, star_id), same image type and algorithm: `dT_T = (T − T_ref)/T_ref`, `de1`,
  `de2`, `dg1`, `dg2`. Raises if paired stars are at different positions (simulations must
  share placement and seed; then they also share the noise draws, so noise largely cancels).
  Two injected references per star: `*_os` (same algorithm, oversampled) and `*_hsm_os`
  (HSM, oversampled). The nominal error (e.g. dilation 1.01) is not used directly, because
  the dilation also widens the MTF and the pixel dilutes it.
- `summary(kind)`: mean, std, error on the mean (std/√N) per (sim, image, algorithm).

## Plots
- `plot_bias(sim, images, algorithms)`: native − oversampled dT/T, de1, de2 per algorithm.
- `plot_recovery(injected='hsm_os' | 'os', images, algorithms)`: measured vs injected
  errors, 1:1 line.
- `images` / `algorithms` restrict what is drawn (a name or a list; default all).
- `plot_vs_phase(kind, quantity, image, sim)`, `plot_vs_snr(...)`: binned dependence on the
  sub-pixel phase and on S/N; `kind='bias'` or `'differences'`. `plotting.plot_vs` for any
  per-star column (e.g. `mag`).

## Saved folder
`moments.fits` (the table) and `config.yaml` (configuration + provenance).

## Limitations
- Each star is re-rendered for the noiseless stamp (always, for the S/N) and for the
  oversampled image: the truth PSF is evaluated twice per star.
- `wmom` sizes depend on the weight FWHM; compare only at equal FWHM.
- The noiseless stamp omits detector effects that the noisy image has.
- Higher-order moments (Piff `calculate_moments`) are not included yet.
