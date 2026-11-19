# Conventions

Units, coordinates and definitions used throughout psf-fomo. The module pages (`sim.md`,
`moments.md`, `piffval.md`, `datafit.md`) refer to this page.

## Units

| quantity | unit |
|---|---|
| positions on the SCA (`x`, `y`) | native pixels (GalSim convention, below) |
| sky offsets, PSF sizes in Piff configs (PixelGrid scale, GaussianMixture sigmas, Piff `Gaussian`) | arcsec |
| second moments, `T`, kernel parameters (`GaussianMoments`, `add_kernel`) | native pix² |
| Piff `GaussianBlur` parameters `[ixx, iyy, ixy]` | arcsec² (converted to pix² in our tables) |
| Zernike aberrations | waves |
| magnitudes | AB |
| simulated images | ADU = e⁻ / `galsim.roman.gain`, background subtracted |
| real L2 images | the file's units (DN/s; see `datafit.md`) |
| jitter | milliarcsec |

Native pixel: `galsim.roman.pixel_scale` = 0.11″. Arcsec are converted to native pixels with
this nominal scale everywhere, also with the Roman WCS, so sizes are comparable between images
and between the simple and the Roman WCS.

## Coordinates

- **Pixels.** GalSim convention: pixel centres at integer coordinates, an SCA spans
  1 … 4088. Sub-pixel phase = position − round(position), in [−0.5, 0.5).
- **World axes.** Shapes, kernels and second moments are along the world (u, v) axes (the
  local WCS Jacobian applied), not the pixel axes.
- **Fitted centres.** Piff's `star.fit.center` is in (u, v) arcsec relative to the catalog
  position; star tables give it as `du`, `dv` [arcsec] and `dx`, `dy` [pix] (converted with
  the local WCS Jacobian).
- **Field coordinates.** Linear error fields use x̂ = x / 4088, ŷ = y / 4088.
- **Real data.** Roman L2 files are in the science frame, which is mirrored relative to
  GalSim's SCA frame (a flip in y or in x depending on the SCA). `L2Image(frame=...)` states
  the frame of the input file; fits with RomanOptics use a flipped working copy (`datafit.md`).

## Moments

The same definitions for every algorithm (`psf_fomo.moments.measure`):

- second-moment matrix M = (Mxx, Mxy, Myy) in native pix², world axes;
- size **T = Mxx + Myy** (trace), native pix²; sigma = det(M)^¼ (native pix), so
  T = 2 sigma² (1 + |g|²) / (1 − |g|²);
- shape as **distortion** e = (Mxx − Myy, 2 Mxy) / T, and **reduced shear** g = e / (1 + √(1 − |e|²));
- relative size error dT/T = (T − T_ref) / T_ref; shape errors de1 = e1 − e1_ref, de2 likewise;
- centroid `du`, `dv` in arcsec relative to the star position.

Algorithms are Piff's: `hsm` (GalSim adaptive moments), `am` (ngmix adaptive moments),
`gauss` (ngmix elliptical Gaussian fit), `wmom` (ngmix moments with a fixed circular Gaussian
weight of a given FWHM in arcsec, not deweighted; labelled `wmom_<fwhm>`). wmom sizes depend
on the weight, so compare them only at the same FWHM. wmom has no centroid fit: data and model
must be measured about the same position. Piff's global shape method is never changed by
these measurements.

## Images

- **Native image**: the star stamp at the detector's 0.11″ sampling.
- **Oversampled image**: the pixel-convolved (effective) PSF sampled on a grid `oversample`
  times finer. The pre-pixel PSF is available as an option (`include_pixel=False`); outputs
  record which one was used.
- **Pixel response** is applied exactly once: profiles drawn with GalSim `method='auto'`
  are convolved with the native pixel; Piff components with `_method = 'no_pixel'`
  (e.g. PixelGrid) already contain it.

## Simulated PSF (truth)

Rendering chain for one star:

optical PSF (with the truth aberrations; `set_psf_model(spatial_interpolation=...)`) →
dilation → shear → Gaussian kernel M → jitter → MTF → shift → × flux → drawn
(`method='auto'`) → noise.

- **Optical PSF modes**: `'fast'` (default) and `'piff'` use Piff's `RomanOpticalModel`
  (`galsim.roman` PSFs at the SCA corners, blended with position weights); `'exact'` uses
  `galsim.roman.getPSF` at the star's position.
- **Chromatic by default**: the optical PSF depends on wavelength (diffraction and
  aberration phases scale with λ; the extra aberrations are in waves at GalSim's reference
  wavelength, 1293 nm), and each star's PSF is integrated over the bandpass weighted by its
  SED (flat by default: constant photons per unit wavelength). `set_chromatic(None)` uses the
  PSF at the filter's effective wavelength instead.
- **Sum over the filter**: `'piff'` draws PSF × SED through the bandpass as Piff does (exact
  in wavelength). `'fast'` and `'exact'` compute the optical PSF at `n_waves` wavelengths
  (default 10) evenly spaced from the blue to the red limit, take it as linear in
  wavelength between them, and sum them with bandpass × SED weights into an achromatic
  effective PSF (`'fast'`: once per corner, SCA and observation; `'exact'`: per star). The
  rest of the chain is achromatic, so this equals drawing through the bandpass. Piff's
  `RomanOptics` (fit side) is exact in wavelength.
- Dilation and shear act on the optical PSF only, before the MTF.
- The Gaussian kernel adds M to the second moments (native pix², world axes, positive
  semi-definite).
- Jitter is the average of the PSF over the line-of-sight trajectory during the exposure
  (it is not dilated with the optics). The default amplitude is a **placeholder**
  (`galsim.roman.jitter_rms`), not a measured Roman value.
- **MTF (charge diffusion)**: three Gaussians with weights (0.17519, 0.53146, 0.29335) and
  relative sigmas (0.4522, 0.8050, 1.4329) of a base σ = 0.3279 native pix; all widths are
  multiplied by `dilation × mtf_scale`. On the fit side the same kernel is Piff's
  `GaussianMixture` with the sigmas converted to arcsec (× 0.11″) and `interp: Fixed`.

Each random stream is seeded from (seed, obs, sca, purpose), so every star is regenerated
exactly from the configuration, and changing one setting does not change the other draws.

## Fitting

- One Piff fit per SCA; observations of that SCA are pooled.
- Piff model structure built by `FitConfig`:
  `Sum[ Convolve[ RomanOptics, kernel, MTF ], Simple(PixelGrid) ]`, pieces left out when not
  requested.
- **Chromatic model**: RomanOptics' `chromatic` option is left at Piff's default (True: the
  PSF integrated over the bandpass with each star's SED) unless set. Simulated stars carry the
  simulation's SED; real-data stars get one assumed SED (flat by default). The other
  components (MTF, kernels, PixelGrid) do not depend on wavelength.
- Linear models (PixelGrid) use `BasisPolynomial`; nonlinear per-star models
  (`GaussianMoments`, Piff `Gaussian`) use `Polynomial` (at least (order+1)(order+2)/2 stars).
  Piff's `GaussianBlur` works with both.
- **Convergence**: Piff has no convergence attribute. `converged` is False when Piff logged
  "PSF fit did not converge" (it reached `max_iter`); `niter` and `max_iter` are reported. A
  fit is never reported as converged otherwise.
- **Star selection cuts** are Piff's rejection step, the same for simulations
  (`PiffValidation.set_cuts`) and real data (`StarCatalog.set_cuts`). Default: `min_sep` =
  2.5″ (stars with any other object within 2.5″ are rejected). Simulations also default to
  `max_snr_weight` = 0 (no S/N cap on the weights); real data keeps Piff's default (100)
  unless set. For simulations the cuts apply to each observation separately.
- **Reserve stars** are chosen at fit time among the stars that pass the cuts (Piff's
  `reserve_frac` and seed), not in the simulation catalog.

## Metrics

- `flux_r` = fitted flux / true flux; `flux_tot_r` = `flux_r` × total flux of the unit model
  (differs from `flux_r` when a component, e.g. a PixelGrid, carries net flux).
- χ²/dof: Piff's own dof is (weighted pixels − 3) per star. The reference solver's dof is
  weighted pixels − N_stars − N_free_coefficients.
- Summaries give the mean, the star-to-star standard deviation and the error on the mean
  (std / √N), using unflagged rows only, with the number of flagged rows.
