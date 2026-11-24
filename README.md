# psf-fomo

PSF Fitting, mOdeling and MOments (psf-fomo), is a test bench and validation tool for PSF modeling of the Nancy Grace Roman Space Telescope Wide Field Instrument (WFI). With this tool, you are able to simulate Roman star images with known PSF errors, measure PSF moments, fit the stars with [Piff](https://github.com/rmjarvis/Piff), and check how well the fitted PSF
recovers the truth. It can also fit Piff PSF models to real Roman L2 images. This tool is in active
development and may be updated with high cadence

## Installation

psf-fomo needs Python ≥ 3.10, as well as GalSim and Piff. To use with most
up-to-date versions of these codes, consider using the main branch of GalSim and Piff until a new
tagged version with current changes is released. The roman module has been updated in GalSim's main branch
and contains a flip to the image that is not in v2.8.5 of the code.
Piff has also had some additions that may not be included in the latest tagged version that are used here.
If using this on real Roman data, beware that changes to both GalSim and Piff are ongoing. As develop, the 
code here may also change. 

1. If you want to create an new environment with the needed
 dependencies, you can use:
   ```
   conda env create -f environment.yml
   conda activate psf-fomo
   ```
2. Install Piff at the pinned commit:
   ```
   git clone https://github.com/rmjarvis/Piff.git
   cd Piff
   git switch main
   pip install .
   ```
3. Install psf-fomo. This also installs its other dependencies, including ngmix and TreeCorr:
   ```
   git clone <this repository> psf-fomo
   cd psf-fomo
   pip install .
   ```

## What it can do

**Simulations** (`psf_fomo.sim`)
- Full-SCA images of stars with the Roman PSF (`galsim.roman` optics through Piff's Roman
  optical model), on any SCAs and over several observations.
- Controlled PSF errors, each with optional variations per observation and across the field:
  extra Zernike aberrations, dilation, shear, a Gaussian convolution kernel, charge-diffusion
  (MTF) width, pointing jitter (random walk or a given time series), per-star aberration
  scatter.
- Star placement (random or grid, fixed or random sub-pixel phase), magnitudes, chromatic
  stars, Gaussian noise or the `galsim.roman` noise and detector effects (each switchable),
  simple or Roman WCS.
- Every star is regenerated exactly from the configuration; runs are saved (FITS images,
  catalogs, YAML configuration with versions) and reloaded, and can be driven from YAML.
- The truth PSF of any star on an oversampled grid.

**Moments** (`psf_fomo.moments`)
- PSF size and shape with Piff's algorithms (`hsm`, `am`, `gauss`, `wmom`) on native
  (undersampled), noiseless and oversampled images, in one common convention.
- Measured vs injected PSF errors between simulations, star by star; dependence on sub-pixel
  phase and S/N.

**Piff fits of simulations** (`psf_fomo.piff_simfit`, `psf_fomo.reference`)
- Piff configurations built from high-level options: Roman optics (free or pinned), the fixed
  MTF, a fitted kernel (Piff's `GaussianBlur` or `Gaussian`), a
  `PixelGrid`, outlier rejection, iteration controls, reserve stars.
- Fits in memory or through Piff's own file-based input.
- Diagnostics: χ², flux and centroid recovery, convergence, moment residuals of model vs
  data and model vs truth (native and oversampled, several algorithms), fitted components vs
  truth, ρ statistics, and plots.
- A reference solver: the joint χ² optimum of the same model, to separate optimizer
  limitations from model limitations.

**Piff fits of real data** (`psf_fomo.piff_datafit`)
- Roman L2 images (converted from ASDF to FITS), detection catalogs, star selection and cuts
  through Piff, Piff fits with the same model builder and diagnostics (no truth), and the
  reference optimum.
- Synthetic L2 files from simulations, to try the workflow without real data.

**Helpers for real data** (`psf_fomo.utils`)
- Conversion of calibrated Roman ASDF files (local, URL or open product) to the FITS layout
  used above, with uniform-weight images and background-only weights.
- SEP detection catalogs of converted files, optionally matched to Gaia, and their plots.
- Optional dependencies (asdf, roman_datamodels, gwcs, sep, astroquery, …):
  `pip install ".[utils]"`.

Features marked **experimental** in the documentation (telescope-attitude orientation,
the stellar-locus selection, `PiffDataFit.initialize`) are still being tested.

## Documentation

- [`docs/conventions.md`](docs/conventions.md): units, coordinates, moment definitions,
  the simulated PSF, fitting conventions.
- [`docs/sim.md`](docs/sim.md), [`docs/moments.md`](docs/moments.md),
  [`docs/piff_simfit.md`](docs/piff_simfit.md),
  [`docs/piff_datafit.md`](docs/piff_datafit.md), [`docs/utils.md`](docs/utils.md): usage
  of each module.
- [`CHANGELOG.md`](CHANGELOG.md): changes between versions.

## License

MIT, see [`LICENSE`](LICENSE).
