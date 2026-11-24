# psf-fomo

A test bench for PSF modeling of the Nancy Grace Roman Space Telescope Wide Field
Instrument (WFI): simulate Roman star images with known PSF errors, measure PSF moments, fit
the stars with [Piff](https://github.com/rmjarvis/Piff), and check how well the fitted PSF
recovers the truth. It can also fit Piff PSF models to real Roman L2 images.

## Installation

psf-fomo needs Python ≥ 3.10, a GalSim with its Roman science-frame update (GalSim's `main`
branch, tested at commit `09b7f5d3f`; not yet in a release: `galsim.roman` WCS and PSFs in
Roman's science frame, like L2 files) and a recent Piff. Install GalSim from a clone of its
`main` branch (`pip install .`) as for Piff below. It is built against Piff's
`main` branch at commit `280064f7e6bb648c82bb41643d8a10fa89034566`, which is newer than the
latest Piff release (it needs, e.g., Piff's `GaussianBlur` model and its moment methods for
undersampled stars), so Piff is installed from a clone rather than as a dependency.

1. Create an environment with the dependencies (conda shown; pip works as well):
   ```
   conda env create -f environment.yml
   conda activate psf-fomo
   ```
2. Install Piff at the pinned commit:
   ```
   git clone https://github.com/rmjarvis/Piff.git
   cd Piff
   git checkout 280064f7e6bb648c82bb41643d8a10fa89034566
   pip install .
   cd ..
   ```
3. Install psf-fomo (editable), with the optional extras you need:
   ```
   git clone <this repository> psf-fomo
   cd psf-fomo
   pip install -e ".[moments,stats,dev]"
   ```
   Extras: `moments` (ngmix: the `am`, `gauss` and `wmom` moment algorithms), `stats`
   (TreeCorr: ρ statistics), `dev` (pytest).

Check the installation by running the fast tests:
```
python -m pytest            # fast tests
python -m pytest -m slow    # slow tests: Roman PSF rendering and full Piff fits
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

**Piff validation** (`psf_fomo.piffval`, `psf_fomo.reference`)
- Piff configurations built from high-level options: Roman optics (free or pinned), the fixed
  MTF, a fitted kernel (signed `GaussianMoments`, Piff's `GaussianBlur` or `Gaussian`), a
  `PixelGrid`, outlier rejection, iteration controls, reserve stars.
- Fits in memory or through Piff's own file-based input.
- Diagnostics: χ², flux and centroid recovery, convergence, moment residuals of model vs
  data and model vs truth (native and oversampled, several algorithms), fitted components vs
  truth, ρ statistics, and plots.
- A reference solver: the joint χ² optimum of the same model, to separate optimizer
  limitations from model limitations.

**Real data** (`psf_fomo.datafit`)
- Roman L2 images (converted from ASDF to FITS), detection catalogs, star selection and cuts
  through Piff, Piff fits with the same model builder and diagnostics (no truth), and the
  reference optimum.
- Synthetic L2 files from simulations, to try the workflow without real data.

Features marked **experimental** in the documentation (telescope-attitude orientation,
the stellar-locus selection, `DataFit.initialize`) are still being tested.

## Documentation

- [`docs/conventions.md`](docs/conventions.md): units, coordinates, moment definitions,
  the simulated PSF, fitting conventions.
- [`docs/sim.md`](docs/sim.md), [`docs/moments.md`](docs/moments.md),
  [`docs/piffval.md`](docs/piffval.md), [`docs/datafit.md`](docs/datafit.md): usage of each
  module.
- [`CHANGELOG.md`](CHANGELOG.md): changes between versions.

## License

MIT, see [`LICENSE`](LICENSE).
