# psf-fomo
PSF fitting, modeling, and moments validation tool for the Roman Space Telescope HLIS. This tool is meant to use GalSim to make image simulations of the Roman PSF, use PIFF for PSF fitting, and test different moment measurement algorithms.

## Install

psf-fomo is built against Piff's `main` branch at commit
`6ada1330bd7b1cb1cf23f50dcb3bdcbc2db69800`, which is newer than the latest Piff release.
Piff is therefore not installed automatically.

1. Install Piff from a clone at that commit:
   ```
   git clone https://github.com/rmjarvis/Piff.git
   cd Piff
   git checkout 6ada1330bd7b1cb1cf23f50dcb3bdcbc2db69800
   pip install .
   ```
2. Install psf-fomo (editable), with the optional extras you need:
   ```
   cd psf-fomo
   pip install -e ".[moments,dev]"
   ```
   Extras: `moments` (ngmix), `stats` (TreeCorr), `dev` (pytest).

## Tests

```
pytest            # fast tests only
pytest -m slow    # slow tests (Roman PSF rendering, Piff fits)
```

## Documentation

Usage pages per module are in `docs/`; Phase 0 study notes are in `docs/notes/`.
