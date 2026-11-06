"""Piff validation module: fit simulations with Piff and compute diagnostics.

- `FitConfig`: builds the Piff model and fit settings (shared with real-data fits).
- `PiffValidation`: fits one SCA of a `Simulation` (observations pooled).
- `FitResult`, `load_fit`: results, saved and reloaded.
- `compare_fits`: oversampled model moments of several fits at the same positions.

Truth-free helpers shared with `psf_fomo.datafit` live in `fitting` (logger, convergence,
shape method, reserve, per-star fit columns, fit summary) and `residuals` (``'data'`` and
``'model_os'`` moments, `model_moments`, `compare_fits`).
"""
from .config import FitConfig
from .residuals import compare_fits
from .run import FitResult, PiffValidation, load_fit, piff_shape_method

__all__ = ["FitConfig", "PiffValidation", "FitResult", "load_fit", "piff_shape_method",
           "compare_fits"]
