"""Piff fits of simulations: fit with Piff and compare the fitted PSF with the truth.

- `PiffFitConfig`: builds the Piff model and fit settings (shared with real-data fits).
- `PiffSimFit`: fits one SCA of a `Simulation` (observations pooled).
- `PiffSimFitResult`, `load_sim_fit`: results, saved and reloaded.
- `compare_fits`: oversampled model moments of several fits at the same positions.

Truth-free helpers shared with `psf_fomo.piff_datafit` live in `fitting` (logger, convergence,
shape method, reserve, per-star fit columns, fit summary) and `residuals` (``'data'`` and
``'model_os'`` moments, `model_moments`, `compare_fits`).
"""
from .config import PiffFitConfig
from .residuals import compare_fits
from .run import PiffSimFitResult, PiffSimFit, load_sim_fit, piff_shape_method

__all__ = ["PiffFitConfig", "PiffSimFit", "PiffSimFitResult", "load_sim_fit",
           "piff_shape_method", "compare_fits"]
