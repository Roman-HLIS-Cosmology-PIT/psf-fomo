"""Piff validation module: fit simulations with Piff and compute diagnostics.

- `FitConfig`: builds the Piff model and fit settings.
- `PiffValidation`: fits one SCA of a `Simulation` (observations pooled).
- `FitResult`, `load_fit`: results, saved and reloaded.
"""
from .config import FitConfig
from .run import FitResult, PiffValidation, load_fit, piff_shape_method

__all__ = ["FitConfig", "PiffValidation", "FitResult", "load_fit", "piff_shape_method"]
