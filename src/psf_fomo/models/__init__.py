"""Piff Model subclasses defined by this package.

Importing this package registers the models with Piff (type names usable in Piff configs):

- ``GaussianMoments`` (experimental): signed convolution kernel with second moments
  (Mxx, Mxy, Myy), native pix².

Model classes must only be defined in this package, never in notebook cells: Piff raises if a
type name is registered twice by a new class definition. Re-importing these modules is safe.
"""
from .gaussian_moments import GaussianMoments

__all__ = ["GaussianMoments"]
