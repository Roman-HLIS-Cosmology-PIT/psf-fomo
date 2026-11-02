"""psf-fomo: Roman WFI PSF modelling test bench.

Three parts, used in this order:

- ``psf_fomo.sim``: simulated Roman star images with known truth and controlled PSF errors.
- ``psf_fomo.moments``: PSF size/shape measurements on native and oversampled images.
- ``psf_fomo.piffval``: Piff fits of the simulations and their diagnostics.
"""
from importlib.metadata import version as _version

__version__ = _version("psf-fomo")
