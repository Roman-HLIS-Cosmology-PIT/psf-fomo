"""psf-fomo: Roman WFI PSF modelling test bench.

- ``psf_fomo.sim``: simulated Roman star images with known truth and controlled PSF errors.
- ``psf_fomo.moments``: PSF size/shape measurements on native and oversampled images.
- ``psf_fomo.piffval``: Piff fits of the simulations and their diagnostics.
- ``psf_fomo.reference``: the joint χ² optimum of the model Piff fits.
- ``psf_fomo.datafit``: Piff fits of real Roman L2 images.
- ``psf_fomo.models``: extra Piff models (``GaussianMoments``), registered on import.

Units and definitions: ``docs/conventions.md``.
"""
from importlib.metadata import version as _version

__version__ = _version("psf-fomo")
