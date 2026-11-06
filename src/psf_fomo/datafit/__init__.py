"""Piff fits of real Roman images (kept separate from the simulation validation, `piffval`).

- `L2Image`: one L2 exposure of one SCA in the FITS layout of the user's ASDF → FITS
  converter (image variant, weights, sky, checks).
- `StarCatalog`: the detection catalog, star selection (Piff SizeMag, all, or a column) and
  Piff's cuts; `StarCatalog.measure` / `measure_objects`: moments of every detection before
  any fit (+ a selection preview); `plot_size_mag`.
- `DataFit`: runs Piff (Input → Select → fit) on an image + catalog with a `FitConfig`;
  `DataFitResult`, `load_data_fit`: results, saved and reloaded.
- `DataReferenceSolver`, `reference_from_data_fit` (also `DataFitResult.reference`): the
  joint χ² optimum of the fitted model, centres held at Piff's fitted ones.

Shares only truth-free functions with `psf_fomo.piffval` (``FitConfig``, ``fitting``,
``residuals``, ``components``, ``plotting``) and the reference solver.
"""
from .catalog import StarCatalog
from .fit import DataFit, DataFitResult, load_data_fit
from .l2 import L2Image, galsim_filter_name
from .measure import measure_objects
from .plotting import plot_size_mag
from .reference import DataReferenceSolver, reference_from_data_fit

__all__ = ["L2Image", "StarCatalog", "DataFit", "DataFitResult", "load_data_fit",
           "measure_objects", "plot_size_mag", "DataReferenceSolver",
           "reference_from_data_fit", "galsim_filter_name"]
