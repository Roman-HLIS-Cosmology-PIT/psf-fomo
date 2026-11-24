"""Piff fits of real Roman images (kept separate from the fits of simulations, `piff_simfit`).

- `L2Image`: one L2 exposure of one SCA, converted from ASDF to FITS (image variant,
  weights, sky, checks).
- `StarCatalog`: the detection catalog, star selection (Piff SizeMag, all, or a column) and
  Piff's cuts; `StarCatalog.measure` / `measure_objects`: moments of every detection before
  any fit (+ a selection preview); `plot_size_mag`; `StarCatalog.select_locus` /
  `select_locus`: stellar-locus selection (experimental; field-dependent locus, saturation
  turn, per-star reasons), `plot_locus`.
- `PiffDataFit`: runs Piff (Input → Select → fit) on an image + catalog with a `PiffFitConfig`;
  `PiffDataFitResult`, `load_data_fit`: results, saved and reloaded.
- `DataReferenceSolver`, `reference_from_data_fit` (also `PiffDataFitResult.reference`): the
  joint χ² optimum of the fitted model, centers held at Piff's fitted ones.
- `psf_fomo.piff_datafit.synthetic`: write a simulation (or any image) as an L2 file and a
  detection catalog, to try this workflow without real data.

Shares only truth-free functions with `psf_fomo.piff_simfit` (``PiffFitConfig``, ``fitting``,
``residuals``, ``components``, ``plotting``) and the reference solver.
"""
from .catalog import StarCatalog
from .fit import PiffDataFit, PiffDataFitResult, attitude_support, load_data_fit
from .l2 import L2Image, galsim_filter_name
from .locus import LocusResult, select_locus
from .measure import measure_objects
from .plotting import plot_locus, plot_size_mag
from .reference import DataReferenceSolver, reference_from_data_fit

__all__ = ["L2Image", "StarCatalog", "PiffDataFit", "PiffDataFitResult", "load_data_fit",
           "measure_objects", "plot_size_mag", "select_locus", "LocusResult", "plot_locus",
           "DataReferenceSolver",
           "reference_from_data_fit", "galsim_filter_name", "attitude_support"]
