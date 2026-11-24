"""Helpers to prepare real Roman data: ASDF → FITS conversion and source detection.

- `convert_asdf_to_fits`: calibrated ``*_cal.asdf`` (path, URL or open product)
  → multi-extension FITS in the layout `psf_fomo.piff_datafit.L2Image` reads.
- `detect_sources`: SEP detection catalog of a converted file, optionally only the stars
  matched to Gaia; its columns are the ones `psf_fomo.piff_datafit.StarCatalog` reads.
- `plot_detections`, `plot_detection_sizes`: detections over the image, size vs magnitude.

The extra dependencies (asdf, roman_datamodels, gwcs, sep, …) are installed with
``pip install ".[utils]"`` and imported only when a function needs them.
"""
from .convert import convert_asdf_to_fits
from .detect import detect_sources
from .plotting import plot_detection_sizes, plot_detections

__all__ = ["convert_asdf_to_fits", "detect_sources", "plot_detections",
           "plot_detection_sizes"]
