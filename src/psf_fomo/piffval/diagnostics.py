"""Fit diagnostics of a simulation fit.

Per-star quantities (`star_table`) and whole-fit quantities (`fit_summary`) for a fitted Piff
PSF and the simulation truth. The truth-free columns come from `psf_fomo.piffval.fitting`;
this module adds the simulation catalog columns and the truth ratios:

- ``flux_r`` = fitted flux / true flux (both ADU). True flux = catalog flux (e⁻) /
  ``galsim.roman.gain``.
- ``flux_tot_r`` = ``flux_r`` × total flux of the unit model (e.g. optics ⊛ kernel + grid).
  It is the physically meaningful flux ratio when a component (PixelGrid in a Sum) carries
  net flux (CLAUDE.md §5.5).
- ``du, dv`` [arcsec], ``dx, dy`` [pix]: fitted centre relative to the catalog position. The
  simulation places stars exactly at the catalog position, so these are centroid errors.
- ``flag`` = 1 if Piff flagged (removed) the star, else 0.
"""
import numpy as np
import galsim.roman
from astropy.table import Table

from ..moments.compare import summarize
from .fitting import fit_columns, fit_summary, kernel_max_nfev, leaf_models, model_flux  # noqa: F401

ROW_COLUMNS = ("obs", "sca", "star_id", "x", "y", "phase_x", "phase_y", "mag")


def star_table(psf, stars, rows):
    """One row per star with flux, centroid and χ² diagnostics.

    Parameters
    ----------
    psf : piff.PSF
        The fitted PSF.
    stars : list of piff.Star
        Fitted stars (``psf.stars``), in the same order as ``rows``.
    rows : astropy.table.Table
        Simulation catalog rows of these stars (``obs``, ``sca``, ``star_id``, ``x``, ``y``,
        ``flux`` in e⁻, …).

    Returns
    -------
    astropy.table.Table
        Columns ``obs``, ``sca``, ``star_id``, ``x``, ``y``, ``phase_x``, ``phase_y``,
        ``mag``, ``reserve`` (bool), ``flag``, ``flux`` (fitted, ADU), ``flux_true`` (ADU),
        ``flux_r``, ``model_flux`` (total flux of the unit model), ``flux_tot_r``, ``du``,
        ``dv`` (arcsec), ``dx``, ``dy`` (pix), ``chisq``, ``dof``, ``chisq_dof``.
    """
    if len(stars) != len(rows):
        raise ValueError("stars and rows have different lengths")
    fit = fit_columns(psf, stars)
    out = Table()
    for name in ROW_COLUMNS:
        out[name] = rows[name]
    for name in ("reserve", "flag", "flux"):
        out[name] = fit[name]
    out["flux_true"] = np.asarray(rows["flux"], dtype=float) / galsim.roman.gain
    out["flux_r"] = out["flux"] / out["flux_true"]
    out["model_flux"] = fit["model_flux"]
    out["flux_tot_r"] = out["flux_r"] * out["model_flux"]
    for name in ("du", "dv", "dx", "dy", "chisq", "dof", "chisq_dof"):
        out[name] = fit[name]
    for name, unit in [("x", "pix"), ("y", "pix"), ("flux", "adu"), ("flux_true", "adu")]:
        out[name].unit = unit
    return out


def summary(table, columns=("flux_r", "flux_tot_r", "dx", "dy", "chisq_dof")):
    """Mean, std, error on the mean, ``n`` and ``n_flagged`` for fit and reserve stars.

    Parameters
    ----------
    table : astropy.table.Table
        A star table (with ``reserve`` and ``flag``).
    columns : sequence of str
        Columns to summarise.

    Returns
    -------
    astropy.table.Table
        One row per star set (``set`` = 'fit' or 'reserve').
    """
    table = table.copy()
    table["set"] = np.where(table["reserve"], "reserve", "fit")
    return summarize(table, ["set"], list(columns))
