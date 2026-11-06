"""Per-star diagnostics of a real-data fit (no truth).

`star_table` joins the truth-free fit columns (`psf_fomo.piffval.fitting.fit_columns`) with
the catalog values of each star:

- ``star_id`` (catalog id), ``sca``, ``x``, ``y`` (GalSim image coordinates, 1-based),
  ``phase_x``, ``phase_y`` (position − round(position), CLAUDE.md §5.1).
- ``mag`` (``ABZP`` − 2.5 log10 ``flux_cat``; NaN without ``ABZP``), ``flux_cat`` (catalog
  flux, DN/s), ``flux_cat_r`` = fitted flux / ``flux_cat``. A diagnostic only: a Kron flux
  misses part of the light, so it is not expected to be 1.
- ``gain`` (e⁻ per DN/s; NaN without model Poisson weights).
- ``sky_resid`` [DN/s per pixel]: median of the unmasked edge pixels of the sky-subtracted
  stamp. ≈ 0 if the sky level is right; a trend across the SCA means a sky gradient, an overall
  offset a wrong level (e.g. ``SKYBKG`` from ``SCI`` applied to ``SCI_UNIF``).
- ``du``, ``dv`` [arcsec], ``dx``, ``dy`` [pix]: fitted centre relative to the catalog position
  (not a centroid error: the catalog position is a measurement too).
"""
import numpy as np
from astropy.table import Table

from ..piffval.fitting import fit_columns
from .catalog import PIFF_ID


def star_table(psf, stars, catalog):
    """One row per fitted star.

    Parameters
    ----------
    psf : piff.PSF
        The fitted PSF.
    stars : list of piff.Star
        Fitted stars (``psf.stars``), with their data images.
    catalog : astropy.table.Table
        The Piff catalog of the run (`StarCatalog.piff_table`), matched by ``id``.

    Returns
    -------
    astropy.table.Table
        Columns ``sca``, ``star_id``, ``x``, ``y``, ``phase_x``, ``phase_y``, ``mag``,
        ``flux_cat``, ``gain``, ``reserve``, ``flag``, ``flux``, ``flux_cat_r``,
        ``model_flux``, ``du``, ``dv``, ``dx``, ``dy``, ``chisq``, ``dof``, ``chisq_dof``,
        ``sky_resid``.
    """
    rows = catalog_rows(stars, catalog)
    fit = fit_columns(psf, stars)
    x = np.array([s.image_pos.x for s in stars], dtype=float)
    y = np.array([s.image_pos.y for s in stars], dtype=float)
    out = Table()
    out["sca"] = [int(s.data.properties.get("sca", s.data.properties["chipnum"])) for s in stars]
    out["star_id"] = rows[PIFF_ID]
    out["x"], out["y"] = x, y
    out["phase_x"], out["phase_y"] = x - np.round(x), y - np.round(y)
    out["mag"] = _column(rows, "mag")
    out["flux_cat"] = _column(rows, "flux_cat")
    out["gain"] = _column(rows, "gain")
    for name in ("reserve", "flag", "flux"):
        out[name] = fit[name]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["flux_cat_r"] = np.asarray(out["flux"]) / np.asarray(out["flux_cat"])
    for name in ("model_flux", "du", "dv", "dx", "dy", "chisq", "dof", "chisq_dof"):
        out[name] = fit[name]
    out["sky_resid"] = [sky_residual(s) for s in stars]
    # DN = ADU; 'adu / s' is a FITS unit, 'DN / s' is not (astropy would drop it on writing).
    for name, unit in [("x", "pix"), ("y", "pix"), ("flux", "adu / s"),
                       ("flux_cat", "adu / s"), ("sky_resid", "adu / s")]:
        out[name].unit = unit
    return out


def catalog_rows(stars, catalog):
    """Catalog rows of the stars, in the stars' order (matched by the ``id`` property)."""
    index = {int(i): k for k, i in enumerate(catalog[PIFF_ID])}
    return catalog[[index[int(s.data.properties[PIFF_ID])] for s in stars]]


def sky_residual(star):
    """Median of the unmasked edge pixels of a (sky-subtracted) stamp, image units."""
    image, weight = star.image.array, star.weight.array
    edge = np.zeros(image.shape, dtype=bool)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
    use = edge & (weight > 0)
    return float(np.median(image[use])) if use.any() else np.nan


def _column(rows, name):
    return (np.asarray(rows[name], dtype=float) if name in rows.colnames
            else np.full(len(rows), np.nan))
