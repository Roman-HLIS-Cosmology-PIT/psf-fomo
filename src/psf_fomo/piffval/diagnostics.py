"""Fit diagnostics.

Per-star quantities (`star_table`) and whole-fit quantities (`fit_summary`) for a fitted Piff
PSF and the simulation truth. Conventions:

- ``flux_r`` = fitted flux / true flux (both ADU). True flux = catalog flux (e⁻) /
  ``galsim.roman.gain``.
- ``flux_tot_r`` = ``flux_r`` × total flux of the unit model (e.g. optics ⊛ kernel + grid).
  It is the physically meaningful flux ratio when a component (PixelGrid in a Sum) carries
  net flux (CLAUDE.md §5.5).
- ``du, dv`` [arcsec]: fitted centre (``star.fit.center``) relative to the catalog position;
  ``dx, dy`` [native pix]: the same through the local WCS. The simulation places stars exactly
  at the catalog position, so these are centroid errors.
- ``chisq``, ``dof``: Piff's per-star values from the last reflux (dof = weighted pixels − 3
  with a free centre).
- ``flag`` = 1 if Piff flagged (removed) the star, else 0.
"""
import numpy as np
import galsim
import galsim.roman
from astropy.table import Table

from ..moments.compare import summarize
from .components import leaf_psfs


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
    out = Table()
    for name in ("obs", "sca", "star_id", "x", "y", "phase_x", "phase_y", "mag"):
        out[name] = rows[name]
    out["reserve"] = [bool(s.is_reserve) for s in stars]
    out["flag"] = [int(bool(s.is_flagged)) for s in stars]
    out["flux"] = [s.fit.flux for s in stars]
    out["flux_true"] = np.asarray(rows["flux"], dtype=float) / galsim.roman.gain
    out["flux_r"] = out["flux"] / out["flux_true"]
    out["model_flux"] = [model_flux(psf, s) for s in stars]
    out["flux_tot_r"] = out["flux_r"] * out["model_flux"]
    du = np.array([s.fit.center[0] for s in stars])
    dv = np.array([s.fit.center[1] for s in stars])
    out["du"], out["dv"] = du, dv
    pix = [s.data.local_wcs.toImage(galsim.PositionD(u, v)) for s, u, v in zip(stars, du, dv)]
    out["dx"] = [p.x for p in pix]
    out["dy"] = [p.y for p in pix]
    out["chisq"] = [s.fit.chisq for s in stars]
    out["dof"] = [s.fit.dof for s in stars]
    out["chisq_dof"] = out["chisq"] / np.maximum(out["dof"], 1)
    for name, unit in [("x", "pix"), ("y", "pix"), ("du", "arcsec"), ("dv", "arcsec"),
                       ("dx", "pix"), ("dy", "pix"), ("flux", "adu"), ("flux_true", "adu")]:
        out[name].unit = unit
    return out


def model_flux(psf, star):
    """Total flux of the unit-flux model at a star (≠ 1 when a component carries net flux)."""
    prof, _ = psf._getRawProfile(star)
    return float(prof.flux)


def fit_summary(psf, converged, max_iter, table, seconds=None):
    """Whole-fit numbers.

    Parameters
    ----------
    psf : piff.PSF
        The fitted PSF.
    converged : bool
        False if Piff stopped at ``max_iter`` ("PSF fit did not converge").
    max_iter : int
        The configured maximum number of iterations.
    table : astropy.table.Table
        From `star_table`.
    seconds : float, optional
        Wall time of the fit.

    Returns
    -------
    dict
        ``chisq``, ``dof``, ``chisq_dof`` (Piff's totals over fitted, unflagged stars; dof =
        pixels − 3 per star), ``niter``, ``max_iter``, ``converged``, ``last_delta_chisq``,
        ``nremoved``, ``n_stars``, ``n_reserve``, ``n_flagged``, ``seconds``, and
        ``kernel_max_nfev`` (GaussianMoments stars that stopped at max_nfev in the last
        iteration; None without that kernel).
    """
    return {
        "chisq": float(psf.chisq),
        "dof": int(psf.dof),
        "chisq_dof": float(psf.chisq / psf.dof) if psf.dof > 0 else float("nan"),
        "niter": int(psf.niter),
        "max_iter": int(max_iter),
        "converged": bool(converged),
        "last_delta_chisq": float(psf.last_delta_chisq),
        "nremoved": int(psf.nremoved),
        "n_stars": len(table),
        "n_reserve": int(np.sum(table["reserve"])),
        "n_flagged": int(np.sum(table["flag"])),
        "seconds": None if seconds is None else float(seconds),
        "kernel_max_nfev": kernel_max_nfev(psf),
    }


def kernel_max_nfev(psf):
    """Stars whose GaussianMoments fit stopped at max_nfev in the last iteration, or None."""
    counts = [m.n_max_nfev for m in leaf_models(psf) if hasattr(m, "max_nfev_history")]
    return int(sum(counts)) if counts else None


def leaf_models(psf):
    """All Piff Model objects of a (possibly composite) PSF, in component order."""
    return [leaf.model for leaf in leaf_psfs(psf)]


def summary(table, columns=("flux_r", "flux_tot_r", "dx", "dy", "chisq_dof")):
    """Mean, std, error on the mean, ``n`` and ``n_flagged`` for fit and reserve stars.

    Parameters
    ----------
    table : astropy.table.Table
        From `star_table`.
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
