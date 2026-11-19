"""Truth-free pieces of a Piff fit, shared by simulation validation and real-data fits.

- Logging and the convergence flag (`make_logger`): Piff has no convergence attribute, so a
  handler on the logger notices its "did not converge" warning (reached ``max_iter``).
- `piff_shape_method`: set Piff's global shape method for the duration of a block.
- `choose_reserve`: Piff's reserve rule, ``int(frac × n)`` stars chosen at random.
- `fit_columns`: per-star flux, centroid and χ² columns of a fitted PSF.
- `fit_summary`: whole-fit numbers (χ², dof, iterations, convergence, ...).

Conventions (docs/conventions.md):

- ``flux``: fitted flux in image units. ``model_flux``: total flux of the unit model (≠ 1 when
  a component, e.g. a PixelGrid in a Sum, carries net flux).
- ``du, dv`` [arcsec]: fitted centre (``star.fit.center``) relative to the catalog position;
  ``dx, dy`` [native pix]: the same through the local WCS.
- ``chisq``, ``dof``: Piff's per-star values from the last reflux (dof = weighted pixels − 3
  with a free centre).
"""
import logging
import sys
from contextlib import contextmanager

import numpy as np
import galsim
import piff
import piff.config
from astropy.table import Table

from .components import leaf_psfs

NOT_CONVERGED = "PSF fit did not converge"  # Piff's warning when max_iter is reached


def fit_columns(psf, stars):
    """Per-star fit diagnostics that need no truth.

    Parameters
    ----------
    psf : piff.PSF
        The fitted PSF.
    stars : list of piff.Star
        Fitted stars (``psf.stars``).

    Returns
    -------
    astropy.table.Table
        Columns ``reserve`` (bool), ``flag`` (1 if Piff flagged the star), ``flux`` (fitted,
        image units), ``model_flux``, ``du``, ``dv`` (arcsec), ``dx``, ``dy`` (pix),
        ``chisq``, ``dof``, ``chisq_dof`` (NaN for a star Piff excluded before computing its
        χ²; such stars have ``flag`` 1).
    """
    out = Table()
    out["reserve"] = [bool(s.is_reserve) for s in stars]
    out["flag"] = [int(bool(s.is_flagged)) for s in stars]
    out["flux"] = [s.fit.flux for s in stars]
    out["model_flux"] = [model_flux(psf, s) for s in stars]
    du = np.array([s.fit.center[0] for s in stars], dtype=float)
    dv = np.array([s.fit.center[1] for s in stars], dtype=float)
    out["du"], out["dv"] = du, dv
    pix = [s.data.local_wcs.toImage(galsim.PositionD(u, v)) for s, u, v in zip(stars, du, dv)]
    out["dx"] = [p.x for p in pix]
    out["dy"] = [p.y for p in pix]
    # Piff leaves χ² and dof unset (None) for a star it excluded before any reflux succeeded
    # (it catches the per-star exception and flags the star): NaN here.
    out["chisq"] = [np.nan if s.fit.chisq is None else s.fit.chisq for s in stars]
    out["dof"] = [np.nan if s.fit.dof is None else s.fit.dof for s in stars]
    out["chisq_dof"] = out["chisq"] / np.maximum(out["dof"], 1)
    for name, unit in [("du", "arcsec"), ("dv", "arcsec"), ("dx", "pix"), ("dy", "pix")]:
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
        Star table with ``reserve`` and ``flag`` columns.
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


DEFAULT_MIN_SEP = 2.5  # arcsec: a 25-pixel stamp reaches 1.9″ at its corners, plus PSF wings
CUT_KEYS = ("min_sep", "min_snr", "max_snr_weight", "max_edge_frac", "stamp_center_size",
            "max_mask_pixels", "max_pixel_cut", "reject_where", "nstars")


def make_cuts(min_sep=DEFAULT_MIN_SEP, min_snr=None, max_snr_weight=None, max_edge_frac=None,
              stamp_center_size=None, max_mask_pixels=None, max_pixel_cut=None,
              reject_where=None, nstars=None):
    """Piff's rejection cuts as a plain dict (used by simulation and real-data fits alike).

    Parameters
    ----------
    min_sep : float or None
        Reject stars with any other object within this distance, arcsec. Default 2.5. None:
        no separation cut.
    min_snr : float, optional
        Minimum S/N (Piff's own estimate from the stamp and weight).
    max_snr_weight : float, optional
        Piff caps each star's effective S/N at this value by inflating its variance. None =
        Piff's default (100); 0 turns the cap off.
    max_edge_frac, stamp_center_size : float, optional
        Reject stars with more than this flux fraction outside a radius of
        ``stamp_center_size`` pixels (Piff's default radius 13, meant for 32-pixel stamps).
    max_mask_pixels : int, optional
        Reject stars with at least this many masked (zero-weight) pixels in the stamp.
    max_pixel_cut : float, optional
        Reject stars brighter than the flux whose typical peak pixel is this value (image
        units).
    reject_where : str, optional
        Piff ``reject_where`` expression on star properties.
    nstars : int, optional
        Keep at most this many stars (highest S/N).

    Returns
    -------
    dict
        Keys `CUT_KEYS`; None means the cut is not applied.
    """
    if min_sep is not None and min_sep <= 0:
        raise ValueError("min_sep must be > 0 (or None)")
    cuts = {"min_sep": min_sep, "min_snr": min_snr, "max_snr_weight": max_snr_weight,
            "max_edge_frac": max_edge_frac, "stamp_center_size": stamp_center_size,
            "max_mask_pixels": max_mask_pixels, "max_pixel_cut": max_pixel_cut,
            "reject_where": reject_where, "nstars": nstars}
    floats = ("min_sep", "min_snr", "max_snr_weight", "max_edge_frac", "max_pixel_cut")
    ints = ("stamp_center_size", "max_mask_pixels", "nstars")
    return {k: (None if v is None else float(v) if k in floats else int(v) if k in ints else v)
            for k, v in cuts.items()}


def select_config(cuts, extra_reject=()):
    """Piff ``select`` entries for these cuts (no selection type, no reserve).

    Parameters
    ----------
    cuts : dict
        From `make_cuts` (missing keys mean "not applied").
    extra_reject : sequence of str
        More ``reject_where`` expressions, combined with the cuts' own by ``or``.

    Returns
    -------
    dict
    """
    out = {}
    rejects = list(extra_reject)
    if cuts.get("reject_where"):
        rejects.append(f"({cuts['reject_where']})")
    if rejects:
        out["reject_where"] = " or ".join(rejects)
    for key in CUT_KEYS:
        if key != "reject_where" and cuts.get(key) is not None:
            out[key] = cuts[key]
    return out


def apply_cuts(stars, cuts, logger=None):
    """Apply Piff's rejection cuts to one exposure's stars, in memory.

    All ``stars`` are the candidate objects (Piff's ``Flag`` selection without a flag) and also
    the neighbours seen by ``min_sep``, so pass the stars of one exposure at a time. As in
    Piff, kept stars get the Poisson variance of their own image added when they have a
    ``gain`` property, and ``snr`` / ``raw_snr`` properties.

    Returns
    -------
    kept : list of piff.Star
        In the order of ``stars``.
    keep : numpy.ndarray of bool
        Which of ``stars`` were kept (Piff returns new objects, matched here by position in
        the list through the ``_cut_index`` property).
    """
    for i, s in enumerate(stars):
        s.data.properties["_cut_index"] = i
    handler = piff.FlagSelect(select_config(cuts), logger=logger)
    kept = handler.rejectStars(handler.selectStars(stars, logger), logger, objects=stars)
    kept = sorted(kept, key=lambda s: s.data.properties["_cut_index"])
    keep = np.zeros(len(stars), dtype=bool)
    keep[[s.data.properties["_cut_index"] for s in kept]] = True
    for s in list(stars) + list(kept):
        s.data.properties.pop("_cut_index", None)
    return kept, keep


def choose_reserve(n, frac=0.0, seed=0):
    """Boolean mask of reserve stars: ``int(frac × n)`` chosen at random (Piff's rule)."""
    mask = np.zeros(n, dtype=bool)
    n_reserve = int(frac * n)
    if n_reserve:
        mask[np.random.default_rng(seed).choice(n, n_reserve, replace=False)] = True
    return mask


@contextmanager
def piff_shape_method(method="hsm", weight_fwhm=None):
    """Set Piff's global shape method inside a ``with`` block; the previous one is restored.

    Parameters
    ----------
    method : {'hsm', 'gauss', 'am', 'wmom'}
        Method used by Piff while the block runs.
    weight_fwhm : float, optional
        Weight FWHM in arcsec (``wmom`` only).
    """
    old = (piff.star.get_shape_method(), piff.star.get_shape_weight_fwhm())
    piff.set_shape_method(method, weight_fwhm)
    try:
        yield
    finally:
        piff.set_shape_method(*old)


class _ConvergenceWatch(logging.Handler):
    """Notices Piff's "did not converge" warning (Piff has no convergence attribute)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.triggered = False

    def emit(self, record):
        if record.getMessage().startswith(NOT_CONVERGED):
            self.triggered = True


def make_logger(verbose=0):
    """A Piff logger printing at the given verbosity, plus a convergence watcher.

    The logger always passes warnings to the watcher, even when only errors are printed.
    """
    levels = {0: logging.ERROR, 1: logging.INFO, 2: piff.config.VERBOSE, 3: logging.DEBUG}
    logger = piff.config.PiffLogger("psf_fomo.piffval")  # not registered: a fresh logger
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(levels[verbose])
    stream.setFormatter(logging.Formatter("%(message)s"))
    watch = _ConvergenceWatch()
    logger.addHandler(stream)
    logger.addHandler(watch)
    logger.setLevel(min(levels[verbose], logging.WARNING))
    return logger, watch
