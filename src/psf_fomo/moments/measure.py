"""Moment measurements on one star image, reported in one common convention.

The algorithms are Piff's own (``piff.Star.hsm``, ``.am``, ``.gauss``, ``.wmom(fwhm)``; Piff
commit 280064f). We call them directly on each star and never change Piff's global shape
method (``piff.set_shape_method``), which would also change Piff's fitting initialization and
star selection.

Piff's am, gauss and wmom start from (wmom: are centered on) ``star.image_pos`` shifted by
``star.fit.center``. On a star made with `make_star` that offset is zero, so everything here is
measured about the star position; pass fresh stars, not stars returned by a Piff fit.

All Piff methods return ``(flux, cenu, cenv, sigma, g1, g2, flag)`` in local world coordinates
(arcsec), with ``sigma`` = det(M)^(1/4) and g the reduced shear. We convert to:

- ``sigma`` [arcsec] and the trace ``T = Mxx + Myy`` [arcsec²],
  ``T = 2 sigma² (1 + g²) / (1 - g²)`` (Piff's ``measureShapes`` convention);
- distortion ``e = (Mxx - Myy, 2 Mxy) / T`` and reduced shear ``g``, world (u, v) axes;
- centroid offset ``du, dv`` [arcsec] from the star position.

Sizes are on the sky, measured through the image's own (local) WCS, so they are comparable
between native and oversampled images and between WCSs. Divide T by
``galsim.roman.pixel_scale**2`` (0.11″²) for nominal native pix² (`SIZE_UNITS` records the
units). Earlier versions reported T in pix² and sigma in pix; tables now carry
``meta['SIZEUNIT'] = 'arcsec'`` and `upgrade_sizes` converts tables saved before.

Algorithm conventions (Piff):

- ``hsm``: GalSim adaptive moments (``FindAdaptiveMom``); weight image used only as a mask.
- ``am``: ngmix adaptive moments; flux doubled to match HSM's.
- ``gauss``: ngmix maximum-likelihood elliptical Gaussian fit.
- ``wmom``: ngmix moments with a fixed circular Gaussian weight of FWHM ``weight_fwhm``
  [arcsec] centered on the star position; raw weighted moments, not deweighted. Size and shape
  therefore depend on the weight and are meaningful only in comparisons at the same FWHM.
"""
import re
import warnings

import numpy as np
import galsim
import galsim.roman
import piff

ALGORITHMS = ("hsm", "am", "gauss", "wmom")
DEFAULT_WMOM_FWHM = 0.4  # arcsec
SIZE_UNITS = {"T": "arcsec2", "sigma": "arcsec"}
SIZE_UNIT_KEY = "SIZEUNIT"   # table meta key (a FITS header keyword when saved)
_SIZE_COLUMN = re.compile(r"(T|sigma)(_|$)")   # T, T_os, T_ref, sigma_model, ... (not dT_T)

# Flag bits: 1, 2, 4 are Piff's (failure, negative flux, centroid > 1″ from the position).
FLAG_EXCEPTION = 8


def algorithm_label(algorithm, weight_fwhm=None):
    """Name used in result tables, e.g. ``'hsm'`` or ``'wmom_0.4'`` (FWHM in arcsec)."""
    if algorithm == "wmom":
        return f"wmom_{weight_fwhm:g}"
    return algorithm


def g_to_e(g1, g2):
    """Reduced shear g → distortion e = 2g / (1 + |g|²)."""
    gsq = g1 * g1 + g2 * g2
    return 2 * g1 / (1 + gsq), 2 * g2 / (1 + gsq)


def e_to_g(e1, e2):
    """Distortion e → reduced shear g = e / (1 + sqrt(1 - |e|²))."""
    denom = 1 + np.sqrt(1 - (e1 * e1 + e2 * e2))
    return e1 / denom, e2 / denom


def sigma_g_to_T(sigma, g1, g2):
    """Trace T = Mxx + Myy from the determinant radius sigma and the reduced shear.

    T = 2 sigma² (1 + |g|²) / (1 - |g|²), in the units of sigma².
    """
    gsq = g1 * g1 + g2 * g2
    return 2 * sigma**2 * (1 + gsq) / (1 - gsq)


def T_e_to_sigma(T, e1, e2):
    """Determinant radius sigma = det(M)^(1/4) = sqrt(T/2 · sqrt(1 - |e|²))."""
    return np.sqrt(T / 2 * np.sqrt(1 - (e1 * e1 + e2 * e2)))


def moments_matrix(T, e1, e2):
    """Second-moment matrix elements (Mxx, Mxy, Myy) from T and distortion e.

    Mxx = T (1 + e1) / 2, Myy = T (1 - e1) / 2, Mxy = T e2 / 2, in the units of T.
    """
    return T * (1 + e1) / 2, T * e2 / 2, T * (1 - e1) / 2


def measure_star(star, algorithm, weight_fwhm=DEFAULT_WMOM_FWHM):
    """Measure flux, centroid, size and shape of one star with one algorithm.

    Parameters
    ----------
    star : piff.Star
        Star with image, weight and position (``star.image_pos``). Measured about
        ``image_pos`` shifted by ``star.fit.center`` (zero for a new star).
    algorithm : {'hsm', 'am', 'gauss', 'wmom'}
        Measurement algorithm (see the module docstring for conventions).
    weight_fwhm : float
        FWHM of the fixed circular Gaussian weight, arcsec (``wmom`` only). Default 0.4.

    Returns
    -------
    dict
        ``flux`` (image units; algorithm-specific normalization), ``du``, ``dv`` (arcsec),
        ``sigma`` (arcsec), ``T`` (arcsec²), ``e1``, ``e2``, ``g1``, ``g2``, ``flag``
        (0 = good; bits 1 failure, 2 negative flux, 4 centroid more than 1″ from the star
        position, 8 the algorithm raised an exception). Size and shape are NaN when the
        measurement failed (sigma ≤ 0).
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    if algorithm != "hsm":
        require_ngmix()
    try:
        if algorithm == "wmom":
            values = star.wmom(weight_fwhm)
        else:
            values = getattr(star, algorithm)
    except Exception as err:  # one bad star should not stop a large run
        warnings.warn(f"{algorithm} raised {type(err).__name__}: {err}")
        values = (np.nan,) * 6 + (FLAG_EXCEPTION,)
    flux, du, dv, sigma, g1, g2, flag = values
    out = {"flux": flux, "du": du, "dv": dv, "flag": int(flag)}
    if not sigma > 0:
        out.update(sigma=np.nan, T=np.nan, e1=np.nan, e2=np.nan, g1=np.nan, g2=np.nan)
        out["flag"] |= 1
        return out
    e1, e2 = g_to_e(g1, g2)
    out.update(sigma=sigma, T=sigma_g_to_T(sigma, g1, g2), e1=e1, e2=e2, g1=g1, g2=g2)
    return out


def mark_sizes(table):
    """Record in ``table.meta`` that its size columns are in arcsec (T in arcsec²)."""
    table.meta[SIZE_UNIT_KEY] = "arcsec"
    return table


def upgrade_sizes(table):
    """Convert, in place, a moments table saved before sizes were reported in arcsec.

    Tables without ``meta['SIZEUNIT'] = 'arcsec'`` have T (and ``T_*``) in native pix² and
    sigma (``sigma_*``) in native pix; they are multiplied by 0.11″² and 0.11″
    (``galsim.roman.pixel_scale``), the scale they were divided by. Returns the table.
    """
    if table.meta.get(SIZE_UNIT_KEY) == "arcsec":
        return table
    scale = galsim.roman.pixel_scale
    names = [n for n in table.colnames if _SIZE_COLUMN.match(n)]
    for name in names:
        table[name] = table[name] * (scale**2 if name.startswith("T") else scale)
    if names:
        warnings.warn("table saved with sizes in native pixels; converted to arcsec "
                      f"(T in arcsec²): {names}")
    return mark_sizes(table)


def require_ngmix():
    """Raise a clear ImportError if ngmix (needed for am, gauss, wmom) is missing."""
    try:
        import ngmix  # noqa: F401
    except ImportError as err:
        raise ImportError("algorithms 'am', 'gauss' and 'wmom' need ngmix: "
                          "pip install -e \".[moments]\" (or install ngmix)") from err


def make_star(image, image_pos, weight=None, pointing=None):
    """Wrap a galsim Image (e.g. a noiseless or oversampled stamp) in a piff.Star.

    Parameters
    ----------
    image : galsim.Image
        Stamp with a WCS.
    image_pos : galsim.PositionD
        Star position in the image coordinates; ``wmom`` centers its weight here and the
        centroids are reported relative to it.
    weight : galsim.Image or float, optional
        Inverse-variance weight image, or a constant. Default 1 everywhere.
    pointing : galsim.CelestialCoord, optional
        Piff needs a pointing for a celestial WCS (e.g. a real image's TAN-SIP). Default: the
        star's own sky position. The moments use only the local WCS, so any nearby pointing
        gives the same measurement.

    Returns
    -------
    piff.Star
    """
    if weight is None or np.isscalar(weight):
        weight = galsim.ImageF(image.bounds, wcs=image.wcs,
                               init_value=1.0 if weight is None else float(weight))
    if pointing is None and image.wcs is not None and image.wcs.isCelestial():
        pointing = image.wcs.toWorld(image_pos)
    return piff.Star(piff.StarData(image, image_pos, weight=weight, pointing=pointing), None)
