"""Our own stellar-locus selection on a pre-fit measurement table (`select_locus`).

Replaces Piff's star finder (SizeMag) for catalogs that are already star lists (e.g. Gaia
matches): it decides which stars are usable PSF stars from their measured size and
magnitude, with every decision recorded per star. Piff's cuts (``min_snr``, ``min_sep``,
``max_mask_pixels``, ``clean_only``) and reserve still apply afterwards.

Steps (each star ends with one ``reason``):

1. ``'flagged'``: the size measurement failed (flag ≠ 0) or size / magnitude is not finite.
2. ``'faint'``: fainter than ``mag_max`` (optional; Piff's ``min_snr`` also removes faint
   stars).
3. Locus center: log(size) fitted as a polynomial of order ``order`` in the normalized SCA
   position (0: constant; 1: plane, default; 2: quadratic), robustly (iterative clipping at
   ``k`` × the locus width). The PSF size varies across an SCA, so a field-dependent center
   keeps the band equally fair everywhere. Residual = size / center(x, y) − 1; locus width =
   robust σ of the residuals of all usable stars (1.4826 × MAD).
4. ``'saturated'``: brighter than the saturation turn, a *change* of the locus with
   magnitude. Residuals are binned in magnitude (``mag_bin``); the reference level is the
   median of the bin medians (so a constant offset between the fitted center and the bin
   medians, e.g. from a skewed size distribution, cancels). A bin deviates when its median
   differs from the reference by more than ``sat_sigma`` × its uncertainty (1.2533 × width /
   √n) **and** by at least ``min_shift`` × width (a real turn grows sizes by many widths; this
   keeps a very tight locus from turning tiny trends into a turn). The turn is the faint edge
   of the run of deviating bins (same sign) that extends to the bright end, at least
   ``sat_bins`` bins long; on a flat faint end this is the first departure seen scanning from
   faint to bright. Bins with fewer than ``min_bin_count`` stars are skipped. If the turn
   would mark more than half of the usable stars, it is not applied (``bright_limit_source``
   ``'unclear'``, a warning): saturation affects the bright tail. ``bright_limit``
   (magnitude) overrides the detection. The center is refitted without the saturated stars.
5. ``'off_locus'``: |residual| > ``k`` × width (k = 3 keeps 99.7 % of a Gaussian locus:
   removes binaries, blends and other outliers without trimming the PSF size distribution).
6. ``'selected'``: the rest.

Sizes are those of `psf_fomo.datafit.measure` (T [arcsec²] or σ [arcsec] of any algorithm, or the
catalog ``size_cat``); the default ``T_wmom_0.4`` uses a fixed weight, which gives a stable
size on native stamps (that wmom sizes depend on the weight does not matter here).
"""
import warnings

import numpy as np
from astropy.table import Table

from .catalog import PIFF_ID

N_PIX = 4088              # SCA side, pixels (positions normalized to [-1, 1])
REASONS = ("selected", "saturated", "off_locus", "faint", "flagged")
MAX_ITER = 20
MAX_SATURATED_FRACTION = 0.5   # a detected turn marking more usable stars is not applied


class LocusResult:
    """Outcome of `select_locus`.

    Attributes
    ----------
    table : astropy.table.Table
        One row per star of the input: ``id``, ``x``, ``y``, ``mag``, ``size``, ``size_ref``
        (locus center at the star's position, same units as ``size``), ``resid`` (size /
        size_ref − 1), ``reason``.
    summary : dict
        ``n_<reason>`` counts, ``bright_limit`` (mag or None) and ``bright_limit_source``
        (``'detected'``, ``'given'``, ``'none'``, ``'unclear'``: a turn was found but would
        mark more than half of the usable stars, so it was not applied), ``detected_limit``
        (the scan's turn, also when not applied), ``scan_reference`` (reference level of the
        scan), ``width`` (robust σ of the fractional residual), ``k``, ``order``,
        ``coefficients`` (of log size), ``size``, ``mag``, ``off_locus_fraction`` (off-locus /
        (selected + off-locus)), ``n_iter``.
    bins : astropy.table.Table
        The saturation scan: ``mag_lo``, ``mag_hi``, ``n``, ``median``, ``shift`` (median −
        reference), ``uncertainty``, ``deviates``, ``used``.
    """

    def __init__(self, table, summary, bins):
        self.table = table
        self.summary = summary
        self.bins = bins

    @property
    def selected_ids(self):
        """Ids of the selected stars (list of int)."""
        return [int(i) for i in self.table[PIFF_ID][self.table["reason"] == "selected"]]

    def plot(self):
        """Size–magnitude, residual–magnitude with the saturation scan, residual map."""
        from .plotting import plot_locus
        return plot_locus(self)


def select_locus(table, size="T_wmom_0.4", order=1, k=3.0, mag="mag", mag_max=None,
                 bright_limit=None, mag_bin=0.25, sat_sigma=3.0, sat_bins=2,
                 min_bin_count=3, min_shift=1.0):
    """Select the stellar-locus stars of a pre-fit measurement table. Experimental.

    Parameters
    ----------
    table : astropy.table.Table
        `StarCatalog.measure` output (``id``, ``x``, ``y``, the size and magnitude columns,
        ``flag_<alg>``).
    size : str
        Size column: ``T_<alg>`` [arcsec²], ``sigma_<alg>`` [arcsec] or ``size_cat``. Default
        ``'T_wmom_0.4'`` (measure with ``algorithms=("wmom",)``).
    order : {0, 1, 2}
        Polynomial order of the locus center across the SCA. Default 1 (a plane).
    k : float
        Band half-width in robust σ of the fractional size residual. Default 3.
    mag : str
        Magnitude column: ``'mag'`` (catalog, default) or ``'mag_<alg>'``.
    mag_max : float, optional
        Faint limit (stars fainter are ``'faint'``). Default None.
    bright_limit : float, optional
        Saturation magnitude set by hand (stars brighter are ``'saturated'``); default None =
        detected from the data.
    mag_bin : float
        Magnitude bin of the saturation scan. Default 0.25.
    sat_sigma : float
        A bin deviates when its median differs from the reference by more than
        ``sat_sigma`` × its uncertainty (and ``min_shift``). Default 3.
    sat_bins : int
        Minimum length of the deviating run at the bright end. Default 2.
    min_bin_count : int
        Bins with fewer stars are skipped in the scan. Default 3.
    min_shift : float
        ... and by at least ``min_shift`` × the locus width. Default 1; 0 turns it off.

    Returns
    -------
    LocusResult

    Example
    -------
    >>> t = cat.measure(img, algorithms=("wmom",))
    >>> res = select_locus(t)            # or cat.select_locus(t) to use it for fits
    >>> res.summary["bright_limit"], res.summary["n_selected"]
    >>> res.plot()
    """
    if order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    if k <= 0:
        raise ValueError("k must be positive")
    for name in (size, mag, "x", "y", PIFF_ID):
        if name not in table.colnames:
            raise ValueError(f"column {name!r} not in the table (have: {table.colnames})")
    s = np.asarray(table[size], dtype=float)
    m = np.asarray(table[mag], dtype=float)
    x, y = np.asarray(table["x"], dtype=float), np.asarray(table["y"], dtype=float)
    flag_col = _flag_column(size)
    flag = (np.asarray(table[flag_col]) if flag_col in table.colnames
            else np.zeros(len(table), dtype=int))
    reason = np.full(len(table), "selected", dtype=object)
    ok = (flag == 0) & np.isfinite(s) & (s > 0) & np.isfinite(m)
    reason[~ok] = "flagged"
    if mag_max is not None:
        reason[ok & (m > mag_max)] = "faint"
    work = reason == "selected"
    basis = _basis(x, y, order)
    if work.sum() <= basis.shape[1]:
        raise ValueError(f"only {int(work.sum())} usable stars for a locus of order {order}")

    coeff, width, n_iter = _robust_fit(basis, np.log(np.where(s > 0, s, 1.0)), work, k)
    resid = s / np.exp(basis @ coeff) - 1
    bins, reference, detected = _saturation_scan(m[work], resid[work], width, mag_bin,
                                                 sat_sigma, sat_bins, min_bin_count, min_shift)
    if bright_limit is not None:
        limit, source = float(bright_limit), "given"
    elif detected is None:
        limit, source = None, "none"
    elif np.mean(m[work] < detected) > MAX_SATURATED_FRACTION:
        warnings.warn(f"the saturation scan found a turn at {detected:.2f} mag that would mark "
                      f"{np.mean(m[work] < detected):.0%} of the usable stars as saturated: not "
                      "applied (look at the residual–magnitude panel; set bright_limit by "
                      "hand if there is a turn)")
        limit, source = None, "unclear"
    else:
        limit, source = detected, "detected"
    if limit is not None:
        saturated = work & (m < limit)
        reason[saturated] = "saturated"
        work &= ~saturated
        if work.sum() <= basis.shape[1]:
            raise ValueError("too few stars fainter than the saturation limit "
                             f"({limit:.2f} mag) to fit the locus")
        coeff, width, n_iter = _robust_fit(basis, np.log(np.where(s > 0, s, 1.0)), work, k)
    size_ref = np.exp(basis @ coeff)
    resid = s / size_ref - 1
    reason[work & (np.abs(resid) > k * width)] = "off_locus"

    out = Table({PIFF_ID: table[PIFF_ID], "x": x, "y": y, "mag": m, "size": s,
                 "size_ref": size_ref, "resid": resid,
                 "reason": np.array(reason, dtype=str)})
    counts = {f"n_{r}": int(np.sum(out["reason"] == r)) for r in REASONS}
    n_locus = counts["n_selected"] + counts["n_off_locus"]
    summary = dict(counts, bright_limit=None if limit is None else float(limit),
                   bright_limit_source=source,
                   detected_limit=None if detected is None else float(detected),
                   scan_reference=float(reference), width=float(width), k=float(k),
                   order=int(order), coefficients=[float(c) for c in coeff], size=size,
                   mag=mag, n_iter=int(n_iter),
                   off_locus_fraction=(counts["n_off_locus"] / n_locus if n_locus else 0.0))
    return LocusResult(out, summary, bins)


def _flag_column(size):
    """``flag_<alg>`` for a measured size column (``T_<alg>``, ``sigma_<alg>``)."""
    for prefix in ("T_", "sigma_"):
        if size.startswith(prefix):
            return "flag_" + size[len(prefix):]
    return None


def _basis(x, y, order):
    """Polynomial terms x̂^i ŷ^j (i + j ≤ order) of positions normalized to [-1, 1]."""
    xn = 2 * (x - 1) / (N_PIX - 1) - 1
    yn = 2 * (y - 1) / (N_PIX - 1) - 1
    terms = [xn ** i * yn ** j for i in range(order + 1) for j in range(order + 1 - i)]
    return np.column_stack(terms)


def _robust_fit(basis, logs, use, k):
    """Least-squares fit of log size on ``use``, clipping at k × width until stable.

    The width is the robust σ (1.4826 × MAD) of the residuals of *all* ``use`` stars, not of
    the clipped members: the MAD already resists outliers, and a width from clipped members
    would shrink at every pass for bands narrower than ≈ 2σ. Clipping only protects the fit
    of the center, and is centered on the median residual: a first least-squares fit over all
    stars can be dragged off a tight locus by a few outliers (e.g. galaxies), so that every
    star would be clipped around zero.

    Returns (coefficients, width of the fractional residual, iterations).

    Raises
    ------
    ValueError
        If clipping leaves fewer stars than the polynomial has terms.
    """
    members = use.copy()
    for n_iter in range(1, MAX_ITER + 1):
        coeff = np.linalg.lstsq(basis[members], logs[members], rcond=None)[0]
        resid = np.exp(logs - basis @ coeff) - 1
        r = resid[use]
        center = np.median(r)
        width = max(1.4826 * np.median(np.abs(r - center)), 1e-6)
        new = use & (np.abs(resid - center) <= k * width)
        if new.sum() <= basis.shape[1]:
            raise ValueError(f"the locus fit kept only {int(new.sum())} stars after clipping "
                             f"at {k:g} × width ({width:.4f}); the size distribution has no "
                             "clear locus")
        if np.array_equal(new, members):
            break
        members = new
    return coeff, float(width), n_iter


def _saturation_scan(m, resid, width, mag_bin, sat_sigma, sat_bins, min_bin_count, min_shift):
    """Binned median residual vs magnitude, the reference level and the turn (or None).

    Returns (bins table, reference, faint edge of the bright-end deviating run or None).
    """
    lo = np.floor(np.min(m) / mag_bin) * mag_bin
    hi = np.ceil(np.max(m) / mag_bin) * mag_bin
    edges = np.arange(lo, hi + mag_bin / 2, mag_bin)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        r = resid[(m >= a) & (m < b)]
        rows.append({"mag_lo": float(a), "mag_hi": float(b), "n": len(r),
                     "median": float(np.median(r)) if len(r) else np.nan,
                     "uncertainty": (float(1.2533 * width / np.sqrt(len(r))) if len(r)
                                     else np.nan),
                     "used": len(r) >= min_bin_count})
    bins = Table(rows=rows)
    used = np.asarray(bins["used"], dtype=bool)
    # The reference is the typical bin, not zero: the scan looks for a change with magnitude.
    reference = float(np.median(np.asarray(bins["median"])[used])) if used.any() else 0.0
    shift = np.asarray(bins["median"]) - reference
    threshold = np.maximum(sat_sigma * np.asarray(bins["uncertainty"]), min_shift * width)
    bins["shift"] = shift
    bins["deviates"] = used & (np.abs(shift) > threshold)
    # Walk from the bright end: the run of deviating bins (same sign) the saturated stars make.
    run, sign, last = 0, 0, None
    for row in bins:
        if not row["used"]:
            continue  # too few stars: neither ends nor counts towards the run
        if not row["deviates"] or (sign and np.sign(row["shift"]) != sign):
            break
        sign = np.sign(row["shift"])
        run += 1
        last = row["mag_hi"]
    return bins, reference, (float(last) if run >= sat_bins else None)
