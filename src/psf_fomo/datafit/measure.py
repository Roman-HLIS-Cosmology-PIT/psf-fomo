"""Moments of every detection before any PSF fit (`measure_objects`, `StarCatalog.measure`).

For examining an image and its catalog before Piff runs: size–magnitude diagrams with any of
the moment algorithms, and optionally a preview of Piff's star selection.

The stamps are the ones a fit would use: Piff's own Input cuts them (same stamp size, sky
subtraction, weights and bad pixels) from a temporary copy of the Piff catalog. Each stamp is
measured with `psf_fomo.moments.measure.measure_star` about the catalog position (conventions
of `psf_fomo.moments`: T = Mxx + Myy in native pix², σ = det(M)^¼ in pix, distortion e, world
axes; flags 1 failure, 2 negative flux, 4 centroid > 1″, 8 exception).

Algorithms on native (undersampled) stamps (CLAUDE.md §9.8): ``gauss`` fits an elliptical
Gaussian (stars form a tight locus); ``wmom`` reports raw moments with a fixed round weight of
the given FWHM, so every size is compressed toward the weight's (stars and galaxies still
separate); ``hsm`` and ``am`` fail on most native stamps.
"""
import copy
import tempfile
import warnings
from pathlib import Path

import numpy as np
import piff
from astropy.table import Table

from ..moments.measure import ALGORITHMS, DEFAULT_WMOM_FWHM, algorithm_label, measure_star
from ..piffval.fitting import make_logger, piff_shape_method
from .catalog import PIFF_ID

CATALOG_COLUMNS = ("x", "y", "flux_cat", "mag", "size_cat", "clean")
MEASURED_KEYS = ("T", "sigma", "e1", "e2", "flux")


def measure_objects(image, catalog, algorithms=("gauss",), wmom_fwhm=(DEFAULT_WMOM_FWHM,),
                    stamp_size=25, mag_range=None, max_objects=None, preview_selection=False,
                    verbose=0):
    """Measure every detection of a catalog on its image, before any fit.

    Parameters
    ----------
    image : L2Image
        The exposure (variant, weights and sky as set on it).
    catalog : StarCatalog
        The detections (its selection and cuts are used by ``preview_selection``).
    algorithms : sequence of str
        Any of ``'hsm'``, ``'am'``, ``'gauss'``, ``'wmom'``. Default ('gauss',).
    wmom_fwhm : float or sequence of float
        wmom weight FWHM(s), arcsec; each gives its own label ``wmom_<fwhm>``. Default (0.4,).
    stamp_size : int
        Stamp side, pixels (as `DataFit.set_stamp_size`). Default 25.
    mag_range : (float, float), optional
        Measure only detections with catalog ``mag`` in this range (needs ``ABZP``).
    max_objects : int, optional
        Measure at most this many detections, the brightest by catalog flux.
    preview_selection : bool
        Also run the catalog's Piff selection (selection type, then the cuts; no reserve, no
        fit) and add ``candidate`` and ``star`` columns. With ``mag_range`` / ``max_objects``
        the selection sees only that subset (fewer neighbours for ``min_sep``, a different
        size–magnitude locus for SizeMag). Default False.
    verbose : int
        Piff log level, 0 (errors only) … 3.

    Returns
    -------
    astropy.table.Table
        One row per detection that got a stamp (inside the image, not fully masked):
        ``id``, catalog columns ``x``, ``y`` (1-based), ``flux_cat`` [DN/s], ``mag``,
        ``size_cat`` [pix], ``clean`` (those available), and per algorithm label ``<a>``
        (e.g. ``gauss``, ``wmom_0.4``): ``T_<a>`` [pix²], ``sigma_<a>`` [pix], ``e1_<a>``,
        ``e2_<a>``, ``flux_<a>`` [DN/s], ``mag_<a>`` (``ABZP`` − 2.5 log10 ``flux_<a>``;
        instrumental −2.5 log10 without ``ABZP``; NaN for flux ≤ 0), ``flag_<a>``; with
        ``preview_selection`` also ``candidate`` and ``star``. ``meta`` holds the settings
        and counts (``n_catalog``, ``n_measured``).

    Example
    -------
    >>> t = measure_objects(img, cat, algorithms=("gauss", "wmom"), preview_selection=True)
    >>> from psf_fomo.datafit.plotting import plot_size_mag
    >>> plot_size_mag(t, algorithm="gauss")
    """
    unknown = [a for a in algorithms if a not in ALGORITHMS]
    if unknown:
        raise ValueError(f"algorithms {unknown} not in {ALGORITHMS}")
    labels = [(a, f) for a in algorithms
              for f in ([float(w) for w in np.atleast_1d(wmom_fwhm)] if a == "wmom" else [None])]
    # The original file in its own frame: moments use only the local WCS (world axes), so the
    # pixel frame does not matter here, and x, y stay those of the L2 file.
    piff_catalog = _subset(catalog.piff_table(image, stamp_size, fit_frame=False), mag_range,
                           max_objects)
    logger, _ = make_logger(verbose)
    with tempfile.TemporaryDirectory() as tmp:
        # Imported here: fit.py imports catalog.py, which calls this module.
        from .fit import piff_input_config
        catalog_file = Path(tmp) / "catalog.fits"
        piff_catalog.write(catalog_file)
        inp = piff_input_config(image, catalog, catalog_file, piff_catalog, stamp_size)
        base = {"input": copy.deepcopy(inp)}
        objects, _, _, _ = piff.Input.process(copy.deepcopy(inp), base, logger=logger)
        # Measure before the selection: Piff's max_snr_weight rescales weights in place.
        rows = [_measure(obj, labels) for obj in objects]
        if preview_selection:
            chosen = _preview(catalog, objects, base, logger)
    out = _join(piff_catalog, objects, rows, labels, catalog.header.get("ABZP"))
    if preview_selection:
        ids = np.asarray(out[PIFF_ID])
        for name, group in chosen.items():
            out[name] = np.isin(ids, list(group))
    out.meta.update(n_catalog=len(catalog.table), n_measured=len(out),
                    stamp_size=int(stamp_size), image=str(image.path),
                    algorithms=[algorithm_label(a, f) for a, f in labels])
    return out


def _subset(table, mag_range, max_objects):
    if mag_range is not None:
        if "mag" not in table.colnames:
            raise ValueError("mag_range needs catalog magnitudes (ABZP in the catalog header)")
        mag = np.asarray(table["mag"], dtype=float)
        table = table[(mag >= mag_range[0]) & (mag <= mag_range[1])]
    if max_objects is not None and len(table) > max_objects:
        if "flux_cat" in table.colnames:
            order = np.argsort(-np.nan_to_num(np.asarray(table["flux_cat"], dtype=float),
                                              nan=-np.inf))
            table = table[np.sort(order[:max_objects])]
        else:
            table = table[:max_objects]
    if len(table) == 0:
        raise ValueError("no detections left to measure")
    return table


def _measure(obj, labels):
    out = {}
    for alg, fwhm in labels:
        m = measure_star(obj, alg, fwhm)
        label = algorithm_label(alg, fwhm)
        for key in MEASURED_KEYS:
            out[f"{key}_{label}"] = m[key]
        out[f"flag_{label}"] = m["flag"]
    return out


def _preview(catalog, objects, base, logger):
    """Ids picked by the selection type (``candidate``) and kept by the cuts (``star``)."""
    from .fit import select_stars
    select = catalog.piff_select()
    try:
        with piff_shape_method(catalog.selection["shape_method"]):
            candidates, stars = select_stars(select, objects, dict(base, select=select), logger)
    except RuntimeError as err:  # Piff found no stars at one of the two stages
        warnings.warn(f"selection preview: {err}")
        return {"candidate": set(), "star": set()}
    return {"candidate": {int(s.data.properties[PIFF_ID]) for s in candidates},
            "star": {int(s.data.properties[PIFF_ID]) for s in stars}}


def _join(piff_catalog, objects, rows, labels, abzp):
    index = {int(i): k for k, i in enumerate(piff_catalog[PIFF_ID])}
    cat = piff_catalog[[index[int(o.data.properties[PIFF_ID])] for o in objects]]
    out = Table({PIFF_ID: cat[PIFF_ID]})
    for name in CATALOG_COLUMNS:
        if name in cat.colnames:
            out[name] = cat[name]
    zero = 0.0 if abzp is None else float(abzp)
    for alg, fwhm in labels:
        label = algorithm_label(alg, fwhm)
        for key in MEASURED_KEYS:
            out[f"{key}_{label}"] = np.array([r[f"{key}_{label}"] for r in rows], dtype=float)
        flux = np.asarray(out[f"flux_{label}"])
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"mag_{label}"] = np.where(flux > 0, zero - 2.5 * np.log10(flux), np.nan)
        out[f"flag_{label}"] = np.array([r[f"flag_{label}"] for r in rows], dtype=int)
        out[f"T_{label}"].unit = "pix2"
        out[f"sigma_{label}"].unit = "pix"
    return out
