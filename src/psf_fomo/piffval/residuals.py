"""Moment residuals of a fitted PSF: model vs data and model vs truth.

Three comparisons, each measured with any of the `psf_fomo.moments` algorithms (Piff's hsm,
am, gauss, wmom; conventions in `psf_fomo.moments.measure`: T = Mxx + Myy in native pix²,
distortion e, world axes):

- ``'data'`` (native): the noisy data stamp vs the model drawn on the same stamp at the fitted
  flux and centre (what Piff's residuals are made of). Both are measured about the catalog
  position, which wmom needs (it has no centroid fit; Phase 3).
- ``'truth_native'``: the noiseless truth stamp (`Simulation.render_stamp`) vs the model drawn
  at the true position (centre offset 0) with the true flux. Same weight. Isolates model error
  from noise. The truth stamp has no detector effects (IPC, ...), the data may.
- ``'truth_os'``: the truth PSF on a grid ``oversample`` × finer (`Simulation.truth_psf`) vs the
  model on the same grid. With ``include_pixel=True`` (default) both are the pixel-convolved
  effective PSF: a model drawn with method ``'auto'`` is convolved with the native pixel first,
  a ``'no_pixel'`` model (e.g. a Sum with a PixelGrid) already contains it. HSM works here.

Residuals: ``dT_T = (T_model − T_ref) / T_ref``, ``de1 = e1_model − e1_ref``, ``de2`` likewise
(ref = data or truth).

Flags: ``flag_ref`` and ``flag_model`` are the measurement flags (`psf_fomo.moments.measure`:
1 failure, 2 negative flux, 4 centroid > 1″, 8 exception); ``flag`` = their OR, plus
``FLAG_PIFF`` (16) if Piff flagged the star. Summaries use ``flag == 0`` only and report
``n_flagged``.
"""
import numpy as np
import galsim
import piff
from astropy.table import Table

from ..moments.compare import summarize
from ..moments.measure import (ALGORITHMS, DEFAULT_WMOM_FWHM, algorithm_label, make_star,
                               measure_star)

COMPARISONS = ("data", "truth_native", "truth_os")
FLAG_PIFF = 16
MOMENT_KEYS = ("T", "e1", "e2", "g1", "g2", "sigma", "du", "dv", "flux")


def moment_residuals(psf, stars, table, sim, native=("wmom", "gauss"),
                     wmom_fwhm=(DEFAULT_WMOM_FWHM,), oversampled=("hsm",), oversample=5,
                     include_pixel=True, comparisons=COMPARISONS):
    """Tidy table of model and reference moments, one row per star × comparison × algorithm.

    Parameters
    ----------
    psf : piff.PSF
        Fitted PSF.
    stars : list of piff.Star
        Fitted stars with their data images (same order as ``table``).
    table : astropy.table.Table
        Star table from `diagnostics.star_table` (``obs``, ``sca``, ``star_id``,
        ``flux_true``, ``reserve``, ``flag``, …).
    sim : Simulation
        The simulation (truth stamps and oversampled truth).
    native : sequence of str
        Algorithms for the native comparisons (``'data'``, ``'truth_native'``). Default
        ('wmom', 'gauss'); hsm/am mostly fail on native H158 stamps (Phase 3).
    wmom_fwhm : float or sequence of float
        wmom weight FWHM(s), arcsec; each gives its own algorithm label ``wmom_<fwhm>``.
        Default (0.4,).
    oversampled : sequence of str
        Algorithms for ``'truth_os'``. Default ('hsm',).
    oversample : int
        Grid refinement of ``'truth_os'``. Default 5.
    include_pixel : bool
        ``'truth_os'`` on the effective (pixel-convolved) PSF. Default True. False needs a
        model without a pixel-including component (no PixelGrid).
    comparisons : sequence of str
        Subset of ``COMPARISONS``. Default all.

    Returns
    -------
    astropy.table.Table
        Columns: ``obs``, ``sca``, ``star_id``, ``x``, ``y``, ``phase_x``, ``phase_y``,
        ``mag``, ``reserve``, ``comparison``, ``algorithm``, ``<key>_ref`` and
        ``<key>_model`` for key in T [pix²], e1, e2, g1, g2, sigma [pix], du, dv [arcsec],
        flux; ``dT_T``, ``de1``, ``de2``, ``flag_ref``, ``flag_model``, ``flag``.
    """
    unknown = set(comparisons) - set(COMPARISONS)
    if unknown:
        raise ValueError(f"unknown comparisons {sorted(unknown)}; allowed: {COMPARISONS}")
    for alg in list(native) + list(oversampled):
        if alg not in ALGORITHMS:
            raise ValueError(f"algorithm {alg!r} not in {ALGORITHMS}")
    if len(stars) != len(table):
        raise ValueError("stars and table have different lengths")
    native_algs = _expand(native, wmom_fwhm)
    os_algs = _expand(oversampled, wmom_fwhm)
    rows = []
    for star, row in zip(stars, table):
        base = {k: row[k] for k in ("obs", "sca", "star_id", "x", "y", "phase_x", "phase_y",
                                    "mag", "reserve")}
        piff_flag = FLAG_PIFF if row["flag"] else 0
        for comparison, (ref, model) in _image_pairs(psf, star, row, sim, comparisons,
                                                     oversample, include_pixel):
            algs = os_algs if comparison == "truth_os" else native_algs
            for alg, fwhm in algs:
                m_ref = measure_star(ref, alg, fwhm)
                m_mod = measure_star(model, alg, fwhm)
                rows.append(_row(base, comparison, algorithm_label(alg, fwhm), m_ref, m_mod,
                                 piff_flag))
    return Table(rows=rows)


def _expand(algorithms, wmom_fwhm):
    """(algorithm, fwhm) pairs: one entry per wmom FWHM, fwhm None for the others."""
    fwhms = [float(f) for f in np.atleast_1d(wmom_fwhm)]
    out = []
    for alg in algorithms:
        out += [(alg, f) for f in fwhms] if alg == "wmom" else [(alg, None)]
    return out


def _image_pairs(psf, star, row, sim, comparisons, oversample, include_pixel):
    """Yield (comparison, (reference star, model star)) for one star."""
    if "data" in comparisons:
        model_image = psf.drawStar(star).image
        yield "data", (star, make_star(model_image, star.image_pos, star.weight))
    if "truth_native" in comparisons:
        # Render a little larger and cut to the star's own stamp: a file-based Piff run cuts
        # its stamps itself, possibly centred differently from Simulation.stars.
        size = max(star.image.array.shape) + 4
        truth_image = sim.render_stamp(int(row["star_id"]), int(row["obs"]), size=size)
        truth_image = truth_image[star.image.bounds]
        at_truth = star.fit.withNew(flux=float(row["flux_true"]), center=(0.0, 0.0))
        model_image = psf.drawStar(piff.Star(star.data, at_truth)).image
        yield "truth_native", (make_star(truth_image, star.image_pos, star.weight),
                               make_star(model_image, star.image_pos, star.weight))
    if "truth_os" in comparisons:
        truth_image = sim.truth_psf(int(row["sca"]), int(row["obs"]), int(row["star_id"]),
                                    oversample=oversample, include_pixel=include_pixel)
        model_image = draw_model_oversampled(psf, star, truth_image, include_pixel)
        yield "truth_os", (make_star(truth_image, truth_image.true_center),
                           make_star(model_image, model_image.true_center))


def draw_model_oversampled(psf, star, template, include_pixel=True):
    """The fitted model of one star on the grid of ``template``, centred, unit model flux.

    Parameters
    ----------
    psf : piff.PSF
        Fitted PSF.
    star : piff.Star
        Star with interpolated fit parameters (its flux and centre are not used).
    template : galsim.Image
        Image whose bounds and WCS (e.g. from `Simulation.truth_psf`) are used.
    include_pixel : bool
        True: the effective PSF (native pixel included once). False: the pre-pixel PSF,
        impossible when a component already includes the pixel (raises ValueError).

    Returns
    -------
    galsim.Image
    """
    prof, method = psf._getRawProfile(star)
    if method == "auto" and include_pixel:
        prof = galsim.Convolve(prof, star.data.local_wcs.toWorld(galsim.Pixel(1.0)))
    elif method == "no_pixel" and not include_pixel:
        raise ValueError("the model includes the pixel response (e.g. a PixelGrid): "
                         "no pre-pixel image possible")
    image = template.copy()
    image.setZero()
    prof.drawImage(image=image, method="no_pixel")  # centred on the image, as truth_psf
    return image


def _row(base, comparison, algorithm, m_ref, m_mod, piff_flag):
    out = dict(base, comparison=comparison, algorithm=algorithm)
    for key in MOMENT_KEYS:
        out[key + "_ref"] = m_ref[key]
        out[key + "_model"] = m_mod[key]
    out["dT_T"] = (m_mod["T"] - m_ref["T"]) / m_ref["T"]
    out["de1"] = m_mod["e1"] - m_ref["e1"]
    out["de2"] = m_mod["e2"] - m_ref["e2"]
    out["flag_ref"] = m_ref["flag"]
    out["flag_model"] = m_mod["flag"]
    out["flag"] = m_ref["flag"] | m_mod["flag"] | piff_flag
    return out


def moment_summary(moments, columns=("dT_T", "de1", "de2")):
    """Mean, std, error on the mean, n and n_flagged per (set, comparison, algorithm).

    ``set`` is 'fit' or 'reserve'.
    """
    moments = moments.copy()
    moments["set"] = np.where(moments["reserve"], "reserve", "fit")
    return summarize(moments, ["set", "comparison", "algorithm"], list(columns))
