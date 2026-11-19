"""Moment residuals of a fitted PSF: model vs data, model vs truth, and model alone.

Four comparisons, each measured with any of the `psf_fomo.moments` algorithms (Piff's hsm,
am, gauss, wmom; conventions in `psf_fomo.moments.measure`: T = Mxx + Myy in native pix²,
distortion e, world axes):

- ``'data'`` (native): the noisy data stamp vs the model drawn on the same stamp at the fitted
  flux and centre (what Piff's residuals are made of). Both are measured about the catalog
  position, which wmom needs (it has no centroid fit).
- ``'truth_native'``: the noiseless truth stamp (`Simulation.render_stamp`) vs the model drawn
  at the true position (centre offset 0) with the true flux. Same weight. Isolates model error
  from noise. The truth stamp has no detector effects (IPC, ...), the data may.
- ``'truth_os'``: the truth PSF on a grid ``oversample`` × finer (`Simulation.truth_psf`) vs the
  model on the same grid. With ``include_pixel=True`` (default) both are the pixel-convolved
  effective PSF: a model drawn with method ``'auto'`` is convolved with the native pixel first,
  a ``'no_pixel'`` model (e.g. a Sum with a PixelGrid) already contains it. HSM works here.
- ``'model_os'``: the model alone on an oversampled grid (same convention as ``'truth_os'``),
  centred, unit flux, at each star. Needs no truth, so it works on real data; the ``*_ref``
  columns and the residuals are NaN. Not in the default set.

``'truth_native'`` and ``'truth_os'`` need the simulation; ``'data'`` and ``'model_os'`` do
not. `model_moments` gives the oversampled model moments at any SCA positions (field maps)
and `compare_fits` the model − model differences between fits at the same positions.

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
from astropy.table import Table, vstack

from ..moments.compare import summarize
from ..moments.measure import (ALGORITHMS, DEFAULT_WMOM_FWHM, algorithm_label, make_star,
                               measure_star)

COMPARISONS = ("data", "truth_native", "truth_os")  # the default set for simulations
TRUTH_COMPARISONS = ("truth_native", "truth_os")
ALL_COMPARISONS = COMPARISONS + ("model_os",)
FLAG_PIFF = 16
MOMENT_KEYS = ("T", "e1", "e2", "g1", "g2", "sigma", "du", "dv", "flux")
BASE_COLUMNS = ("obs", "sca", "star_id", "x", "y", "phase_x", "phase_y", "mag", "reserve")
MODEL_OS_SIZE = 25  # native pixels of the model_moments grid when there is no stamp
NO_REFERENCE = {**{key: np.nan for key in MOMENT_KEYS}, "flag": 0}
NOT_DRAWN = {**{key: np.nan for key in MOMENT_KEYS}, "flag": 8}   # measurement flag "exception"


def drawable(star):
    """False for a star Piff left with a non-finite flux or centre (it cannot be drawn)."""
    return bool(np.isfinite(star.fit.flux) and np.all(np.isfinite(star.fit.center)))


def moment_residuals(psf, stars, table, sim=None, native=("wmom", "gauss"),
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
        Star table (``flag``, ``reserve`` and, for the truth comparisons, ``obs``, ``sca``,
        ``star_id``, ``flux_true``). Identifier columns among ``obs``, ``sca``, ``star_id``,
        ``x``, ``y``, ``phase_x``, ``phase_y``, ``mag`` are copied when present.
    sim : Simulation, optional
        The simulation (truth stamps and oversampled truth). Required for ``'truth_native'``
        and ``'truth_os'`` only.
    native : sequence of str
        Algorithms for the native comparisons (``'data'``, ``'truth_native'``). Default
        ('wmom', 'gauss'); hsm/am can fail on undersampled stamps (flagged).
    wmom_fwhm : float or sequence of float
        wmom weight FWHM(s), arcsec; each gives its own algorithm label ``wmom_<fwhm>``.
        Default (0.4,).
    oversampled : sequence of str
        Algorithms for ``'truth_os'`` and ``'model_os'``. Default ('hsm',).
    oversample : int
        Grid refinement of ``'truth_os'`` and ``'model_os'``. Default 5.
    include_pixel : bool
        Oversampled images are the effective (pixel-convolved) PSF. Default True. False needs
        a model without a pixel-including component (no PixelGrid).
    comparisons : sequence of str
        Subset of ``ALL_COMPARISONS`` (``'data'``, ``'truth_native'``, ``'truth_os'``,
        ``'model_os'``). Default ``COMPARISONS`` (the first three).

    Returns
    -------
    astropy.table.Table
        Columns: the identifier columns, ``reserve``, ``comparison``, ``algorithm``,
        ``<key>_ref`` and ``<key>_model`` for key in T [pix²], e1, e2, g1, g2, sigma [pix],
        du, dv [arcsec], flux; ``dT_T``, ``de1``, ``de2``, ``flag_ref``, ``flag_model``,
        ``flag``. For ``'model_os'`` the ``*_ref`` columns and residuals are NaN.
    """
    unknown = set(comparisons) - set(ALL_COMPARISONS)
    if unknown:
        raise ValueError(f"unknown comparisons {sorted(unknown)}; allowed: {ALL_COMPARISONS}")
    if sim is None and set(comparisons) & set(TRUTH_COMPARISONS):
        raise ValueError(f"comparisons {TRUTH_COMPARISONS} need the simulation (sim=...)")
    for alg in list(native) + list(oversampled):
        if alg not in ALGORITHMS:
            raise ValueError(f"algorithm {alg!r} not in {ALGORITHMS}")
    if len(stars) != len(table):
        raise ValueError("stars and table have different lengths")
    native_algs = _expand(native, wmom_fwhm)
    os_algs = _expand(oversampled, wmom_fwhm)
    base_columns = [k for k in BASE_COLUMNS if k in table.colnames]
    rows = []
    for star, row in zip(stars, table):
        base = {k: row[k] for k in base_columns}
        piff_flag = FLAG_PIFF if row["flag"] else 0
        if not drawable(star):
            # Piff excluded the star with a non-finite flux or centre: nothing to draw.
            for comparison in comparisons:
                algs = native_algs if comparison in ("data", "truth_native") else os_algs
                for alg, fwhm in algs:
                    rows.append(_row(base, comparison, algorithm_label(alg, fwhm),
                                     NO_REFERENCE, NOT_DRAWN, piff_flag))
            continue
        for comparison, (ref, model) in _image_pairs(psf, star, row, sim, comparisons,
                                                     oversample, include_pixel):
            algs = native_algs if comparison in ("data", "truth_native") else os_algs
            for alg, fwhm in algs:
                m_ref = NO_REFERENCE if ref is None else measure_star(ref, alg, fwhm)
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
        # Fresh stars for both sides: Piff measures about star.fit.center, which is the
        # fitted centre on Piff's star but zero (the catalog position) on a new one.
        yield "data", (make_star(star.image, star.image_pos, star.weight),
                       make_star(model_image, star.image_pos, star.weight))
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
    if "model_os" in comparisons:
        size = max(star.image.array.shape)
        model_image = model_os_image(psf, star, oversample, include_pixel, size)
        yield "model_os", (None, make_star(model_image, model_image.true_center))


def model_os_image(psf, star, oversample=5, include_pixel=True, size=MODEL_OS_SIZE,
                   sed_eff=None):
    """The fitted model of one star alone on a grid ``oversample`` × finer, centred.

    Same convention as `Simulation.truth_psf`: a ``size × oversample`` square image whose WCS
    is the star's local WCS scaled by 1/oversample; unit model flux.

    Parameters
    ----------
    psf : piff.PSF
        Fitted PSF.
    star : piff.Star
        Star (or target) with interpolated fit parameters.
    oversample : int
        Fine pixels per native pixel. Default 5.
    include_pixel : bool
        Effective PSF (True, default) or pre-pixel PSF (see `draw_model_oversampled`).
    size : int
        Size in native pixels. Default 25.
    sed_eff : galsim.SED, optional
        For a chromatic model, when the star has no SED (see `draw_model_oversampled`).

    Returns
    -------
    galsim.Image
    """
    local = star.data.local_wcs
    fine = galsim.JacobianWCS(*(local.jacobian().getMatrix().ravel() / oversample))
    template = galsim.ImageD(size * oversample, size * oversample, wcs=fine)
    return draw_model_oversampled(psf, star, template, include_pixel, sed_eff)


def model_moments(psf, x, y, sca, algorithms=("hsm",), wmom_fwhm=(DEFAULT_WMOM_FWHM,),
                  oversample=5, include_pixel=True, size=MODEL_OS_SIZE, sed_eff=None):
    """Oversampled moments of the fitted model at any positions on an SCA (no data needed).

    Parameters
    ----------
    psf : piff.PSF
        Fitted PSF (with its context: ``psf.wcs`` keyed by SCA, ``psf.pointing``).
    x, y : float or array of float
        Positions in native pixels (GalSim image coordinates of the fitted image).
    sca : int
        SCA (chipnum) of the fit.
    algorithms : sequence of str
        Any of ``'hsm'``, ``'am'``, ``'gauss'``, ``'wmom'``. Default ('hsm',).
    wmom_fwhm : float or sequence of float
        wmom weight FWHM(s), arcsec. Default (0.4,).
    oversample : int
        Grid refinement. Default 5.
    include_pixel : bool
        Effective PSF (True, default) or pre-pixel PSF.
    size : int
        Image size in native pixels. Default 25.
    sed_eff : galsim.SED, optional
        For a chromatic model: the effective SED drawn at every position. Default: that of
        the fit's stars (`fit_sed`).

    Returns
    -------
    astropy.table.Table
        One row per position × algorithm: ``x``, ``y``, ``algorithm``, T [pix²], e1, e2, g1,
        g2, sigma [pix], du, dv [arcsec], flux, ``flag``.
    """
    for alg in algorithms:
        if alg not in ALGORITHMS:
            raise ValueError(f"algorithm {alg!r} not in {ALGORITHMS}")
    if sed_eff is None:
        sed_eff = fit_sed(psf)
    x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    algs = _expand(algorithms, wmom_fwhm)
    rows = []
    for xi, yi in zip(x, y):
        target = piff.Star.makeTarget(x=xi, y=yi, wcs=psf.wcs[sca], pointing=psf.pointing,
                                      properties={"sca": sca, "chipnum": sca})
        target = psf.interpolateStar(target)
        image = model_os_image(psf, target, oversample, include_pixel, size, sed_eff)
        star = make_star(image, image.true_center)
        for alg, fwhm in algs:
            m = measure_star(star, alg, fwhm)
            rows.append({"x": xi, "y": yi, "algorithm": algorithm_label(alg, fwhm),
                         **{key: m[key] for key in MOMENT_KEYS}, "flag": m["flag"]})
    return Table(rows=rows)


def sca_grid(n_grid=5, n_pix=4088):
    """Centres of an ``n_grid`` × ``n_grid`` grid of cells covering an SCA.

    Returns flattened (x, y) arrays in GalSim image coordinates (pixel centres 1 … n_pix).
    """
    centres = (np.arange(n_grid) + 0.5) * n_pix / n_grid + 0.5
    x, y = np.meshgrid(centres, centres)
    return x.ravel(), y.ravel()


def compare_fits(results, x=None, y=None, algorithms=("hsm",), wmom_fwhm=(DEFAULT_WMOM_FWHM,),
                 oversample=5, include_pixel=True, size=MODEL_OS_SIZE):
    """Oversampled model moments of several fits at the same positions, and their differences.

    Parameters
    ----------
    results : dict
        ``{label: result}``; any result with ``.psf``, ``.config['sca']`` and ``.table``
        (simulation `FitResult` or real-data result). The first entry is the reference. All
        fits must be of the same SCA.
    x, y : array of float, optional
        Positions in native pixels. Default: the unflagged stars of the first fit.
    algorithms, wmom_fwhm, oversample, include_pixel, size
        As `model_moments`.

    Returns
    -------
    astropy.table.Table
        One row per fit × position × algorithm: ``fit``, the `model_moments` columns,
        ``dT_T`` = (T − T_first)/T_first, ``de1``, ``de2`` (fit − first fit; 0 for the first),
        and ``flag`` (OR of this fit's and the first fit's measurement flags).
    """
    if not results:
        raise ValueError("results is empty")
    labels = list(results)
    first = results[labels[0]]
    sca = first.config["sca"]
    if any(r.config["sca"] != sca for r in results.values()):
        raise ValueError("all fits must be of the same SCA")
    if x is None or y is None:
        good = first.table[first.table["flag"] == 0]
        x, y = np.asarray(good["x"], dtype=float), np.asarray(good["y"], dtype=float)
    tables = {label: model_moments(r.psf, x, y, sca, algorithms, wmom_fwhm, oversample,
                                   include_pixel, size,
                                   sed_eff=fit_sed(r.psf, getattr(r, "_stars", None) or []))
              for label, r in results.items()}
    ref = tables[labels[0]]
    out = []
    for label, t in tables.items():
        t = t.copy()
        t.add_column(label, name="fit", index=0)
        t["dT_T"] = (t["T"] - ref["T"]) / ref["T"]
        t["de1"] = t["e1"] - ref["e1"]
        t["de2"] = t["e2"] - ref["e2"]
        t["flag"] = t["flag"] | ref["flag"]
        out.append(t)
    return vstack(out)


def draw_model_oversampled(psf, star, template, include_pixel=True, sed_eff=None):
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
    sed_eff : galsim.SED, optional
        For a chromatic model: the effective SED (SED × bandpass) to draw with when the star
        has none (e.g. a target at an arbitrary position). Default: the star's own.

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
    own = star.data.sed_eff
    draw_profile(prof, image, psf, own if own is not None else sed_eff)
    return image


def draw_profile(prof, image, psf, sed_eff=None):
    """Draw a model profile without adding a pixel, centred on ``image``.

    A chromatic profile (Piff's chromatic RomanOptics) is drawn as Piff does: times the
    effective SED (SED × bandpass, unit flux), through a flat bandpass over the filter's range.
    """
    if isinstance(prof, galsim.ChromaticObject):
        if sed_eff is None:
            raise ValueError("a chromatic model needs an SED to be drawn (sed_eff)")
        (prof * sed_eff).drawImage(psf.flat_bandpass, image=image, method="no_pixel")
    else:
        prof.drawImage(image=image, method="no_pixel")


def fit_sed(psf, stars=()):
    """The effective SED of the fit's stars (the first one that has one), or None.

    Used to draw a chromatic model where there is no star (field maps): in this package all
    stars of a fit share one SED (the simulation's, or the real-data SED).
    """
    for star in list(stars) + list(getattr(psf, "stars", None) or []):
        if star.data.sed_eff is not None:
            return star.data.sed_eff
    return None


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
