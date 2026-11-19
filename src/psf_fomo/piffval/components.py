"""Per-component summaries of a fitted Piff PSF, compared with the simulation truth.

With ``truth=False`` (real data) the tables hold the fitted values only.

- optics (``RomanOptics``): fitted extra Zernikes at each star vs the catalog truth (waves).
- kernel (``GaussianMoments``, Piff ``GaussianBlur`` or Piff ``Gaussian``): the kernel's
  second moments M at each star in native pix² (world axes) vs the truth kernel (catalog
  ``kernel_*``, 0 if none). Piff's ``Gaussian`` (scale σ in arcsec, reduced shear g) is
  converted with M = σ² A Aᵀ / pixel_scale², A = the shear matrix. Piff's ``GaussianBlur``
  ([ixx, iyy, ixy] arcsec²) reports the PSD projection it actually draws, converted to pix²
  (``clipped`` marks stars where that differs from the parameters).
- pixelgrid (``PixelGrid``): the grid's net flux in the unit model and its largest value.
"""
import numpy as np
import galsim
import galsim.roman
import piff
from astropy.table import Table

from ..sim.psf import gaussian_kernel
from .config import blur_moments

PIXEL_SCALE = galsim.roman.pixel_scale


def leaf_psfs(psf):
    """The leaf PSF components (RomanOptics, Simple) of a possibly composite PSF, in order."""
    if hasattr(psf, "components"):
        return [leaf for comp in psf.components for leaf in leaf_psfs(comp)]
    return [psf]


def kind_of(leaf):
    """'optics', 'kernel', 'pixelgrid', 'mtf' or 'other' for a leaf PSF component."""
    name = leaf.model._type_name
    return {"Roman": "optics", "GaussianMoments": "kernel", "Gaussian": "kernel",
            "GaussianBlur": "kernel", "PixelGrid": "pixelgrid",
            "GaussianMixture": "mtf"}.get(name, "other")


def components(psf, stars, rows, truth=True):
    """Summary tables of the fitted components.

    Parameters
    ----------
    psf : piff.PSF
        Fitted PSF.
    stars : list of piff.Star
        Fitted stars (interpolated parameters), same order as ``rows``.
    rows : astropy.table.Table
        Catalog rows of these stars: with ``truth``, simulation rows (``obs``, ``star_id``,
        ``z4`` …, ``kernel_*``); otherwise any table whose ``obs`` / ``star_id`` columns (if
        present) identify the stars.
    truth : bool
        Compare with the simulation truth (default True; a missing truth column means 0).
        False: fitted values only (real data).

    Returns
    -------
    dict
        ``'optics'``: one row per Noll index; ``'kernel'`` and ``'pixelgrid'``: one row per
        star. Only the components present are included (a second component of the same
        kind gets the key ``'kernel_2'`` etc.).
    """
    out = {}
    makers = {"optics": optics_table, "kernel": kernel_table, "pixelgrid": pixelgrid_table}
    for leaf in leaf_psfs(psf):
        kind = kind_of(leaf)
        if kind not in makers:
            continue
        key, i = kind, 2
        while key in out:
            key, i = f"{kind}_{i}", i + 1
        out[key] = (makers[kind](leaf, stars, rows) if kind == "pixelgrid"
                    else makers[kind](leaf, stars, rows, truth))
    return out


def _id_columns(rows):
    """The star-identifying columns (``obs``, ``star_id``) present in ``rows``."""
    return Table({k: rows[k] for k in ("obs", "star_id") if k in rows.colnames})


def optics_table(leaf, stars, rows, truth=True):
    """Fitted (and true) extra Zernikes, one row per fitted Noll index.

    Columns: ``noll``, ``fit_mean``, ``fit_std`` (over stars; nonzero only with a linear
    field or several SCAs), with ``truth``: ``truth_mean``, ``diff_mean``, ``diff_std``
    (fit − truth at each star); then ``fit_err`` (Piff's parameter σ, mean over stars),
    ``prior_sigma`` (NaN = none). Waves.
    """
    model = leaf.model
    n = model._single_point_param_len
    fitted, errors = [], []
    for star in stars:
        p = np.asarray(star.fit.get_params(leaf._num), dtype=float)
        var = star.fit.get_params_var(leaf._num)
        var = np.full(p.size, np.nan) if var is None else np.asarray(var, dtype=float)
        if model.aberration_interp == "linear":
            xh, yh = star.image_pos.x / model.sca_size, star.image_pos.y / model.sca_size
            a, b, c = p.reshape(3, n)
            p = a + b * xh + c * yh
            var = var.reshape(3, n)[0]  # σ of the constant term
        fitted.append(p)
        errors.append(np.sqrt(var))
    fitted, errors = np.array(fitted), np.array(errors)
    out = Table({"noll": np.arange(4, 4 + n),
                 "fit_mean": fitted.mean(axis=0), "fit_std": fitted.std(axis=0)})
    if truth:
        true = np.array([[float(row[f"z{k}"]) if f"z{k}" in rows.colnames else 0.0
                          for k in range(4, 4 + n)] for row in rows])
        diff = fitted - true
        out["truth_mean"] = true.mean(axis=0)
        out["diff_mean"] = diff.mean(axis=0)
        out["diff_std"] = diff.std(axis=0)
    prior = model.orig_prior_sigma
    out["fit_err"] = np.nanmean(errors, axis=0)
    out["prior_sigma"] = (np.full(n, np.nan) if prior is None
                          else np.asarray(prior, dtype=float))
    return out


def kernel_moments(leaf, params):
    """(Mxx, Mxy, Myy) in native pix² of a kernel component's parameters, and a clip flag."""
    params = np.asarray(params, dtype=float)
    if leaf.model._type_name == "Gaussian":
        scale, g1, g2 = params[-3:]  # (flux scaling, du, dv) come first when present
        a = galsim.Shear(g1=g1, g2=g2).getMatrix()
        m = scale**2 * (a @ a.T) / PIXEL_SCALE**2
        return (m[0, 0], m[0, 1], m[1, 1]), False
    if leaf.model._type_name == "GaussianBlur":
        # Raw [ixx, iyy, ixy] arcsec² may be non-PSD; the model draws their PSD projection.
        drawn = leaf.model.constrain_params(params)
        return blur_moments(drawn), not np.allclose(drawn, params, rtol=0, atol=1e-16)
    return tuple(params), False  # GaussianMoments: signed, drawn as given


def kernel_table(leaf, stars, rows, truth=True):
    """Kernel second moments at each star (and vs truth), native pix².

    Columns: ``obs``, ``star_id`` (when in ``rows``), ``mxx``, ``mxy``, ``myy``, with
    ``truth``: ``mxx_true`` …, ``dmxx`` … (fit − truth); ``T`` (= mxx + myy), with
    ``truth``: ``T_true``, ``dT``; ``clipped``.
    """
    fitted, clipped = [], []
    for star in stars:
        m, c = kernel_moments(leaf, star.fit.get_params(leaf._num))
        fitted.append(m)
        clipped.append(c)
    fitted = np.array(fitted)
    out = _id_columns(rows)
    for j, key in enumerate(("mxx", "mxy", "myy")):
        out[key] = fitted[:, j]
        if truth:
            true = (np.asarray(rows[f"kernel_{key}"], dtype=float)
                    if f"kernel_{key}" in rows.colnames else np.zeros(len(rows)))
            out[key + "_true"] = true
            out["d" + key] = fitted[:, j] - true
    out["T"] = out["mxx"] + out["myy"]
    if truth:
        out["T_true"] = out["mxx_true"] + out["myy_true"]
        out["dT"] = out["T"] - out["T_true"]
    out["clipped"] = clipped
    return out


def pixelgrid_table(leaf, stars, rows):
    """PixelGrid at each star: net flux in the unit model and largest |value|.

    Columns: ``obs``, ``star_id``, ``grid_flux``, ``grid_max``.
    """
    flux, peak = [], []
    for star in stars:
        prof, _ = leaf._getRawProfile(star)
        flux.append(float(prof.flux))
        peak.append(float(np.max(np.abs(star.fit.get_params(leaf._num)))))
    out = _id_columns(rows)
    out["grid_flux"] = flux
    out["grid_max"] = peak
    return out


def component_image(psf, leaf, x, y, sca, oversample=5, size=25, sed_eff=None):
    """Image of one component's unit profile at an SCA position, on a fine grid.

    A chromatic component (Piff's chromatic RomanOptics) is drawn with the effective SED
    ``sed_eff``, by default that of the fit's stars (`residuals.fit_sed`).

    Parameters
    ----------
    psf : piff.PSF
        The fitted (full) PSF, used to interpolate all parameters to (x, y).
    leaf : piff.PSF
        One of `leaf_psfs(psf)`.
    x, y : float
        Position in native pixels.
    sca : int
        SCA (chipnum) of the fit.
    oversample : int
        Fine pixels per native pixel. Default 5.
    size : int
        Size in native pixels. Default 25.

    Returns
    -------
    galsim.Image
        Drawn without adding a pixel (a PixelGrid already contains it; optics / kernel /
        MTF are shown pre-pixel), except Piff's GaussianBlur, which is integrated over each
        fine pixel: near zero blur it is a near-delta that point sampling cannot represent.
        For the drawing only, its moments get a floor of (0.1 fine pixel)² per axis, so a
        zero blur is drawn as a Gaussian 0.1 fine pixel wide (the fitted values are not
        changed).
    """
    wcs = psf.wcs[sca]
    target = piff.Star.makeTarget(x=x, y=y, wcs=wcs, pointing=psf.pointing,
                                  properties={"sca": sca, "chipnum": sca})
    target = psf.interpolateStar(target)
    method = "no_pixel"
    if leaf.model._type_name == "GaussianBlur":
        # GaussianBlur exists only in k-space; near zero blur its maxk is unbounded, so a
        # pre-pixel drawing would need a huge FFT. Draw the same Gaussian in real space, and
        # integrate it over each fine pixel: at zero blur it is a near-delta, which point
        # sampling at pixel centres turns into one enormous value. Integrating an exact delta
        # needs an FFT of the pixel itself (~80000² at 3× oversampling), so the drawn kernel
        # gets a floor of 0.1 fine pixel in σ: invisible in a plot, image sum still 1.
        m, _ = kernel_moments(leaf, target.fit.get_params(leaf._num))
        floor = (0.1 / oversample) ** 2           # native pix², added per axis
        prof = gaussian_kernel(m[0] + floor, m[1], m[2] + floor)
        method = "auto"
    else:
        prof, _ = leaf._getRawProfile(target)
    local = wcs.local(galsim.PositionD(x, y))
    fine = galsim.JacobianWCS(*(local.jacobian().getMatrix().ravel() / oversample))
    image = galsim.ImageD(size * oversample, size * oversample, wcs=fine)
    if isinstance(prof, galsim.ChromaticObject):
        from .residuals import draw_profile, fit_sed
        draw_profile(prof, image, psf, sed_eff if sed_eff is not None else fit_sed(psf))
    else:
        prof.drawImage(image=image, method=method)
    return image
