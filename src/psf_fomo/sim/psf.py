"""Truth PSF: Piff's Roman optical model plus the PSF error chain.

Rendering chain for one star (CLAUDE.md §5.2):
optical PSF (Piff ``RomanOpticalModel.getProfile`` with the truth Zernikes) → dilate → shear →
Gaussian-kernel convolution → jitter convolution → MTF convolution → × flux → drawn with ``method='auto'`` (pixel applied
once). Profiles are in world coordinates (arcsec), as returned by Piff/GalSim for the image WCS.
"""
import numpy as np
import galsim
import galsim.roman
import piff

# Three-Gaussian charge-diffusion kernel (pyimcom OpenUniverse2024 genpsf.py), also used in
# Piff's tests/test_roman.py::test_roman_diffusion.
MTF_WEIGHTS = (0.17519, 0.53146, 0.29335)
MTF_REL_SIGMAS = (0.4522, 0.8050, 1.4329)
MTF_BASE_SIGMA = 0.3279  # native pixels


def make_optical_model(filter_name, bandpass, max_zernike, linear, nominal_interp="bilinear",
                       chromatic=False):
    """Piff's Roman optical model, configured to render truth PSFs.

    Parameters
    ----------
    filter_name : str
        GalSim Roman filter name, e.g. ``'H158'``.
    bandpass : galsim.Bandpass
        The filter bandpass.
    max_zernike : int
        Highest Noll index of the extra aberrations (4 … 22).
    linear : bool
        If True, the extra aberrations vary linearly across each SCA (Piff
        ``aberration_interp='linear'``: a + b x̂ + c ŷ with x̂ = x / n_pix); else constant.
    nominal_interp : {'bilinear', 'five_point'}
        How Piff interpolates sample-point PSFs across the SCA. Default ``'bilinear'``.
    chromatic : bool
        If True the optical PSF is chromatic (wavelength-dependent); else it is evaluated at
        the bandpass effective wavelength. Default False.

    Returns
    -------
    piff.roman.RomanOpticalModel
    """
    model = piff.roman.RomanOpticalModel(
        chromatic=chromatic,
        filter_name=filter_name,
        max_zernike=max_zernike,
        aberration_interp="linear" if linear else "constant",
        nominal_interp=nominal_interp,
        aberration_prior_sigma=None,
    )
    model.set_bandpass(bandpass)
    return model


def optical_profile(model, params, x, y, sca, wcs, pointing=None):
    """Optical PSF at an SCA position from Piff's Roman model.

    Parameters
    ----------
    model : piff.roman.RomanOpticalModel
        Model from `make_optical_model`.
    params : numpy.ndarray
        Extra Zernike aberrations Z4 … Z_max in waves (length max_zernike − 3), or for a
        linear model the concatenated (a, b, c) blocks.
    x, y : float
        Position in native pixels.
    sca : int
        SCA number, 1 … 18.
    wcs : galsim.BaseWCS
        WCS of the SCA image.
    pointing : galsim.CelestialCoord, optional
        Telescope pointing; required for a celestial WCS.

    Returns
    -------
    galsim.GSObject or galsim.ChromaticObject
        Unit-flux optical PSF in world coordinates, without the pixel response.
    """
    target = piff.Star.makeTarget(x=x, y=y, wcs=wcs, pointing=pointing, stamp_size=8,
                                  properties={"sca": sca})
    return model.getProfile(np.asarray(params, dtype=float), star=target)


def mtf_kernel(width_factor=1.0):
    """Charge-diffusion (MTF) kernel in native-pixel units.

    Parameters
    ----------
    width_factor : float
        Multiplies all Gaussian widths (``psf_dilation × mtf_scale``). Default 1.

    Returns
    -------
    galsim.GSObject
        Unit-flux sum of three Gaussians, sizes in native pixels. Convert to world
        coordinates with ``local_wcs.toWorld(kernel)``.
    """
    return galsim.Sum([w * galsim.Gaussian(sigma=r * MTF_BASE_SIGMA * width_factor)
                       for w, r in zip(MTF_WEIGHTS, MTF_REL_SIGMAS)])


def clip_to_psd(mxx, mxy, myy):
    """Nearest positive semi-definite moment matrix (negative eigenvalues set to 0).

    Parameters
    ----------
    mxx, mxy, myy : float
        Second-moment matrix elements, any units.

    Returns
    -------
    tuple of float
        (mxx, mxy, myy) of the clipped matrix; unchanged if already PSD.
    """
    w, v = np.linalg.eigh(np.array([[mxx, mxy], [mxy, myy]], dtype=float))
    if w[0] >= 0:
        return float(mxx), float(mxy), float(myy)
    m = v @ np.diag(np.clip(w, 0.0, None)) @ v.T
    return float(m[0, 0]), float(m[0, 1]), float(m[1, 1])


def gaussian_kernel(mxx, mxy, myy, pixel_scale=galsim.roman.pixel_scale):
    """Unit-flux elliptical Gaussian with second-moment matrix M, in world coordinates.

    M is given in native pix² along the world (u, v) axes and converted to arcsec² with
    ``pixel_scale`` (not with the local WCS), so the same M means the same sky-plane kernel
    with the simple and the Roman WCS.

    Parameters
    ----------
    mxx, mxy, myy : float
        Elements of M, native pix². M must be positive semi-definite (see `clip_to_psd`);
        a zero eigenvalue gives a kernel that is a line (drawn as a very thin Gaussian).
    pixel_scale : float
        Arcsec per native pixel. Default ``galsim.roman.pixel_scale`` (0.11″).

    Returns
    -------
    galsim.GSObject
        The kernel, sizes in arcsec. Convolving a PSF with it adds M to the PSF's second
        moments.
    """
    w, v = np.linalg.eigh(np.array([[mxx, mxy], [mxy, myy]], dtype=float) * pixel_scale**2)
    if w[0] < -1e-12 * max(abs(w[1]), 1e-30):
        raise ValueError(f"kernel moments ({mxx}, {mxy}, {myy}) are not positive semi-definite")
    # A tiny floor keeps the transform invertible for a rank-1 (line) kernel; 1e-6 pix is far
    # below anything a 0.11″ pixel can resolve.
    floor = (1e-6 * pixel_scale) ** 2
    a = v @ np.diag(np.sqrt(np.clip(w, floor, None)))
    return galsim.Gaussian(sigma=1.0).transform(a[0, 0], a[0, 1], a[1, 0], a[1, 1]).withFlux(1.0)


def random_walk_offsets(rms_mas, n_steps, rng):
    """Pointing offsets along a random-walk trajectory during an exposure.

    The walk is re-centred on its mean (a mean offset only moves the star, it does not change
    the PSF) and scaled so the RMS about the mean is exactly ``rms_mas`` in each axis.

    Parameters
    ----------
    rms_mas : float
        RMS offset per axis, milliarcsec.
    n_steps : int
        Number of equally spaced time samples (≥ 2).
    rng : numpy.random.Generator
        Random generator for the walk.

    Returns
    -------
    dx, dy : numpy.ndarray
        Offsets in milliarcsec.
    weights : numpy.ndarray
        Time weights, summing to 1.
    """
    path = np.cumsum(rng.normal(size=(n_steps, 2)), axis=0)
    path -= path.mean(axis=0)
    path *= rms_mas / path.std(axis=0)
    return path[:, 0], path[:, 1], np.full(n_steps, 1.0 / n_steps)


def timeseries_offsets(t, dx_mas, dy_mas):
    """Pointing offsets from a user time series.

    Parameters
    ----------
    t : array_like
        Sample times in seconds (increasing).
    dx_mas, dy_mas : array_like
        Offsets in milliarcsec along the world u and v axes.

    Returns
    -------
    dx, dy : numpy.ndarray
        Offsets in milliarcsec, re-centred on their time-weighted mean.
    weights : numpy.ndarray
        Time weights from the sample spacing, summing to 1.
    """
    t = np.asarray(t, dtype=float)
    weights = np.gradient(t)
    weights /= weights.sum()
    dx = np.asarray(dx_mas, dtype=float)
    dy = np.asarray(dy_mas, dtype=float)
    return dx - np.sum(weights * dx), dy - np.sum(weights * dy), weights


def jitter_kernel(dx_mas, dy_mas, weights):
    """Jitter convolution kernel: the time-weighted distribution of pointing offsets.

    Parameters
    ----------
    dx_mas, dy_mas : array_like
        Offsets in milliarcsec along world u and v.
    weights : array_like
        Weights summing to 1.

    Returns
    -------
    galsim.GSObject
        Sum of weighted delta functions, in arcsec.
    """
    return galsim.Sum([galsim.DeltaFunction(flux=w).shift(x / 1000.0, y / 1000.0)
                       for x, y, w in zip(dx_mas, dy_mas, weights)])


def apply_errors(optical, local_wcs, dilation=1.0, g1=0.0, g2=0.0, kernel=None, jitter=None,
                 mtf_width=None):
    """Apply the PSF error chain to an optical PSF.

    Order: dilate → shear → Gaussian kernel → jitter → MTF. Dilation and shear act on the
    optical PSF only; the kernel and jitter are convolutions fixed on the sky, so they are not
    dilated with the optics (convolutions commute, so their order does not matter); the MTF is
    an isotropic detector kernel defined in pixels.

    Parameters
    ----------
    optical : galsim.GSObject or galsim.ChromaticObject
        Unit-flux optical PSF in world coordinates.
    local_wcs : galsim.JacobianWCS
        Local WCS at the star, used to put the pixel-based MTF kernel on the sky.
    dilation : float
        Size factor of the optical PSF (1 = no change).
    g1, g2 : float
        Reduced shear applied to the optical PSF (world u, v axes).
    kernel : galsim.GSObject, optional
        Gaussian kernel from `gaussian_kernel`. Default None (no kernel).
    jitter : galsim.GSObject, optional
        Jitter kernel from `jitter_kernel`. Default None (no jitter).
    mtf_width : float, optional
        MTF width factor (``psf_dilation × mtf_scale``). Default None (no MTF).

    Returns
    -------
    galsim.GSObject or galsim.ChromaticObject
        Unit-flux PSF without the pixel response.
    """
    prof = optical
    if dilation != 1.0:
        prof = prof.dilate(dilation)
    if g1 != 0.0 or g2 != 0.0:
        prof = prof.shear(g1=g1, g2=g2)
    if kernel is not None:
        prof = galsim.Convolve(prof, kernel)
    if jitter is not None:
        prof = galsim.Convolve(prof, jitter)
    if mtf_width is not None:
        prof = galsim.Convolve(prof, local_wcs.toWorld(mtf_kernel(mtf_width)))
    return prof


def linear_field(coeffs, x, y, n_pix=galsim.roman.n_pix):
    """Evaluate a linear field a + b x̂ + c ŷ across an SCA (x̂ = x / n_pix, as in Piff).

    Parameters
    ----------
    coeffs : array_like
        (a, b, c), each a scalar or an array (e.g. one value per Zernike).
    x, y : float or numpy.ndarray
        Positions in native pixels.
    n_pix : int
        SCA size. Default ``galsim.roman.n_pix``.

    Returns
    -------
    numpy.ndarray
        Field values; shape broadcast from ``x`` and the coefficient shape.
    """
    a, b, c = (np.asarray(v, dtype=float) for v in coeffs)
    xh = np.asarray(x, dtype=float)[..., None] / n_pix
    yh = np.asarray(y, dtype=float)[..., None] / n_pix
    out = a + b * xh + c * yh
    return out[..., 0] if a.ndim == 0 else out
