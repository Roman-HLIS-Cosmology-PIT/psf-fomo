"""Truth PSF: the Roman optical PSF plus the PSF error chain.

Rendering chain for one star (docs/conventions.md):
optical PSF → dilate → shear → Gaussian-kernel convolution → jitter convolution → MTF
convolution → × flux → drawn with ``method='auto'`` (pixel applied once). Profiles are in world
coordinates (arcsec), as returned by Piff/GalSim for the image WCS.

The optical PSF of a star (``spatial_interpolation``):

- ``'piff'``: Piff's ``RomanOpticalModel`` exactly — GalSim Roman PSFs at the SCA corners,
  blended with position weights; a chromatic PSF is drawn through the bandpass as Piff draws it.
- ``'fast'``: the same corner blending, but each chromatic corner PSF is first summed over the
  filter (weighted by bandpass × SED) into an achromatic effective PSF, once per SCA and
  observation (`effective_profile`). Stars then cost as much as monochromatic ones.
- ``'exact'``: ``galsim.roman.getPSF`` at the star's own position, summed over the filter
  the same way, per star.
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

SPATIAL_MODES = ("fast", "piff", "exact")

DEFAULT_N_WAVES = 10
# Wavelength spacing (nm) of the sum over the filter when n_waves is None.
EXACT_STEP_NM = 5.0
# Effective-PSF grid: the Nyquist scale at the filter's blue limit divided by this. Finer than
# the oversampled truth images (native / 5) and than GalSim's own intermediate grid (Nyquist / 2),
# which biases sizes by ~1e-4 when it is the coarser of two resamplings.
EFFECTIVE_REFINE = 4


def wavelength_nodes(bandpass, n_waves):
    """Wavelengths (nm) at which the optical PSF is evaluated for the sum over the filter.

    Parameters
    ----------
    bandpass : galsim.Bandpass
        The filter; its blue and red limits are the first and last nodes.
    n_waves : int or None
        Number of evenly spaced nodes (≥ 2). None: one node every `EXACT_STEP_NM` nm.
    """
    blue, red = bandpass.blue_limit, bandpass.red_limit
    if n_waves is None:
        n_waves = int(round((red - blue) / EXACT_STEP_NM)) + 1
    return np.linspace(blue, red, int(n_waves))


def band_weights(bandpass, sed, waves):
    """Weights of the node PSFs in the integral over the filter.

    The PSF is taken as linear in wavelength between nodes (as GalSim's ``n_waves``
    interpolation), so the integral of PSF × bandpass × SED is Σ w_k PSF(λ_k) with
    w_k = ∫ h_k(λ) bandpass(λ) SED(λ) dλ, h_k the piecewise-linear "hat" function of node k.
    Normalised to Σ w_k = 1 (a unit-flux PSF stays unit flux).

    Parameters
    ----------
    bandpass : galsim.Bandpass
    sed : galsim.SED
        Spectral SED (any normalisation).
    waves : numpy.ndarray
        Nodes in nm, increasing, from the blue to the red limit.

    Returns
    -------
    numpy.ndarray
        One weight per node.
    """
    fine = np.union1d(np.linspace(bandpass.blue_limit, bandpass.red_limit, 4001), waves)
    density = bandpass(fine) * sed(fine)
    weights = []
    for k in range(len(waves)):
        integrand = np.interp(fine, waves, np.eye(len(waves))[k]) * density
        weights.append(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(fine)))
    weights = np.array(weights)
    return weights / weights.sum()


def effective_profile(chromatic, sed, bandpass, n_waves, extent):
    """Achromatic PSF: a chromatic PSF summed over the filter, weighted by bandpass × SED.

    The monochromatic PSFs at the nodes (`wavelength_nodes`) are drawn, with their weights
    (`band_weights`), directly onto one fine grid (`EFFECTIVE_REFINE`), which becomes an
    ``InterpolatedImage``. Every later step of the truth chain is achromatic, so drawing this
    profile equals drawing the chromatic PSF × SED through the bandpass.

    Parameters
    ----------
    chromatic : galsim.ChromaticObject
        Unit-flux chromatic PSF, world coordinates.
    sed : galsim.SED
        The stars' SED (any normalisation).
    bandpass : galsim.Bandpass
    n_waves : int or None
        Number of wavelength nodes; None = every `EXACT_STEP_NM` nm.
    extent : float
        Width of the grid in arcsec. Light beyond it is dropped (choose it larger than any
        image drawn from the profile).

    Returns
    -------
    galsim.InterpolatedImage
        Its flux is the fraction of light inside the grid (slightly below 1).
    """
    waves = wavelength_nodes(bandpass, n_waves)
    weights = band_weights(bandpass, sed, waves)
    # The bluest PSF is the sharpest: its Nyquist scale sets the grid.
    first = chromatic.evaluateAtWavelength(waves[0])
    scale = first.nyquist_scale / EFFECTIVE_REFINE
    n = 2 * int(np.ceil(extent / scale / 2))
    image = galsim.ImageD(n, n, scale=scale)
    for k, (wave, weight) in enumerate(zip(waves, weights)):
        prof = first if k == 0 else chromatic.evaluateAtWavelength(wave)
        prof.withScaledFlux(weight).drawImage(image=image, method="no_pixel", add_to_image=True)
    return galsim.InterpolatedImage(image)


class TruthRomanModel(piff.roman.RomanOpticalModel):
    """Piff's Roman optical model, optionally returning effective (band-summed) corner PSFs.

    Without ``sed`` it behaves exactly as Piff's ``RomanOpticalModel`` (``'piff'`` mode).
    With ``sed`` and a chromatic model (``'fast'`` mode), each sample-point (corner) PSF is
    replaced by its `effective_profile`, built once and reused while Piff reuses its sample
    PSFs; Piff's position weights then blend achromatic profiles. Used for the truth only; it is
    not a Piff configuration type.

    Parameters
    ----------
    sed : galsim.SED, optional
        The stars' SED. Default None (Piff's behaviour).
    n_waves : int or None
        Wavelength nodes of the effective PSFs (see `effective_profile`). Default 5.
    extent : float
        Width of the effective-PSF grids, arcsec. Default 14.
    **kwargs
        Passed to ``piff.roman.RomanOpticalModel``.
    """

    _type_name = None   # not registered as a Piff model type

    def __init__(self, sed=None, n_waves=DEFAULT_N_WAVES, extent=14.0, **kwargs):
        if n_waves is not None and int(n_waves) < 2:
            raise ValueError("n_waves must be at least 2 (or None)")
        self.sed = sed
        self.n_waves = None if n_waves is None else int(n_waves)
        self.extent = float(extent)
        super().__init__(**kwargs)

    def clear_cache(self):
        super().clear_cache()
        self._effective = {}   # sca → (Piff's sample profiles, their effective versions)

    def _get_sample_profiles(self, star, params, cache=True, sca=None):
        profiles = super()._get_sample_profiles(star, params, cache=cache, sca=sca)
        if self.sed is None or not self.chromatic:
            return profiles
        # Piff returns the same tuple object while its own cache is valid, so the identity
        # check reuses the effective PSFs exactly as long as Piff reuses its profiles.
        key = sca if sca is not None else int(star.data.properties["sca"])
        cached = self._effective.get(key)
        if cached is not None and cached[0] is profiles:
            return cached[1]
        effective = tuple(effective_profile(p, self.sed, self.bandpass, self.n_waves,
                                            self.extent) for p in profiles)
        if cache:
            self._effective[key] = (profiles, effective)
        return effective


def make_optical_model(filter_name, bandpass, max_zernike, linear, nominal_interp="bilinear",
                       chromatic=False, sed=None, n_waves=DEFAULT_N_WAVES, extent=14.0):
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
    sed : galsim.SED, optional
        Give it for ``'fast'`` mode: corner PSFs become effective (band-summed) PSFs. Default
        None (Piff's model, ``'piff'`` mode).
    n_waves : int or None
        Wavelength nodes of the effective PSFs. Default 5; None = every 5 nm.
    extent : float
        Width of the effective-PSF grids, arcsec. Default 14.

    Returns
    -------
    TruthRomanModel
        A ``piff.roman.RomanOpticalModel`` subclass.
    """
    model = TruthRomanModel(
        sed=sed,
        n_waves=n_waves,
        extent=extent,
        chromatic=chromatic,
        filter_name=filter_name,
        max_zernike=max_zernike,
        aberration_interp="linear" if linear else "constant",
        nominal_interp=nominal_interp,
        aberration_prior_sigma=None,
    )
    model.set_bandpass(bandpass)
    return model


def position_profile(sca, filter_name, bandpass, x, y, wcs, zernikes, sed=None,
                     n_waves=DEFAULT_N_WAVES, extent=14.0):
    """Optical PSF from ``galsim.roman.getPSF`` at the star's own position (``'exact'`` mode).

    Called as Piff's model calls ``getPSF`` (same pupil binning, the image WCS at the
    position, extra aberrations indexed by Noll number), but at the star instead of the SCA
    corners: GalSim interpolates its design aberrations to the position.

    Parameters
    ----------
    sca : int
        SCA number, 1 … 18.
    filter_name : str
        GalSim Roman filter name.
    bandpass : galsim.Bandpass
    x, y : float
        Position in native pixels.
    wcs : galsim.BaseWCS
        WCS of the SCA image.
    zernikes : array_like
        Extra aberrations Z4 … Z_max at the star, waves.
    sed : galsim.SED, optional
        For a chromatic PSF: the stars' SED; the result is the `effective_profile`. None: the
        monochromatic PSF at the bandpass effective wavelength.
    n_waves : int or None
        Wavelength nodes of the effective PSF. Default 5.
    extent : float
        Width of the effective-PSF grid, arcsec. Default 14.

    Returns
    -------
    galsim.GSObject
        Unit-flux (effective) optical PSF in world coordinates, without the pixel response.
    """
    zernikes = np.asarray(zernikes, dtype=float)
    extra = np.zeros(len(zernikes) + 4)
    extra[4:] = zernikes
    psf = galsim.roman.getPSF(
        sca, filter_name, SCA_pos=galsim.PositionD(x, y),
        pupil_bin=piff.roman.roman_psf.pupil_bin, wcs=wcs, extra_aberrations=extra,
        wavelength=None if sed is not None else bandpass.effective_wavelength)
    if sed is None:
        return psf
    return effective_profile(psf, sed, bandpass, n_waves, extent)


def optical_profile(model, params, x, y, sca, wcs, pointing=None):
    """Optical PSF at an SCA position from Piff's Roman model.

    Parameters
    ----------
    model : TruthRomanModel
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
