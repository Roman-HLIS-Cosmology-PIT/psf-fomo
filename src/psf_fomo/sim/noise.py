"""Noise and detector effects.

Images are rendered in electrons (e⁻). Noise functions add noise in e⁻, then divide by
``galsim.roman.gain`` so the stored images are in ADU. The expected background (sky, thermal,
dark current) is subtracted after the detector effects, so the background-subtracted images
are approximately zero-mean away from stars.

Roman effects are applied in the order of GalSim's Roman demo (demo13):
sky (+ thermal, + stray light) → Poisson → reciprocity failure → dark current → nonlinearity →
IPC → read noise → gain → quantisation. Persistence is not available (it needs the history of
previous exposures).
"""
import numpy as np
import galsim
import galsim.roman

ROMAN_EFFECTS = ("sky", "poisson", "reciprocity", "dark", "nonlinearity", "ipc", "read",
                 "quantize")


def background_image(image, filter_name, bandpass, world_pos):
    """Expected sky + thermal background of one exposure, in e⁻ per pixel.

    Parameters
    ----------
    image : galsim.Image
        Template image (bounds and WCS of the SCA); not modified.
    filter_name : str
        GalSim Roman filter name, e.g. ``'H158'``.
    bandpass : galsim.Bandpass
        The filter bandpass.
    world_pos : galsim.CelestialCoord
        Sky position used for the zodiacal sky level.

    Returns
    -------
    galsim.ImageF
        Background in e⁻ per pixel for ``galsim.roman.exptime``.
    """
    sky_level = galsim.roman.getSkyLevel(bandpass, world_pos=world_pos)  # e⁻/arcsec²
    sky_level *= 1.0 + galsim.roman.stray_light_fraction
    bkg = galsim.ImageF(image.bounds, wcs=image.wcs)
    image.wcs.makeSkyImage(bkg, sky_level)
    bkg += galsim.roman.thermal_backgrounds[filter_name] * galsim.roman.exptime
    return bkg


def add_gaussian_noise(image, sigma, rng):
    """Add Gaussian noise in e⁻ and convert the image to ADU (in place).

    Parameters
    ----------
    image : galsim.Image
        Image in e⁻; returned in ADU.
    sigma : float
        Noise standard deviation in e⁻ per pixel.
    rng : numpy.random.Generator
        Source of the GalSim noise seed.
    """
    image.addNoise(galsim.GaussianNoise(_galsim_rng(rng), sigma=sigma))
    image /= galsim.roman.gain


def add_roman_noise(image, effects, background, rng):
    """Apply the selected Roman noise and detector effects; convert to ADU (in place).

    Parameters
    ----------
    image : galsim.Image
        Noiseless image in e⁻; returned in ADU with the expected background subtracted.
    effects : sequence of str
        Subset of `ROMAN_EFFECTS`.
    background : galsim.Image
        Expected sky + thermal background in e⁻ (from `background_image`).
    rng : numpy.random.Generator
        Source of the GalSim noise seed.
    """
    gs_rng = _galsim_rng(rng)
    dark_mean = galsim.roman.dark_current * galsim.roman.exptime
    if "sky" in effects:
        image += background
    if "poisson" in effects:
        image.addNoise(galsim.PoissonNoise(gs_rng))
    if "reciprocity" in effects:
        galsim.roman.addReciprocityFailure(image)
    if "dark" in effects:
        image.addNoise(galsim.DeviateNoise(galsim.PoissonDeviate(gs_rng, dark_mean)))
    if "nonlinearity" in effects:
        galsim.roman.applyNonlinearity(image)
    if "ipc" in effects:
        galsim.roman.applyIPC(image)
    if "read" in effects:
        image.addNoise(galsim.GaussianNoise(gs_rng, sigma=galsim.roman.read_noise))
    image /= galsim.roman.gain
    if "quantize" in effects:
        image.quantize()
    if "sky" in effects:
        image -= background / galsim.roman.gain
    if "dark" in effects:
        image -= dark_mean / galsim.roman.gain


def background_variance(noise, background_e=0.0):
    """Expected per-pixel noise variance away from stars, in ADU².

    Star shot noise is not included; Piff adds it from the model when stars carry a ``gain``
    property (see `star_gain`).

    Parameters
    ----------
    noise : dict
        Noise configuration: ``{'kind': 'gaussian', 'sigma': σ}`` or
        ``{'kind': 'roman', 'effects': [...]}``.
    background_e : float or numpy.ndarray
        Expected sky + thermal background in e⁻ at the star(s) (Roman noise only).

    Returns
    -------
    float or numpy.ndarray
        Variance in ADU².
    """
    gain = galsim.roman.gain
    if noise["kind"] == "gaussian":
        return np.zeros_like(background_e, dtype=float) + (noise["sigma"] / gain) ** 2
    effects = noise["effects"]
    var_e = np.zeros_like(background_e, dtype=float)
    if "sky" in effects and "poisson" in effects:
        var_e = var_e + background_e
    if "dark" in effects:
        var_e = var_e + galsim.roman.dark_current * galsim.roman.exptime
    if "read" in effects:
        var_e = var_e + galsim.roman.read_noise ** 2
    return var_e / gain ** 2


def star_gain(noise):
    """Gain (e⁻/ADU) to attach to Piff stars, or None if star shot noise is not simulated."""
    if noise["kind"] == "roman" and "poisson" in noise["effects"]:
        return galsim.roman.gain
    return None


def _galsim_rng(rng):
    return galsim.BaseDeviate(int(rng.integers(1, 2**31 - 1)))
