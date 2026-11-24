"""Star positions and brightness on an SCA.

Positions use GalSim image coordinates: pixel centers are at integer values and an SCA spans
1 … ``galsim.roman.n_pix`` in x and y. The sub-pixel phase of a position is
``position - round(position)``, in [-0.5, 0.5].
"""
import numpy as np
import galsim
import galsim.roman


def draw_positions(n_stars, kind, rng, margin, phase=None, n_pix=galsim.roman.n_pix):
    """Draw star positions on one SCA.

    Parameters
    ----------
    n_stars : int
        Number of stars.
    kind : {'random', 'grid'}
        ``'random'``: uniform continuous positions. ``'grid'``: an evenly spaced
        ceil(sqrt(n)) × ceil(sqrt(n)) grid (truncated to ``n_stars``), each point given a random
        sub-pixel phase.
    rng : numpy.random.Generator
        Random generator for positions and phases.
    margin : float
        Distance in native pixels kept clear of each SCA edge, so drawn stamps stay on-chip.
    phase : tuple of float, optional
        Fixed sub-pixel phase (fx, fy), each in [-0.5, 0.5]; e.g. (0, 0) puts every star at a
        pixel center. Default None (random phases).
    n_pix : int
        SCA size in pixels. Default ``galsim.roman.n_pix``.

    Returns
    -------
    x, y : numpy.ndarray
        Positions in native pixels (GalSim image coordinates).
    """
    lo, hi = 1 + margin, n_pix - margin
    if kind == "random":
        x = rng.uniform(lo, hi, n_stars)
        y = rng.uniform(lo, hi, n_stars)
    elif kind == "grid":
        n_side = int(np.ceil(np.sqrt(n_stars)))
        grid = np.array([(lo + hi) / 2]) if n_side == 1 else np.linspace(lo, hi, n_side)
        xx, yy = np.meshgrid(grid, grid)
        x = np.round(xx.ravel()[:n_stars]) + rng.uniform(-0.5, 0.5, n_stars)
        y = np.round(yy.ravel()[:n_stars]) + rng.uniform(-0.5, 0.5, n_stars)
    else:
        raise ValueError(f"placement kind must be 'random' or 'grid', got {kind!r}")
    if phase is not None:
        x = np.round(x) + phase[0]
        y = np.round(y) + phase[1]
    return x, y


def sub_pixel_phase(position):
    """Sub-pixel phase ``position - round(position)`` (native pixels, in [-0.5, 0.5])."""
    position = np.asarray(position, dtype=float)
    return position - np.round(position)


def draw_magnitudes(n_stars, rng, mag=None, mag_range=None):
    """Star magnitudes: a constant, or uniform random in a range.

    Parameters
    ----------
    n_stars : int
        Number of stars.
    rng : numpy.random.Generator
        Random generator (used only with ``mag_range``).
    mag : float, optional
        Constant AB magnitude for every star.
    mag_range : tuple of float, optional
        (faint, bright) or (bright, faint) limits of a uniform distribution, AB magnitudes.

    Returns
    -------
    numpy.ndarray
        AB magnitudes.
    """
    if mag_range is not None:
        return rng.uniform(min(mag_range), max(mag_range), n_stars)
    return np.full(n_stars, float(mag))


def mag_to_flux(mag, bandpass, exptime=galsim.roman.exptime,
                area=galsim.roman.collecting_area):
    """Convert AB magnitudes to the number of detected electrons in one exposure.

    With an AB zeropoint, GalSim defines ``mag = -2.5 log10(F) + zeropoint`` where F is the
    photon flux through the bandpass in photons/s/cm². The Roman bandpasses are total
    throughputs, so detected photons are counted as electrons (as in GalSim's Roman demo).

    Parameters
    ----------
    mag : float or array_like
        AB magnitudes.
    bandpass : galsim.Bandpass
        Bandpass with an AB zeropoint, e.g. ``galsim.roman.getBandpasses(AB_zeropoint=True)``.
    exptime : float
        Exposure time in seconds. Default ``galsim.roman.exptime``.
    area : float
        Collecting area in cm². Default ``galsim.roman.collecting_area``.

    Returns
    -------
    numpy.ndarray or float
        Flux in electrons (e⁻).
    """
    return 10 ** (-0.4 * (np.asarray(mag, dtype=float) - bandpass.zeropoint)) * area * exptime
