"""World coordinate systems for the simulated SCAs."""
import datetime

import galsim
import galsim.roman

# A Roman-observable pointing used in prev_code (RA, Dec in degrees).
DEFAULT_RA = 61.24
DEFAULT_DEC = -48.42


def pointing_coord(wcs_config):
    """Telescope pointing as a CelestialCoord.

    Parameters
    ----------
    wcs_config : dict
        WCS configuration with ``ra`` and ``dec`` in degrees.

    Returns
    -------
    galsim.CelestialCoord
    """
    return galsim.CelestialCoord(wcs_config["ra"] * galsim.degrees,
                                 wcs_config["dec"] * galsim.degrees)


def make_wcs(wcs_config, scas):
    """WCS for each SCA.

    Parameters
    ----------
    wcs_config : dict
        ``{'kind': 'simple', ...}`` for a plain pixel scale (``galsim.roman.pixel_scale``,
        0.11″, identical for every SCA), or ``{'kind': 'roman', 'ra', 'dec', 'pa', 'date'}``
        for ``galsim.roman.getWCS`` (degrees; ``pa`` None lets GalSim choose; ``date`` an ISO
        string or None for GalSim's default).
    scas : sequence of int
        SCA numbers.

    Returns
    -------
    wcs : dict
        SCA number → galsim WCS.
    pointing : galsim.CelestialCoord or None
        The pointing for a celestial (Roman) WCS; None for the simple pixel scale, matching
        Piff's convention for Euclidean WCSs.
    """
    if wcs_config["kind"] == "simple":
        scale = galsim.PixelScale(galsim.roman.pixel_scale)
        return {sca: scale for sca in scas}, None
    pointing = pointing_coord(wcs_config)
    pa = None if wcs_config["pa"] is None else wcs_config["pa"] * galsim.degrees
    date = (None if wcs_config["date"] is None
            else datetime.datetime.fromisoformat(wcs_config["date"]))
    wcs = galsim.roman.getWCS(world_pos=pointing, PA=pa, date=date, SCAs=list(scas))
    return {sca: wcs[sca] for sca in scas}, pointing
