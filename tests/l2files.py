"""Test helper: write images in the FITS layout of the user's ASDF → FITS L2 converter.

Mirrors the documented layout (see ``docs/datafit.md``): PRIMARY with DETECTOR / FILTER /
EXPTIME; SCI, DQ, ERR, MASK, IVAR, VAR_POISSON, VAR_RNOISE, GAIN, DUMO, SCI_UNIF and, with
``uniform``, MASK_UNIF, IVAR_UNIF, GAIN_UNIF; SKYBKG in every header; the WCS in the image
headers. Units DN/s; GAIN in electrons per DN/s.
"""
import numpy as np
import galsim
import galsim.roman
from astropy.io import fits

from psf_fomo.sim.wcs import DEFAULT_DEC, DEFAULT_RA


def open_l2(path, **kwargs):
    """`L2Image` of a test file: these are written in GalSim's frame (``frame='galsim'``)."""
    from psf_fomo.datafit import L2Image
    return L2Image(path, frame="galsim", **kwargs)


def to_science_frame(src, dst):
    """Write a GalSim-frame test file as Roman's science frame (the per-SCA flip).

    The flip is its own inverse, so flipping a GalSim-frame file with
    `L2Image.write_fit_copy` gives the science-frame file; that function is tested separately
    against sky positions.
    """
    from psf_fomo.datafit import L2Image
    return L2Image(src, frame="science").write_fit_copy(dst)


def roman_wcs(sca=5, pa_deg=0.0):
    """galsim.roman WCS of one SCA at the repo's default pointing (an observable one)."""
    pointing = galsim.CelestialCoord(DEFAULT_RA * galsim.degrees, DEFAULT_DEC * galsim.degrees)
    return galsim.roman.getWCS(world_pos=pointing, PA=pa_deg * galsim.degrees,
                               SCAs=[sca])[sca]


def mirrored_wcs(wcs, centre):
    """A TAN WCS equal to ``wcs`` at ``centre`` but with the x axis mirrored."""
    j = wcs.local(centre).jacobian().getMatrix()
    affine = galsim.AffineTransform(-j[0, 0], j[0, 1], -j[1, 0], j[1, 1], origin=centre)
    return galsim.TanWCS(affine, world_origin=wcs.toWorld(centre))


def scene(wcs, shape=(256, 256), stars=(), sky=2.0, read_var=0.01, gain=250.0, seed=0,
          profile=None, noise=True):
    """Image in DN/s: sky + stars, with read and Poisson noise.

    Parameters
    ----------
    wcs : galsim.BaseWCS
    shape : (ny, nx)
    stars : sequence of (x, y, flux) or (x, y, flux, profile)
        1-based positions, flux in DN/s; an object's own unit-flux profile (e.g. a galaxy)
        replaces ``profile``.
    sky, read_var, gain : float
        Sky (DN/s per pixel), read variance ((DN/s)²), gain (e⁻ per DN/s).
    profile : galsim.GSObject, optional
        Unit-flux PSF; default a round Gaussian of σ = 0.15″.

    Returns
    -------
    numpy.ndarray (float32)
    """
    profile = profile or galsim.Gaussian(sigma=0.15)
    image = galsim.ImageD(shape[1], shape[0], wcs=wcs)
    for obj in stars:
        x, y, flux = obj[:3]
        prof = obj[3] if len(obj) > 3 else profile
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        stamp = galsim.ImageD(galsim.BoundsI(ix - 20, ix + 20, iy - 20, iy + 20), wcs=wcs)
        (prof * flux).drawImage(stamp, center=galsim.PositionD(x, y), method="auto")
        b = stamp.bounds & image.bounds
        image[b] += stamp[b]
    image += sky
    if noise:
        rng = np.random.default_rng(seed)
        electrons = rng.poisson(np.clip(image.array, 0, None) * gain) / gain
        image.array[:] = electrons + rng.normal(0, np.sqrt(read_var), image.array.shape)
    return image.array.astype(np.float32)


PSF = galsim.Gaussian(sigma=0.15)   # the test PSF (σ ≈ 1.36 native pixels)


def field(n_stars=60, n_gals=60, shape=(700, 700), margin=25, min_dist=30.0, seed=0):
    """Random stars (the test PSF) and galaxies (sheared exponentials ⊛ PSF), well separated.

    ``min_dist`` (pixels) must exceed a stamp's half-diagonal plus the PSF wings: a 25-pixel
    stamp reaches 17 pixels from its centre at the corners, so an object 18 pixels away on
    the diagonal puts light into the corner pixels that no single-star model can fit.

    Returns a list of dicts: ``id`` (1…N), ``x``, ``y`` (1-based), ``flux`` [DN/s],
    ``is_star``, ``profile`` (unit flux), ``r50`` [pix] (approximate half-light radius).
    """
    rng = np.random.default_rng(seed)
    objects = []
    while len(objects) < n_stars + n_gals:
        x, y = rng.uniform(margin, shape[1] - margin), rng.uniform(margin, shape[0] - margin)
        if any(np.hypot(x - o["x"], y - o["y"]) < min_dist for o in objects):
            continue
        is_star = len(objects) < n_stars
        flux = float(np.exp(rng.uniform(np.log(300), np.log(5000))))
        if is_star:
            prof, r50 = PSF, 1.6
        else:
            g1, g2 = rng.normal(0, 0.2, 2).clip(-0.6, 0.6)
            gal = galsim.Exponential(half_light_radius=0.35).shear(g1=g1, g2=g2)
            prof, r50 = galsim.Convolve(gal, PSF), 3.6
        objects.append({"id": len(objects) + 1, "x": x, "y": y, "flux": flux,
                        "is_star": is_star, "profile": prof,
                        "r50": r50 * rng.normal(1, 0.03)})
    return objects


def sep_catalog(path, objects, abzp=26.0, sky=2.0, flux_scale=0.95, unclean=()):
    """A detection catalog shaped like the user's SEP output (0-based x, y; HDU 1).

    Columns ``id``, ``x``, ``y`` (0-based), ``flux_auto`` (true flux × ``flux_scale``, DN/s),
    ``snr_auto_bkg``, ``r50`` [pix], ``clean`` (False for ids in ``unclean``), and
    ``is_star`` (1/0, a stand-in for an external star flag); header ``ABZP``, ``SKY``.
    """
    from astropy.table import Table
    t = Table({"id": [o["id"] for o in objects],
               "x": [o["x"] - 1 for o in objects], "y": [o["y"] - 1 for o in objects],
               "flux_auto": [o["flux"] * flux_scale for o in objects],
               "snr_auto_bkg": [o["flux"] / 2.0 for o in objects],
               "r50": [o["r50"] for o in objects],
               "clean": [o["id"] not in set(unclean) for o in objects],
               "is_star": [int(o["is_star"]) for o in objects]})
    t.meta.update({"ABZP": abzp, "SKY": sky})
    t.write(path, overwrite=True)
    return path


def write_field(folder, objects, shape=(700, 700), sca=5, sky=2.0, read_var=0.01, gain=250.0,
                unclean=(), seed=1, primary=None):
    """Draw ``objects`` (from `field`), write the L2 file and the SEP-style catalog.

    ``primary``: extra PRIMARY header cards (see `write_l2_file`). The folder is created if
    needed. Returns (image path, catalog path).
    """
    folder.mkdir(parents=True, exist_ok=True)
    wcs = roman_wcs(sca=sca)
    sci = scene(wcs, shape=shape, stars=[(o["x"], o["y"], o["flux"], o["profile"])
                                        for o in objects],
                sky=sky, read_var=read_var, gain=gain, seed=seed)
    image = write_l2_file(folder / "l2.fits", sci, wcs, sca=sca, sky=sky, read_var=read_var,
                          gain=gain, primary=primary)
    catalog = sep_catalog(folder / "det.fits", objects, sky=sky, unclean=unclean)
    return image, catalog


def write_l2_file(path, sci, wcs, sca=5, filter_name="F158", sky=2.0, read_var=0.01,
                  gain=250.0, mask=None, dumo=None, uniform=True, background_weights=True,
                  primary=None, planes="all"):
    """Write ``sci`` (DN/s, sky included) in the converter's layout.

    Parameters
    ----------
    planes : {'all', 'unif'}
        ``'all'``: every plane of the layout. ``'unif'``: only SCI_UNIF, MASK_UNIF,
        IVAR_UNIF, GAIN_UNIF (what a ``variant='unif'`` fit reads; for full-SCA tests).
    background_weights : bool
        IVAR = 1/(VAR_RNOISE + SKYBKG/GAIN) (converter's ``--background-weights``) if True,
        else 1/ERR² (contains the stars' Poisson noise).
    uniform : bool
        Write MASK_UNIF, IVAR_UNIF, GAIN_UNIF (converter's ``--uniform``).
    primary : dict, optional
        PRIMARY header cards replacing the defaults (DETECTOR, FILTER, EXPTIME); a value None
        removes the card.
    """
    sci = np.asarray(sci, dtype=np.float32)
    mask = np.zeros(sci.shape, np.uint8) if mask is None else mask.astype(np.uint8)
    dumo = np.zeros_like(sci) if dumo is None else dumo.astype(np.float32)
    gain_map = np.full(sci.shape, gain, np.float32)
    var_rnoise = np.full(sci.shape, read_var, np.float32)
    var_poisson = (np.clip(sci, 0, None) / gain).astype(np.float32)
    err = np.sqrt(var_rnoise + var_poisson)
    ivar = (1 / (var_rnoise + sky / gain_map) if background_weights else 1 / err**2)

    header = galsim.FitsHeader()
    ny, nx = sci.shape
    wcs.writeToFitsHeader(header, galsim.BoundsI(1, nx, 1, ny))
    wcs_cards = header.header

    cards = {"DETECTOR": f"WFI{sca:02d}", "FILTER": filter_name, "EXPTIME": 139.8}
    cards.update(primary or {})
    hdus = [fits.PrimaryHDU()]
    for key, value in cards.items():
        if value is not None:
            hdus[0].header[key] = value

    def image_hdu(name, data, with_wcs=False):
        h = fits.ImageHDU(np.asarray(data), name=name)
        if with_wcs:
            h.header.extend(wcs_cards, unique=True)
        return h

    if planes == "all":
        hdus += [image_hdu("SCI", sci, True), image_hdu("DQ", mask.astype(np.int32)),
                 image_hdu("ERR", err), image_hdu("MASK", mask),
                 image_hdu("IVAR", ivar.astype(np.float32)),
                 image_hdu("VAR_POISSON", var_poisson), image_hdu("VAR_RNOISE", var_rnoise),
                 image_hdu("GAIN", gain_map), image_hdu("DUMO", dumo)]
    hdus += [image_hdu("SCI_UNIF", sci + dumo, True)]
    if uniform or planes == "unif":
        hdus += [image_hdu("MASK_UNIF", mask),
                 image_hdu("IVAR_UNIF", (1 / (var_rnoise + sky / gain_map)).astype(np.float32)),
                 image_hdu("GAIN_UNIF", gain_map)]
    for h in hdus:
        h.header["SKYBKG"] = sky
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path
