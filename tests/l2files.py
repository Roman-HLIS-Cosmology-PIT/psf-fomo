"""Test helper: small fields of stars and galaxies written in the L2 layout.

The file writers are the package's (`psf_fomo.datafit.synthetic`); this module adds the
analytic test scenes (Gaussian PSF, sheared exponential galaxies) and opens test files. Like
real L2 files and galsim.roman's WCS, they are in Roman's science frame.
"""
import numpy as np
import galsim
import galsim.roman

from psf_fomo.datafit.synthetic import write_l2_file  # noqa: F401
from psf_fomo.datafit.synthetic import write_sep_catalog as sep_catalog
from psf_fomo.sim.wcs import DEFAULT_DEC, DEFAULT_RA


def open_l2(path, **kwargs):
    """`L2Image` of a test file."""
    from psf_fomo.datafit import L2Image
    return L2Image(path, **kwargs)


def roman_wcs(sca=5, pa_deg=0.0):
    """galsim.roman WCS of one SCA at the repo's default pointing (an observable one)."""
    pointing = galsim.CelestialCoord(DEFAULT_RA * galsim.degrees, DEFAULT_DEC * galsim.degrees)
    return galsim.roman.getWCS(world_pos=pointing, PA=pa_deg * galsim.degrees,
                               SCAs=[sca])[sca]


def mirrored_wcs(wcs, center):
    """A TAN WCS equal to ``wcs`` at ``center`` but with the x axis mirrored."""
    j = wcs.local(center).jacobian().getMatrix()
    affine = galsim.AffineTransform(-j[0, 0], j[0, 1], -j[1, 0], j[1, 1], origin=center)
    return galsim.TanWCS(affine, world_origin=wcs.toWorld(center))


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
    stamp reaches 17 pixels from its center at the corners, so an object 18 pixels away on
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
