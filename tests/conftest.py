"""Shared test helpers: synthetic Piff stars that need no galsim.roman.getPSF call."""
import numpy as np
import galsim
import piff
import pytest
from astropy.table import Table


def make_synthetic_stars(prof, n=8, flux=1e4, stamp_size=25, scale=0.11, sigma=None, seed=0,
                         sca=1):
    """Stars drawn from a known profile (method 'auto'), at random positions and phases.

    Returns (stars, rows): Piff stars (weight 1, or 1/σ² with Gaussian noise σ) and a catalog
    table shaped like `Simulation.catalog` (flux in e⁻, i.e. × galsim.roman.gain).
    """
    rng = np.random.default_rng(seed)
    wcs = galsim.PixelScale(scale)
    stars, rows = [], []
    for i in range(n):
        x, y = rng.uniform(100, 3900, 2)
        ix, iy = int(round(x)), int(round(y))
        lo = stamp_size // 2
        bounds = galsim.BoundsI(ix - lo, ix - lo + stamp_size - 1, iy - lo, iy - lo + stamp_size - 1)
        image = galsim.ImageF(bounds, wcs=wcs)
        pos = galsim.PositionD(x, y)
        (prof * flux).drawImage(image, method="auto", center=pos)
        w = 1.0
        if sigma is not None:
            image.addNoise(galsim.GaussianNoise(galsim.BaseDeviate(seed * 1000 + i), sigma))
            w = 1.0 / sigma**2
        weight = galsim.ImageF(bounds, wcs=wcs, init_value=w)
        data = piff.StarData(image, pos, weight=weight,
                             properties={"sca": sca, "chipnum": sca, "star_id": i, "obs": 0})
        stars.append(piff.Star(data, None))
        rows.append({"obs": 0, "sca": sca, "star_id": i, "x": x, "y": y,
                     "phase_x": x - ix, "phase_y": y - iy, "mag": 18.0,
                     "flux": flux * galsim.roman.gain})
    return stars, Table(rows=rows)


class AnalyticSim:
    """Stand-in for `Simulation` (render_stamp, truth_psf) for stars of one known profile.

    ``rows`` and ``flux`` (ADU) as returned by / passed to `make_synthetic_stars`.
    """

    def __init__(self, prof, rows, flux=1e4, stamp_size=25, scale=0.11):
        self.prof, self.rows, self.flux = prof, rows, flux
        self.stamp_size, self.scale = stamp_size, scale

    def _row(self, star_id, obs):
        return self.rows[(self.rows["star_id"] == star_id) & (self.rows["obs"] == obs)][0]

    def render_stamp(self, star_id, obs=0, size=None):
        row = self._row(star_id, obs)
        x, y = float(row["x"]), float(row["y"])
        ix, iy, lo = int(round(x)), int(round(y)), self.stamp_size // 2
        bounds = galsim.BoundsI(ix - lo, ix - lo + self.stamp_size - 1,
                                iy - lo, iy - lo + self.stamp_size - 1)
        image = galsim.ImageD(bounds, wcs=galsim.PixelScale(self.scale))
        (self.prof * self.flux).drawImage(image, method="auto", center=galsim.PositionD(x, y))
        return image

    def truth_psf(self, sca, obs, star_id, oversample=5, include_pixel=True, size=None):
        prof = self.prof
        if include_pixel:
            prof = galsim.Convolve(prof, galsim.Pixel(self.scale))
        n = (size or self.stamp_size) * oversample
        image = galsim.ImageD(n, n, wcs=galsim.JacobianWCS(self.scale / oversample, 0, 0,
                                                           self.scale / oversample))
        prof.drawImage(image=image, method="no_pixel")
        return image


@pytest.fixture
def synthetic_stars():
    return make_synthetic_stars


@pytest.fixture
def analytic_sim():
    return AnalyticSim
