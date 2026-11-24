"""MomentsValidation / MomentsResult with a Gaussian stand-in for the Roman simulation."""
import numpy as np
import galsim
import galsim.roman
import piff
import pytest
from astropy.table import Table

from psf_fomo.moments import MomentsValidation, load_moments
from psf_fomo.moments.measure import make_star, sigma_g_to_T
from psf_fomo.sim import Simulation

PIX = galsim.roman.pixel_scale
FLUX = 1.0e4  # ADU


class FakeSimulation(Simulation):
    """Elliptical Gaussian PSF (pre-pixel) instead of the Roman PSF; same access methods.

    Positions come from ``seed``; the noise from ``noise_seed`` (same default), so two fakes
    with the same seeds and different ``dilation`` are paired like real simulations.
    """

    def __init__(self, n_stars=20, dilation=1.0, seed=0, noise_seed=None, sigma_pix=2.0,
                 g1=0.03, noise=1.0):
        rng = np.random.default_rng(seed)
        x = rng.uniform(100, 4000, n_stars)
        y = rng.uniform(100, 4000, n_stars)
        table = Table({"star_id": np.arange(n_stars), "obs": np.zeros(n_stars, int),
                       "sca": np.ones(n_stars, int), "x": x, "y": y,
                       "phase_x": x - np.round(x), "phase_y": y - np.round(y),
                       "mag": np.full(n_stars, 18.0), "bkg_var": np.full(n_stars, noise**2)})
        config = {"scas": [1], "n_obs": 1, "n_stars": n_stars, "stamp_size": 25}
        super().__init__(config, {}, {}, [table])
        self.profile = galsim.Gaussian(sigma=sigma_pix * PIX * dilation).shear(g1=g1, g2=0.0)
        self.noise = noise
        self.noise_seed = seed if noise_seed is None else noise_seed
        self.wcs = galsim.PixelScale(PIX)

    def render_stamp(self, star_id, obs=0, size=None):
        row = self.catalogs[obs][star_id]
        ix, iy = int(round(row["x"])), int(round(row["y"]))
        image = galsim.ImageD(galsim.BoundsI(ix - 12, ix + 12, iy - 12, iy + 12), wcs=self.wcs)
        self.profile.withFlux(FLUX).drawImage(
            image, center=galsim.PositionD(row["x"], row["y"]))
        return image

    def stars(self, sca=None, obs=0):
        rows = self.catalog(obs, sca)
        stars = []
        for row in rows:
            image = self.render_stamp(int(row["star_id"]), obs)
            rng = np.random.default_rng([self.noise_seed, int(row["star_id"])])
            image.array[:] += rng.normal(0.0, self.noise, image.array.shape)
            pos = galsim.PositionD(row["x"], row["y"])
            stars.append(make_star(image, pos, 1.0 / self.noise**2))
        return stars, rows

    def truth_psf(self, sca, obs, star_id, oversample=5, include_pixel=True, size=None):
        prof = self.profile
        if include_pixel:
            prof = galsim.Convolve(prof, galsim.Pixel(PIX))
        fine = galsim.ImageD(25 * oversample, 25 * oversample, scale=PIX / oversample)
        prof.drawImage(fine, method="no_pixel")
        return fine


def expected_T_os(sigma_pix, g1, dilation):
    """T [arcsec²] of the pixel-convolved sheared Gaussian: Gaussian trace + 2 × 1/12 pix²."""
    return (sigma_g_to_T(sigma_pix * dilation, g1, 0.0) + 2.0 / 12.0) * PIX**2


@pytest.fixture(scope="module")
def result():
    sims = {"base": FakeSimulation(), "dil": FakeSimulation(dilation=1.02)}
    mv = MomentsValidation(sims)
    mv.set_algorithms(["hsm"])
    return mv.run()


def test_table_layout(result):
    table = result.table
    assert len(table) == 2 * 20 * 3  # sims × stars × image types
    assert set(table["image"]) == {"noisy", "noiseless", "oversampled"}
    assert set(table["algorithm"]) == {"hsm"}
    for name in ("sim", "obs", "sca", "star_id", "phase_x", "snr", "T", "e1", "flag"):
        assert name in table.colnames
    assert np.all(table["flag"] == 0)


def test_oversampled_size_is_analytic(result):
    rows = result.table[(result.table["image"] == "oversampled")
                        & (result.table["sim"] == "base")]
    np.testing.assert_allclose(rows["T"], expected_T_os(2.0, 0.03, 1.0), rtol=3e-3)


def test_differences_recover_injected_dilation(result):
    diff = result.differences()
    assert set(diff["sim"]) == {"dil"}
    expected = expected_T_os(2.0, 0.03, 1.02) / expected_T_os(2.0, 0.03, 1.0) - 1
    over = diff[diff["image"] == "oversampled"]
    np.testing.assert_allclose(over["dT_T"], expected, rtol=0.03)
    np.testing.assert_allclose(over["dT_T_hsm_os"], over["dT_T"])
    # Paired noise largely cancels: noisy differences are close to the noiseless ones.
    noisy = diff[diff["image"] == "noisy"]
    np.testing.assert_allclose(np.mean(noisy["dT_T"]), expected, rtol=0.05)
    # A dilation does not change the shape.
    np.testing.assert_allclose(over["de1"], 0.0, atol=1e-4)


def test_bias_well_sampled_is_small(result):
    bias = result.bias()
    assert set(bias["image"]) == {"noisy", "noiseless"}
    noiseless = bias[bias["image"] == "noiseless"]
    assert np.all(np.abs(noiseless["dT_T"]) < 5e-3)


def test_summary_excludes_flagged(result):
    table = result.table.copy()
    table["flag"][0] = 1
    flagged = type(result)(table, result.config, result.provenance)
    summary = flagged.summary("moments")
    row = summary[(summary["sim"] == table["sim"][0]) & (summary["image"] == table["image"][0])]
    assert row["n_flagged"][0] == 1 and row["n"][0] == 19
    for kind in ("bias", "differences"):
        assert "dT_T_mean" in result.summary(kind).colnames


def test_mismatched_positions_raise():
    sims = {"a": FakeSimulation(n_stars=3, seed=0), "b": FakeSimulation(n_stars=3, seed=1)}
    mv = MomentsValidation(sims)
    mv.set_algorithms(["hsm"])
    mv.set_images(["oversampled"])
    with pytest.raises(ValueError, match="different positions"):
        mv.run().differences()


def test_global_shape_method_untouched():
    before = piff.star.get_shape_method()
    mv = MomentsValidation({"a": FakeSimulation(n_stars=2)})
    mv.set_algorithms(["hsm", "wmom"] if _has_ngmix() else ["hsm"])
    mv.run()
    assert piff.star.get_shape_method() == before


def test_wmom_labels():
    mv = MomentsValidation({"a": FakeSimulation(n_stars=2)})
    mv.set_algorithms(["wmom"], wmom_fwhm=[0.3, 0.5])
    mv.set_images(["noiseless"])
    assert set(mv.run().table["algorithm"]) == {"wmom_0.3", "wmom_0.5"}


def test_save_load_round_trip(result, tmp_path):
    result.save(tmp_path)
    loaded = load_moments(tmp_path)
    assert loaded.config == result.config
    assert loaded.table.colnames == result.table.colnames
    for name in result.table.colnames:
        np.testing.assert_array_equal(loaded.table[name], result.table[name])
    assert len(loaded.differences()) == len(result.differences())


def test_load_converts_tables_saved_in_pixels(result, tmp_path):
    # A table saved before sizes were in arcsec: no SIZEUNIT, T in pix², sigma in pix.
    result.save(tmp_path)
    old = result.table.copy()
    old.meta.clear()
    old["T"] = old["T"] / PIX**2
    old["sigma"] = old["sigma"] / PIX
    old.write(tmp_path / "moments.fits", overwrite=True)
    with pytest.warns(UserWarning, match="native pixels"):
        loaded = load_moments(tmp_path)
    np.testing.assert_allclose(loaded.table["T"], result.table["T"], rtol=1e-12)
    np.testing.assert_allclose(loaded.table["sigma"], result.table["sigma"], rtol=1e-12)


def test_builder_validation(tmp_path):
    mv = MomentsValidation({"a": FakeSimulation(n_stars=1)})
    with pytest.raises(ValueError):
        mv.set_algorithms(["unweighted"])
    with pytest.raises(ValueError):
        mv.set_images(["native"])
    with pytest.raises(ValueError):
        mv.set_algorithms(["wmom"], wmom_fwhm=0.0)
    with pytest.raises(ValueError, match="given as objects"):
        mv.to_yaml(tmp_path / "m.yaml")
    with pytest.raises(ValueError):
        MomentsValidation({"a": FakeSimulation(n_stars=1)}, reference="b")


def _has_ngmix():
    try:
        import ngmix  # noqa: F401
    except ImportError:
        return False
    return True
