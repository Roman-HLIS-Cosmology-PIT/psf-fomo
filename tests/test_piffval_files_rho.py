"""File-based Piff runs (config generation, memory vs files) and ρ statistics."""
import numpy as np
import galsim
import galsim.roman
import piff
import pytest

from psf_fomo.piffval import FitConfig, PiffValidation, load_fit
from psf_fomo.piffval.rho import MomentRhoStats, rho_stats
from psf_fomo.piffval.run import make_logger
from psf_fomo.sim import RomanSimulation

PIX = galsim.roman.pixel_scale


def file_config(sim, sca):
    # The builder carries the same config as a Simulation; nothing is rendered here.
    pv = PiffValidation(sim, FitConfig(), sca=sca, obs=0)
    pv.set_reserve(0.25, seed=9)
    return pv.piff_file_config("simdir", 0, "outdir")


def test_file_config_simple_wcs():
    sim = RomanSimulation(scas=[7, 3], n_obs=2)
    config = file_config(sim, sca=3)
    inp = config["input"]
    assert inp["image_hdu"] == 2              # second SCA in the simulation's list
    assert inp["cat_hdu"] == 2                # SCA 3 is first in sorted order (after STARS)
    assert (inp["x_col"], inp["y_col"]) == ("x", "y") and "ra_col" not in inp
    assert inp["chipnum"] == 3 and inp["properties"] == {"sca": 3, "obs": 0}
    assert inp["property_cols"] == ["star_id"] and inp["noise"] == "BKGVAR"
    assert "gain" not in inp                  # Gaussian noise: no star shot noise
    # Default cuts (min_sep 2.5″, no S/N cap), as in memory.
    assert config["select"] == {"min_sep": 2.5, "max_snr_weight": 0.0, "reserve_frac": 0.25,
                                "seed": 9}
    assert config["modules"] == ["psf_fomo.models"] and config["shape_method"] == "hsm"
    assert config["psf"]["components"][0]["filter_name"] == "H158"


def test_file_config_roman_wcs_and_gain():
    sim = RomanSimulation(scas=[5])
    sim.set_wcs("roman", ra=30.0, dec=-40.0)
    sim.set_noise("roman-L2")
    inp = file_config(sim, sca=5)["input"]
    assert inp["cat_hdu"] == 1 and inp["ra_col"] == "ra" and "x_col" not in inp
    assert inp["ra"] == pytest.approx(2.0) and inp["dec"] == -40.0  # Piff: RA in hours
    assert inp["gain"] == "GAIN"


def test_files_mode_needs_one_observation_and_a_folder():
    sim = RomanSimulation(n_obs=2)
    with pytest.raises(ValueError):
        PiffValidation(sim, FitConfig()).run(mode="files", save_to="x")
    with pytest.raises(ValueError):
        PiffValidation(sim, FitConfig(), obs=0).run(mode="files")


@pytest.fixture
def gaussian_fit(synthetic_stars):
    stars, rows = synthetic_stars(galsim.Gaussian(sigma=0.15), n=30, flux=2e4, sigma=1.0)
    psf = piff.PSF.process({"type": "Simple", "model": {"type": "Gaussian"},
                            "interp": {"type": "Mean"}, "chisq_thresh": 1e-6})
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    psf.fit(stars, logger=make_logger(0)[0])
    return psf


def test_rho_stats_table_and_global_method_untouched(gaussian_fit):
    pytest.importorskip("treecorr")
    pytest.importorskip("ngmix")
    psf = gaussian_fit
    table = rho_stats(psf, psf.stars, algorithms=("wmom", "gauss"), wmom_fwhm=(0.4,))
    assert piff.star.get_shape_method() == "hsm"
    assert set(table["algorithm"]) == {"wmom_0.4", "gauss"}
    assert set(table["rho"]) == {"rho1", "rho2", "rho3", "rho4", "rho5"}
    assert np.all(table["n_used"] == 30) and np.all(table["n_input"] == 30)
    # An exact model: δg is noise only, ρ1 consistent with zero.
    rho1 = table[(table["rho"] == "rho1") & (table["npairs"] > 0)]
    assert np.all(np.abs(rho1["xip"]) < 5 * rho1["xip_err"] + 1e-8)


def test_moment_rho_stats_uses_its_algorithm(gaussian_fit):
    pytest.importorskip("ngmix")
    star = gaussian_fit.stars[0]
    stats = MomentRhoStats("wmom", 0.5)
    assert stats._measure_size_shape(star) == star.wmom(0.5)
    assert MomentRhoStats("gauss")._measure_size_shape(star) == star.gauss
    with pytest.raises(ValueError):
        MomentRhoStats("ksb")


def test_moment_rho_stats_passes_piff_keywords(gaussian_fit):
    # Our measureShapes override must forward include_flagged and logger by name (Piff added
    # include_flagged before logger): flagged stars are dropped unless requested.
    pytest.importorskip("ngmix")
    psf = gaussian_fit
    stars = list(psf.stars[:5])
    stars[0] = stars[0].flag_if(True)
    stats = MomentRhoStats("wmom", 0.4)
    positions, data, model = stats.measureShapes(psf, stars, logger=make_logger(0)[0])
    assert len(positions) == len(data) == len(model) == 4 and stats.n_used == 4
    positions, _, _ = stats.measureShapes(psf, stars, include_flagged=True)
    assert len(positions) == 5


def small_simulation(wcs="simple"):
    sim = RomanSimulation(scas=[1, 5], n_stars=12, seed=21)
    sim.set_brightness(mag_range=(19.5, 20.5))
    sim.set_noise("gaussian", sigma=10.0)
    if wcs == "roman":
        sim.set_wcs("roman")
    return sim


@pytest.mark.slow
@pytest.mark.parametrize("wcs", ["simple", "roman"])
def test_files_mode_matches_memory(tmp_path, wcs):
    sim = small_simulation(wcs).run(save_to=tmp_path / "sim")
    cfg = FitConfig()
    cfg.set_optics("pinned")
    mem = PiffValidation(tmp_path / "sim", cfg, sca=5).run()
    pv = PiffValidation(tmp_path / "sim", cfg, sca=5, obs=0)
    files = pv.run(save_to=tmp_path / "fit", mode="files")
    assert (tmp_path / "fit" / "piff.yaml").exists()
    # Chromatic by default on both sides: Piff reads the SED saved with the simulation.
    assert (tmp_path / "sim" / "sed.fits").exists()
    assert files.config["piff_psf"]["components"][0].get("chromatic", True)
    # Both modes apply the same cuts: the same stars are kept and rejected.
    assert len(files.table) == len(mem.table) == 12 - mem.fit_summary["n_rejected"]
    np.testing.assert_array_equal(np.sort(files.table["star_id"]),
                                  np.sort(mem.table["star_id"]))
    np.testing.assert_array_equal(np.sort(files.rejected["star_id"]),
                                  np.sort(mem.rejected["star_id"]))
    # Same data, weights (Gaussian noise: BKGVAR = per-star variance) and model.
    assert abs(files.fit_summary["chisq_dof"] - mem.fit_summary["chisq_dof"]) < 0.01
    f = files.table[np.argsort(files.table["star_id"])]
    m = mem.table[np.argsort(mem.table["star_id"])]
    np.testing.assert_allclose(f["flux"], m["flux"], rtol=1e-3)
    back = load_fit(tmp_path / "fit")
    stars, rows = back.fitted_stars()
    assert list(rows["star_id"]) == list(files.table["star_id"])
    assert files.moments is not None and len(files.moments) > 0
