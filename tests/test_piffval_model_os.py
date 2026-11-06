"""Truth-free diagnostics shared with real-data fits: 'model_os', model_moments, compare_fits,
fit_columns, components without truth."""
from types import SimpleNamespace

import numpy as np
import galsim
import galsim.roman
import piff
import pytest

from psf_fomo.models import GaussianMoments  # noqa: F401  (registered)
from psf_fomo.piffval import compare_fits, diagnostics
from psf_fomo.piffval.components import components
from psf_fomo.piffval.fitting import fit_columns, make_logger
from psf_fomo.piffval.residuals import model_moments, moment_residuals, sca_grid

PIX = galsim.roman.pixel_scale
TRUTH = galsim.Gaussian(sigma=0.15).shear(g1=0.03, g2=-0.02)


def fitted_gaussian(stars):
    psf = piff.PSF.process({"type": "Simple", "model": {"type": "Gaussian"},
                            "interp": {"type": "Mean"}, "max_iter": 20})
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    psf.fit(stars, logger=make_logger(0)[0])
    return psf


@pytest.fixture
def fit(synthetic_stars, analytic_sim):
    stars, rows = synthetic_stars(TRUTH, n=8, flux=2e4, sigma=1.0, seed=3)
    psf = fitted_gaussian(stars)
    table = diagnostics.star_table(psf, psf.stars, rows)
    return psf, psf.stars, rows, table, analytic_sim(TRUTH, rows, flux=2e4)


def test_star_table_columns_unchanged(fit):
    # The refactor into fitting.fit_columns must keep the simulation star table as it was.
    *_, table, _ = fit
    assert table.colnames == ["obs", "sca", "star_id", "x", "y", "phase_x", "phase_y", "mag",
                              "reserve", "flag", "flux", "flux_true", "flux_r", "model_flux",
                              "flux_tot_r", "du", "dv", "dx", "dy", "chisq", "dof",
                              "chisq_dof"]
    assert str(table["du"].unit) == "arcsec" and str(table["dx"].unit) == "pix"


def test_fit_columns_need_no_truth(fit):
    psf, stars, *_ = fit
    cols = fit_columns(psf, stars)
    assert len(cols) == len(stars)
    assert "flux_true" not in cols.colnames and "flux_r" not in cols.colnames
    np.testing.assert_allclose(cols["flux"], [s.fit.flux for s in stars])


def test_model_os_matches_truth_os_model(fit):
    # 'model_os' draws the same model on the same grid as the model side of 'truth_os'.
    psf, stars, _, table, sim = fit
    m = moment_residuals(psf, stars, table, sim, native=(), oversampled=("hsm",),
                         comparisons=("truth_os", "model_os"))
    os_truth = m[m["comparison"] == "truth_os"]
    os_model = m[m["comparison"] == "model_os"]
    np.testing.assert_allclose(os_model["T_model"], os_truth["T_model"], rtol=1e-10)
    np.testing.assert_allclose(os_model["e1_model"], os_truth["e1_model"], atol=1e-12)
    assert np.all(np.isnan(os_model["T_ref"])) and np.all(np.isnan(os_model["dT_T"]))
    assert np.all(os_model["flag_ref"] == 0)


def test_data_and_model_os_without_simulation(fit):
    pytest.importorskip("ngmix")
    psf, stars, _, table, _ = fit
    # A real-data table: no truth columns at all.
    data_table = table["star_id", "x", "y", "reserve", "flag"]
    m = moment_residuals(psf, stars, data_table, sim=None, native=("wmom",),
                         oversampled=("hsm",), comparisons=("data", "model_os"))
    assert set(m["comparison"]) == {"data", "model_os"}
    assert "obs" not in m.colnames and "star_id" in m.colnames
    with pytest.raises(ValueError, match="need the simulation"):
        moment_residuals(psf, stars, data_table, sim=None, comparisons=("truth_os",))


def test_model_moments_constant_model(fit):
    # Mean interpolation: the same model everywhere, equal to the model at the stars.
    psf, stars, _, table, sim = fit
    x, y = sca_grid(3)
    mm = model_moments(psf, x, y, sca=1, algorithms=("hsm",))
    assert len(mm) == 9 and set(mm["algorithm"]) == {"hsm"}
    m = moment_residuals(psf, stars, table, sim, native=(), comparisons=("model_os",))
    np.testing.assert_allclose(mm["T"], m["T_model"][0], rtol=1e-8)
    T_expected = 2 * (0.15 / PIX) ** 2 + 2 / 12  # Gaussian + pixel, round-ish
    np.testing.assert_allclose(mm["T"], T_expected, rtol=0.02)


def test_sca_grid():
    x, y = sca_grid(4)
    assert x.size == 16
    assert x.min() > 1 and x.max() < 4088 and np.allclose(sorted(set(x)), sorted(set(y)))


def test_compare_fits(synthetic_stars):
    stars_a, rows_a = synthetic_stars(TRUTH, n=8, flux=2e4, sigma=1.0, seed=3)
    stars_b, rows_b = synthetic_stars(TRUTH.dilate(1.05), n=8, flux=2e4, sigma=1.0, seed=4)
    results = {}
    for label, stars, rows in (("a", stars_a, rows_a), ("b", stars_b, rows_b)):
        psf = fitted_gaussian(stars)
        results[label] = SimpleNamespace(psf=psf, config={"sca": 1},
                                         table=diagnostics.star_table(psf, psf.stars, rows))
    out = compare_fits(results, algorithms=("hsm",))
    a, b = out[out["fit"] == "a"], out[out["fit"] == "b"]
    assert len(a) == len(b) == 8  # default positions: the first fit's unflagged stars
    assert np.all(a["dT_T"] == 0)
    # Dilating the pre-pixel Gaussian by 5 % raises T by < 10.25 % (the pixel is not dilated).
    assert np.all((b["dT_T"] > 0.05) & (b["dT_T"] < 0.1025))
    results["c"] = SimpleNamespace(psf=results["a"].psf, config={"sca": 2},
                                   table=results["a"].table)
    with pytest.raises(ValueError, match="same SCA"):
        compare_fits(results)


def test_components_without_truth(fit):
    psf, stars, rows, *_ = fit
    with_truth = components(psf, stars, rows)["kernel"]
    without = components(psf, stars, rows["star_id", "x", "y"], truth=False)["kernel"]
    assert "mxx_true" in with_truth.colnames and "dT" in with_truth.colnames
    assert not any(c.endswith("_true") for c in without.colnames)
    assert "dT" not in without.colnames and "obs" not in without.colnames
    np.testing.assert_allclose(without["T"], with_truth["T"])
