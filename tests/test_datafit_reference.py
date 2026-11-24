"""Reference optimum of real-data fits (DataReferenceSolver): fitted centers, no truth."""
import importlib.util

import numpy as np
import pytest

from l2files import field, open_l2, write_field
from psf_fomo.datafit import DataFit, DataReferenceSolver, StarCatalog
from psf_fomo.piffval import FitConfig
from psf_fomo.reference import comparison_table

GAUSSIAN_MODEL = {"type": "Simple", "model": {"type": "Gaussian"},
                  "interp": {"type": "Mean"}, "max_iter": 10}


@pytest.fixture(scope="module")
def data_fit(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ref")
    image, catalog = write_field(folder, field(n_stars=30, n_gals=0, seed=7), seed=8)
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(clean_only=False, max_snr_weight=0)  # χ² at the noise scale
    fit = DataFit(open_l2(image), cat, FitConfig.from_piff_config(GAUSSIAN_MODEL))
    fit.set_reserve(0.2, seed=1)
    if importlib.util.find_spec("ngmix") is None:  # wmom / gauss moments need ngmix
        fit.set_diagnostics(moments=False)
    return fit.run(save_to=folder / "fit")


@pytest.fixture(scope="module")
def reference(data_fit):
    return data_fit.reference(starts=("given", "zero"), max_nfev=30)


def test_given_is_piff_with_profiled_fluxes(data_fit, reference):
    given = reference["given"]
    # Same centers and parameters, fluxes re-profiled exactly: χ² at most Piff's (whose last
    # flux came from a linearized step), and close to it.
    piff_chisq = data_fit.fit_summary["chisq"]
    assert 0.98 * piff_chisq <= given.summary["chisq"] <= 1.001 * piff_chisq
    assert given.summary["centers"] == "fitted"
    # Held centers are Piff's fitted ones (same stars, same order).
    np.testing.assert_allclose(given.table["du"], data_fit.table["du"])
    np.testing.assert_allclose(given.table["dv"], data_fit.table["dv"])
    assert given.summary["n_fit_stars"] == int((~data_fit.table["reserve"]).sum())


def test_optimum_of_exact_model(reference):
    given = reference["given"]
    for start, opt in reference["optimum"].items():
        assert opt.summary["converged"], start
        assert opt.summary["chisq"] <= given.summary["chisq"] * (1 + 1e-6)
        assert abs(opt.summary["chisq_dof"] - 1) < 0.06  # the model is exact
    chis = [opt.summary["chisq"] for opt in reference["optimum"].values()]
    np.testing.assert_allclose(chis, chis[0], rtol=1e-4)  # both starts: the same optimum


def test_data_diagnostics_of_solutions(data_fit, reference):
    opt = reference["optimum"]["given"]
    assert "flux_cat_r" in opt.table.colnames and "flux_true" not in opt.table.colnames
    kernel = opt.components["kernel"]
    assert not any(c.endswith("_true") for c in kernel.colnames)
    if data_fit.moments is not None:
        assert set(opt.moments["comparison"]) == {"data", "model_os"}
    t = comparison_table({"Piff": data_fit, "Piff, profiled": reference["given"],
                          "optimum": opt}, comparison="data", algorithm="gauss")
    assert {"chisq", "flux_cat_r"} <= set(t.colnames)
    assert np.all(np.isnan(t["flux_r"]))  # no truth flux on real data
    np.testing.assert_allclose(t["flux_cat_r"], t["flux_cat_r"][0], rtol=0.01)


def test_catalog_centers_option(data_fit):
    stars, _ = data_fit.fitted_stars()
    solver = DataReferenceSolver(data_fit.psf, stars, data_fit.selection, centers="catalog")
    out = solver.solve(starts=())
    assert np.all(out["given"].table["du"] == 0) and np.all(out["given"].table["dv"] == 0)
    with pytest.raises(ValueError, match="centers"):
        DataReferenceSolver(data_fit.psf, stars, data_fit.selection, centers="truth")
