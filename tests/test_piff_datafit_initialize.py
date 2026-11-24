"""PiffDataFit.initialize (Piff's initialization, no fit) and the reference's refresh rounds.

The test field has model Poisson weights (a gain per star), so weights depend on the model:
the refresh rounds must bring the reference started from Piff's initialization to the same
optimum as the reference of a converged Piff fit.
"""
from pathlib import Path

import numpy as np
import pytest

from l2files import field, open_l2, write_field
from psf_fomo.piff_datafit import PiffDataFit, StarCatalog, load_data_fit
from psf_fomo.piff_simfit import PiffFitConfig

GAUSSIAN_MODEL = {"type": "Simple", "model": {"type": "Gaussian"},
                  "interp": {"type": "Mean"}, "max_iter": 10}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    folder = tmp_path_factory.mktemp("init")
    image, catalog = write_field(folder, field(n_stars=30, n_gals=0, seed=7), seed=8)
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(clean_only=False, max_snr_weight=0)
    fit = PiffDataFit(open_l2(image), cat, PiffFitConfig.from_piff_config(GAUSSIAN_MODEL))
    fit.set_reserve(0.2, seed=1)
    fit.set_diagnostics(moments=False)
    return folder, fit


@pytest.fixture(scope="module")
def fitted(data):
    folder, fit = data
    return fit.run(save_to=folder / "fit")


@pytest.fixture(scope="module")
def initialized(data):
    return data[1].initialize()


def test_initialize_is_not_a_fit(initialized, fitted):
    s = initialized.fit_summary
    assert s["initialized_only"] and s["niter"] == 0 and not s["converged"]
    assert "initialized_only" not in fitted.fit_summary
    # Same stars, same reserve choice as the fit.
    assert len(initialized.table) == len(fitted.table)
    np.testing.assert_array_equal(initialized.table["reserve"], fitted.table["reserve"])
    # Not saved: Piff's inputs were in a temporary folder that is gone.
    assert not Path(initialized.config["piff"]["input"]["cat_file_name"]).exists()
    stars, _ = initialized.fitted_stars()   # kept in memory
    assert len(stars) == len(initialized.table)


def test_reference_from_initialization_reaches_the_fit_optimum(initialized, fitted):
    from_fit = fitted.reference(starts=("given",), max_nfev=30, report="fit", given=False)
    from_init = initialized.reference(starts=("given",), max_nfev=30, report="fit",
                                      given=False)        # refresh 2 by default
    a, b = from_fit["optimum"]["given"], from_init["optimum"]["given"]
    assert "rounds" not in a.summary and len(b.summary["rounds"]) == 3
    assert b.summary["converged"]
    assert b.summary["rounds"][-1]["max_dx"] < 3e-4
    assert b.summary["rounds"][-1]["max_dx"] < b.summary["rounds"][1]["max_dx"]
    # Same model parameters (Gaussian σ, g1, g2 with a Mean interpolator) and χ²/dof.
    np.testing.assert_allclose(b.x, a.x, atol=5e-4)
    assert abs(b.summary["chisq_dof"] - a.summary["chisq_dof"]) < 0.01
    assert len(b.table) == int((~fitted.table["reserve"]).sum())


def test_refresh_after_a_fit_changes_little_and_reports_reserve(fitted):
    plain = fitted.reference(starts=("given",), max_nfev=30, given=False)
    refreshed = fitted.reference(starts=("given",), max_nfev=30, given=False, refresh=1)
    a, b = plain["optimum"]["given"], refreshed["optimum"]["given"]
    np.testing.assert_allclose(b.x, a.x, atol=5e-4)
    assert len(b.table) == len(fitted.table) and np.any(b.table["reserve"])
    with pytest.raises(ValueError, match="refresh"):
        fitted.reference(refresh=-1)


def test_initialize_saves_when_asked(data):
    folder, fit = data
    res = fit.initialize(save_to=folder / "init")
    for name in ("psf.fits", "stars.fits", "config.yaml", "catalog.fits"):
        assert (folder / "init" / name).exists(), name
    back = load_data_fit(folder / "init")
    assert back.fit_summary["initialized_only"]
    stars, table = back.fitted_stars()
    assert len(stars) == len(res.table)
    np.testing.assert_allclose(table["flux"], res.table["flux"])
