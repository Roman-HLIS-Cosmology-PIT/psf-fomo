"""PiffSimFit runs, reserve stars, shape-method context, convergence flag, diagnostics."""
import numpy as np
import galsim
import piff
import pytest

from psf_fomo.piff_simfit import PiffFitConfig, PiffSimFit, load_sim_fit, piff_shape_method
from psf_fomo.piff_simfit import diagnostics
from psf_fomo.piff_simfit.components import kernel_moments
from psf_fomo.piff_simfit.run import NOT_CONVERGED, choose_reserve, make_logger
from psf_fomo.sim import RomanSimulation


def test_choose_reserve():
    mask = choose_reserve(30, frac=0.2, seed=4)
    assert mask.sum() == 6
    np.testing.assert_array_equal(mask, choose_reserve(30, frac=0.2, seed=4))
    assert choose_reserve(9, frac=0.2).sum() == 1  # rounded down, as Piff
    assert choose_reserve(10, frac=0.0).sum() == 0


def test_shape_method_restored_after_error():
    assert piff.star.get_shape_method() == "hsm"
    with pytest.raises(RuntimeError):
        with piff_shape_method("wmom", 0.4):
            assert piff.star.get_shape_method() == "wmom"
            raise RuntimeError("boom")
    assert piff.star.get_shape_method() == "hsm"
    assert piff.star.get_shape_weight_fwhm() is None


def test_logger_watch_and_quiet_output(capsys):
    logger, watch = make_logger(0)
    logger.warning("some other warning")
    assert not watch.triggered
    logger.warning(NOT_CONVERGED + ".  Max iterations = %d reached.", 3)
    assert watch.triggered
    assert capsys.readouterr().out == ""  # verbose=0 prints errors only


def gaussian_psf(max_iter):
    return piff.PSF.process({"type": "Simple", "model": {"type": "Gaussian"},
                             "interp": {"type": "Mean"}, "max_iter": max_iter})


@pytest.mark.parametrize("max_iter, converged", [(1, False), (20, True)])
def test_convergence_flag_from_piff(max_iter, converged, synthetic_stars):
    stars, _ = synthetic_stars(galsim.Gaussian(sigma=0.15), n=5, sigma=1.0)
    psf = gaussian_psf(max_iter)
    psf.set_context({1: galsim.PixelScale(0.11)}, None, None)
    logger, watch = make_logger(0)
    psf.fit(stars, logger=logger)
    # With max_iter=1 Piff cannot test convergence (no previous chi²) and warns.
    assert (not watch.triggered) == converged


def test_star_table_on_exact_model(synthetic_stars):
    stars, rows = synthetic_stars(galsim.Gaussian(sigma=0.15), n=6, flux=2e4, sigma=1.0)
    stars[1].data.properties["is_reserve"] = True
    psf = gaussian_psf(20)
    psf.set_context({1: galsim.PixelScale(0.11)}, None, None)
    psf.fit(stars, logger=make_logger(0)[0])
    table = diagnostics.star_table(psf, psf.stars, rows)
    assert list(table["reserve"]) == [False, True, False, False, False, False]
    np.testing.assert_allclose(table["flux_r"], 1.0, atol=2e-3)
    np.testing.assert_allclose(table["model_flux"], 1.0, atol=1e-6)
    np.testing.assert_allclose(table["dx"], 0.0, atol=5e-3)
    # dx is the arcsec center through the WCS.
    np.testing.assert_allclose(table["dx"], table["du"] / 0.11)
    assert 0.8 < np.mean(table["chisq_dof"]) < 1.2
    summary = diagnostics.summary(table)
    assert list(summary["set"]) == ["fit", "reserve"]
    assert list(summary["n"]) == [5, 1]
    fs = diagnostics.fit_summary(psf, True, 20, table)
    assert fs["n_reserve"] == 1 and fs["kernel_max_nfev"] is None


def test_validation_settings():
    # The builder carries the same config as a Simulation; nothing is rendered here.
    sim = RomanSimulation(scas=[1, 5], n_obs=2)
    with pytest.raises(ValueError):
        PiffSimFit(sim, PiffFitConfig(), sca=3)
    with pytest.raises(ValueError):
        PiffSimFit(sim, PiffFitConfig(), obs=2)
    pv = PiffSimFit(sim, PiffFitConfig(), sca=5, obs=1)
    assert pv.sca == 5 and pv.obs == [1]
    with pytest.raises(ValueError):
        pv.set_reserve(1.0)


def faint_simulation(n_stars=20, **errors):
    """Moderate S/N so that tiny rendering differences do not dominate χ²."""
    sim = RomanSimulation(n_stars=n_stars, seed=11)
    sim.set_brightness(mag_range=(19.5, 20.5))
    sim.set_noise("gaussian", sigma=10.0)
    if "kernel" in errors:
        sim.add_kernel(*errors["kernel"])
    return sim.run()


@pytest.mark.slow
def test_closed_loop_no_error(tmp_path):
    sim = faint_simulation()
    cfg = PiffFitConfig()
    cfg.set_optics("pinned")
    pv = PiffSimFit(sim, cfg, sca=1)
    pv.set_reserve(0.2, seed=2)
    res = pv.run(save_to=tmp_path / "fit")
    fs = res.fit_summary
    # Random placement may put a pair within the default min_sep (2.5″): both are rejected.
    n_kept = fs["n_input"] - fs["n_rejected"]
    assert fs["n_input"] == 20 and len(res.table) == n_kept
    n_fit = n_kept - int(0.2 * n_kept)
    assert fs["converged"] and fs["n_reserve"] == int(0.2 * n_kept) and fs["n_flagged"] == 0
    # dof ≈ 16 × 622: 3σ of χ²/dof is ≈ 0.03.
    assert abs(fs["chisq_dof"] - 1) < 0.05
    good = res.table[res.table["flag"] == 0]
    assert abs(np.mean(good["flux_r"]) - 1) < 2e-3
    assert np.all(np.abs(good["dx"]) < 0.05) and np.all(np.abs(good["dy"]) < 0.05)

    # Moment residuals of an exact model: zero up to noise, for every comparison.
    ms = res.moment_summary()
    fit_rows = ms[ms["set"] == "fit"]
    for row in fit_rows:
        assert row["n"] + row["n_flagged"] == n_fit
        if row["comparison"] != "data":  # noiseless truth: model error only
            assert abs(row["dT_T_mean"]) < 2e-3, (row["comparison"], row["algorithm"])
    os_hsm = fit_rows[(fit_rows["comparison"] == "truth_os") & (fit_rows["algorithm"] == "hsm")]
    assert os_hsm["n_flagged"][0] == 0
    optics = res.components()["optics"]
    assert list(optics["noll"]) == [4] and abs(optics["diff_mean"][0]) < 1e-6  # pinned at 0

    back = load_sim_fit(tmp_path / "fit", sim=sim)
    assert back.fit_summary == fs
    np.testing.assert_allclose(back.table["flux"], res.table["flux"])
    np.testing.assert_allclose(back.moments["dT_T"], res.moments["dT_T"])
    a = res.psf.draw(x=2000.0, y=2100.0, chipnum=1, stamp_size=15).array
    b = back.psf.draw(x=2000.0, y=2100.0, chipnum=1, stamp_size=15).array
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-12)
    # Stars rebuilt from the simulation carry the saved fits and reserve marks.
    stars, _ = back.fitted_stars()
    assert [s.is_reserve for s in stars] == list(res.table["reserve"])
    redo = back.compute_moments(native=("wmom",), oversampled=(), comparisons=("data",))
    orig = res.moments[(res.moments["comparison"] == "data")
                       & (res.moments["algorithm"] == "wmom_0.4")]
    np.testing.assert_allclose(redo["T_model"], orig["T_model"], rtol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["blur", "signed"])
def test_closed_loop_kernel_recovered(kind):
    # Truth = optics ⊛ Gaussian kernel M ⊛ MTF; fit with Piff's GaussianBlur (in family) or
    # our signed GaussianMoments.
    true_m = (0.03, 0.005, 0.02)
    sim = faint_simulation(kernel=true_m)
    cfg = PiffFitConfig()
    cfg.set_optics("pinned")
    cfg.add_kernel(kind, order=0)
    res = PiffSimFit(sim, cfg).run()
    assert res.fit_summary["converged"]
    # None for GaussianBlur (no per-star scipy fit), 0 for GaussianMoments.
    assert res.fit_summary["kernel_max_nfev"] == (None if kind == "blur" else 0)
    kernel = res.psf.components[1]
    fitted, _ = kernel_moments(kernel, res.psf.stars[0].fit.get_params(kernel._num))
    # Signed is first order in M / T_psf (T_psf ≈ 0.6 pix² pre-pixel for H158).
    np.testing.assert_allclose(fitted, true_m, atol=3e-3 if kind == "blur" else 5e-3)
    assert abs(res.fit_summary["chisq_dof"] - 1) < 0.05
