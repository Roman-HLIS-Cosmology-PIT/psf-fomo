"""Reference solver on synthetic stars (no getPSF), plus slow Roman closed loops."""
import copy

import numpy as np
import galsim
import galsim.roman
import piff
import pytest

from psf_fomo.models import GaussianMoments  # noqa: F401  (registered)
from psf_fomo.piffval import FitConfig, FitResult, PiffValidation, diagnostics
from psf_fomo.piffval.config import blur_params
from psf_fomo.piffval.run import make_logger
from psf_fomo.reference import ReferenceSolver, comparison_table
from psf_fomo.reference.solver import basis_matrix
from psf_fomo.sim import RomanSimulation

PIX = galsim.roman.pixel_scale
KERNEL_M = (0.04, 0.01, 0.02)
FIXED_BASE = {"type": "Simple",
              "model": {"type": "GaussianMixture", "weights": [1.0], "sigmas": [0.12]},
              "interp": {"type": "Fixed"}}


def kernel_truth():
    cov = np.array([[KERNEL_M[0], KERNEL_M[1]], [KERNEL_M[1], KERNEL_M[2]]]) * PIX**2
    kernel = galsim.Gaussian(sigma=1.0).transform(*np.linalg.cholesky(cov).ravel()).withFlux(1)
    return galsim.Convolve(galsim.Gaussian(sigma=0.12), kernel)


def fit(config, stars):
    psf = piff.PSF.process(copy.deepcopy(config))
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    psf.fit(stars, logger=make_logger(0)[0])
    return psf


def kernel_config(signed=False, init=None, **iterations):
    """Convolve[fixed base, kernel]: Piff's GaussianBlur (exact in family for a Gaussian
    truth kernel) or, with ``signed``, our GaussianMoments."""
    if signed:
        model = {"type": "GaussianMoments"}
    else:
        model = {"type": "GaussianBlur"}
        if init is not None:
            model["init"] = blur_params(*init)
    return {"type": "Convolve", **iterations, "components": [
        FIXED_BASE,
        {"type": "Simple", "model": model, "interp": {"type": "Polynomial", "order": 0}}]}


def test_reference_finishes_an_early_stopped_fit(synthetic_stars):
    # Piff stopped after one iteration from a kernel 5× too large. The joint optimum of the
    # same model is the truth, from Piff's solution and from zero.
    stars, rows = synthetic_stars(kernel_truth(), n=6)
    for key, value in zip(("mxx", "mxy", "myy"), KERNEL_M):
        rows[f"kernel_{key}"] = value  # truth columns, as a simulation catalog has them
    psf = fit(kernel_config(init=(0.2, 0.0, 0.1), max_iter=1, min_iter=1), stars)
    solver = ReferenceSolver(psf, psf.stars, rows)
    out = solver.solve(starts=("given", "zero"), max_nfev=60)
    # Piff's early-stopped solution is measurably off; the reference must improve on it.
    piff_kernel = out["given"].components["kernel"]
    assert np.max(np.abs(piff_kernel["dmxx"])) > 5e-4
    given_chi2 = out["given"].summary["chisq"]
    for start, res in out["optimum"].items():
        assert res.summary["converged"], start
        assert res.summary["chisq"] <= given_chi2
        k = res.components["kernel"]
        for key in ("dmxx", "dmxy", "dmyy"):
            assert np.all(np.abs(k[key]) < 2e-4), (start, key)
        np.testing.assert_allclose(res.table["flux_r"], 1.0, atol=1e-4)
    # The Fixed base component is frozen; only the kernel's 3 coefficients are free.
    assert solver.nfree == 3
    assert [lf.frozen for lf in solver.leaves] == [True, False]
    assert out["given"].summary["leaves"][1]["proj_err"] < 1e-10


def test_reference_does_not_beat_a_converged_exact_fit(synthetic_stars):
    stars, rows = synthetic_stars(galsim.Gaussian(sigma=0.15), n=6, flux=2e4, sigma=1.0)
    psf = fit({"type": "Simple", "model": {"type": "Gaussian"}, "interp": {"type": "Mean"},
               "chisq_thresh": 1e-6, "max_iter": 20}, stars)
    solver = ReferenceSolver(psf, psf.stars, rows)
    out = solver.solve()
    given, best = out["given"].summary, out["optimum"]["given"].summary
    assert best["chisq"] <= given["chisq"] + 1e-6
    # Piff also fits centers, the reference holds them at truth: only a small difference.
    assert given["chisq"] - best["chisq"] < 0.01 * given["dof"]
    n_pix = sum(np.count_nonzero(s.weight.array) for s in stars)
    assert given["dof"] == n_pix - 6 - 3
    assert abs(best["chisq_dof"] - 1) < 0.1


def test_reserve_stars_do_not_enter_chi2(synthetic_stars):
    stars, rows = synthetic_stars(galsim.Gaussian(sigma=0.15), n=5, sigma=1.0)
    stars[0].data.properties["is_reserve"] = True
    psf = fit({"type": "Simple", "model": {"type": "Gaussian"}, "interp": {"type": "Mean"}},
              stars)
    solver = ReferenceSolver(psf, psf.stars, rows)
    x = solver.x_start("given")
    chi2 = solver.profiled(solver.images(x))[1]
    solver.D[0] += 100.0  # spoil the reserve star's data
    assert solver.profiled(solver.images(x))[1] == chi2
    assert solver.dof == sum(np.count_nonzero(s.weight.array) for s in stars[1:]) - 4 - 3


def test_reserve_stars_are_not_drawn_during_optimization(synthetic_stars):
    # Reserve stars have zero weight in χ²: the optimum must not depend on them, and they are
    # drawn only at the reported solutions (report='all'), never during the optimization.
    stars, rows = synthetic_stars(kernel_truth(), n=8, sigma=1.0)
    for s in stars[:5]:
        s.data.properties["is_reserve"] = True
    psf = fit(kernel_config(signed=True, max_iter=3), stars)
    fit_only = [k for k, s in enumerate(psf.stars) if not s.is_reserve]
    assert len(fit_only) == 3

    def run(stars, rows, report):
        solver = ReferenceSolver(psf, stars, rows)
        out = solver.solve(starts=("zero",), max_nfev=30, report=report, given=False)
        return solver, out

    full, out_full = run(psf.stars, rows, "fit")
    sub, out_sub = run([psf.stars[k] for k in fit_only], rows[fit_only], "fit")
    best_full, best_sub = out_full["optimum"]["zero"], out_sub["optimum"]["zero"]
    assert "given" not in out_full
    assert best_full.summary["dof"] == best_sub.summary["dof"]
    np.testing.assert_allclose(best_full.summary["chisq"], best_sub.summary["chisq"],
                               rtol=1e-8)
    for key in ("mxx", "mxy", "myy"):
        np.testing.assert_allclose(best_full.components["kernel"][key],
                                   best_sub.components["kernel"][key], rtol=1e-6, atol=1e-10)
    # Same draws as without the reserve stars; report='fit' gives only the fit stars.
    assert full.n_draws == sub.n_draws
    assert len(best_full.table) == 3 and not np.any(best_full.table["reserve"])
    assert best_full.summary["n_reported"] == 3

    # report='all': the same optimum, plus one exact draw of each reserve star.
    every, out_every = run(psf.stars, rows, "all")
    best_every = out_every["optimum"]["zero"]
    assert every.n_draws == full.n_draws + 5
    assert len(best_every.table) == 8 and np.sum(best_every.table["reserve"]) == 5
    np.testing.assert_allclose(best_every.summary["chisq"], best_full.summary["chisq"],
                               rtol=1e-12)
    with pytest.raises(ValueError, match="report"):
        full.solve(report="reserve")


def test_basis_spans_piff_interpolation(synthetic_stars):
    # Piff's BasisPolynomial (order 1) and Polynomial (order 1) solutions lie in our basis.
    stars, rows = synthetic_stars(kernel_truth(), n=8, sigma=1.0)
    config = {"type": "Sum", "max_iter": 3, "components": [
        {"type": "Convolve", "components": [
            FIXED_BASE,
            {"type": "Simple", "model": {"type": "GaussianMoments"},
             "interp": {"type": "Polynomial", "order": 1}}]},
        {"type": "Simple", "model": {"type": "PixelGrid", "size": 5, "scale": 0.11},
         "interp": {"type": "BasisPolynomial", "order": 1}}]}
    psf = fit(config, stars)
    solver = ReferenceSolver(psf, psf.stars, rows)
    kernel, grid = solver.leaves[1], solver.leaves[2]
    assert kernel.K.shape == (8, 3) and grid.K.shape == (8, 3)
    assert kernel.proj_err < 1e-8 and grid.proj_err < 1e-8
    np.testing.assert_allclose(basis_matrix(grid.psf, psf.stars)[:, 0], 1.0)


def test_linearity_detection_and_freeze(synthetic_stars):
    stars, rows = synthetic_stars(kernel_truth(), n=5, sigma=1.0)
    config = {"type": "Sum", "max_iter": 2, "components": [
        kernel_config(),
        {"type": "Simple", "model": {"type": "PixelGrid", "size": 5, "scale": 0.11},
         "interp": {"type": "BasisPolynomial", "order": 0}}]}
    psf = fit(config, stars)
    solver = ReferenceSolver(psf, psf.stars, rows, freeze=("kernel",))
    assert [lf.kind for lf in solver.free] == ["pixelgrid"]
    solver.jacobian(solver.x_start("given"))
    assert solver.free[0].linear  # a PixelGrid in a Sum (fit_flux) enters linearly
    out = solver.solve(max_nfev=10)
    np.testing.assert_allclose(out["optimum"]["given"].components["kernel"]["mxx"],
                               out["given"].components["kernel"]["mxx"])


def test_comparison_table(synthetic_stars):
    stars, rows = synthetic_stars(kernel_truth(), n=6)
    psf = fit(kernel_config(), stars)
    out = ReferenceSolver(psf, psf.stars, rows).solve(max_nfev=20)
    table = diagnostics.star_table(psf, psf.stars, rows)
    piff_result = FitResult(psf, table, {"chisq_dof": 0.0, "converged": True}, {}, {})
    t = comparison_table({"piff": piff_result, "piff, true centers": out["given"],
                          "optimum": out["optimum"]["given"]})
    assert list(t["label"]) == ["piff", "piff, true centers", "optimum"]
    assert set(t.colnames) >= {"chisq_dof", "flux_r", "dT_T", "converged"}
    assert np.isnan(t["dT_T"][0])  # no moments computed
    assert t["converged"][0] == "True" and t["converged"][1] == ""


@pytest.mark.slow
def test_reference_from_config_in_family_kernel():
    # Roman truth = optics ⊛ kernel M ⊛ MTF; same model, no Piff fit: the optimum is the truth.
    s = RomanSimulation(n_stars=12, seed=5)
    s.set_brightness(mag_range=(19.5, 20.5))
    s.set_noise("gaussian", sigma=10.0)
    s.add_kernel(*KERNEL_M)
    cfg = FitConfig()
    cfg.set_optics("pinned")
    cfg.add_kernel("blur", order=0)
    pv = PiffValidation(s.run(), cfg)
    from psf_fomo.reference import reference_from_config
    out = reference_from_config(pv, starts=("zero",), max_nfev=40)
    best = out["optimum"]["zero"]
    assert best.summary["converged"]
    assert abs(best.summary["chisq_dof"] - 1) < 0.06
    k = best.components["kernel"]
    for key in ("dmxx", "dmxy", "dmyy"):
        assert abs(k[key][0]) < 3e-3, key
