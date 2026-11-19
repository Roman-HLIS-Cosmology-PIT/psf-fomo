"""Moment residuals, the oversampled-model convention, component summaries, plots."""
import copy

import matplotlib

matplotlib.use("Agg")

import numpy as np
import galsim
import galsim.roman
import piff
import pytest

from psf_fomo.models import GaussianMoments  # noqa: F401  (registered)
from psf_fomo.piffval import FitResult, diagnostics, plotting
from psf_fomo.piffval.components import components, kernel_moments, leaf_psfs
from psf_fomo.piffval.residuals import (FLAG_PIFF, _image_pairs, draw_model_oversampled,
                                        moment_residuals, moment_summary)
from psf_fomo.piffval.run import make_logger

PIX = galsim.roman.pixel_scale
TRUTH = galsim.Gaussian(sigma=0.15).shear(g1=0.03, g2=-0.02)


def fitted_gaussian(stars, max_iter=20):
    psf = piff.PSF.process({"type": "Simple", "model": {"type": "Gaussian"},
                            "interp": {"type": "Mean"}, "max_iter": max_iter})
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    psf.fit(stars, logger=make_logger(0)[0])
    return psf


@pytest.fixture
def exact_fit(synthetic_stars, analytic_sim):
    stars, rows = synthetic_stars(TRUTH, n=8, flux=2e4, sigma=1.0, seed=3)
    stars[2].data.properties["is_reserve"] = True
    psf = fitted_gaussian(stars)
    table = diagnostics.star_table(psf, psf.stars, rows)
    return psf, psf.stars, rows, table, analytic_sim(TRUTH, rows, flux=2e4)


def test_oversampled_model_is_effective_psf_sampled_finer(exact_fit):
    # "Oversampled image" convention: the pixel-convolved PSF sampled on a finer grid. At the
    # native-pixel centres it must equal the native drawing (× fine/native pixel area), i.e.
    # the pixel is applied exactly once.
    psf, stars, _, _, sim = exact_fit
    star = stars[0]
    template = sim.truth_psf(1, 0, 0, oversample=5)
    fine = draw_model_oversampled(psf, star, template)
    prof, method = psf._getRawProfile(star)
    assert method == "auto"
    native = prof.drawImage(nx=25, ny=25, scale=PIX, method="auto").array
    # GalSim's FFT drawing is accurate to ~1e-3 of the peak; a second pixel would be ~10 %.
    np.testing.assert_allclose(fine.array[2::5, 2::5] * 25, native, atol=2e-3 * native.max())


def test_pre_pixel_request_with_pixel_model_raises(exact_fit, synthetic_stars):
    stars, _ = synthetic_stars(TRUTH, n=4)
    psf = piff.PSF.process({"type": "Simple",
                            "model": {"type": "Gaussian", "include_pixel": False},
                            "interp": {"type": "Mean"}})
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    psf.fit(stars, logger=make_logger(0)[0])
    template = exact_fit[4].truth_psf(1, 0, 0)
    with pytest.raises(ValueError):
        draw_model_oversampled(psf, psf.stars[0], template, include_pixel=False)


def test_moment_residuals_exact_model(exact_fit):
    pytest.importorskip("ngmix")
    psf, stars, _, table, sim = exact_fit
    m = moment_residuals(psf, stars, table, sim, native=("wmom", "gauss"),
                         wmom_fwhm=(0.4, 0.5), oversampled=("hsm",))
    assert len(m) == len(stars) * (3 + 3 + 1)  # data, truth_native: 3 labels; truth_os: 1
    assert set(m["algorithm"]) == {"wmom_0.4", "wmom_0.5", "gauss", "hsm"}
    good = m[m["flag"] == 0]
    truth = good[good["comparison"] != "data"]
    # The fitted Gaussian is the truth profile: model − truth vanishes up to the fit noise.
    assert np.all(np.abs(truth["dT_T"]) < 5e-3)
    assert np.all(np.abs(truth["de1"]) < 5e-3) and np.all(np.abs(truth["de2"]) < 5e-3)
    # wmom about the same position on both sides; truth_os T is HSM's trace in pix².
    os_rows = good[good["comparison"] == "truth_os"]
    T_expected = 2 * (0.15 / PIX) ** 2 + 2 / 12  # Gaussian + pixel, round-ish
    np.testing.assert_allclose(os_rows["T_ref"], T_expected, rtol=0.02)
    summary = moment_summary(m)
    assert set(summary["set"]) == {"fit", "reserve"}
    assert "dT_T_err" in summary.colnames and "n_flagged" in summary.colnames


def test_data_comparison_measures_both_about_the_catalog_position(exact_fit):
    # Piff (≥ 280064f) measures am/gauss/wmom about star.image_pos + star.fit.center. The data
    # and model sides of 'data' must use the same position: both are fresh stars (centre 0).
    psf, stars, _, table, sim = exact_fit
    star = piff.Star(stars[0].data, stars[0].fit.withNew(center=(0.02, -0.01)))
    pairs = dict(_image_pairs(psf, star, table[0], sim, ("data",), 5, True))
    data, model = pairs["data"]
    assert tuple(data.fit.center) == (0.0, 0.0) and tuple(model.fit.center) == (0.0, 0.0)
    assert data.image_pos == model.image_pos == star.image_pos
    np.testing.assert_array_equal(data.image.array, star.image.array)


def test_piff_flag_propagates(exact_fit):
    psf, stars, _, table, sim = exact_fit
    table = table.copy()
    table["flag"][1] = 1
    m = moment_residuals(psf, stars, table, sim, native=(), oversampled=("hsm",),
                         comparisons=("truth_os",))
    assert m["flag"][1] & FLAG_PIFF and not m["flag"][0] & FLAG_PIFF


def test_invalid_algorithm_or_comparison(exact_fit):
    psf, stars, _, table, sim = exact_fit
    with pytest.raises(ValueError):
        moment_residuals(psf, stars, table, sim, native=("ksb",))
    with pytest.raises(ValueError):
        moment_residuals(psf, stars, table, sim, comparisons=("oversampled",))


def test_chi2_by_radius_noise_level(exact_fit):
    psf, stars, *_ = exact_fit
    r, chi2, n = plotting.chi2_by_radius(psf, stars)
    good = n > 50
    np.testing.assert_allclose(chi2[good], 1.0, atol=0.35)


def test_gaussian_kernel_moments_conversion():
    psf = piff.PSF.process({"type": "Simple", "model": {"type": "Gaussian"},
                            "interp": {"type": "Mean"}})
    leaf = leaf_psfs(psf)[0]
    scale, g1, g2 = 0.05, 0.1, -0.05
    (mxx, mxy, myy), clipped = kernel_moments(leaf, [scale, g1, g2])
    prof = galsim.Gaussian(sigma=1.0).dilate(scale).shear(g1=g1, g2=g2)
    a = prof.drawImage(nx=400, ny=400, scale=0.002, method="no_pixel").array
    y, x = (np.indices(a.shape) - 199.5) * 0.002
    f = a.sum()
    expected = np.array([(a * x * x).sum(), (a * x * y).sum(), (a * y * y).sum()]) / f / PIX**2
    np.testing.assert_allclose([mxx, mxy, myy], expected, rtol=2e-3, atol=1e-7)
    assert not clipped


KERNEL_M = (0.04, 0.01, 0.02)
# Noiseless stars with weight 1: χ² has no noise scale, so Piff's default chisq_thresh (0.1 ×
# dof) would stop after min_iter iterations. Use a tiny threshold for these tests.
ITERATIONS = {"max_iter": 10, "min_iter": 2, "chisq_thresh": 1e-6}
KERNEL_PART = {"type": "Convolve", "components": [
    {"type": "Simple",
     "model": {"type": "GaussianMixture", "weights": [1.0], "sigmas": [0.12]},
     "interp": {"type": "Fixed"}},
    {"type": "Simple", "model": {"type": "GaussianBlur"},   # exact for a Gaussian kernel
     "interp": {"type": "Polynomial", "order": 0}}]}


def kernel_stars(synthetic_stars):
    """Stars = round Gaussian ⊛ Gaussian kernel KERNEL_M; truth columns in the rows."""
    cov = np.array([[KERNEL_M[0], KERNEL_M[1]], [KERNEL_M[1], KERNEL_M[2]]]) * PIX**2
    kernel = galsim.Gaussian(sigma=1.0).transform(*np.linalg.cholesky(cov).ravel()).withFlux(1)
    stars, rows = synthetic_stars(galsim.Convolve(galsim.Gaussian(sigma=0.12), kernel), n=6)
    for key, value in zip(("mxx", "mxy", "myy"), KERNEL_M):
        rows[f"kernel_{key}"] = value
    return stars, rows


def test_kernel_table(synthetic_stars):
    stars, rows = kernel_stars(synthetic_stars)
    psf = piff.PSF.process(dict(copy.deepcopy(KERNEL_PART), **ITERATIONS))
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    psf.fit(stars, logger=make_logger(0)[0])
    comps = components(psf, psf.stars, rows)
    assert set(comps) == {"kernel"}
    k = comps["kernel"]
    np.testing.assert_allclose(k["T_true"], 0.06)
    for key in ("dmxx", "dmxy", "dmyy", "dT"):
        assert np.all(np.abs(k[key]) < 1e-3), key
    assert not np.any(k["clipped"])


def test_pixelgrid_table_in_sum(synthetic_stars):
    # In a Sum the kernel/grid split is not unique (the grid can absorb part of a
    # broadening), so check only identities: the Convolve part has unit flux, so the
    # grid flux is the model's total flux minus 1, star by star.
    stars, rows = kernel_stars(synthetic_stars)
    psf = piff.PSF.process({"type": "Sum", **ITERATIONS, "components": [
        copy.deepcopy(KERNEL_PART),
        {"type": "Simple", "model": {"type": "PixelGrid", "size": 5, "scale": 0.11},
         "interp": {"type": "BasisPolynomial", "order": 0}}]})
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    psf.fit(stars, logger=make_logger(0)[0])
    comps = components(psf, psf.stars, rows)
    assert set(comps) == {"kernel", "pixelgrid"}
    total = [diagnostics.model_flux(psf, s) for s in psf.stars]
    np.testing.assert_allclose(comps["pixelgrid"]["grid_flux"], np.array(total) - 1, atol=1e-10)
    assert np.all(comps["pixelgrid"]["grid_max"] > 0)
    assert len(comps["kernel"]) == len(stars)


def test_plots_run(exact_fit):
    pytest.importorskip("ngmix")
    psf, stars, rows, table, sim = exact_fit
    config = {"sca": 1, "obs": [0]}
    result = FitResult(psf, table, {}, config, {}, sim=sim, stars=stars, rows=rows)
    result.compute_moments(native=("wmom",), oversampled=("hsm",))
    for fig in (result.plot_mean_images(oversampled=True),
                result.plot_mean_images(which="reserve"),
                result.plot_chi2_radius(),
                result.plot_moment_residuals("truth_os"),
                result.plot_moment_residuals("data", algorithms="wmom_0.4"),
                result.plot_field("flux_r"),
                result.plot_field("dT_T", comparison="truth_os", algorithm="hsm"),
                result.plot_components("kernel", n_grid=2)):
        assert fig is not None
    stars_fit, rows_fit = result.selected_stars("fit")
    assert len(stars_fit) == len(stars) - 1 and len(rows_fit) == len(stars_fit)
