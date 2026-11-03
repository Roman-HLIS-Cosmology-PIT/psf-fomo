"""Moment measurements on analytic Gaussians (no Roman PSF rendering)."""
import numpy as np
import galsim
import galsim.roman
import pytest

from psf_fomo.moments.measure import (T_e_to_sigma, algorithm_label, e_to_g, g_to_e,
                                      make_star, measure_star, moments_matrix, sigma_g_to_T)

PIX = galsim.roman.pixel_scale
NGMIX_ALGORITHMS = ["am", "gauss", "wmom"]


def gaussian_star(sigma_pix, g1, g2, wcs=None, n=64, method="no_pixel", flux=1000.0, noise=0.0):
    """Star of an elliptical Gaussian (world sigma in native pixels), centred on the stamp.

    ``noise`` > 0 adds fixed-seed Gaussian noise of that rms with the matching weight.
    """
    wcs = galsim.PixelScale(PIX) if wcs is None else wcs
    prof = galsim.Gaussian(sigma=sigma_pix * PIX, flux=flux).shear(g1=g1, g2=g2)
    image = galsim.ImageD(n, n, wcs=wcs)
    prof.drawImage(image, method=method)
    if noise > 0:
        image.addNoise(galsim.GaussianNoise(galsim.BaseDeviate(1234), sigma=noise))
        return make_star(image, image.true_center, 1.0 / noise**2)
    return make_star(image, image.true_center)


def test_converters_round_trip():
    g1, g2 = 0.12, -0.07
    e1, e2 = g_to_e(g1, g2)
    np.testing.assert_allclose(e_to_g(e1, e2), (g1, g2), rtol=1e-12)
    T = sigma_g_to_T(2.0, g1, g2)
    np.testing.assert_allclose(T_e_to_sigma(T, e1, e2), 2.0, rtol=1e-12)
    mxx, mxy, myy = moments_matrix(T, e1, e2)
    np.testing.assert_allclose(mxx + myy, T)
    np.testing.assert_allclose(np.sqrt(mxx * myy - mxy**2), 4.0, rtol=1e-12)  # sigma² = det^½


def test_algorithm_label():
    assert algorithm_label("hsm") == "hsm"
    assert algorithm_label("wmom", 0.4) == "wmom_0.4"


def test_hsm_recovers_gaussian():
    res = measure_star(gaussian_star(3.0, 0.1, -0.05), "hsm")
    assert res["flag"] == 0
    np.testing.assert_allclose(res["sigma"], 3.0, rtol=1e-3)
    np.testing.assert_allclose((res["g1"], res["g2"]), (0.1, -0.05), atol=1e-3)
    np.testing.assert_allclose(res["T"], sigma_g_to_T(3.0, 0.1, -0.05), rtol=2e-3)
    np.testing.assert_allclose(res["flux"], 1000.0, rtol=1e-3)


@pytest.mark.parametrize("algorithm", ["am", "gauss"])
def test_ngmix_adaptive_recovers_gaussian(algorithm):
    pytest.importorskip("ngmix")
    # Small noise (peak S/N ~ 2000): on this exactly noiseless star ngmix's Gaussian fit
    # converges to the right parameters but returns flag 512, which Piff reports as failure.
    res = measure_star(gaussian_star(3.0, 0.1, -0.05, noise=0.01), algorithm)
    assert res["flag"] == 0
    np.testing.assert_allclose(res["sigma"], 3.0, rtol=1e-3)
    np.testing.assert_allclose((res["g1"], res["g2"]), (0.1, -0.05), atol=1e-3)
    # am doubles ngmix's Gaussian-weighted flux to match HSM's convention.
    np.testing.assert_allclose(res["flux"], 1000.0, rtol=1e-2)


def test_wmom_matches_analytic_weighted_moments():
    pytest.importorskip("ngmix")
    sigma, g1, g2, fwhm = 2.0, 0.1, -0.05, 0.4
    res = measure_star(gaussian_star(sigma, g1, g2), "wmom", weight_fwhm=fwhm)
    assert res["flag"] == 0
    # Gaussian × Gaussian weight: covariance (C⁻¹ + W⁻¹)⁻¹, in arcsec².
    e1, e2 = g_to_e(g1, g2)
    mxx, mxy, myy = moments_matrix(sigma_g_to_T(sigma * PIX, g1, g2), e1, e2)
    cov = np.array([[mxx, mxy], [mxy, myy]])
    sw2 = (fwhm / (2 * np.sqrt(2 * np.log(2))))**2
    cw = np.linalg.inv(np.linalg.inv(cov) + np.eye(2) / sw2)
    T = np.trace(cw)
    np.testing.assert_allclose(res["T"], T / PIX**2, rtol=1e-3)
    np.testing.assert_allclose(res["e1"], (cw[0, 0] - cw[1, 1]) / T, atol=1e-4)
    np.testing.assert_allclose(res["e2"], 2 * cw[0, 1] / T, atol=1e-4)


@pytest.mark.parametrize("algorithm", ["hsm"] + NGMIX_ALGORITHMS)
def test_rotated_wcs_gives_world_shapes(algorithm):
    if algorithm != "hsm":
        pytest.importorskip("ngmix")
    theta = np.radians(30.0)
    c, s = np.cos(theta), np.sin(theta)
    rotated = galsim.JacobianWCS(PIX * c, -PIX * s, PIX * s, PIX * c)
    plain = measure_star(gaussian_star(2.5, 0.08, 0.03), algorithm)
    rot = measure_star(gaussian_star(2.5, 0.08, 0.03, wcs=rotated), algorithm)
    np.testing.assert_allclose(rot["T"], plain["T"], rtol=2e-3)
    np.testing.assert_allclose((rot["e1"], rot["e2"]), (plain["e1"], plain["e2"]), atol=1e-3)


@pytest.mark.parametrize("algorithm", ["hsm"] + NGMIX_ALGORITHMS)
def test_native_and_oversampled_agree_when_well_sampled(algorithm):
    if algorithm != "hsm":
        pytest.importorskip("ngmix")
    # Native: drawn with the pixel ('auto'). Oversampled: pixel-convolved profile sampled on
    # a 5× finer grid. Both images represent the same effective PSF.
    native = measure_star(gaussian_star(4.0, 0.05, 0.0, method="auto"), algorithm,
                          weight_fwhm=1.0)
    prof = galsim.Convolve(galsim.Gaussian(sigma=4.0 * PIX, flux=1000.0).shear(g1=0.05, g2=0),
                           galsim.Pixel(PIX))
    fine = galsim.ImageD(320, 320, scale=PIX / 5)
    prof.drawImage(fine, method="no_pixel")
    over = measure_star(make_star(fine, fine.true_center), algorithm, weight_fwhm=1.0)
    np.testing.assert_allclose(native["T"], over["T"], rtol=1e-3)
    np.testing.assert_allclose(native["e1"], over["e1"], atol=1e-4)


def test_failed_measurement_is_flagged():
    star = make_star(galsim.ImageD(25, 25, scale=PIX), galsim.PositionD(13, 13))
    res = measure_star(star, "hsm")
    assert res["flag"] != 0
    assert np.isnan(res["T"]) and np.isnan(res["e1"])


def test_unknown_algorithm():
    with pytest.raises(ValueError):
        measure_star(gaussian_star(2.0, 0, 0), "unweighted")
