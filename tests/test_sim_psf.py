import numpy as np
import galsim
import galsim.roman

import pytest

from psf_fomo.sim.psf import (MTF_BASE_SIGMA, MTF_REL_SIGMAS, MTF_WEIGHTS, apply_errors,
                              clip_to_psd, gaussian_kernel, jitter_kernel, linear_field,
                              mtf_kernel, random_walk_offsets, timeseries_offsets)

PIX = galsim.roman.pixel_scale


def unweighted_moments(prof, scale, n):
    """Unweighted second moments (Ixx, Ixy, Iyy) of a profile drawn without a pixel."""
    image = prof.drawImage(nx=n, ny=n, scale=scale, method="no_pixel")
    a = image.array
    y, x = (np.indices(a.shape) - (n - 1) / 2) * scale
    f = a.sum()
    xc, yc = (a * x).sum() / f, (a * y).sum() / f
    x, y = x - xc, y - yc
    return (a * x * x).sum() / f, (a * x * y).sum() / f, (a * y * y).sum() / f


def test_mtf_kernel_variance():
    # Per-axis variance ≈ 0.106 pix²; mtf_scale 1.1 adds ≈ 0.022 pix² (docs/conventions.md).
    analytic = sum(w * (r * MTF_BASE_SIGMA) ** 2 for w, r in zip(MTF_WEIGHTS, MTF_REL_SIGMAS))
    np.testing.assert_allclose(analytic, 0.106, atol=1e-3)
    ixx, ixy, iyy = unweighted_moments(mtf_kernel(1.0), 0.02, 300)
    np.testing.assert_allclose([ixx, iyy], analytic, rtol=1e-3)
    assert abs(ixy) < 1e-6
    ixx11, _, _ = unweighted_moments(mtf_kernel(1.1), 0.02, 300)
    np.testing.assert_allclose(ixx11 - ixx, 0.022, atol=1e-3)
    np.testing.assert_allclose(mtf_kernel(1.0).flux, 1.0)


def test_random_walk_rms_and_centering():
    dx, dy, w = random_walk_offsets(14.0, 200, np.random.default_rng(0))
    np.testing.assert_allclose([dx.std(), dy.std()], 14.0)
    np.testing.assert_allclose([dx.mean(), dy.mean()], 0.0, atol=1e-12)
    np.testing.assert_allclose(w.sum(), 1.0)


def test_timeseries_offsets_weighted_mean_zero():
    t = np.array([0.0, 1.0, 3.0, 4.0])
    dx, dy, w = timeseries_offsets(t, [5.0, 6.0, 7.0, 8.0], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose([np.sum(w * dx), np.sum(w * dy)], 0.0, atol=1e-12)


def test_error_chain_second_moments():
    # Convolutions add second moments: Gaussian (dilated) + jitter + MTF.
    sigma, dilation, rms_mas = 0.1, 1.05, 14.0
    dx, dy, w = random_walk_offsets(rms_mas, 100, np.random.default_rng(1))
    local = galsim.PixelScale(PIX).local()
    prof = apply_errors(galsim.Gaussian(sigma=sigma), local, dilation=dilation,
                        jitter=jitter_kernel(dx, dy, w), mtf_width=1.0)
    mtf_var = sum(wt * (r * MTF_BASE_SIGMA * PIX) ** 2
                  for wt, r in zip(MTF_WEIGHTS, MTF_REL_SIGMAS))
    expected = (sigma * dilation) ** 2 + (rms_mas / 1000) ** 2 + mtf_var
    ixx, _, iyy = unweighted_moments(prof, 0.004, 600)
    np.testing.assert_allclose([ixx, iyy], expected, rtol=2e-3)
    np.testing.assert_allclose(prof.flux, 1.0)


def test_gaussian_kernel_moments():
    # M is in native pix² along world axes; on the sky it is M × 0.11″².
    m = (0.03, 0.01, 0.02)
    kern = gaussian_kernel(*m)
    ixx, ixy, iyy = unweighted_moments(kern, 0.002, 400)
    np.testing.assert_allclose(np.array([ixx, ixy, iyy]) / PIX**2, m, rtol=2e-3, atol=1e-6)
    np.testing.assert_allclose(kern.flux, 1.0)


def test_gaussian_kernel_adds_moments_in_chain():
    m = (0.022, 0.0, 0.022)
    local = galsim.PixelScale(PIX).local()
    base = galsim.Gaussian(sigma=0.1)
    prof = apply_errors(base, local, kernel=gaussian_kernel(*m))
    ixx, ixy, iyy = unweighted_moments(prof, 0.004, 400)
    np.testing.assert_allclose([ixx, iyy], 0.01 + 0.022 * PIX**2, rtol=1e-3)
    # Not dilated with the optics: with dilation the kernel's contribution is unchanged.
    prof = apply_errors(base, local, dilation=1.1, kernel=gaussian_kernel(*m))
    ixx, _, _ = unweighted_moments(prof, 0.004, 400)
    np.testing.assert_allclose(ixx, 0.0121 + 0.022 * PIX**2, rtol=1e-3)


def test_gaussian_kernel_rank_one_and_non_psd():
    # A line along u cannot be sampled on its own: convolve it with a round Gaussian.
    prof = galsim.Convolve(galsim.Gaussian(sigma=0.05), gaussian_kernel(0.02, 0.0, 0.0))
    ixx, _, iyy = unweighted_moments(prof, 0.002, 400)
    np.testing.assert_allclose(ixx, 0.05**2 + 0.02 * PIX**2, rtol=2e-3)
    np.testing.assert_allclose(iyy, 0.05**2, rtol=2e-3)
    with pytest.raises(ValueError):
        gaussian_kernel(0.01, 0.02, 0.01)


def test_clip_to_psd():
    assert clip_to_psd(0.02, 0.005, 0.01) == (0.02, 0.005, 0.01)
    mxx, mxy, myy = clip_to_psd(0.01, 0.02, 0.01)  # eigenvalues 0.03, -0.01
    np.testing.assert_allclose([mxx, mxy, myy], [0.015, 0.015, 0.015])


def test_shear_changes_ellipticity():
    g1 = 0.05
    prof = apply_errors(galsim.Gaussian(sigma=0.1), galsim.PixelScale(PIX).local(), g1=g1)
    ixx, ixy, iyy = unweighted_moments(prof, 0.004, 400)
    e1 = (ixx - iyy) / (ixx + iyy)
    np.testing.assert_allclose(e1, 2 * g1 / (1 + g1 ** 2), rtol=1e-3)
    assert abs(ixy) < 1e-8


def test_no_errors_returns_input():
    gauss = galsim.Gaussian(sigma=0.1)
    assert apply_errors(gauss, galsim.PixelScale(PIX).local()) is gauss


def test_linear_field():
    x = np.array([0.0, 4088.0, 0.0])
    y = np.array([0.0, 0.0, 4088.0])
    np.testing.assert_allclose(linear_field((1.0, 2.0, 3.0), x, y), [1.0, 3.0, 4.0])
    vec = linear_field((np.ones(3), np.zeros(3), np.ones(3)), x, y)
    assert vec.shape == (3, 3)
    np.testing.assert_allclose(vec[2], 2.0)
