"""GaussianMoments kernel: moments of the profile, per-star fit, Piff loop, registration."""
import importlib

import numpy as np
import galsim
import galsim.roman
import piff
import pytest

import psf_fomo.models.gaussian_moments as gm_module
from psf_fomo.models import GaussianMoments
from psf_fomo.piffval.run import make_logger

PIX = galsim.roman.pixel_scale


def moments_pix2(prof, oversample=40, n=400):
    """Flux and unweighted (Mxx, Mxy, Myy) in native pix² of a profile drawn without pixel."""
    scale = PIX / oversample
    a = prof.drawImage(nx=n, ny=n, scale=scale, method="no_pixel").array
    y, x = (np.indices(a.shape) - (n - 1) / 2) / oversample
    f = a.sum()
    return f, (a * x * x).sum() / f, (a * x * y).sum() / f, (a * y * y).sum() / f


def test_registered_and_reimport_is_safe():
    assert piff.Model.valid_model_types["GaussianMoments"] is GaussianMoments
    try:
        reloaded = importlib.reload(gm_module)  # would raise in Piff without the guard
        assert piff.Model.valid_model_types["GaussianMoments"] is reloaded.GaussianMoments
        model = piff.Model.process({"type": "GaussianMoments", "signed": True})
        assert model.signed
    finally:  # keep one class for the other tests (isinstance checks)
        gm_module.GaussianMoments = GaussianMoments
        piff.Model.valid_model_types["GaussianMoments"] = GaussianMoments


def test_positive_profile_moments():
    m = (0.30, 0.05, 0.10)
    f, mxx, mxy, myy = moments_pix2(GaussianMoments().getProfile(m), oversample=20, n=256)
    np.testing.assert_allclose(f, 1.0, atol=1e-3)
    np.testing.assert_allclose([mxx, mxy, myy], m, atol=2e-3)


def test_positive_profile_clips_non_psd():
    # Eigenvalues 0.03 and -0.01 → clipped to 0.03 along the diagonal: 0.015 each.
    f, mxx, mxy, myy = moments_pix2(
        galsim.Convolve(galsim.Gaussian(sigma=0.05), GaussianMoments().getProfile((0.01, 0.02, 0.01))))
    base = (0.05 / PIX) ** 2
    np.testing.assert_allclose([mxx - base, mxy, myy - base], [0.015, 0.015, 0.015], atol=5e-4)


@pytest.mark.parametrize("m", [(0.0, 0.0, 0.0), (0.02, 0.005, -0.015), (-0.01, -0.004, 0.03)])
def test_signed_profile_has_exact_moments(m):
    f, mxx, mxy, myy = moments_pix2(GaussianMoments(signed=True).getProfile(m),
                                    oversample=200, n=400)
    np.testing.assert_allclose(f, 1.0, atol=1e-4)
    np.testing.assert_allclose([mxx, mxy, myy], m, atol=2e-4)


def test_invalid_init():
    with pytest.raises(ValueError):
        GaussianMoments(init=(0.0, 0.0, 0.0))  # positive mode must start inside the cone
    with pytest.raises(ValueError):
        GaussianMoments(init=(0.1, 0.2))


def kernel_fit_psf(signed, max_iter=10, max_nfev=30):
    """Convolve[ fixed round Gaussian (GaussianMixture), GaussianMoments ]."""
    config = {
        "type": "Convolve", "max_iter": max_iter, "min_iter": 2, "chisq_thresh": 1e-6,
        "components": [
            {"type": "Simple",
             "model": {"type": "GaussianMixture", "weights": [1.0], "sigmas": [0.12]},
             "interp": {"type": "Fixed"}},
            {"type": "Simple",
             "model": {"type": "GaussianMoments", "signed": signed, "max_nfev": max_nfev},
             "interp": {"type": "Polynomial", "order": 0}},
        ]}
    return piff.PSF.process(config)


@pytest.mark.parametrize("signed", [False, True])
def test_kernel_recovered_in_piff_fit(signed, synthetic_stars):
    true_m = (0.04, 0.01, 0.02)
    kernel = galsim.Gaussian(sigma=1.0).transform(
        *np.linalg.cholesky(np.array([[0.04, 0.01], [0.01, 0.02]]) * PIX**2).ravel())
    truth = galsim.Convolve(galsim.Gaussian(sigma=0.12), kernel.withFlux(1.0))
    stars, _ = synthetic_stars(truth, n=6, flux=1e4)
    psf = kernel_fit_psf(signed)
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    logger, watch = make_logger(0)
    psf.fit(stars, logger=logger)
    fitted = psf.stars[0].fit.get_params(psf.components[1]._num)
    # Positive mode is exact; signed is first order in M / T_psf (T_psf ≈ 2.4 pix² here).
    tol = 5e-4 if not signed else 2e-3
    np.testing.assert_allclose(fitted, true_m, atol=tol)
    np.testing.assert_allclose([s.fit.flux for s in psf.stars], 1e4, rtol=2e-3)


def test_max_nfev_hits_are_counted(synthetic_stars):
    kernel = galsim.Gaussian(sigma=0.05)
    stars, _ = synthetic_stars(galsim.Convolve(galsim.Gaussian(sigma=0.12), kernel), n=3)
    psf = kernel_fit_psf(signed=False, max_iter=2, max_nfev=2)
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    logger, _ = make_logger(0)
    psf.fit(stars, logger=logger)
    model = psf.components[1].model
    assert len(model.max_nfev_history) == psf.niter
    assert model.n_max_nfev == 3


def test_write_read_round_trip(tmp_path, synthetic_stars):
    stars, _ = synthetic_stars(galsim.Gaussian(sigma=0.13), n=4)
    psf = kernel_fit_psf(signed=True, max_iter=3)
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    logger, _ = make_logger(0)
    psf.fit(stars, logger=logger)
    psf.write(str(tmp_path / "psf.fits"))
    back = piff.read(str(tmp_path / "psf.fits"))
    model = back.components[1].model
    assert isinstance(model, GaussianMoments) and model.signed
    a = psf.draw(x=500.0, y=600.0, chipnum=1, stamp_size=15).array
    b = back.draw(x=500.0, y=600.0, chipnum=1, stamp_size=15).array
    np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-10)
