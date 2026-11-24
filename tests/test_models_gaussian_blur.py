"""Piff's GaussianBlur as our kernel kind ``'blur'``.

Checks our assumptions about it (units, parameter order, PSD projection) against the
simulation's Gaussian kernel (`psf_fomo.sim.psf.gaussian_kernel` of the PSD-clipped M), and
the package glue: PiffFitConfig, components, reference solver.
"""
import copy

import numpy as np
import galsim
import galsim.roman
import piff
import pytest

from psf_fomo.piff_simfit import PiffFitConfig
from psf_fomo.piff_simfit.components import component_image, kernel_moments, kind_of, leaf_psfs
from psf_fomo.piff_simfit.config import blur_moments, blur_params
from psf_fomo.piff_simfit.run import make_logger
from psf_fomo.reference import ReferenceSolver
from psf_fomo.reference.solver import BLUR_STEP
from psf_fomo.sim.psf import clip_to_psd, gaussian_kernel

PIX = galsim.roman.pixel_scale
KERNEL_M = (0.04, 0.01, 0.02)   # pix², inside the PSD cone
BASE = galsim.Gaussian(sigma=0.12)
FIXED_BASE = {"type": "Simple",
              "model": {"type": "GaussianMixture", "weights": [1.0], "sigmas": [0.12]},
              "interp": {"type": "Fixed"}}


def draw(prof, n=25):
    return prof.drawImage(nx=n, ny=n, scale=PIX, method="auto").array


def test_param_conversion_order_and_units():
    p = blur_params(0.04, 0.01, 0.02)
    np.testing.assert_allclose(p, np.array([0.04, 0.02, 0.01]) * PIX**2)
    np.testing.assert_allclose(blur_moments(p), (0.04, 0.01, 0.02))


@pytest.mark.parametrize("m", [KERNEL_M, (0.0, 0.0, 0.0), (0.03, 0.0, 0.0),
                               (0.01, 0.02, 0.01)])  # last one is non-PSD: both clip it
def test_blur_draws_like_clipped_gaussian_kernel(m):
    blur = piff.GaussianBlur().getProfile(np.array(blur_params(*m)))
    ours = gaussian_kernel(*clip_to_psd(*m))
    a, b = draw(galsim.Convolve(BASE, blur)), draw(galsim.Convolve(BASE, ours))
    np.testing.assert_allclose(a, b, atol=2e-5 * b.max())


def test_config_builds_blur():
    cfg = PiffFitConfig()
    cfg.set_optics("pinned")
    cfg.add_kernel("blur", init=KERNEL_M)
    kernel = cfg.piff_config("H158")["components"][1]
    assert kernel["model"] == {"type": "GaussianBlur", "init": blur_params(*KERNEL_M)}
    assert kernel["interp"] == {"type": "Polynomial", "order": 0}
    cfg.add_kernel("blur", interp="BasisPolynomial")
    kernel = cfg.piff_config("H158")["components"][1]
    assert kernel["model"] == {"type": "GaussianBlur"}
    assert kernel["interp"] == {"type": "BasisPolynomial", "order": 0}
    with pytest.raises(ValueError):
        cfg.add_kernel("signed", interp="BasisPolynomial")
    with pytest.raises(ValueError):
        cfg.add_kernel("blur", interp="Mean")
    psf = piff.PSF.process(cfg.piff_config("H158"))
    assert [kind_of(leaf) for leaf in leaf_psfs(psf)] == ["optics", "kernel", "mtf"]


def test_old_settings_without_interp_mean_polynomial():
    cfg = PiffFitConfig()
    cfg.add_kernel("signed")
    del cfg.config["kernel"]["interp"]
    assert cfg.piff_config("H158")["components"][1]["interp"]["type"] == "Polynomial"


def kernel_psf(interp="Polynomial", **iterations):
    config = {"type": "Convolve", "min_iter": 2, "max_iter": 20, "chisq_thresh": 1e-8,
              **iterations, "components": [
                  FIXED_BASE,
                  {"type": "Simple", "model": {"type": "GaussianBlur"},
                   "interp": {"type": interp, "order": 0}}]}
    psf = piff.PSF.process(copy.deepcopy(config))
    psf.set_context({1: galsim.PixelScale(PIX)}, None, None)
    return psf


def truth_profile(m=KERNEL_M):
    return galsim.Convolve(BASE, gaussian_kernel(*m))


def test_kernel_moments_projection_and_clip_flag():
    psf = kernel_psf()
    leaf = leaf_psfs(psf)[1]
    m, clipped = kernel_moments(leaf, blur_params(*KERNEL_M))
    np.testing.assert_allclose(m, KERNEL_M)
    assert not clipped
    m, clipped = kernel_moments(leaf, blur_params(0.01, 0.0, -0.01))
    np.testing.assert_allclose(m, (0.01, 0.0, 0.0), atol=1e-12)
    assert clipped


@pytest.mark.parametrize("interp", ["Polynomial", "BasisPolynomial"])
def test_blur_recovered_in_piff_fit(interp, synthetic_stars):
    stars, _ = synthetic_stars(truth_profile(), n=6, flux=1e4)
    psf = kernel_psf(interp)
    logger, watch = make_logger(0)
    psf.fit(stars, logger=logger)
    assert not watch.triggered, "Piff reached max_iter"
    leaf = leaf_psfs(psf)[1]
    m, _ = kernel_moments(leaf, psf.stars[0].fit.get_params(leaf._num))
    np.testing.assert_allclose(m, KERNEL_M, atol=5e-4)


def test_component_image_at_zero_blur(synthetic_stars):
    # A bare FourierGaussian at zero blur has maxk = 1e100: drawn in real space instead.
    stars, _ = synthetic_stars(BASE, n=3)
    psf = kernel_psf(max_iter=1, min_iter=1)
    psf.fit(stars, logger=make_logger(0)[0])
    leaf = leaf_psfs(psf)[1]
    for oversample, include_pixel in ((3, False), (3, True), (1, True)):
        image = component_image(psf, leaf, 1000.0, 1000.0, sca=1, oversample=oversample,
                                size=7, include_pixel=include_pixel)
        # Unit flux whether integrated over image pixels or convolved with the native pixel,
        # and no negative values (a Gaussian is positive; an FFT of the pixel would ring).
        np.testing.assert_allclose(image.array.sum(), 1.0, atol=2e-3)
        assert image.array.min() >= -1e-8
    # With the native pixel, a zero blur is the pixel itself: flat over the central pixel.
    native = component_image(psf, leaf, 1000.0, 1000.0, sca=1, oversample=1, size=7).array
    assert native[3, 3] > 0.99 and native.sum() - native[3, 3] < 0.01


def test_reference_recovers_blur(synthetic_stars):
    stars, rows = synthetic_stars(truth_profile(), n=6)
    for key, value in zip(("mxx", "mxy", "myy"), KERNEL_M):
        rows[f"kernel_{key}"] = value
    psf = kernel_psf(max_iter=2, chisq_thresh=0.1)   # deliberately stopped early
    psf.fit(stars, logger=make_logger(0)[0])
    solver = ReferenceSolver(psf, psf.stars, rows)
    kernel = solver.leaves[1]
    assert kernel.step == BLUR_STEP and kernel.kind == "kernel"
    np.testing.assert_allclose(solver.x_start("zero"), blur_params(0.01, 0.0, 0.01))
    out = solver.solve(starts=("given", "zero"), max_nfev=60)
    for start, res in out["optimum"].items():
        assert res.summary["converged"], start
        k = res.components["kernel"]
        for key in ("dmxx", "dmxy", "dmyy"):
            assert np.all(np.abs(k[key]) < 2e-4), (start, key)
    assert out["optimum"]["zero"].summary["leaves"][1]["linear"] is False
