"""FitConfig: generated Piff structure, validation, YAML round trip, Piff accepts it."""
import numpy as np
import galsim
import galsim.roman
import piff
import pytest

from psf_fomo.piffval import FitConfig
from psf_fomo.piffval.config import PINNED_MAX_ZERNIKE, PINNED_PRIOR_SIGMA
from psf_fomo.sim.psf import MTF_BASE_SIGMA, MTF_REL_SIGMAS, MTF_WEIGHTS, mtf_kernel


def types(cfg):
    """Nested list of component type names."""
    if "components" in cfg:
        return [cfg["type"], [types(c) for c in cfg["components"]]]
    return cfg["type"] if cfg["type"] != "Simple" else cfg["model"]["type"]


def test_default_is_optics_with_mtf():
    psf = FitConfig().piff_config("H158")
    assert types(psf) == ["Convolve", ["RomanOptics", "GaussianMixture"]]
    optics = psf["components"][0]
    assert "chromatic" not in optics and optics["filter_name"] == "H158"   # Piff's default
    assert optics["max_zernike"] == 22 and optics["aberration_prior_sigma"] == 0.05
    assert (psf["max_iter"], psf["min_iter"], psf["chisq_thresh"]) == (30, 2, 0.1)


def test_full_structure_and_order():
    cfg = FitConfig()
    cfg.add_kernel("signed", order=1)
    cfg.add_pixelgrid(size=7, scale=0.11, order=2)
    cfg.set_outliers(chisq_nsigma=4, max_remove=0.05, max_centroid=0.5)
    psf = cfg.piff_config("H158")
    assert types(psf) == ["Sum", [["Convolve", ["RomanOptics", "GaussianMoments",
                                               "GaussianMixture"]], "PixelGrid"]]
    kernel = psf["components"][0]["components"][1]
    assert "signed" not in kernel["model"]   # GaussianMoments is always signed
    assert kernel["interp"] == {"type": "Polynomial", "order": 1}
    assert psf["components"][1]["interp"] == {"type": "BasisPolynomial", "order": 2}
    assert psf["outliers"] == [{"type": "Chisq", "nsigma": 4.0, "max_remove": 0.05},
                               {"type": "Centroid", "max_offset": 0.5}]


def test_no_mtf_no_convolve():
    cfg = FitConfig()
    cfg.set_mtf(False)
    assert cfg.piff_config("H158")["type"] == "RomanOptics"


def test_pinned_optics():
    cfg = FitConfig()
    cfg.set_optics("pinned", max_zernike=12, prior_sigma=0.1)
    optics = cfg.piff_config("H158")["components"][0]
    assert optics["max_zernike"] == PINNED_MAX_ZERNIKE
    assert optics["aberration_prior_sigma"] == PINNED_PRIOR_SIGMA


def test_prior_sigma_options():
    cfg = FitConfig()
    cfg.set_optics("free", max_zernike=8, prior_sigma=[0.1, 0.1, 0.05, 0.05, 0.02])
    assert cfg.config["optics"]["prior_sigma"] == [0.1, 0.1, 0.05, 0.05, 0.02]
    cfg.set_optics("free", prior_sigma=None)
    assert cfg.piff_config("H158")["components"][0]["aberration_prior_sigma"] is None
    with pytest.raises(ValueError):
        cfg.set_optics("free", max_zernike=8, prior_sigma=[0.1, 0.1])
    with pytest.raises(ValueError):
        cfg.set_optics("free", prior_sigma=-1.0)


def test_invalid_settings():
    cfg = FitConfig()
    with pytest.raises(ValueError):
        cfg.set_optics("fixed")
    with pytest.raises(ValueError):
        cfg.add_kernel("moffat")
    with pytest.raises(ValueError, match="'blur'"):
        cfg.add_kernel("positive")   # replaced by Piff's GaussianBlur
    with pytest.raises(ValueError):
        cfg.add_pixelgrid(size=0)


def test_saved_positive_kernel_is_rejected():
    cfg = FitConfig()
    cfg.add_kernel("signed")
    cfg.config["kernel"]["kind"] = "positive"   # as in settings saved before the removal
    old = FitConfig.from_config(cfg.config)
    with pytest.raises(ValueError, match="not available"):
        old.piff_config("H158")
    with pytest.raises(ValueError):
        cfg.set_piff_shape_method("wmom")  # needs weight_fwhm


def test_yaml_round_trip(tmp_path):
    cfg = FitConfig()
    cfg.set_optics("free", max_zernike=11, prior_sigma=0.02, aberration_interp="linear")
    cfg.add_kernel("blur", order=0, init=(0.02, 0.0, 0.02))
    cfg.add_pixelgrid(size=5)
    cfg.set_piff_shape_method("gauss")
    cfg.to_yaml(tmp_path / "fit.yaml")
    back = FitConfig.from_yaml(tmp_path / "fit.yaml")
    assert back.config == cfg.config
    assert back.piff_config("H158") == cfg.piff_config("H158")


def test_from_piff_config_is_used_as_is():
    raw = {"type": "Simple", "model": {"type": "Gaussian"}, "interp": {"type": "Mean"}}
    assert FitConfig.from_piff_config(raw).piff_config("H158") == raw


@pytest.mark.parametrize("kernel", [None, "signed", "blur", "gaussian"])
@pytest.mark.parametrize("pixelgrid", [False, True])
def test_piff_builds_every_combination(kernel, pixelgrid):
    cfg = FitConfig()
    # Alternate between no prior and a scalar prior to check both reach Piff.
    cfg.set_optics("free", max_zernike=8, prior_sigma=None if kernel else 0.05)
    cfg.add_kernel(kernel)
    if pixelgrid:
        cfg.add_pixelgrid(size=5)
    cfg.set_outliers(chisq_nsigma=4)
    psf = piff.PSF.process(cfg.piff_config("H158"))
    psf.set_context({1: galsim.PixelScale(0.11)}, None,
                    galsim.roman.getBandpasses(AB_zeropoint=True)["H158"])
    assert psf.max_iter == 30


def test_mtf_component_matches_simulation_kernel():
    # Piff's GaussianMixture with our constants = the truth MTF at scale 1.
    cfg = FitConfig()
    mixture = piff.Model.process(cfg.piff_config("H158")["components"][1]["model"])
    np.testing.assert_allclose(mixture.weights, np.array(MTF_WEIGHTS) / sum(MTF_WEIGHTS))
    np.testing.assert_allclose(mixture.sigmas, np.array(MTF_REL_SIGMAS) * MTF_BASE_SIGMA
                               * galsim.roman.pixel_scale)
    local = galsim.PixelScale(galsim.roman.pixel_scale).local()
    truth = local.toWorld(mtf_kernel(1.0))
    fit_side = mixture.getProfile(np.concatenate([mixture.weights, mixture.sigmas, mixture.g1,
                                                  mixture.g2, mixture.du, mixture.dv]))
    kw = dict(nx=64, ny=64, scale=0.011, method="no_pixel")
    a, b = truth.drawImage(**kw).array, fit_side.drawImage(**kw).array
    # The weights sum to 1.00000 to 5 digits, so the profiles agree to ~1e-5 of the peak.
    np.testing.assert_allclose(a, b, atol=1e-4 * a.max())
