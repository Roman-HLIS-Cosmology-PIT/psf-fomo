"""Telescope attitude: read from the L2 header (or given), passed to RomanOptics by DataFit.

The converter copies ASDF ``meta.pointing`` into PRIMARY (RA_V1, DEC_V1, PA_V3, TARG_RA,
TARG_DEC, PA_APER; degrees). With a RomanOptics model, DataFit writes boresight_ra,
boresight_dec, telescope_pa into the RomanOptics config when the installed GalSim and Piff
support them (GalSim's getPSF(boresight=..., telescope_PA=...)). The software support is
mocked here; no RomanOptics fit is run.
"""
import numpy as np
import pytest

import psf_fomo.datafit.fit as fit_module
from l2files import field, open_l2, write_field
from psf_fomo.datafit import DataFit, L2Image, StarCatalog, load_data_fit
from psf_fomo.datafit.fit import add_attitude
from psf_fomo.io import read_yaml
from psf_fomo.piffval import FitConfig

HEADER = {"RA_V1": 10.5, "DEC_V1": -40.25, "PA_V3": 123.0, "TARG_RA": 10.6,
          "TARG_DEC": -40.2, "PA_APER": 183.0}
ATTITUDE = {"boresight_ra": 10.5, "boresight_dec": -40.25, "telescope_pa": 123.0}
GAUSSIAN_MODEL = {"type": "Simple", "model": {"type": "Gaussian"},
                  "interp": {"type": "Mean"}, "max_iter": 5}


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("attitude")
    objects = field(n_stars=20, n_gals=0, seed=7)
    with_pointing = write_field(folder / "with", objects, primary=HEADER)
    without = write_field(folder / "without", objects)
    return with_pointing, without


def roman_fit(image, catalog):
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cfg = FitConfig()
    cfg.set_optics(mode="pinned")
    return DataFit(image, cat, cfg), cat


def roman_blocks(psf_config):
    out = []
    if isinstance(psf_config, dict):
        if psf_config.get("type") == "RomanOptics":
            out.append(psf_config)
        for v in psf_config.values():
            out += roman_blocks(v)
    elif isinstance(psf_config, list):
        for v in psf_config:
            out += roman_blocks(v)
    return out


def psf_config(fit, cat, tmp_path):
    t = cat.piff_table(fit.image)
    return fit.piff_config(tmp_path / "catalog.fits", t)["psf"]


def test_pointing_from_header_and_argument(files):
    (image, _), (bare, _) = files
    img = open_l2(image)
    assert img.pointing == pytest.approx({"ra_v1": 10.5, "dec_v1": -40.25, "pa_v3": 123.0,
                                          "target_ra": 10.6, "target_dec": -40.2,
                                          "pa_aperture": 183.0})
    assert img.attitude == ATTITUDE
    # The argument replaces header values; unknown keys (as in meta.pointing) are ignored.
    img2 = open_l2(image, pointing={"pa_v3": 124.0, "quaternion": [1, 0, 0, 0]})
    assert img2.attitude["telescope_pa"] == 124.0 and img2.pointing["ra_v1"] == 10.5
    # No pointing in the header: none known, unless given.
    assert open_l2(bare).pointing == {} and open_l2(bare).attitude is None
    assert open_l2(bare, pointing={"ra_v1": 1.0, "dec_v1": 2.0}).attitude is None   # no PA
    given = open_l2(bare, pointing={"ra_v1": 1.0, "dec_v1": 2.0, "pa_v3": 3.0})
    assert given.attitude == {"boresight_ra": 1.0, "boresight_dec": 2.0, "telescope_pa": 3.0}
    # Saved with the image config and rebuilt.
    again = L2Image.from_config(given.config())
    assert again.attitude == given.attitude


def test_compare_frames_needs_the_whole_pointing(files):
    (_, _), (bare, _) = files
    with pytest.raises(ValueError, match="pointing values"):
        open_l2(bare).compare_frames()


def test_add_attitude_reaches_nested_roman_optics():
    config = {"type": "Convolve", "components": [
        {"type": "RomanOptics", "max_zernike": 4}, {"type": "GaussianMixture"}]}
    add_attitude(config, ATTITUDE)
    assert config["components"][0]["telescope_pa"] == 123.0
    assert "boresight_ra" not in config["components"][1]


def test_piff_config_with_and_without_attitude(files, tmp_path, monkeypatch):
    (image, catalog), (bare, _) = files
    monkeypatch.setattr(fit_module, "attitude_support", lambda: [])
    fit, cat = roman_fit(open_l2(image), catalog)
    blocks = roman_blocks(psf_config(fit, cat, tmp_path))
    assert len(blocks) == 1 and {k: blocks[0][k] for k in ATTITUDE} == ATTITUDE
    assert fit.attitude() == ATTITUDE

    fit.set_attitude(False)
    assert fit.attitude() is None
    assert "boresight_ra" not in roman_blocks(psf_config(fit, cat, tmp_path))[0]

    no_pointing, cat2 = roman_fit(open_l2(bare), catalog)
    assert no_pointing.attitude() is None
    assert "boresight_ra" not in roman_blocks(psf_config(no_pointing, cat2, tmp_path))[0]
    no_pointing.set_attitude(True)
    with pytest.raises(ValueError, match="RA_V1"):
        no_pointing.attitude()

    # A model without RomanOptics never gets the attitude.
    gauss = DataFit(open_l2(image), cat, FitConfig.from_piff_config(GAUSSIAN_MODEL))
    assert gauss.attitude() is None
    with pytest.raises(ValueError, match="'auto', True or False"):
        fit.set_attitude("yes")


def test_attitude_without_software_support(files, tmp_path, monkeypatch):
    (image, catalog), _ = files
    monkeypatch.setattr(fit_module, "attitude_support", lambda: ["a newer GalSim"])
    fit, cat = roman_fit(open_l2(image), catalog)
    with pytest.warns(UserWarning, match="oriented from the WCS"):
        assert fit.attitude() is None
    assert "boresight_ra" not in roman_blocks(psf_config(fit, cat, tmp_path))[0]
    fit.set_attitude(True)
    with pytest.raises(RuntimeError, match="a newer GalSim"):
        fit.attitude()
    with pytest.raises(RuntimeError, match="a newer GalSim"):     # before any work
        fit.run(save_to=tmp_path / "run")
    assert not (tmp_path / "run" / "catalog.fits").exists()


def test_attitude_support_reports_strings():
    missing = fit_module.attitude_support()
    assert isinstance(missing, list) and all(isinstance(m, str) for m in missing)


def test_run_records_attitude_and_pointing(files, tmp_path):
    (image, catalog), _ = files
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(max_snr_weight=0)
    fit = DataFit(open_l2(image), cat, FitConfig.from_piff_config(GAUSSIAN_MODEL))
    fit.set_diagnostics(moments=False)
    res = fit.run(save_to=tmp_path)
    assert res.checks["attitude"] == {"requested": "auto", "used": False, "values": None}
    saved = read_yaml(tmp_path / "config.yaml")
    assert saved["config"]["attitude"] == "auto"
    assert saved["config"]["image"]["pointing"]["pa_v3"] == 123.0
    back = load_data_fit(tmp_path)
    assert back.checks["attitude"]["used"] is False
    assert np.all(np.isfinite(back.table["flux"]))
