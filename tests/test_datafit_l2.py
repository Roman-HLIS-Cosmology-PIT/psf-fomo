"""L2Image: the user's L2 FITS layout, weights, sky, orientation, and Piff reading it."""
import numpy as np
import galsim
import galsim.roman
import piff
import pytest
from astropy.table import Table

from l2files import mirrored_wcs, open_l2, roman_wcs, scene, write_l2_file
from psf_fomo.datafit import L2Image, galsim_filter_name
from psf_fomo.piffval.fitting import make_logger

SKY, READ_VAR, GAIN = 2.0, 0.01, 250.0
STARS = [(60.3, 70.6, 400.0), (180.8, 150.2, 600.0), (90.5, 200.1, 300.0)]  # x, y (1-based), DN/s


@pytest.fixture(scope="module")
def wcs():
    return roman_wcs(sca=5)


@pytest.fixture
def l2_file(tmp_path, wcs):
    def make(name="l2.fits", **kwargs):
        sci = scene(wcs, stars=STARS, sky=SKY, read_var=READ_VAR, gain=GAIN, seed=1)
        return write_l2_file(tmp_path / name, sci, wcs, sky=SKY, read_var=READ_VAR, gain=GAIN,
                             **kwargs)
    return make


def test_metadata_and_hdus(l2_file):
    img = open_l2(l2_file())
    assert img.variant == "unif" and img.sca == 5 and img.filter_name == "H158"
    assert img.sky_header == SKY and img.sky == SKY and img.exptime == pytest.approx(139.8)
    assert img.weights == "model_poisson"
    std = open_l2(img.path, variant="standard")
    names = ["PRIMARY", "SCI", "DQ", "ERR", "MASK", "IVAR", "VAR_POISSON", "VAR_RNOISE", "GAIN",
             "DUMO", "SCI_UNIF", "MASK_UNIF", "IVAR_UNIF", "GAIN_UNIF"]
    assert img.hdus["image"] == names.index("SCI_UNIF")
    assert img.hdus["ivar"] == names.index("IVAR_UNIF")
    assert std.hdus["image"] == names.index("SCI") and std.hdus["gain"] == names.index("GAIN")


def test_missing_uniform_planes(l2_file):
    path = l2_file(uniform=False)
    with pytest.raises(ValueError, match="--uniform"):
        open_l2(path)
    assert open_l2(path, variant="standard").hdus["mask"] == 4


def test_header_overrides_and_errors(l2_file):
    path = l2_file(primary={"DETECTOR": None, "FILTER": None, "SCA": 7, "OPT_ELEM": "J129"})
    img = open_l2(path)
    assert img.sca == 7 and img.filter_name == "J129"
    path = l2_file("bare.fits", primary={"DETECTOR": None, "FILTER": None})
    with pytest.raises(ValueError, match="SCA"):
        open_l2(path)
    img = open_l2(path, sca=5, filter_name="F158")
    assert (img.sca, img.filter_name) == (5, "H158")


def test_filter_names():
    assert galsim_filter_name("f158") == "H158" and galsim_filter_name("H158") == "H158"
    assert galsim_filter_name("F184") == "F184"
    with pytest.raises(ValueError):
        galsim_filter_name("F999")


def test_check_weights(l2_file):
    # Uniform planes and 'ivar' mode need no check.
    assert open_l2(l2_file()).check_weights() is None
    good = open_l2(l2_file("bg.fits"), variant="standard")
    assert good.check_weights() == pytest.approx(1.0, abs=1e-5)
    data_based = open_l2(l2_file("err.fits", background_weights=False), variant="standard")
    with pytest.raises(ValueError, match="--background-weights"):
        data_based.check_weights()
    data_based.set_weights("ivar")
    assert data_based.check_weights() is None


def test_check_orientation(tmp_path, wcs):
    sci = scene(wcs, stars=STARS, noise=False)
    img = open_l2(write_l2_file(tmp_path / "ok.fits", sci, wcs))
    out = img.check_orientation(pa_deg=0.0)
    assert out["parity_ok"]
    # The local scale differs from the nominal 0.11″ by a few % (Roman's field distortion; at
    # this SCA corner ≈ 0.107″): compare with the written WCS itself.
    assert out["scale"] == pytest.approx(local_scale(wcs, 128.5, 128.5), rel=1e-6)
    assert abs(out["rotation_deg"]) < 2.0  # same frame; only distortion across the SCA
    flipped = mirrored_wcs(wcs, galsim.PositionD(128.5, 128.5))
    bad = open_l2(write_l2_file(tmp_path / "flip.fits", sci, flipped))
    with pytest.raises(ValueError, match="mirrored"):
        bad.check_orientation()


def test_gain_at(l2_file):
    mask = np.zeros((256, 256), np.uint8)
    mask[60:80, 50:70] = 1  # covers the whole 11×11 stamp around star 0
    img = open_l2(l2_file(mask=mask))
    x, y = [s[0] for s in STARS], [s[1] for s in STARS]
    gains = img.gain_at(x, y, stamp_size=25)
    np.testing.assert_allclose(gains, GAIN)
    assert np.isnan(img.gain_at(60.3, 70.6, stamp_size=11))[0]


def test_sky_setting_and_piff_input(l2_file):
    img = open_l2(l2_file())
    inp = img.piff_input()
    assert inp["sky"] == SKY and inp["chipnum"] == 5 and inp["properties"] == {"sca": 5}
    assert inp["bandpass"] == {"type": "RomanBandpass", "name": "H158"}
    img.set_sky(0)
    assert "sky" not in img.piff_input()
    img.set_sky()
    assert img.sky == SKY
    assert L2Image.from_config(img.config()).config() == img.config()


def local_scale(wcs, x, y):
    """√|pixel area| of a WCS at (x, y), arcsec."""
    return np.sqrt(abs(wcs.local(galsim.PositionD(x, y)).pixelArea()))


def test_piff_reads_the_file(tmp_path, l2_file, wcs):
    # Piff's own Files input on this layout: sky subtracted, IVAR_UNIF as weight, MASK_UNIF as
    # bad pixels, the TAN-SIP WCS from the image header.
    mask = np.zeros((256, 256), np.uint8)
    mask[149, 179] = 1  # 0-based [y-1, x-1]: pixel (180, 150), next to star 1's centre
    img = open_l2(l2_file(mask=mask))
    cat = Table({"x": [s[0] for s in STARS], "y": [s[1] for s in STARS]})
    cat.write(tmp_path / "cat.fits")
    config = dict(img.piff_input(), cat_file_name=str(tmp_path / "cat.fits"), x_col="x",
                  y_col="y", stamp_size=25)
    stars, piff_wcs, _, bandpass = piff.Input.process(config, logger=make_logger(0)[0])
    assert len(stars) == 3 and set(piff_wcs) == {5} and bandpass is not None
    expected_weight = 1 / (READ_VAR + SKY / GAIN)
    for star in stars:
        image, weight = star.image.array, star.weight.array
        edge = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
        assert abs(np.median(edge)) < 0.05  # sky (2 DN/s) removed; noise σ ≈ 0.13
        assert np.allclose(weight[weight > 0], expected_weight, rtol=1e-5)
        # The TAN-SIP WCS read back from the header equals the one written.
        scale = np.sqrt(abs(star.data.local_wcs.pixelArea()))
        expected = local_scale(wcs, star.image_pos.x, star.image_pos.y)
        assert scale == pytest.approx(expected, rel=1e-6)
    star1 = min(stars, key=lambda s: abs(s.image_pos.x - 180.8))
    assert star1.weight[galsim.PositionI(180, 150)] == 0
