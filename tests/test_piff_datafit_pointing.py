"""Telescope pointing of an L2 image: read from the PRIMARY header (or given), saved, rebuilt.

The converter copies ASDF ``meta.pointing`` into PRIMARY (RA_V1, DEC_V1, PA_V3, TARG_RA,
TARG_DEC, PA_APER; degrees). Orienting RomanOptics by the attitude needs unreleased GalSim and
Piff changes and is off by default; only that default is tested here.
"""
import pytest

from l2files import field, open_l2, write_field
from psf_fomo.piff_datafit import PiffDataFit, L2Image, StarCatalog
from psf_fomo.piff_simfit import PiffFitConfig

HEADER = {"RA_V1": 10.5, "DEC_V1": -40.25, "PA_V3": 123.0, "TARG_RA": 10.6,
          "TARG_DEC": -40.2, "PA_APER": 183.0}
ATTITUDE = {"boresight_ra": 10.5, "boresight_dec": -40.25, "telescope_pa": 123.0}


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pointing")
    objects = field(n_stars=20, n_gals=0, seed=7)
    with_pointing = write_field(folder / "with", objects, primary=HEADER)
    without = write_field(folder / "without", objects)
    return with_pointing, without


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


def test_attitude_is_off_by_default(files):
    (image, catalog), _ = files
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cfg = PiffFitConfig()
    cfg.set_optics(mode="pinned")
    fit = PiffDataFit(open_l2(image), cat, cfg)
    assert fit.use_attitude is False
    assert fit.attitude() is None    # image has an attitude, RomanOptics is used: still off
