"""Moments on real Roman simulations (slow: galsim.roman.getPSF at pupil_bin=4)."""
import numpy as np
import pytest

from psf_fomo.moments import MomentsValidation
from psf_fomo.sim import RomanSimulation

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def saved_sims(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sims")
    paths = {}
    for name, dilation in [("base", 1.0), ("dil", 1.01)]:
        sim = RomanSimulation(n_stars=4, seed=5)
        sim.add_dilation(dilation)
        paths[name] = str(folder / name)
        sim.run(save_to=paths[name])
    return paths


def test_dilation_recovered_on_roman_psf(saved_sims):
    mv = MomentsValidation(saved_sims)
    mv.set_algorithms(["hsm"])
    res = mv.run()
    diff = res.differences()
    assert np.all(diff["flag"] == 0)
    injected = diff[diff["image"] == "oversampled"]["dT_T"]
    # 1 % dilation of optics and MTF gives just under 2 % in T once the pixel is included.
    assert np.all((injected > 0.01) & (injected < 0.021))
    noiseless = diff[diff["image"] == "noiseless"]
    np.testing.assert_allclose(noiseless["dT_T"], noiseless["dT_T_hsm_os"], rtol=0.3)


def test_yaml_round_trip(saved_sims, tmp_path):
    mv = MomentsValidation(saved_sims, reference="base")
    mv.set_algorithms(["hsm", "wmom"], wmom_fwhm=[0.3, 0.4])
    mv.set_images(["noiseless"], oversample=3)
    mv.select(scas=[1], obs=[0])
    mv.to_yaml(tmp_path / "moments.yaml")
    again = MomentsValidation.from_yaml(tmp_path / "moments.yaml")
    assert again.config == mv.config
