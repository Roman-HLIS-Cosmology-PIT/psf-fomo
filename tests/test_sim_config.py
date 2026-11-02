"""Builder, YAML round trip and the deterministic truth (no PSF rendering)."""
import numpy as np
import pytest

from psf_fomo.sim import RomanSimulation
from psf_fomo.sim.simulation import _Truth


def test_defaults():
    sim = RomanSimulation()
    c = sim.config
    assert c["scas"] == [1] and c["n_stars"] == 1 and c["n_obs"] == 1
    assert c["mtf"] == {"scale": 1.0, "sigma_obs": 0.0, "sigma_field": 0.0}
    assert c["jitter"] is None and c["chromatic"] is None
    assert c["noise"] == {"kind": "gaussian", "sigma": 1.0}
    assert c["wcs"]["kind"] == "simple"


@pytest.mark.parametrize("kwargs", [dict(scas=[0]), dict(scas=[19]), dict(scas=[3, 3]),
                                    dict(n_stars=0), dict(filter="X999"), dict(seed=-1),
                                    dict(max_zernike=23), dict(stamp_size=80)])
def test_invalid_construction(kwargs):
    with pytest.raises(ValueError):
        RomanSimulation(**kwargs)


def test_invalid_settings():
    sim = RomanSimulation()
    with pytest.raises(ValueError):
        sim.set_brightness(mag=18, mag_range=(17, 19))
    with pytest.raises(ValueError):
        sim.set_placement(phase=(0.7, 0))
    with pytest.raises(ValueError):
        sim.set_zernikes({3: 0.1})
    with pytest.raises(ValueError):
        sim.set_noise("roman", effects=["sky", "ipc"])
    with pytest.raises(ValueError):
        sim.add_jitter("timeseries", t=[0, 0], dx_mas=[1, 2], dy_mas=[1, 2])


def configured_sim(**kwargs):
    sim = RomanSimulation(scas=[1, 5], n_obs=2, n_stars=6, seed=7, **kwargs)
    sim.set_placement("grid")
    sim.set_brightness(mag_range=(17, 20))
    sim.set_zernikes({4: 0.02, 6: -0.01}, sigma_obs=0.003, sigma_field={5: 0.004})
    sim.add_dilation(1.01, sigma_obs=0.002, sigma_field=0.001)
    sim.add_shear(0.01, -0.005, sigma_obs=0.001)
    sim.set_mtf(1.05, sigma_field=0.01)
    sim.add_aberration_scatter(0.001)
    sim.add_jitter("timeseries", t=[0, 1, 2], dx_mas=[0, 5, 0], dy_mas=[1, 0, -1])
    sim.set_noise("roman", effects=["sky", "poisson", "read"])
    return sim


def test_yaml_round_trip(tmp_path):
    sim = configured_sim()
    sim.to_yaml(tmp_path / "sim.yaml")
    assert RomanSimulation.from_yaml(tmp_path / "sim.yaml").config == sim.config


def test_truth_is_deterministic():
    config = configured_sim().config
    t1, p1 = _Truth(config).sca_truth(1, 5)
    t2, p2 = _Truth(config).sca_truth(1, 5)
    for name in t1.colnames:
        np.testing.assert_array_equal(t1[name], t2[name])
    np.testing.assert_array_equal(p1, p2)


def test_truth_table_contents():
    sim = configured_sim()
    truth = _Truth(sim.config)
    table, params = truth.sca_truth(0, 5)
    assert list(table["star_id"]) == list(range(6, 12))  # SCA 5 is the second SCA
    assert set(table["sca"]) == {5}
    for name in ["x", "y", "phase_x", "u", "v", "mag", "flux", "dilation", "g1", "g2",
                 "mtf_scale", "z4", "z22"]:
        assert name in table.colnames
    assert np.all(np.abs(table["phase_x"]) <= 0.5)
    # A Zernike field (sigma_field on Z5) switches Piff to the linear (a, b, c) layout.
    assert truth.linear and params.shape == (6, 3 * 19)


def test_obs_offsets_shared_across_scas_and_differ_between_obs():
    truth = _Truth(configured_sim().config)
    assert truth.obs_truth(0)["dilation"] != truth.obs_truth(1)["dilation"]
    # Without a field term the dilation is the same on both SCAs within an observation.
    sim = configured_sim()
    sim.add_dilation(1.01, sigma_obs=0.002, sigma_field=0.0)
    truth = _Truth(sim.config)
    d1 = truth.sca_truth(0, 1)[0]["dilation"]
    d5 = truth.sca_truth(0, 5)[0]["dilation"]
    np.testing.assert_allclose(d1, d5[0])
    np.testing.assert_allclose(d1, d1[0])


def test_zernike_field_is_linear_in_position():
    sim = configured_sim()
    sim.add_aberration_scatter(0.0)
    table, params = _Truth(sim.config).sca_truth(0, 1)
    a, b, c = params[0].reshape(3, 19)
    z5 = a[1] + b[1] * table["x"] / 4088 + c[1] * table["y"] / 4088
    np.testing.assert_allclose(table["z5"], z5)
    assert np.std(table["z5"]) > 0
    # Z4 has no field term: constant across the SCA.
    np.testing.assert_allclose(table["z4"], table["z4"][0])


def test_changing_one_setting_keeps_other_draws():
    base = configured_sim()
    other = configured_sim()
    other.add_dilation(1.03, sigma_obs=0.01, sigma_field=0.005)
    t1 = _Truth(base.config).sca_truth(0, 1)[0]
    t2 = _Truth(other.config).sca_truth(0, 1)[0]
    np.testing.assert_array_equal(t1["x"], t2["x"])
    np.testing.assert_array_equal(t1["mag"], t2["mag"])
    np.testing.assert_array_equal(t1["z5"], t2["z5"])
