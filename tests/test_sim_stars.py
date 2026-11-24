import numpy as np
import galsim
import galsim.roman
import pytest

from psf_fomo.sim.stars import draw_magnitudes, draw_positions, mag_to_flux, sub_pixel_phase


def test_random_positions_respect_margin():
    rng = np.random.default_rng(1)
    x, y = draw_positions(500, "random", rng, margin=33)
    assert len(x) == len(y) == 500
    assert x.min() >= 34 and x.max() <= galsim.roman.n_pix - 33
    assert y.min() >= 34 and y.max() <= galsim.roman.n_pix - 33


def test_grid_positions_count_and_random_phases():
    rng = np.random.default_rng(2)
    x, y = draw_positions(10, "grid", rng, margin=33)
    assert len(x) == 10
    # 4x4 grid truncated to 10: x takes the 4 grid columns, phases are not all equal
    assert len(np.unique(np.round(x))) == 4
    assert np.std(sub_pixel_phase(x)) > 0


def test_single_grid_star_is_central():
    x, y = draw_positions(1, "grid", np.random.default_rng(3), margin=33)
    center = (1 + galsim.roman.n_pix) / 2
    assert abs(x[0] - center) < 1 and abs(y[0] - center) < 1


def test_fixed_phase():
    rng = np.random.default_rng(4)
    for kind in ("random", "grid"):
        x, y = draw_positions(20, kind, rng, margin=33, phase=(0.25, -0.5))
        np.testing.assert_allclose(sub_pixel_phase(x), 0.25)
        np.testing.assert_allclose(np.abs(sub_pixel_phase(y)), 0.5)


def test_bad_kind():
    with pytest.raises(ValueError):
        draw_positions(3, "spiral", np.random.default_rng(), margin=33)


def test_magnitudes():
    rng = np.random.default_rng(5)
    np.testing.assert_array_equal(draw_magnitudes(3, rng, mag=18.0), [18.0] * 3)
    mags = draw_magnitudes(1000, rng, mag_range=(20, 17))
    assert mags.min() >= 17 and mags.max() <= 20


def test_mag_to_flux_matches_galsim_sed_route():
    # Independent route: a flat-fnu SED normalized to the magnitude, its photon flux through
    # the bandpass, times area and exposure time (as in GalSim's Roman demo).
    bandpass = galsim.roman.getBandpasses(AB_zeropoint=True)["H158"]
    sed = galsim.SED("1", wave_type="nm", flux_type="fnu").withMagnitude(18.0, bandpass)
    expected = sed.calculateFlux(bandpass) * galsim.roman.collecting_area * galsim.roman.exptime
    np.testing.assert_allclose(mag_to_flux(18.0, bandpass), expected, rtol=1e-6)
    # 1 magnitude brighter = factor 10**0.4
    np.testing.assert_allclose(mag_to_flux(17.0, bandpass) / mag_to_flux(18.0, bandpass),
                               10 ** 0.4)
