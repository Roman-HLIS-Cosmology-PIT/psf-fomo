import numpy as np
import galsim
import galsim.roman

from psf_fomo.sim.noise import (ROMAN_EFFECTS, add_gaussian_noise, background_variance,
                                star_gain)

GAIN = galsim.roman.gain


def test_gaussian_noise_in_adu():
    image = galsim.ImageF(300, 300)
    add_gaussian_noise(image, 5.0, np.random.default_rng(0))
    np.testing.assert_allclose(image.array.std(), 5.0 / GAIN, rtol=0.02)
    np.testing.assert_allclose(background_variance({"kind": "gaussian", "sigma": 5.0}),
                               (5.0 / GAIN) ** 2)


def test_roman_background_variance():
    dark = galsim.roman.dark_current * galsim.roman.exptime
    read2 = galsim.roman.read_noise ** 2
    all_effects = {"kind": "roman", "effects": list(ROMAN_EFFECTS)}
    np.testing.assert_allclose(background_variance(all_effects, 100.0),
                               (100.0 + dark + read2) / GAIN ** 2)
    # Without Poisson noise the sky adds no variance.
    no_poisson = {"kind": "roman", "effects": ["sky", "read"]}
    np.testing.assert_allclose(background_variance(no_poisson, 100.0), read2 / GAIN ** 2)


def test_star_gain():
    assert star_gain({"kind": "gaussian", "sigma": 1.0}) is None
    assert star_gain({"kind": "roman", "effects": ["read"]}) is None
    assert star_gain({"kind": "roman", "effects": ["poisson", "read"]}) == GAIN
