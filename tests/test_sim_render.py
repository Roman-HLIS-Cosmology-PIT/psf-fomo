"""End-to-end rendering with the real Roman PSF (slow: galsim.roman.getPSF at pupil_bin=4)."""
import numpy as np
import galsim
import galsim.roman
import piff
import pytest

from psf_fomo.sim import RomanSimulation, load_simulation

pytestmark = pytest.mark.slow
GAIN = galsim.roman.gain


@pytest.fixture(scope="module")
def default_run():
    return RomanSimulation(seed=11).run(save_noiseless=True)


def test_default_run_flux(default_run):
    row = default_run.catalogs[0][0]
    noiseless = default_run.noiseless[0, 1]
    total_e = noiseless.array.sum() * GAIN
    # The 64x64 patch holds almost all the flux; a little is in the wings beyond it.
    assert 0.97 * row["flux"] < total_e < row["flux"]


def test_star_regenerates_exactly(default_run):
    stamp = default_run.render_stamp(0)
    np.testing.assert_allclose(default_run.noiseless[0, 1][stamp.bounds].array, stamp.array,
                               rtol=1e-6, atol=1e-6 * stamp.array.max())


def test_image_is_noisy_version_of_truth(default_run):
    stamp = default_run.render_stamp(0)
    resid = default_run.images[0, 1][stamp.bounds].array - stamp.array
    np.testing.assert_allclose(resid.std(), 1.0 / GAIN, rtol=0.2)  # default sigma = 1 e⁻


def test_run_is_reproducible(default_run):
    again = RomanSimulation(seed=11).run()
    np.testing.assert_array_equal(again.images[0, 1].array, default_run.images[0, 1].array)


def test_truth_psf(default_run):
    with_pixel = default_run.truth_psf(1, 0, 0, oversample=5)
    no_pixel = default_run.truth_psf(1, 0, 0, oversample=5, include_pixel=False)
    # The PSF wings extend beyond a 25-pixel box, so compare enclosed flux on the same
    # 64-pixel box: oversampled truth vs. the native-pixel noiseless image.
    big = default_run.truth_psf(1, 0, 0, oversample=5, size=64)
    native_fraction = (default_run.noiseless[0, 1].array.sum() * GAIN
                       / default_run.catalogs[0][0]["flux"])
    np.testing.assert_allclose(big.array.sum(), native_fraction, rtol=5e-3)
    assert with_pixel.array.sum() < big.array.sum() < 1.0
    # The pixel response adds size: HSM sigma is larger with the pixel.
    s_pix = with_pixel.FindAdaptiveMom().moments_sigma
    s_nopix = no_pixel.FindAdaptiveMom().moments_sigma
    assert s_pix > s_nopix


def test_stars_for_piff(default_run):
    stars, rows = default_run.stars(sca=1)
    assert len(stars) == 1 and isinstance(stars[0], piff.Star)
    star = stars[0]
    assert star.image.array.shape == (25, 25)
    assert star["sca"] == 1 and star["star_id"] == 0
    np.testing.assert_allclose(star.weight.array, 1.0 / rows["bkg_var"][0])


def test_save_load_real_run(default_run, tmp_path):
    default_run.save(tmp_path)
    loaded = load_simulation(tmp_path)
    np.testing.assert_array_equal(loaded.images[0, 1].array, default_run.images[0, 1].array)
    np.testing.assert_array_equal(loaded.noiseless[0, 1].array,
                                  default_run.noiseless[0, 1].array)
    # A loaded simulation re-renders the same truth.
    np.testing.assert_allclose(loaded.render_stamp(0).array, default_run.render_stamp(0).array)


def test_dilation_and_mtf_make_psf_larger():
    base = RomanSimulation(seed=1).run()
    sim = RomanSimulation(seed=1)
    sim.add_dilation(1.05)
    sim.set_mtf(1.2)
    bigger = sim.run()
    s0 = base.truth_psf(1, 0, 0).FindAdaptiveMom().moments_sigma
    s1 = bigger.truth_psf(1, 0, 0).FindAdaptiveMom().moments_sigma
    assert s1 > 1.02 * s0


def test_jitter_makes_psf_larger():
    base = RomanSimulation(seed=1).run()
    sim = RomanSimulation(seed=1)
    sim.add_jitter(rms_mas=30.0)
    jittered = sim.run()
    s0 = base.truth_psf(1, 0, 0).FindAdaptiveMom().moments_sigma
    s1 = jittered.truth_psf(1, 0, 0).FindAdaptiveMom().moments_sigma
    assert s1 > s0


def test_roman_noise_and_wcs():
    sim = RomanSimulation(scas=[3], n_stars=4, seed=2)
    sim.set_wcs("roman")
    sim.set_noise("roman")
    result = sim.run()
    image = result.images[0, 3]
    rows = result.catalog()
    assert "ra" in rows.colnames
    # Background subtracted: the image median is close to zero (in ADU).
    assert abs(np.median(image.array)) < 5.0
    read_var = (galsim.roman.read_noise / GAIN) ** 2
    assert np.all(rows["bkg_var"] > read_var)
    stars, _ = result.stars()
    assert all(s["gain"] == GAIN for s in stars)


def test_chromatic_flat_sed_flux():
    sim = RomanSimulation(seed=4)
    sim.set_chromatic("flat")
    result = sim.run(save_noiseless=True)
    total_e = result.noiseless[0, 1].array.sum() * GAIN
    assert 0.97 * result.catalogs[0][0]["flux"] < total_e < result.catalogs[0][0]["flux"]
