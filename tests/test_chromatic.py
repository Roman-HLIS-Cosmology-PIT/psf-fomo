"""Chromatic defaults: simulation SED settings, the truth's spatial-interpolation modes and
sum over the filter, Piff's chromatic default, SEDs for Piff."""
import copy
import time

import numpy as np
import galsim
import galsim.roman
import pytest
from astropy.table import Table

from l2files import field, open_l2, write_field
from psf_fomo.datafit import DataFit, StarCatalog
from psf_fomo.io import to_plain
from psf_fomo.piffval import FitConfig, PiffValidation
from psf_fomo.piffval.config import needs_star_seds, piff_default_chromatic
from psf_fomo.piffval.residuals import draw_profile
from psf_fomo.sim import RomanSimulation
from psf_fomo.sim import storage
from psf_fomo.sim.psf import (band_weights, effective_profile, make_optical_model,
                             optical_profile, position_profile, wavelength_nodes)
from psf_fomo.sim.simulation import _Truth, _make_sed

BANDPASS = galsim.roman.getBandpasses()["H158"]


def test_simulation_sed_settings():
    sim = RomanSimulation()
    assert sim.config["chromatic"]["sed"] == "flat"           # chromatic by default
    sim.set_chromatic(None)
    assert sim.config["chromatic"] is None                    # monochromatic
    assert _make_sed(None, BANDPASS) is None
    sim.set_chromatic("my_sed.dat", wave_type="Angstrom", flux_type="flambda")
    assert sim.config["chromatic"] == {"sed": "my_sed.dat", "wave_type": "Angstrom",
                                       "flux_type": "flambda"}


def test_psf_model_settings():
    sim = RomanSimulation()
    assert sim.config["psf_model"] == {"spatial_interpolation": "fast",       # defaults
                                       "nominal_interp": "bilinear", "n_waves": 10}
    sim.set_psf_model("exact", n_waves=5)
    assert sim.config["psf_model"] == {"spatial_interpolation": "exact",
                                       "nominal_interp": "bilinear", "n_waves": 5}
    sim.set_psf_model(n_waves=None)                                 # every 5 nm
    assert sim.config["psf_model"]["n_waves"] is None
    sim.set_psf_model("piff")                                       # always exact in wavelength
    assert sim.config["psf_model"]["n_waves"] is None
    with pytest.raises(ValueError, match="piff"):
        sim.set_psf_model("piff", n_waves=5)
    with pytest.raises(ValueError, match="spatial_interpolation"):
        sim.set_psf_model("nearest")
    with pytest.raises(ValueError, match="n_waves"):
        sim.set_psf_model(n_waves=1)
    wide = RomanSimulation(filter="W146")
    assert wide.config["psf_model"]["n_waves"] == 10                # same for every filter
    wide.set_psf_model(n_waves=6)
    wide.set_psf_model()                                            # back to the default
    assert wide.config["psf_model"]["n_waves"] == 10


def test_truth_model_per_mode():
    fast = _Truth(RomanSimulation().config)
    assert fast.spatial == "fast" and fast.model.sed is not None and fast.model.n_waves == 10
    sim = RomanSimulation()
    sim.set_psf_model("piff")
    piff_truth = _Truth(sim.config)
    assert piff_truth.model.sed is None and piff_truth.model.n_waves is None
    old = RomanSimulation().config
    old["psf_model"] = {"nominal_interp": "bilinear", "n_waves": 5}  # saved before the modes
    assert _Truth(old).spatial == "piff" and _Truth(old).model.n_waves is None


def test_band_weights_integrate_linear_functions_exactly():
    # The PSF is taken as linear in wavelength between nodes, so a linear function of
    # wavelength is integrated exactly, whatever the number of nodes.
    sed = galsim.SED("wave", wave_type="nm", flux_type="fphotons")      # not flat
    fine = np.linspace(BANDPASS.blue_limit, BANDPASS.red_limit, 20001)
    density = BANDPASS(fine) * sed(fine)
    f = 3.0 + 0.01 * fine
    expected = np.sum(0.5 * ((f * density)[1:] + (f * density)[:-1]) * np.diff(fine)) / \
        np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(fine))
    for n_waves in (2, 5, None):
        waves = wavelength_nodes(BANDPASS, n_waves)
        weights = band_weights(BANDPASS, sed, waves)
        assert weights.sum() == pytest.approx(1.0)
        assert np.sum(weights * (3.0 + 0.01 * waves)) == pytest.approx(expected, rel=1e-6)
    assert len(wavelength_nodes(BANDPASS, None)) == \
        int(round((BANDPASS.red_limit - BANDPASS.blue_limit) / 5.0)) + 1


def chromatic_gaussian():
    """A chromatic test PSF that needs no getPSF call."""
    return galsim.ChromaticObject(galsim.Gaussian(sigma=0.12)).dilate(
        lambda w: (w / 1500.0) ** 0.6)


def test_effective_profile_matches_galsim_chromatic_drawing():
    sed = galsim.SED("1", wave_type="nm", flux_type="fphotons").withFlux(1.0, BANDPASS)
    pixel = galsim.Pixel(galsim.roman.pixel_scale)
    effective = effective_profile(chromatic_gaussian(), sed, BANDPASS, None, extent=5.0)
    a = galsim.Convolve(effective, pixel).drawImage(nx=64, ny=64, scale=0.11 / 4,
                                                    method="no_pixel")
    b = (galsim.Convolve(chromatic_gaussian(), pixel) * sed).drawImage(
        BANDPASS, nx=64, ny=64, scale=0.11 / 4, method="no_pixel")
    assert a.array.sum() == pytest.approx(b.array.sum(), rel=1e-4)
    ma, mb = a.FindAdaptiveMom(), b.FindAdaptiveMom()
    assert ma.moments_sigma == pytest.approx(mb.moments_sigma, rel=1e-4)
    assert abs(ma.observed_shape.e1 - mb.observed_shape.e1) < 1e-5


def test_truth_draw_chromatic_as_piff_and_achromatic_scaled():
    truth = _Truth(RomanSimulation().config)
    image = galsim.ImageD(48, 48, scale=0.11 / 4)
    truth.draw(chromatic_gaussian(), 1000.0, image, center=None, method="no_pixel")
    direct = (chromatic_gaussian() * truth.sed.withFlux(1000.0, truth.bandpass)).drawImage(
        truth.bandpass, nx=48, ny=48, scale=0.11 / 4, method="no_pixel")
    assert image.array.sum() == pytest.approx(direct.array.sum(), rel=1e-4)
    assert image.FindAdaptiveMom().moments_sigma == pytest.approx(
        direct.FindAdaptiveMom().moments_sigma, rel=1e-4)
    # Achromatic (effective) profiles: flux scaled, so a profile holding 90 % of the light
    # (the rest outside its grid) keeps 90 %.
    image = galsim.ImageD(64, 64, scale=0.11 / 4)
    truth.draw(galsim.Gaussian(sigma=0.12, flux=0.9), 1000.0, image, None, "no_pixel")
    assert image.array.sum() == pytest.approx(900.0, rel=1e-3)


def effective_moments(truth, i, sca=5):
    """HSM of star i's truth PSF × pixel (no noise), 4× oversampled; and the drawing time."""
    pixel = galsim.Pixel(galsim.roman.pixel_scale)    # simple WCS: 0.11″ pixels
    image = galsim.ImageD(96, 96, scale=galsim.roman.pixel_scale / 4)
    t0 = time.perf_counter()
    truth.draw(galsim.Convolve(truth.profile(0, sca, i), pixel), 1.0, image, None, "no_pixel")
    return image.FindAdaptiveMom(), time.perf_counter() - t0


def moment_differences(a, b):
    return ((a.moments_sigma / b.moments_sigma) ** 2 - 1,
            a.observed_shape.e1 - b.observed_shape.e1,
            a.observed_shape.e2 - b.observed_shape.e2)


@pytest.mark.slow
@pytest.mark.parametrize("filter_name", ["H158", "W146"])      # W146: the widest filter
def test_default_n_waves_is_converged(filter_name):
    # Convergence in wavelength sampling of the 'fast' truth (optics + MTF + pixel, flat
    # SED): the default n_waves (10) against n_waves = 20, at two positions on one SCA.
    # Tolerances: |dT/T| < 1e-3, |de| < 3e-4.
    config = RomanSimulation(scas=[5], n_stars=2, filter=filter_name).config
    fine_config = copy.deepcopy(config)
    fine_config["psf_model"]["n_waves"] = 20
    default, fine = _Truth(config), _Truth(fine_config)
    assert default.model.n_waves == 10 and fine.model.n_waves == 20
    for i in range(2):
        (a, _), (b, _) = effective_moments(default, i), effective_moments(fine, i)
        if i == 0:
            cached = default.model._effective[5]
        dt, de1, de2 = moment_differences(a, b)
        msg = f"{filter_name} star {i}: dT/T {dt:.2e}, de1 {de1:.2e}, de2 {de2:.2e}"
        print(msg)
        assert abs(dt) < 1e-3 and abs(de1) < 3e-4 and abs(de2) < 3e-4, msg
    # Same corner PSFs for both stars: effective PSFs built once, then reused.
    assert default.model._effective[5] is cached


@pytest.mark.slow
@pytest.mark.parametrize("filter_name", ["H158", "W146"])
def test_fast_matches_piff(filter_name):
    # Same spatial model (Piff's corner blending); 'fast' sums each corner over the filter
    # once (default n_waves), 'piff' draws as Piff does. Tolerances as the convergence test.
    config = RomanSimulation(scas=[5], n_stars=2, filter=filter_name).config
    piff_config = copy.deepcopy(config)
    piff_config["psf_model"].update(spatial_interpolation="piff", n_waves=None)
    fast, piff_truth = _Truth(config), _Truth(piff_config)
    for i in range(2):
        (a, ta), (b, tb) = effective_moments(fast, i), effective_moments(piff_truth, i)
        dt, de1, de2 = moment_differences(a, b)
        msg = (f"{filter_name} star {i}: dT/T {dt:.2e}, de1 {de1:.2e}, de2 {de2:.2e}; "
               f"draw {ta:.2f} s fast, {tb:.2f} s piff")
        print(msg)
        assert abs(dt) < 1e-3 and abs(de1) < 3e-4 and abs(de2) < 3e-4, msg


@pytest.mark.slow
def test_exact_equals_fast_at_an_sca_corner():
    # At a corner, Piff's blending weight is 1 for that corner's PSF, so 'fast' and 'exact'
    # must give the same profile: checks that 'exact' calls getPSF as Piff does (pupil
    # binning, WCS, aberration indexing).
    wcs = galsim.PixelScale(galsim.roman.pixel_scale)
    sed = galsim.SED("1", wave_type="nm", flux_type="fphotons").withFlux(1.0, BANDPASS)
    zernikes = np.zeros(19)
    zernikes[0] = 0.05                                              # some defocus
    model = make_optical_model("H158", BANDPASS, 22, False, chromatic=True, sed=sed)
    fast = optical_profile(model, zernikes, 0.0, 0.0, 5, wcs)
    exact = position_profile(5, "H158", BANDPASS, 0.0, 0.0, wcs, zernikes, sed=sed)
    a = fast.drawImage(nx=64, ny=64, scale=0.11 / 4, method="no_pixel")
    b = exact.drawImage(nx=64, ny=64, scale=0.11 / 4, method="no_pixel")
    np.testing.assert_allclose(a.array, b.array, rtol=0, atol=1e-9 * b.array.max())


@pytest.mark.slow
def test_exact_mode_renders_a_star():
    sim = RomanSimulation(scas=[5], n_stars=1)
    sim.set_psf_model("exact")
    truth = _Truth(sim.config)
    patch = truth.render_patch(0, 5, 0)
    flux = float(truth.sca_truth(0, 5)[0]["flux"][0])
    assert np.all(np.isfinite(patch.array))
    assert 0.9 * flux < patch.array.sum() < flux                    # wings beyond the patch


def test_galsim_sed_object_is_stored_as_a_table():
    # A smooth SED (photons ∝ λ): the table reproduces it, and the config stays plain YAML.
    sed = galsim.SED(galsim.LookupTable([500.0, 2500.0], [1.0, 5.0], interpolant="linear"),
                     wave_type="nm", flux_type="fphotons")
    sim = RomanSimulation()
    sim.set_chromatic(sed)
    c = sim.config["chromatic"]
    assert c["sed"] == "table" and to_plain(c) == c
    wave = np.asarray(c["wave_nm"])
    assert wave[0] == pytest.approx(BANDPASS.blue_limit)
    assert wave[-1] == pytest.approx(BANDPASS.red_limit)
    assert np.allclose(np.diff(wave)[:-1], storage.SED_STEP_NM)
    rebuilt = _make_sed(c, BANDPASS)
    w = np.linspace(BANDPASS.blue_limit + 1, BANDPASS.red_limit - 1, 7)
    expected = sed.withFlux(1.0, BANDPASS)
    # Same shape; the normalizations differ by ~1e-5 because GalSim integrates the original
    # two-point SED on the bandpass's coarser wavelength grid.
    np.testing.assert_allclose(rebuilt(w), expected(w), rtol=1e-4)
    ratio = rebuilt(w) / expected(w)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)
    with pytest.raises(ValueError, match="spectral"):
        sim.set_chromatic(galsim.SED("1", wave_type="nm", flux_type="1"))   # dimensionless


def test_sed_file_for_piff(tmp_path):
    wave, flux = storage.sample_sed(galsim.SED("1", wave_type="nm", flux_type="fphotons"),
                                    BANDPASS)
    storage.write_sed(tmp_path / storage.SED_FILE, wave, flux)
    t = Table.read(tmp_path / storage.SED_FILE)          # the columns Piff reads by default
    assert t.colnames == ["WAVE", "FLUX"] and np.all(t["FLUX"] == 1.0)
    assert storage.piff_sed_input(None) == {}
    assert storage.piff_sed_input({"sed": "flat"}) == {
        "sed_file_name": "sed.fits", "sed_wave_type": "nm", "sed_flux_type": "fphotons"}
    user = storage.sed_setting("/data/seds/vega.dat", wave_type="nm", flux_type="flambda")
    assert storage.piff_sed_input(user)["sed_file_name"] == "vega.dat"   # copied by save


def test_fit_is_monochromatic_by_default():
    assert piff_default_chromatic() is True       # Piff's RomanOptics default (Piff main)
    psf = FitConfig().piff_config("H158")         # ours: monochromatic
    assert psf["components"][0]["chromatic"] is False and not needs_star_seds(psf)
    alone = FitConfig()
    alone.set_optics("free", chromatic=True)
    alone.set_mtf(False)
    psf = alone.piff_config("H158")
    assert psf["type"] == "RomanOptics" and psf["chromatic"] is True and needs_star_seds(psf)
    piffs_default = FitConfig()
    piffs_default.set_optics("free", chromatic=None)
    piffs_default.set_mtf(False)
    assert "chromatic" not in piffs_default.piff_config("H158")
    gaussian = {"type": "Simple", "model": {"type": "Gaussian"}, "interp": {"type": "Mean"}}
    assert not needs_star_seds(gaussian)


def test_chromatic_optics_in_a_composite_model_is_refused():
    # Piff's Convolve / Sum draw without the stars' SEDs: every star would be flagged.
    for extra in ("mtf", "kernel", "pixelgrid"):
        cfg = FitConfig()
        cfg.set_optics("pinned", chromatic=True)
        cfg.set_mtf(extra == "mtf")
        if extra == "kernel":
            cfg.add_kernel("blur")
        if extra == "pixelgrid":
            cfg.add_pixelgrid()
        with pytest.raises(ValueError, match="on its own"):
            cfg.piff_config("H158")


def chromatic_optics_alone():
    cfg = FitConfig()
    cfg.set_optics("pinned", chromatic=True)
    cfg.set_mtf(False)
    return cfg


def test_monochromatic_simulation_with_chromatic_fit_is_refused():
    sim = RomanSimulation()                          # configuration only, nothing rendered
    sim.set_chromatic(None)
    pv = PiffValidation(sim, chromatic_optics_alone())
    with pytest.raises(ValueError, match="chromatic=False"):
        pv.check_chromatic()
    cfg = FitConfig()
    cfg.set_optics("pinned", chromatic=False)
    PiffValidation(sim, cfg).check_chromatic()       # consistent: no error


def test_file_config_gives_stars_the_simulation_sed():
    sim = RomanSimulation()
    inp = PiffValidation(sim, FitConfig()).piff_file_config("simdir", 0, "outdir")["input"]
    assert inp["sed_file_name"] == "sed.fits" and inp["sed_flux_type"] == "fphotons"
    sim.set_chromatic(None)
    cfg = FitConfig()
    cfg.set_optics("pinned", chromatic=False)
    inp = PiffValidation(sim, cfg).piff_file_config("simdir", 0, "outdir")["input"]
    assert "sed_file_name" not in inp


def test_real_data_assumes_a_flat_sed(tmp_path):
    image, catalog = write_field(tmp_path / "f", field(n_stars=10, n_gals=0, seed=3))
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    fit = DataFit(open_l2(image), cat, chromatic_optics_alone())
    t = cat.piff_table(fit.image)
    inp = fit.piff_config(tmp_path / "catalog.fits", t)["input"]
    assert inp["sed_file_name"] == "sed.fits" and fit.config()["sed"]["sed"] == "flat"
    fit.set_sed(None)
    with pytest.raises(ValueError, match="set_sed"):
        fit.piff_config(tmp_path / "catalog.fits", t)
    cfg = FitConfig()
    cfg.set_optics("pinned", chromatic=False)
    mono = DataFit(open_l2(image), cat, cfg)
    mono.set_sed(None)
    assert "sed_file_name" not in mono.piff_config(tmp_path / "catalog.fits", t)["input"]


def test_draw_profile_handles_chromatic_profiles():
    class Context:                                   # what draw_profile needs from a Piff PSF
        flat_bandpass = galsim.Bandpass(
            galsim.LookupTable([BANDPASS.blue_limit, BANDPASS.red_limit], [1, 1],
                               interpolant="linear"), "nm")
    prof = galsim.ChromaticObject(galsim.Gaussian(sigma=0.15)).dilate(
        lambda w: (w / 1500.0) ** 0.6)                                       # chromatic
    sed_eff = galsim.SED("1", "nm", "fphotons").withFlux(1.0, BANDPASS) * BANDPASS
    image = galsim.ImageD(64, 64, scale=0.02)
    draw_profile(prof, image, Context, sed_eff)
    assert image.array.sum() == pytest.approx(1.0, rel=2e-3)
    with pytest.raises(ValueError, match="SED"):
        draw_profile(prof, galsim.ImageD(64, 64, scale=0.02), Context, None)
    plain = galsim.ImageD(64, 64, scale=0.02)
    draw_profile(galsim.Gaussian(sigma=0.15), plain, Context)        # achromatic: as before
    assert plain.array.sum() == pytest.approx(1.0, rel=2e-3)
