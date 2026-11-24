"""Pixel frame: L2 files and galsim.roman share Roman's science frame.

Files are fitted as they are. A file whose pixel frame is mirrored relative to
galsim.roman's (e.g. with a GalSim from before its science-frame update) is fitted with a
warning, and fits saved on the old flipped working copy are refused when loaded.
"""
import numpy as np
import galsim
import galsim.roman
import pytest

from l2files import field, mirrored_wcs, roman_wcs, scene, sep_catalog, write_l2_file
from psf_fomo.piff_datafit import PiffDataFit, L2Image, StarCatalog, load_data_fit
from psf_fomo.piff_datafit.fit import PiffDataFitResult
from psf_fomo.piff_simfit import PiffFitConfig


def test_file_frame_matches_galsim(tmp_path):
    # A file written with galsim.roman's WCS is in GalSim's (= the science) frame.
    wcs = roman_wcs(sca=5)
    path = write_l2_file(tmp_path / "l2.fits", scene(wcs, stars=[(128.0, 128.0, 1000.0)]), wcs)
    img = L2Image(path)
    assert img.check_orientation()["parity_ok"]
    assert not hasattr(img, "flip") and "frame" not in img.config()


def test_romanoptics_warns_on_a_mirrored_file(tmp_path):
    wcs = roman_wcs(sca=5)
    objects = field(n_stars=10, n_gals=0, shape=(256, 256), seed=4)
    mirrored = mirrored_wcs(wcs, galsim.PositionD(128.5, 128.5))
    sci = scene(mirrored, stars=[(o["x"], o["y"], o["flux"]) for o in objects])
    image = write_l2_file(tmp_path / "mirrored.fits", sci, mirrored)
    cat = StarCatalog(sep_catalog(tmp_path / "det.fits", objects))
    cat.select_stars("all")             # SizeMag finds no locus among 10 stars
    cat.set_cuts(max_snr_weight=0)
    cfg = PiffFitConfig()
    cfg.set_optics("pinned")
    # The fit runs (e.g. with a GalSim from before its science-frame update) and says why its
    # field dependence may be wrong.
    with pytest.warns(UserWarning, match="mirrored"):
        res = PiffDataFit(L2Image(image), cat, cfg).run(save_to=tmp_path / "fit")
    assert not res.checks["orientation"]["parity_ok"]


def test_old_flipped_fits_are_refused(tmp_path):
    wcs = roman_wcs(sca=5)
    path = write_l2_file(tmp_path / "l2.fits", scene(wcs), wcs)
    config = L2Image(path).config()
    config["flip"] = "y"                       # as saved by the earlier flipped workflow
    with pytest.raises(ValueError, match="flipped"):
        L2Image.from_config(config)
    fit_folder = tmp_path / "fit"
    fit_folder.mkdir()
    result = PiffDataFitResult(None, None, {}, {"image": config}, {}, {}, None)
    from psf_fomo.io import write_yaml
    write_yaml(fit_folder / "config.yaml", {"config": result.config, "fit_summary": {},
                                            "checks": {}, "provenance": {}})
    with pytest.raises(ValueError, match="flipped"):
        load_data_fit(fit_folder)


@pytest.mark.slow
def test_romanoptics_closed_loop_on_a_simulated_sca(tmp_path):
    # Full SCA 11 with the Roman PSF, written as it is (the simulation's WCS is galsim.roman's,
    # in the science frame): pinned optics + MTF (the truth model) must fit at the noise level.
    from psf_fomo.sim import RomanSimulation
    s = RomanSimulation(scas=[11], n_stars=36, filter="H158", seed=21)
    s.set_wcs("roman")
    s.set_placement("grid")
    s.set_brightness(mag_range=(18, 19.5))
    s.set_chromatic(None)                      # the fit model is monochromatic
    sim = s.run(save_noiseless=True)
    exptime = galsim.roman.exptime
    g_eff = galsim.roman.gain * exptime
    read_var = (galsim.roman.read_noise / g_eff) ** 2
    sky = 0.6
    noiseless = sim.noiseless[0, 11]
    rng = np.random.default_rng(3)
    expected = noiseless.array / exptime + sky
    sci = (rng.poisson(expected * g_eff) / g_eff
           + rng.normal(0, np.sqrt(read_var), expected.shape)).astype(np.float32)
    image = write_l2_file(tmp_path / "l2.fits", sci, noiseless.wcs, sca=11, sky=sky,
                          read_var=read_var, gain=g_eff, planes="unif")
    truth = sim.catalogs[0]
    objects = [{"id": int(r["star_id"]) + 1, "x": float(r["x"]), "y": float(r["y"]),
                "flux": float(r["flux"]) / galsim.roman.gain / exptime, "is_star": True,
                "r50": 1.3} for r in truth]
    catalog = sep_catalog(tmp_path / "det.fits", objects, sky=sky)

    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(clean_only=False, max_snr_weight=0)
    cfg = PiffFitConfig()
    cfg.set_optics("pinned")
    fit = PiffDataFit(L2Image(image), cat, cfg)
    fit.set_diagnostics(moments=False)
    res = fit.run(save_to=tmp_path / "fit")
    assert res.fit_summary["converged"]
    assert abs(res.fit_summary["chisq_dof"] - 1) < 0.1, res.fit_summary["chisq_dof"]
    assert abs(np.median(res.table["dx"])) < 0.05 and abs(np.median(res.table["dy"])) < 0.05
