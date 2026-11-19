"""Roman science frame (real L2 files) vs GalSim's SCA frame: flips, WCS, fits.

Real L2 files are in Roman's science frame, mirrored relative to galsim.roman's frame: a y flip
for SCA 11, an x flip for SCA 3 (see docs/datafit.md for all SCAs). Test files here are written
in GalSim's frame and flipped into the science frame.
"""
import hashlib

import numpy as np
import galsim
import galsim.roman
import pytest
from astropy.io import fits

from l2files import (field, open_l2, roman_wcs, scene, sep_catalog, to_science_frame,
                     write_field, write_l2_file)
from psf_fomo.datafit import (DataFit, L2Image, StarCatalog, compare_frames, flip_axis,
                              flip_wcs_header, load_data_fit)
from psf_fomo.piffval import FitConfig
from psf_fomo.sim.wcs import DEFAULT_DEC, DEFAULT_RA

GAUSSIAN_MODEL = {"type": "Simple", "model": {"type": "Gaussian"},
                  "interp": {"type": "Mean"}, "max_iter": 10}


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def test_flip_rule():
    assert [s for s in range(1, 19) if flip_axis(s) == "x"] == [3, 6, 9, 12, 15, 18]
    assert flip_axis(11) == "y" and flip_axis(1) == "y"


@pytest.mark.parametrize("sca, axis", [(11, "y"), (3, "x")])
def test_fit_copy_keeps_sky_positions(tmp_path, sca, axis):
    # Every flipped pixel of the working copy lands where the original pixel lands on the
    # sky (exact header transform, incl. the SIP distortion), and the data are flipped.
    wcs = roman_wcs(sca=sca)
    sci = scene(wcs, shape=(200, 260), stars=[(50.3, 60.7, 500.0)], noise=False)
    src = write_l2_file(tmp_path / "src.fits", sci, wcs, sca=sca)
    before = md5(src)
    img = L2Image(src)                       # frame='science': the copy is flipped
    assert img.flip == axis
    dst = img.write_fit_copy(tmp_path / "copy.fits")
    assert md5(src) == before                # the original file is not changed
    hdu = img.hdus["image"]
    w0 = galsim.GSFitsWCS(str(src), hdu=hdu)
    w1 = galsim.GSFitsWCS(str(dst), hdu=hdu)
    for x, y in [(1, 1), (260, 1), (1, 200), (260, 200), (37.25, 151.5), (130.5, 100.5)]:
        fx, fy = img.to_fit_frame(x, y)
        p0 = w0.toWorld(galsim.PositionD(x, y))
        p1 = w1.toWorld(galsim.PositionD(float(fx), float(fy)))
        assert p0.distanceTo(p1) / galsim.arcsec < 1e-6
    a0, a1 = fits.getdata(src, hdu), fits.getdata(dst, hdu)
    np.testing.assert_array_equal(a1, np.flip(a0, axis=1 if axis == "x" else 0))
    m1 = fits.getdata(dst, img.hdus["mask"])
    assert m1.shape == a0.shape
    # Flipping twice restores the file.
    back = L2Image(dst, sca=sca).write_fit_copy(tmp_path / "back.fits")
    np.testing.assert_array_equal(fits.getdata(back, hdu), a0)
    h0, h2 = fits.getheader(src, hdu), fits.getheader(back, hdu)
    for key in ("CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "A_2_0", "B_0_2"):
        if key in h0:
            assert h2[key] == pytest.approx(h0[key], rel=1e-12, abs=1e-15)


def test_galsim_frame_needs_no_copy(tmp_path):
    wcs = roman_wcs(sca=5)
    src = write_l2_file(tmp_path / "g.fits", scene(wcs, noise=False), wcs)
    img = open_l2(src)
    assert img.flip is None and img.write_fit_copy(tmp_path / "unused.fits") == img.path
    assert not (tmp_path / "unused.fits").exists()
    x, y = img.to_fit_frame([10.0], [20.0])
    assert (x[0], y[0]) == (10.0, 20.0)


def test_orientation_by_declared_frame(tmp_path):
    wcs = roman_wcs(sca=11)
    src = write_l2_file(tmp_path / "g.fits", scene(wcs, noise=False), wcs, sca=11)
    science = to_science_frame(src, tmp_path / "s.fits")
    assert open_l2(src).check_orientation()["parity_ok"]                # GalSim file
    assert L2Image(science).check_orientation()["parity_ok"]            # science, flipped
    with pytest.raises(ValueError, match="mirrored"):
        open_l2(science).check_orientation()                            # declared wrongly
    with pytest.warns(UserWarning, match="mirrored"):
        out = open_l2(science).check_orientation(strict=False)
    assert not out["parity_ok"] and out["frame"] == "galsim"


@pytest.mark.parametrize("sca, axis", [(11, "y"), (3, "x")])
def test_compare_frames_finds_the_flip(sca, axis):
    # A science-frame WCS made from GalSim's own (full SCA); with the same pointing the right
    # mapping is exact, the wrong parity is off by arcminutes.
    n = galsim.roman.n_pix
    header = galsim.FitsHeader()
    roman_wcs(sca=sca).writeToFitsHeader(header, galsim.BoundsI(1, n, 1, n))
    science = galsim.GSFitsWCS(header=galsim.FitsHeader(
        header=flip_wcs_header(header.header, axis, (n, n))))
    pointing = {"ra_v1": DEFAULT_RA, "dec_v1": DEFAULT_DEC, "target_ra": DEFAULT_RA,
                "target_dec": DEFAULT_DEC, "pa_v3": 0.0, "pa_aperture": 0.0}
    t = compare_frames(science, sca, (n, n), pointing)
    block = t[(t["world_pos"] == "target") & (t["pa"] == "pa_v3")]
    rows = {r["mapping"]: r for r in block}
    right = "flip x" if axis == "x" else "flip y"
    wrong = "flip y" if axis == "x" else "flip x"
    assert t.meta["best"] == right
    assert rows[right]["rms"] < 1e-3 and abs(rows[right]["extra_rotation"]) < 1e-3
    assert rows[right]["centre_offset"] < 1e-3
    # The other flip differs by a 180° rotation: removed by the best rotation up to the part
    # of GalSim's distortion that is not symmetric under it (≈ 1 mas on SCA 3).
    assert rows[wrong]["rms"] > 100 and rows[wrong]["rms_rotation_free"] < 0.05
    for name in ("as-is", "rotate 180"):
        assert rows[name]["rms_rotation_free"] > 100     # a mirror cannot be rotated away
    with pytest.raises(ValueError, match="full SCA"):
        compare_frames(science, sca, (100, 100), pointing)


@pytest.fixture(scope="module")
def science_field(tmp_path_factory):
    """The same stars as a GalSim-frame file and as a science-frame file + catalog (SCA 5)."""
    folder = tmp_path_factory.mktemp("frames")
    objects = field(n_stars=40, n_gals=0, seed=12)
    galsim_image, galsim_catalog = write_field(folder, objects, seed=13)
    science_image = to_science_frame(galsim_image, folder / "science.fits")
    ny = 700  # write_field's default shape; SCA 5 flips y
    flipped = [dict(o, y=ny + 1 - o["y"]) for o in objects]
    science_catalog = sep_catalog(folder / "det_science.fits", flipped)
    return galsim_image, galsim_catalog, science_image, science_catalog, objects


def gaussian_fit(image, catalog, folder):
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(clean_only=False, max_snr_weight=0)
    return DataFit(image, cat, FitConfig.from_piff_config(GAUSSIAN_MODEL)).run(save_to=folder)


def test_piff_table_and_measure_frames(science_field):
    _, _, science_image, science_catalog, objects = science_field
    img = L2Image(science_image)
    t = StarCatalog(science_catalog).piff_table(img)
    # Fit frame = GalSim frame (the original positions); x_l2, y_l2 = the file's.
    np.testing.assert_allclose(t["x"], [o["x"] for o in objects])
    np.testing.assert_allclose(t["y"], [o["y"] for o in objects])
    np.testing.assert_allclose(t["y_l2"], [701 - o["y"] for o in objects])
    np.testing.assert_allclose(t["gain"], 250.0)
    own = StarCatalog(science_catalog).piff_table(img, fit_frame=False)
    assert "x_l2" not in own.colnames
    np.testing.assert_allclose(own["y"], t["y_l2"])


def test_fit_in_science_frame_equals_galsim_frame(science_field, tmp_path):
    # Flipping the science file back gives exactly the GalSim-frame pixels and WCS, so the fit
    # is the same as on the GalSim-frame file.
    galsim_image, galsim_catalog, science_image, science_catalog, _ = science_field
    ref = gaussian_fit(open_l2(galsim_image), galsim_catalog, tmp_path / "galsim")
    res = gaussian_fit(L2Image(science_image), science_catalog, tmp_path / "science")
    assert (tmp_path / "science" / "image_galsim_frame.fits").exists()
    assert not (tmp_path / "galsim" / "image_galsim_frame.fits").exists()
    assert res.checks["orientation"]["parity_ok"] and res.checks["orientation"]["flip"] == "y"
    a, b = ref.table, res.table
    assert list(a["star_id"]) == list(b["star_id"])
    np.testing.assert_allclose(b["x"], a["x"])
    np.testing.assert_allclose(b["y"], a["y"])
    np.testing.assert_allclose(b["y_l2"], 701 - a["y"])
    np.testing.assert_allclose(b["flux"], a["flux"], rtol=1e-6)
    np.testing.assert_allclose(b["chisq"], a["chisq"], rtol=1e-5)
    assert abs(res.fit_summary["chisq_dof"] - 1) < 0.06
    # Reload: the stamps come from the working copy in the folder.
    again = load_data_fit(tmp_path / "science")
    s0, _ = res.fitted_stars()
    s1, _ = again.fitted_stars()
    for p, q in zip(s0, s1):
        np.testing.assert_array_equal(p.image.array, q.image.array)


def test_romanoptics_refuses_a_mirrored_frame(science_field, tmp_path):
    # Declaring the science file as GalSim-frame would give RomanOptics mirrored positions.
    _, _, science_image, science_catalog, _ = science_field
    cat = StarCatalog(science_catalog)
    cfg = FitConfig()
    cfg.set_optics("pinned")
    with pytest.raises(ValueError, match="mirrored"):
        DataFit(open_l2(science_image), cat, cfg).run(save_to=tmp_path)


@pytest.mark.slow
def test_romanoptics_closed_loop_in_science_frame(tmp_path):
    # Full SCA 11 (a y-flip SCA) with the Roman PSF, written in the science frame: pinned
    # optics + MTF (the truth model) must fit at the noise level after the flip.
    from psf_fomo.sim import RomanSimulation
    s = RomanSimulation(scas=[11], n_stars=36, filter="H158", seed=21)
    s.set_wcs("roman")
    s.set_placement("grid")
    s.set_brightness(mag_range=(18, 19.5))
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
    galsim_file = write_l2_file(tmp_path / "g.fits", sci, noiseless.wcs, sca=11, sky=sky,
                                read_var=read_var, gain=g_eff, planes="unif")
    science_file = to_science_frame(galsim_file, tmp_path / "s.fits")
    galsim_file.unlink()
    n = galsim.roman.n_pix
    truth = sim.catalogs[0]
    objects = [{"id": int(r["star_id"]) + 1, "x": float(r["x"]), "y": n + 1 - float(r["y"]),
                "flux": float(r["flux"]) / galsim.roman.gain / exptime, "is_star": True,
                "r50": 1.3} for r in truth]
    catalog = sep_catalog(tmp_path / "det.fits", objects, sky=sky)

    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(clean_only=False, max_snr_weight=0)
    cfg = FitConfig()
    cfg.set_optics("pinned")
    fit = DataFit(L2Image(science_file), cat, cfg)
    fit.set_diagnostics(moments=False)
    res = fit.run(save_to=tmp_path / "fit")
    assert res.fit_summary["converged"]
    assert abs(res.fit_summary["chisq_dof"] - 1) < 0.1, res.fit_summary["chisq_dof"]
    assert abs(np.median(res.table["dx"])) < 0.05 and abs(np.median(res.table["dy"])) < 0.05
