"""Save/load of simulations, using small fake images (no PSF rendering)."""
import numpy as np
import galsim
from astropy.io import fits
from astropy.table import vstack

from psf_fomo.sim import RomanSimulation, Simulation, load_simulation
from psf_fomo.sim import storage
from psf_fomo.sim.simulation import _Truth


def fake_simulation(wcs_kind="simple"):
    sim = RomanSimulation(scas=[2, 7], n_obs=2, n_stars=3, seed=3)
    sim.set_wcs(wcs_kind)
    config = sim.config
    truth = _Truth(config)
    rng = np.random.default_rng(0)
    images, catalogs = {}, []
    for obs in range(2):
        tables = []
        for sca in config["scas"]:
            image = galsim.ImageF(40, 30, xmin=11, ymin=21, wcs=truth.wcs[sca])
            image.array[:] = rng.normal(size=image.array.shape)
            images[obs, sca] = image
            table = truth.sca_truth(obs, sca)[0].copy()
            table["bkg_var"] = np.full(len(table), 2.0)
            tables.append(table)
        catalogs.append(vstack(tables))
    return Simulation(config, {"psf_fomo": "test"}, images, catalogs)


def test_save_load_round_trip(tmp_path):
    sim = fake_simulation()
    sim.save(tmp_path)
    loaded = load_simulation(tmp_path)
    assert loaded.config == sim.config
    assert loaded.provenance == sim.provenance
    assert loaded.noiseless is None
    for key, image in sim.images.items():
        np.testing.assert_array_equal(loaded.images[key].array, image.array)
        assert loaded.images[key].bounds == image.bounds
        # FITS round trip may return an equivalent WCS of another class (e.g. a PixelScale
        # comes back as an OffsetWCS with zero offsets): compare the mapping, not the type.
        pos = galsim.PositionD(20.0, 30.0)
        assert loaded.images[key].wcs.toWorld(pos) == image.wcs.toWorld(pos)
        np.testing.assert_allclose(loaded.images[key].wcs.jacobian(pos).getMatrix(),
                                   image.wcs.jacobian(pos).getMatrix())
    for t1, t2 in zip(sim.catalogs, loaded.catalogs):
        assert t1.colnames == t2.colnames
        for name in t1.colnames:
            np.testing.assert_array_equal(t1[name], t2[name])


def test_file_layout_simple_wcs(tmp_path):
    fake_simulation("simple").save(tmp_path)
    with fits.open(tmp_path / storage.image_file_name(0)) as hdus:
        assert [h.name for h in hdus[1:]] == ["SCA02", "SCA07"]
        assert hdus[1].header["GAIN"] == galsim.roman.gain
        assert hdus[1].header["BKGVAR"] == 2.0
    with fits.open(tmp_path / storage.catalog_file_name(1)) as hdus:
        assert [h.name for h in hdus[1:]] == ["STARS", "SCA02", "SCA07"]
        assert set(hdus["SCA07"].data["sca"]) == {7}
        assert len(hdus["STARS"].data) == 6


def test_file_layout_roman_wcs(tmp_path):
    sim = fake_simulation("roman")
    assert "ra" in sim.catalogs[0].colnames
    sim.save(tmp_path)
    with fits.open(tmp_path / storage.catalog_file_name(0)) as hdus:
        # With a real WCS, Piff can select stars per SCA from ra/dec: no per-SCA HDUs.
        assert [h.name for h in hdus[1:]] == ["STARS"]
    loaded = load_simulation(tmp_path)
    pos = galsim.PositionD(20.0, 30.0)
    assert (loaded.images[0, 7].wcs.toWorld(pos).distanceTo(
        sim.images[0, 7].wcs.toWorld(pos)) < 1e-6 * galsim.arcsec)
