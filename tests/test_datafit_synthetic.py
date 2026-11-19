"""simulation_to_l2: a simulated SCA written as an L2 file and a detection catalog."""
import numpy as np
import galsim.roman
import pytest
from astropy.table import Table

from psf_fomo.datafit import L2Image, StarCatalog
from psf_fomo.datafit.synthetic import simulation_to_l2
from psf_fomo.sim import RomanSimulation


@pytest.fixture(scope="module")
def sim():
    s = RomanSimulation(scas=[5], n_stars=4, seed=3)
    s.set_wcs("roman")
    s.set_brightness(mag=19.0)
    return s.run(save_noiseless=True)


def test_needs_noiseless_images(tmp_path):
    class NoNoiseless:      # what run() without save_noiseless gives
        noiseless = None
    with pytest.raises(ValueError, match="save_noiseless"):
        simulation_to_l2(NoNoiseless(), sca=5)


@pytest.mark.slow
def test_temporary_folder_is_removed(sim, tmp_path):
    synth = simulation_to_l2(sim, sca=5)
    assert synth.temporary and synth.image_file.exists() and synth.catalog_file.exists()
    folder = synth.folder
    synth.cleanup()
    assert not folder.exists()
    kept = simulation_to_l2(sim, sca=5, folder=tmp_path / "kept")
    kept.cleanup()                              # a chosen folder is never removed
    assert not kept.temporary and kept.image_file.exists()


@pytest.mark.slow
@pytest.mark.parametrize("frame", ["science", "galsim"])
def test_stars_land_on_catalog_positions(sim, frame):
    synth = simulation_to_l2(sim, sca=5, frame=frame, sky=0.6)
    image_file, catalog_file = synth.image_file, synth.catalog_file
    img = L2Image(image_file, frame=frame)
    assert img.sca == 5 and img.sky == pytest.approx(0.6)
    assert img.check_orientation()["parity_ok"]   # the fit frame is GalSim's in both cases
    cat = Table.read(catalog_file)
    truth = sim.catalogs[0]
    # Catalog fluxes are the true fluxes in DN/s.
    np.testing.assert_allclose(cat["flux_auto"],
                               truth["flux"] / galsim.roman.gain / galsim.roman.exptime)
    # Each catalog position (0-based, file frame) maps back to the simulated position.
    x, y = img.to_fit_frame(np.asarray(cat["x"]) + 1, np.asarray(cat["y"]) + 1)
    np.testing.assert_allclose(x, truth["x"], atol=1e-9)
    np.testing.assert_allclose(y, truth["y"], atol=1e-9)
    # In the file's own frame, the brightest pixel near each catalog position is within one
    # pixel of it.
    data = img.array("image")
    for xi, yi in zip(cat["x"], cat["y"]):          # 0-based array indices
        ix, iy = int(round(xi)), int(round(yi))
        stamp = data[iy - 3:iy + 4, ix - 3:ix + 4]
        dy, dx = np.unravel_index(np.argmax(stamp), stamp.shape)
        assert abs(dx - 3) <= 1 and abs(dy - 3) <= 1
    StarCatalog(catalog_file)                 # readable as a detection catalog
