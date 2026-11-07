"""NaN pixels in L2 images and stars Piff cannot fit.

Real L2 files have NaN in masked pixels (first real file: 1036, all masked). Piff zero-weights
masked pixels but multiplies weight × image in its sums, so one NaN in a stamp made that star's
flux and centre NaN and Piff excluded it, leaving χ² and dof unset. The fit's working copy now
cleans such pixels, and the diagnostics accept stars without a χ².
"""
import hashlib

import numpy as np
import piff
import pytest
from astropy.io import fits

from l2files import field, open_l2, roman_wcs, scene, sep_catalog, write_l2_file
from psf_fomo.datafit import DataFit, DataReferenceSolver, StarCatalog
from psf_fomo.datafit.diagnostics import star_table
from psf_fomo.piffval import FitConfig
from psf_fomo.piffval.fitting import fit_columns
from psf_fomo.piffval.residuals import moment_residuals

GAUSSIAN_MODEL = {"type": "Simple", "model": {"type": "Gaussian"},
                  "interp": {"type": "Mean"}, "max_iter": 10}
N_NAN = 40


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def nan_field(tmp_path_factory):
    """Stars only (GalSim frame), N_NAN NaN pixels, all masked; one 3 pix from star 1."""
    folder = tmp_path_factory.mktemp("nan")
    objects = field(n_stars=20, n_gals=0, seed=41)
    wcs = roman_wcs(sca=5)
    sci = scene(wcs, shape=(700, 700), seed=42,
                stars=[(o["x"], o["y"], o["flux"], o["profile"]) for o in objects])
    mask = np.zeros(sci.shape, np.uint8)
    o = objects[0]
    ix, iy = int(round(o["x"])) + 3, int(round(o["y"]))       # 1-based, inside its stamp
    rng = np.random.default_rng(43)
    ys = np.concatenate([[iy - 1], rng.integers(0, 700, N_NAN - 1)])
    xs = np.concatenate([[ix - 1], rng.integers(0, 700, N_NAN - 1)])
    sci[ys, xs] = np.nan
    mask[ys, xs] = 1
    n_nan = int(np.isnan(sci).sum())                           # random pixels may repeat
    image = write_l2_file(folder / "nan.fits", sci, wcs, mask=mask)
    catalog = sep_catalog(folder / "det.fits", objects)
    return image, catalog, n_nan, (ys, xs)


def test_check_finite_and_cleaned_copy(nan_field, tmp_path):
    image, _, n_nan, (ys, xs) = nan_field
    img = open_l2(image)
    assert img.check_finite() == {"total": n_nan, "unmasked": 0}
    before = md5(image)
    copy = img.write_fit_copy(tmp_path / "copy.fits")
    assert copy != img.path and md5(image) == before          # original untouched
    data = fits.getdata(copy, img.hdus["image"])
    assert np.all(np.isfinite(data)) and np.all(data[ys, xs] == 0)
    assert np.all(fits.getdata(copy, img.hdus["ivar"])[ys, xs] == 0)
    assert np.all(fits.getdata(copy, img.hdus["mask"])[ys, xs] == 1)
    # Elsewhere the copy equals the original (no flip for a GalSim-frame file).
    orig = fits.getdata(image, img.hdus["image"])
    good = np.isfinite(orig)
    np.testing.assert_array_equal(data[good], orig[good])


@pytest.fixture(scope="module")
def nan_fit(nan_field, tmp_path_factory):
    image, catalog, _, _ = nan_field
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(clean_only=False, max_snr_weight=0)
    fit = DataFit(open_l2(image), cat, FitConfig.from_piff_config(GAUSSIAN_MODEL))
    fit.set_diagnostics(moments=False)
    return fit.run(save_to=tmp_path_factory.mktemp("nanfit"))


def test_star_with_nan_in_stamp_is_fitted(nan_field, nan_fit):
    _, _, n_nan, _ = nan_field
    t = nan_fit.table
    assert len(t) == 20 and np.all(t["flag"] == 0)            # nobody excluded by Piff
    assert np.all(np.isfinite(t["flux"])) and np.all(np.isfinite(t["chisq"]))
    assert nan_fit.checks["nonfinite"] == {"total": n_nan, "unmasked": 0}
    assert abs(nan_fit.fit_summary["chisq_dof"] - 1) < 0.08


def test_diagnostics_accept_a_star_piff_could_not_fit(nan_fit):
    # A star as Piff leaves it after a failed reflux: NaN flux and centre, χ² and dof unset.
    stars, _ = nan_fit.fitted_stars()
    s0 = stars[0]
    broken = piff.Star(s0.data, piff.StarFit(s0.fit.params, flux=np.nan,
                                             center=(np.nan, np.nan)))
    stars = [broken] + list(stars[1:])
    cols = fit_columns(nan_fit.psf, stars)
    assert np.isnan(cols["chisq"][0]) and np.isnan(cols["dof"][0])
    assert np.isnan(cols["chisq_dof"][0]) and np.all(np.isfinite(cols["chisq_dof"][1:]))

    table = star_table(nan_fit.psf, stars, nan_fit.selection)
    m = moment_residuals(nan_fit.psf, stars, table, native=(), oversampled=("hsm",),
                         comparisons=("model_os",))
    first = m[m["star_id"] == table["star_id"][0]]
    assert len(first) == 1 and first["flag"][0] & 8 and np.isnan(first["T_model"][0])
    assert np.all(m["flag"][1:] == 0)

    solver = DataReferenceSolver(nan_fit.psf, stars, nan_fit.selection)
    assert solver.n_excluded == 1 and len(solver.stars) == len(stars) - 1
