"""StarCatalog and DataFit on files in the user's L2 layout (closed loop, no real data).

A fast analytic model (Piff Gaussian, Mean interpolation) is fitted to images drawn with the
same Gaussian PSF, so the model is exact and χ²/dof ≈ 1 tests the noise model.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import piff
import pytest
from astropy.table import Table

from l2files import field, open_l2, write_field
from psf_fomo.datafit import DataFit, StarCatalog, load_data_fit
from psf_fomo.io import read_yaml
from psf_fomo.piffval import FitConfig
from psf_fomo.piffval.fitting import make_logger

GAIN = 250.0
SHAPE = (700, 700)
GAUSSIAN_MODEL = {"type": "Simple", "model": {"type": "Gaussian"},
                  "interp": {"type": "Mean"}, "max_iter": 10}


@pytest.fixture(scope="module")
def stars_only(tmp_path_factory):
    folder = tmp_path_factory.mktemp("stars")
    objects = field(n_stars=50, n_gals=0, shape=SHAPE, seed=2)
    return (*write_field(folder, objects, unclean=(3, 7, 11)), objects)


@pytest.fixture(scope="module")
def stars_and_galaxies(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mixed")
    objects = field(n_stars=60, n_gals=60, shape=SHAPE, seed=4)
    # One extra star 9 pix (≈ 1″) from a galaxy, for the min_sep test; the galaxy is chosen
    # so that no other object is within 25 pix of the new star.
    for gal in (o for o in objects if not o["is_star"]):
        x, y = gal["x"] + 9.0, gal["y"]
        others = [o for o in objects if o is not gal]
        if x < SHAPE[1] - 25 and all(np.hypot(x - o["x"], y - o["y"]) > 25 for o in others):
            break
    else:
        raise RuntimeError("no galaxy with a free spot for the min_sep test star")
    objects.append(dict(objects[0], id=len(objects) + 1, x=x, y=y, flux=2000.0))
    return (*write_field(folder, objects, seed=5), objects)


def gaussian_config():
    return FitConfig.from_piff_config(GAUSSIAN_MODEL)


@pytest.fixture(scope="module")
def basic_fit(stars_only, tmp_path_factory):
    image, catalog, _ = stars_only
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(max_snr_weight=0)  # no S/N cap: χ² keeps its noise meaning
    fit = DataFit(open_l2(image), cat, gaussian_config(), name="basic")
    fit.set_reserve(0.2, seed=3)
    out = tmp_path_factory.mktemp("fit")
    return fit.run(save_to=out), out


def test_piff_table_and_select(stars_only):
    image, catalog, objects = stars_only
    img = open_l2(image)
    cat = StarCatalog(catalog)
    t = cat.piff_table(img)
    np.testing.assert_allclose(t["x"], [o["x"] for o in objects])  # 0-based + 1
    assert list(t["id"]) == [o["id"] for o in objects]
    assert set(t["clean"]) == {0, 1} and t["clean"].dtype.kind == "i"
    np.testing.assert_allclose(t["mag"], 26.0 - 2.5 * np.log10(t["flux_cat"]))
    np.testing.assert_allclose(t["gain"], GAIN)
    assert "gain" not in cat.property_cols(t) and "id" in cat.property_cols(t)
    img.set_weights("ivar")
    assert "gain" not in cat.piff_table(img).colnames

    # Piff's own max_snr_weight default unless set
    assert cat.piff_select() == {"type": "SizeMag", "reject_where": "clean == 0"}
    cat.select_stars("sizemag", impurity=0.02)
    cat.set_cuts(clean_only=True, min_snr=20, min_sep=1.5, reject_where="r50 > 5")
    sel = cat.piff_select()
    assert sel["impurity"] == 0.02 and sel["min_sep"] == 1.5 and sel["min_snr"] == 20
    assert sel["reject_where"] == "clean == 0 or (r50 > 5)"
    cat.select_stars("column", column="is_star", use_flag=1)
    assert cat.piff_select()["type"] == "Flag"
    assert cat.piff_select()["flag_name"] == "is_star"
    assert "is_star" in cat.piff_table().colnames
    cat.select_stars("all")
    assert cat.piff_select()["type"] == "Flag" and "flag_name" not in cat.piff_select()
    with pytest.raises(ValueError):
        cat.select_stars("column")
    assert StarCatalog.from_config(cat.config()).config() == cat.config()


def test_catalog_from_table(stars_only, tmp_path):
    # A detection table in memory (e.g. from detect_sources) works like its file.
    image, catalog, _ = stars_only
    img = open_l2(image)
    from_file = StarCatalog(catalog)
    table = Table.read(catalog)
    table.meta = {str(k).lower(): v for k, v in table.meta.items()}  # keys matched any case
    from_table = StarCatalog(table)
    assert from_table.path is None and from_table.header == from_file.header
    t_file, t_table = from_file.piff_table(img), from_table.piff_table(img)
    assert t_file.colnames == t_table.colnames
    for name in t_file.colnames:
        np.testing.assert_array_equal(t_file[name], t_table[name])
    table["x"][0] += 100.0  # the catalog keeps its own copy
    assert from_table.table["x"][0] != table["x"][0]

    with pytest.raises(ValueError, match="table="):
        StarCatalog.from_config(from_table.config())
    rebuilt = StarCatalog.from_config(from_table.config(), table=from_table.table)
    assert rebuilt.config() == from_table.config()

    # A fit from the table saves it as detections.fits and records that path.
    from_table.select_stars("all")
    res = DataFit(img, from_table, gaussian_config()).run(save_to=tmp_path)
    assert res.config["catalog"]["path"] == str((tmp_path / "detections.fits").resolve())
    again = StarCatalog.from_config(res.config["catalog"])
    assert len(again.table) == len(from_table.table)
    assert again.header == from_table.header


def test_table_catalog_with_numpy_meta_saves(stars_only, tmp_path):
    # A table from a Python detection step can carry NumPy scalars in its meta (a real run
    # failed to save config.yaml on np.float64 ABZP): the header is plain, the fit saves.
    image, catalog, _ = stars_only
    table = Table.read(catalog)
    table.meta = {"ABZP": np.float64(25.775), "SKY": np.float32(2.0), "NMATCH": np.int64(50)}
    cat = StarCatalog(table)
    assert cat.header == {"ABZP": 25.775, "SKY": 2.0, "NMATCH": 50}
    assert all(type(v) in (float, int) for v in cat.header.values())
    cat.select_stars("all")
    cat.set_cuts(clean_only=False)
    fit = DataFit(open_l2(image), cat, gaussian_config())
    fit.set_diagnostics(moments=False)
    res = fit.run(save_to=tmp_path)
    saved = read_yaml(tmp_path / "config.yaml")
    assert saved["config"]["catalog"]["header"]["ABZP"] == 25.775
    again = load_data_fit(tmp_path)
    assert list(again.table["star_id"]) == list(res.table["star_id"])


def test_catalog_errors(stars_only, tmp_path):
    _, catalog, _ = stars_only
    with pytest.raises(ValueError, match="columns"):
        StarCatalog(catalog, flux_col="flux_iso")
    t = Table.read(catalog)
    t["id"][1] = t["id"][0]
    t.write(tmp_path / "dup.fits")
    with pytest.raises(ValueError, match="not unique"):
        StarCatalog(tmp_path / "dup.fits")


def test_fit_noise_model_and_positions(basic_fit):
    res, _ = basic_fit
    s = res.fit_summary
    assert s["converged"] and s["n_catalog"] == 50
    # Exact model + model Poisson weights: χ²/dof ≈ 1 (σ ≈ 0.01 for ~25000 pixels).
    assert abs(s["chisq_dof"] - 1) < 0.05
    t = res.table
    # unclean ids 3, 7, 11 rejected; the rest fitted or reserve
    assert not set(t["star_id"]) & {3, 7, 11} and len(t) == 47
    assert int(t["reserve"].sum()) == int(0.2 * 47)
    # The 0-based catalog was shifted correctly: fitted centres ≈ catalog positions.
    assert abs(np.median(t["dx"])) < 0.05 and abs(np.median(t["dy"])) < 0.05
    assert abs(np.median(t["sky_resid"])) < 0.02  # SKYBKG subtracted
    np.testing.assert_allclose(np.median(t["flux_cat_r"]), 1 / 0.95, rtol=0.02)
    np.testing.assert_allclose(t["gain"], GAIN)
    assert res.checks["weights_ratio"] is None and res.checks["orientation"]["parity_ok"]
    sel = res.selection
    assert sel["on_image"].all() and sel["candidate"].all() and sel["star"].sum() == 47


def test_moments_components_plots(basic_fit):
    pytest.importorskip("ngmix")
    res, _ = basic_fit
    m = res.moments
    assert set(m["comparison"]) == {"data", "model_os"}
    data = m[(m["comparison"] == "data") & (m["flag"] == 0)]
    assert abs(np.mean(data["dT_T"])) < 0.01  # exact model
    assert "dT_T_mean" in res.moment_summary().colnames
    mm = res.model_moments(n_grid=2)
    assert len(mm) == 4 and np.all(mm["flag"] == 0) and np.all(np.isfinite(mm["T"]))
    kernel = res.components()["kernel"]
    assert not any(c.endswith("_true") for c in kernel.colnames)
    for fig in (res.plot_mean_images(), res.plot_chi2_radius(), res.plot_size_mag(),
                res.plot_field("sky_resid"), res.plot_moment_residuals(),
                res.plot_field("T_model", "model_os", "hsm")):
        assert fig is not None


def test_wrong_pixel_convention_shows_in_centres(stars_only, tmp_path):
    # Treating the 0-based SEP positions as 1-based puts every stamp 1 pixel off: Piff's
    # centroid fit then finds the stars at +1 pixel in x and y.
    image, catalog, _ = stars_only
    cat = StarCatalog(catalog, zero_based=False)
    cat.select_stars("all")
    res = DataFit(open_l2(image), cat, gaussian_config()).run(save_to=tmp_path)
    assert abs(np.median(res.table["dx"]) - 1) < 0.1
    assert abs(np.median(res.table["dy"]) - 1) < 0.1


def test_ivar_mode_has_no_star_poisson(stars_only, tmp_path):
    # IVAR_UNIF holds read + sky noise only; without Piff's per-star Poisson term the core
    # pixels of bright stars have residuals far above that noise. Only a few dozen of a
    # stamp's 625 pixels are affected, so the excess shows mostly in the bright stars.
    image, catalog, _ = stars_only
    img = open_l2(image)
    img.set_weights("ivar")
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(max_snr_weight=0)
    res = DataFit(img, cat, gaussian_config()).run(save_to=tmp_path)
    assert res.fit_summary["chisq_dof"] > 1.2  # model_poisson gives 1.00 ± 0.05
    t = res.table
    order = np.argsort(t["flux"])
    quarter = len(t) // 4
    faint = np.mean(t["chisq_dof"][order[:quarter]])
    bright = np.mean(t["chisq_dof"][order[-quarter:]])
    assert bright > faint + 0.3, (faint, bright)
    assert np.all(np.isnan(t["gain"]))


def test_piff_snr_cap_and_reload(stars_only, tmp_path):
    # Piff's default max_snr_weight (100) inflates the variance of bright stars: χ²/dof < 1.
    # A reload re-applies that scaling from the saved snr / raw_snr.
    image, catalog, _ = stars_only
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    res = DataFit(open_l2(image), cat, gaussian_config()).run(save_to=tmp_path)
    capped = [s for s in res.psf.stars if s.data.properties["snr"] < s.data.properties["raw_snr"]]
    assert capped and res.fit_summary["chisq_dof"] < 0.9
    stars0, _ = res.fitted_stars()
    stars1, _ = load_data_fit(tmp_path).fitted_stars()
    for a, b in zip(stars0, stars1):
        np.testing.assert_allclose(a.weight.array, b.weight.array, rtol=0.02)


def test_save_load(basic_fit):
    res, folder = basic_fit
    assert (folder / "piff.yaml").exists() and (folder / "catalog.fits").exists()
    again = load_data_fit(folder)
    assert list(again.table["star_id"]) == list(res.table["star_id"])
    assert again.fit_summary == res.fit_summary and again.config["sca"] == 5
    stars0, _ = res.fitted_stars()
    stars1, _ = again.fitted_stars()
    for a, b in zip(stars0, stars1):
        np.testing.assert_array_equal(a.image.array, b.image.array)
        assert a.is_reserve == b.is_reserve
        # Poisson term re-added from the final model (Piff's last update preceded its final
        # reflux): weights agree closely, not exactly.
        np.testing.assert_allclose(a.weight.array, b.weight.array, rtol=0.02)
    pytest.importorskip("ngmix")
    m0 = res.moments[res.moments["comparison"] == "model_os"]
    m1 = again.compute_moments()
    m1 = m1[m1["comparison"] == "model_os"]
    np.testing.assert_allclose(m1["T_model"], m0["T_model"], rtol=1e-8)


def test_piff_yaml_reproduces_with_piff(basic_fit):
    # The written config runs through Piff's own pipeline (as piffify would) to the same fit.
    res, folder = basic_fit
    config = read_yaml(folder / "piff.yaml")
    psf = piff.config.process(config, logger=make_logger(0)[0])
    assert len(psf.stars) == len(res.table)
    np.testing.assert_allclose(psf.chisq, res.fit_summary["chisq"], rtol=1e-6)


def test_sizemag_picks_the_stars(stars_and_galaxies, tmp_path):
    pytest.importorskip("ngmix")
    image, catalog, objects = stars_and_galaxies
    cat = StarCatalog(catalog)
    cat.select_stars("sizemag")   # with shape method gauss
    cat.set_cuts(clean_only=False)
    res = DataFit(open_l2(image), cat, gaussian_config()).run(save_to=tmp_path)
    is_star = {o["id"]: o["is_star"] for o in objects}
    chosen = [int(i) for i in res.selection["id"][res.selection["candidate"]]]
    purity = np.mean([is_star[i] for i in chosen])
    completeness = len([i for i in chosen if is_star[i]]) / sum(is_star.values())
    assert purity > 0.9 and completeness > 0.5, (purity, completeness)


def test_min_sep_counts_galaxies_as_neighbours(stars_and_galaxies, tmp_path):
    # Stars are chosen by a column; the galaxies are not candidates but still neighbours.
    image, catalog, objects = stars_and_galaxies
    cat = StarCatalog(catalog)
    cat.select_stars("column", column="is_star", use_flag=1)
    cat.set_cuts(clean_only=False, min_sep=1.5)
    res = DataFit(open_l2(image), cat, gaussian_config()).run(save_to=tmp_path)
    near = objects[-1]["id"]
    sel = res.selection[res.selection["id"] == near][0]
    assert sel["candidate"] and not sel["star"]
    n_star_objects = sum(o["is_star"] for o in objects)
    assert res.selection["candidate"].sum() == n_star_objects
    assert res.selection["star"].sum() == n_star_objects - 1


def test_truth_comparisons_rejected(stars_only):
    image, catalog, _ = stars_only
    fit = DataFit(open_l2(image), StarCatalog(catalog), gaussian_config())
    with pytest.raises(ValueError, match="no truth"):
        fit.set_diagnostics(comparisons=("data", "truth_os"))
