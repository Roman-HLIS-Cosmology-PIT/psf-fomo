"""Pre-fit moments of every detection (StarCatalog.measure) and size–magnitude plots."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

pytest.importorskip("ngmix")  # gauss and wmom are ngmix measurements

from l2files import field, open_l2, write_field
from psf_fomo.datafit import DataFit, StarCatalog, plot_size_mag
from psf_fomo.piffval import FitConfig

ABZP = 26.0  # sep_catalog's zero point
CHEAP = {"type": "Simple", "model": {"type": "Gaussian"}, "interp": {"type": "Mean"},
         "max_iter": 3}


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    folder = tmp_path_factory.mktemp("measure")
    objects = field(n_stars=60, n_gals=60, seed=4)
    image, catalog = write_field(folder, objects, seed=5)
    return image, catalog, {o["id"]: o["is_star"] for o in objects}


@pytest.fixture(scope="module")
def measured(mixed):
    image, catalog, _ = mixed
    cat = StarCatalog(catalog)
    cat.set_cuts(clean_only=False)
    return cat.measure(open_l2(image), algorithms=("gauss", "wmom"), wmom_fwhm=(0.4, 0.6),
                       preview_selection=True)


def test_columns_and_magnitudes(measured, mixed):
    t = measured
    assert len(t) == 120 and t.meta["n_measured"] == 120
    for label in ("gauss", "wmom_0.4", "wmom_0.6"):
        for key in ("T", "sigma", "e1", "e2", "flux", "mag", "flag"):
            assert f"{key}_{label}" in t.colnames
    assert {"id", "x", "y", "flux_cat", "mag", "size_cat", "clean", "candidate",
            "star"} <= set(t.colnames)
    assert str(t["T_gauss"].unit) == "pix2"
    np.testing.assert_allclose(t["mag"], ABZP - 2.5 * np.log10(t["flux_cat"]))
    ok = t["flux_gauss"] > 0
    np.testing.assert_allclose(t["mag_gauss"][ok], ABZP - 2.5 * np.log10(t["flux_gauss"][ok]))


def test_stars_form_a_tight_locus_below_galaxies(measured, mixed):
    t = measured
    is_star = np.array([mixed[2][int(i)] for i in t["id"]])
    good = t["flag_gauss"] == 0
    T = np.asarray(t["T_gauss"])
    stars, gals = T[good & is_star], T[good & ~is_star]
    assert np.median(gals) > 2 * np.median(stars)
    bright = good & is_star & (np.asarray(t["flux_cat"]) > 1000)
    assert np.std(T[bright]) / np.median(T[bright]) < 0.1
    # wmom sizes are compressed toward the weight, but still separate the two populations.
    w = np.asarray(t["T_wmom_0.4"])
    wgood = t["flag_wmom_0.4"] == 0
    assert np.median(w[wgood & ~is_star]) > 1.2 * np.median(w[wgood & is_star])
    # The measured flux follows the catalog flux (0.95 × true in the test catalog).
    ratio = np.asarray(t["flux_gauss"])[bright] / np.asarray(t["flux_cat"])[bright]
    assert 0.9 < np.median(ratio) < 1.2


def test_preview_matches_the_fit_selection(measured, mixed, tmp_path):
    image, catalog, _ = mixed
    cat = StarCatalog(catalog)      # same default selection (SizeMag) and cuts as `measured`
    cat.set_cuts(clean_only=False)
    res = DataFit(open_l2(image), cat, FitConfig.from_piff_config(CHEAP)).run(save_to=tmp_path)
    sel = res.selection[res.selection["on_image"]]
    by_id = {int(i): (bool(c), bool(s)) for i, c, s in zip(sel["id"], sel["candidate"],
                                                            sel["star"])}
    preview = {int(i): (bool(c), bool(s)) for i, c, s in zip(measured["id"],
                                                              measured["candidate"],
                                                              measured["star"])}
    assert preview == by_id
    assert measured["star"].sum() > 0


def test_preview_reports_each_stage(mixed):
    # Cuts that reject every candidate: the candidates are still reported (Piff would stop
    # with "All stars were rejected"), with a warning and the counts in meta.
    image, catalog, _ = mixed
    cat = StarCatalog(catalog)
    cat.select_stars("all")
    cat.set_cuts(clean_only=False, reject_where="flux_cat > 0")
    with pytest.warns(UserWarning, match="rejected all"):
        t = cat.measure(open_l2(image), algorithms=(), preview_selection=True)
    assert t["candidate"].all() and not t["star"].any()
    assert t.meta["n_candidates"] == len(t) and t.meta["n_stars"] == 0
    assert t.meta["selection"]["method"] == "all"


def test_subsets(mixed):
    image, catalog, _ = mixed
    cat = StarCatalog(catalog)
    img = open_l2(image)
    t = cat.measure(img, mag_range=(17.0, 18.5))
    assert len(t) > 0 and np.all((t["mag"] >= 17.0) & (t["mag"] <= 18.5))
    assert "candidate" not in t.colnames and "T_gauss" in t.colnames
    t10 = cat.measure(img, max_objects=10)
    full = np.sort(np.asarray(cat.piff_table()["flux_cat"]))[::-1]
    np.testing.assert_allclose(np.sort(np.asarray(t10["flux_cat"]))[::-1], full[:10])
    with pytest.raises(ValueError, match="not in"):
        cat.measure(img, algorithms=("ksb",))
    with pytest.raises(ValueError, match="no detections"):
        cat.measure(img, mag_range=(40, 41))


def test_plots(measured):
    for kwargs in ({"algorithm": "gauss"}, {"algorithm": "wmom_0.4", "mag": "measured"},
                   {"algorithm": "gauss", "size": "sigma"}, {}):
        assert plot_size_mag(measured, **kwargs) is not None
    with pytest.raises(ValueError, match="needs an algorithm"):
        plot_size_mag(measured, mag="measured")
    with pytest.raises(ValueError, match="not in the table"):
        plot_size_mag(measured, algorithm="hsm")
