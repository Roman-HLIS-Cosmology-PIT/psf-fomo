"""Our stellar-locus selection (select_locus, StarCatalog.select_locus)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from astropy.table import Table

from psf_fomo.datafit import StarCatalog, plot_locus, select_locus

SIZE = "T_wmom_0.4"


def locus_table(n=700, seed=0, sat_mag=18.5, slope=0.03, noise=0.01, n_out=10, n_flag=5):
    """A measurement table with a known locus: size = 2 pix² × (1 + slope x̂) × noise, grown by
    15 % per magnitude brighter than ``sat_mag`` (saturation; None = none), ``n_out`` stars
    30 % too large, ``n_flag`` failed measurements."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(1, 4088, n), rng.uniform(1, 4088, n)
    mag = rng.uniform(17, 23, n)
    xn = 2 * (x - 1) / 4087 - 1
    T = 2.0 * (1 + slope * xn) * (1 + noise * rng.normal(size=n))
    if sat_mag is not None:
        T *= np.where(mag < sat_mag, 1 + 0.15 * (sat_mag - mag), 1.0)
    normal = np.flatnonzero(mag > (sat_mag or 0) + 0.5)
    out = rng.choice(normal, n_out, replace=False)
    T[out] *= 1.3
    flag = np.zeros(n, dtype=int)
    flagged = rng.choice(np.setdiff1d(normal, out), n_flag, replace=False)
    flag[flagged] = 1
    t = Table({"id": np.arange(1, n + 1), "x": x, "y": y, "mag": mag, SIZE: T,
               f"flag_{SIZE[2:]}": flag})
    return t, out, flagged


def test_saturation_outliers_and_flags():
    t, out, flagged = locus_table()
    res = select_locus(t)
    s = res.summary
    assert s["bright_limit_source"] == "detected"
    assert abs(s["bright_limit"] - 18.5) <= 0.25
    reason = dict(zip(res.table["id"], res.table["reason"]))
    assert all(reason[int(t["id"][i])] == "off_locus" for i in out)
    assert all(reason[int(t["id"][i])] == "flagged" for i in flagged)
    sat = res.table["reason"] == "saturated"
    assert np.all(res.table["mag"][sat] < s["bright_limit"])
    # The band only removes clear outliers: 99.7 % of a Gaussian locus is kept.
    assert s["off_locus_fraction"] < (len(out) + 0.01 * s["n_selected"]) / s["n_selected"]
    assert s["width"] == pytest.approx(0.01, rel=0.2)       # the injected noise
    assert s["n_selected"] == len(res.selected_ids)
    assert sum(s[f"n_{r}"] for r in ("selected", "saturated", "off_locus", "faint",
                                     "flagged")) == len(t)
    assert plot_locus(res) is not None


def test_no_saturation_no_turn():
    t, _, _ = locus_table(sat_mag=None)
    s = select_locus(t).summary
    assert s["bright_limit"] is None and s["bright_limit_source"] == "none"
    assert s["n_saturated"] == 0


def test_tight_skewed_locus_has_no_turn():
    # A very tight locus (0.2 %) with a skewed scatter (mean ≠ median, as sub-pixel phase can
    # make), no saturation: the fitted centre is offset from every bin median, but that offset
    # is constant in magnitude, so it is no turn.
    rng = np.random.default_rng(7)
    n = 700
    t = Table({"id": np.arange(1, n + 1), "x": rng.uniform(1, 4088, n),
               "y": rng.uniform(1, 4088, n), "mag": rng.uniform(17, 23, n),
               SIZE: 2.0 * (1 + 0.002 * (rng.exponential(1.0, n) - 1.0)),
               f"flag_{SIZE[2:]}": np.zeros(n, dtype=int)})
    s = select_locus(t, order=0).summary
    assert s["bright_limit"] is None and s["n_saturated"] == 0
    assert s["bright_limit_source"] == "none"


@pytest.mark.parametrize("order", [0, 1, 2])
def test_tight_locus_with_outliers(order):
    # A very tight locus (0.3 %) and a few objects 25–50 % larger: a least-squares fit over
    # everything is dragged ≈ 3 % off the stars, i.e. outside a ±0.9 % band. The clipping is
    # centred on the median residual, so the centre still lands on the stars.
    rng = np.random.default_rng(9)
    n_star, n_out = 60, 6
    n = n_star + n_out
    T = np.concatenate([2.14 * (1 + 0.003 * rng.normal(size=n_star)),
                        2.14 * rng.uniform(1.25, 1.5, n_out)])
    t = Table({"id": np.arange(1, n + 1), "x": rng.uniform(25, 675, n),
               "y": rng.uniform(25, 675, n), "mag": rng.uniform(17, 20, n), SIZE: T,
               f"flag_{SIZE[2:]}": np.zeros(n, dtype=int)})
    res = select_locus(t, order=order)
    r = res.table
    np.testing.assert_allclose(np.median(r["size_ref"][:n_star]), 2.14, rtol=2e-3)
    assert np.all(r["reason"][n_star:] == "off_locus")
    assert np.sum(r["reason"][:n_star] == "selected") >= n_star - 2


def test_no_locus_raises():
    # Sizes with no common value (uniform over a factor 3) and a band far too narrow (k = 0.01):
    # clipping keeps (almost) no star. That must be a clear error, never a fit of an empty set
    # (which gave a zero centre).
    rng = np.random.default_rng(10)
    n = 40
    t = Table({"id": np.arange(1, n + 1), "x": rng.uniform(1, 4088, n),
               "y": rng.uniform(1, 4088, n), "mag": rng.uniform(17, 20, n),
               SIZE: rng.uniform(1, 3, n), f"flag_{SIZE[2:]}": np.zeros(n, dtype=int)})
    with pytest.raises(ValueError, match="no clear locus"):
        select_locus(t, order=0, k=0.01)


def test_turn_covering_most_stars_is_not_applied():
    # 70 % of the stars are bright and saturated: a scan "turn" would remove most of the
    # catalog, so it is reported but not applied.
    rng = np.random.default_rng(8)
    n_bright, n_faint = 490, 210
    mag = np.concatenate([rng.uniform(17, 18, n_bright), rng.uniform(18.5, 23, n_faint)])
    n = len(mag)
    T = 2.0 * (1 + 0.01 * rng.normal(size=n)) * np.where(mag < 18.6, 1 + 0.15 * (18.6 - mag), 1)
    t = Table({"id": np.arange(1, n + 1), "x": rng.uniform(1, 4088, n),
               "y": rng.uniform(1, 4088, n), "mag": mag, SIZE: T,
               f"flag_{SIZE[2:]}": np.zeros(n, dtype=int)})
    with pytest.warns(UserWarning, match="not applied"):
        s = select_locus(t, order=0).summary
    assert s["bright_limit_source"] == "unclear" and s["bright_limit"] is None
    assert s["detected_limit"] is not None and s["n_saturated"] == 0


def test_bright_limit_and_faint_limit_by_hand():
    t, _, _ = locus_table()
    res = select_locus(t, bright_limit=19.0, mag_max=22.0)
    s = res.summary
    assert s["bright_limit"] == 19.0 and s["bright_limit_source"] == "given"
    r = res.table
    assert np.all(r["reason"][r["mag"] < 19.0] != "selected")
    assert np.all((r["reason"] == "faint") == ((r["mag"] > 22.0) & (r["reason"] != "flagged")))


def test_field_dependent_centre_keeps_edges_fair():
    # PSF size ±6 % across the SCA: a constant centre with a tight band cuts the edges, a
    # plane does not.
    t, _, _ = locus_table(n=3000, sat_mag=None, slope=0.06, n_out=0, n_flag=0, seed=4)
    xn = 2 * (np.asarray(t["x"]) - 1) / 4087 - 1
    edge, centre = np.abs(xn) > 0.8, np.abs(xn) < 0.2

    def kept(order):
        sel = select_locus(t, order=order, k=1.0).table["reason"] == "selected"
        return sel[edge].mean(), sel[centre].mean()

    e0, c0 = kept(0)
    e1, c1 = kept(1)
    assert e0 < c0 - 0.3
    assert abs(e1 - c1) < 0.1


def test_bad_inputs():
    t, _, _ = locus_table()
    with pytest.raises(ValueError, match="order"):
        select_locus(t, order=3)
    with pytest.raises(ValueError, match="not in the table"):
        select_locus(t, size="T_gauss")


def test_catalog_locus_selection_end_to_end(tmp_path):
    pytest.importorskip("ngmix")
    from l2files import field, open_l2, write_field
    from psf_fomo.datafit import DataFit
    from psf_fomo.piffval import FitConfig

    objects = field(n_stars=60, n_gals=6, seed=31)
    image, catalog = write_field(tmp_path, objects, seed=32)
    img = open_l2(image)
    cat = StarCatalog(catalog)
    cat.set_cuts(clean_only=False)
    t = cat.measure(img, algorithms=("wmom",))
    res = cat.select_locus(t, order=0)
    is_star = {o["id"]: o["is_star"] for o in objects}
    chosen = set(res.selected_ids)
    assert chosen and all(is_star[i] for i in chosen)                  # no galaxy kept
    galaxies = {o["id"] for o in objects if not o["is_star"]}
    assert all(r["reason"] == "off_locus" for r in res.table if r["id"] in galaxies)

    sel = cat.piff_select()
    assert sel["type"] == "Flag" and sel["flag_name"] == "psf_locus" and sel["use_flag"] == 1
    assert set(cat.piff_table()["id"][cat.piff_table()["psf_locus"] == 1]) == chosen
    preview = cat.measure(img, algorithms=(), preview_selection=True)
    assert set(preview["id"][preview["candidate"]]) == chosen

    again = StarCatalog.from_config(cat.config())
    assert list(again.table["psf_locus"]) == list(cat.table["psf_locus"])
    assert again.selection["locus"]["summary"] == res.summary

    model = FitConfig.from_piff_config({"type": "Simple", "model": {"type": "Gaussian"},
                                        "interp": {"type": "Mean"}, "max_iter": 3})
    fit = DataFit(img, cat, model)
    fit.set_diagnostics(moments=False)
    out = fit.run(save_to=tmp_path / "fit")
    assert set(out.selection["id"][out.selection["candidate"]]) == chosen
    assert out.config["catalog"]["selection"]["method"] == "locus"
