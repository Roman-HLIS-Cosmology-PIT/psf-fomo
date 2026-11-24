"""Piff's rejection cuts in simulation fits (min_sep per observation) and the shared builder."""
import numpy as np
import galsim
import piff
import pytest
from astropy.table import Table

from psf_fomo.piff_simfit import PiffFitConfig, PiffSimFit
from psf_fomo.piff_simfit.fitting import apply_cuts, make_cuts, select_config
from psf_fomo.piff_simfit.run import cut_simulation_stars
from psf_fomo.sim import RomanSimulation

PIX = 0.11
PSF = galsim.Gaussian(sigma=0.15)
# Two stars 10 pixels (1.1″) apart, one far away.
POSITIONS = [(1000.3, 1000.2), (1010.1, 1000.4), (2500.6, 2500.5)]


def star_at(x, y, star_id, obs, flux=2e4, stamp=25):
    wcs = galsim.PixelScale(PIX)
    ix, iy = int(round(x)), int(round(y))
    bounds = galsim.BoundsI(ix - stamp // 2, ix + stamp // 2, iy - stamp // 2, iy + stamp // 2)
    image = galsim.ImageF(bounds, wcs=wcs)
    (PSF * flux).drawImage(image, center=galsim.PositionD(x, y), method="auto")
    weight = galsim.ImageF(bounds, wcs=wcs, init_value=1.0)
    data = piff.StarData(image, galsim.PositionD(x, y), weight=weight,
                         properties={"sca": 1, "chipnum": 1, "star_id": star_id, "obs": obs})
    return piff.Star(data, None)


class FakeSimulation:
    """`Simulation.stars` only: the same three positions in every observation."""

    def stars(self, sca, obs):
        stars = [star_at(x, y, i, obs) for i, (x, y) in enumerate(POSITIONS)]
        rows = Table({"obs": [obs] * 3, "sca": [sca] * 3, "star_id": [0, 1, 2],
                      "x": [p[0] for p in POSITIONS], "y": [p[1] for p in POSITIONS],
                      "mag": [18.0] * 3, "flux": [1.0] * 3})
        return stars, rows


def test_make_cuts_and_select_config():
    cuts = make_cuts()
    assert cuts["min_sep"] == 2.5 and all(v is None for k, v in cuts.items() if k != "min_sep")
    assert select_config(cuts) == {"min_sep": 2.5}
    assert select_config(make_cuts(min_sep=None)) == {}
    sel = select_config(make_cuts(min_snr=20, max_snr_weight=0, reject_where="mag > 20"),
                        extra_reject=["clean == 0"])
    assert sel == {"min_sep": 2.5, "min_snr": 20.0, "max_snr_weight": 0.0,
                   "reject_where": "clean == 0 or (mag > 20)"}
    with pytest.raises(ValueError):
        make_cuts(min_sep=0)


def test_min_sep_rejects_both_stars_of_a_close_pair():
    stars = [star_at(x, y, i, 0) for i, (x, y) in enumerate(POSITIONS)]
    kept, keep = apply_cuts(stars, make_cuts())
    assert list(keep) == [False, False, True]
    assert [s["star_id"] for s in kept] == [2]
    assert "_cut_index" not in stars[0].data.properties
    kept, keep = apply_cuts(stars, make_cuts(min_sep=1.0))     # 1.1″ apart: both kept
    assert keep.all() and [s["star_id"] for s in kept] == [0, 1, 2]


def test_cuts_apply_per_observation_and_reserve_among_kept():
    # The same positions in two observations: stars of another exposure are not neighbors.
    stars, rows, rejected = cut_simulation_stars(FakeSimulation(), 1, [0, 1], make_cuts(),
                                                 frac=0.5, seed=1)
    assert list(rows["obs"]) == [0, 1] and list(rows["star_id"]) == [2, 2]
    assert [(s["obs"], s["star_id"]) for s in stars] == [(0, 2), (1, 2)]
    assert sorted(zip(rejected["obs"], rejected["star_id"])) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert set(rejected.colnames) == {"obs", "sca", "star_id", "x", "y", "mag"}
    assert sum(s.is_reserve for s in stars) == 1               # int(0.5 × 2 kept stars)
    with pytest.raises(ValueError, match="rejected every star"):
        cut_simulation_stars(FakeSimulation(), 1, [0], make_cuts(reject_where="star_id >= 0"))


def test_validation_cut_settings():
    sim = RomanSimulation(scas=[1])                 # configuration only, nothing rendered
    pv = PiffSimFit(sim, PiffFitConfig())
    assert pv.cuts["min_sep"] == 2.5 and pv.cuts["max_snr_weight"] == 0.0
    assert pv.config()["cuts"] == pv.cuts
    pv.set_cuts(None)                               # every cut off
    assert select_config(pv.cuts) == {"max_snr_weight": 0.0}
    pv.set_cuts(min_sep=3.0, min_snr=50)
    select = pv.piff_file_config("simdir", 0, "outdir")["select"]
    assert select["min_sep"] == 3.0 and select["min_snr"] == 50.0
