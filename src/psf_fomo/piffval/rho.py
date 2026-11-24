"""ρ statistics (Rowe 2010; Jarvis et al. 2016) with a chosen moment algorithm.

Piff's ``RhoStats`` measures data and model stars with the process-wide shape method (``hsm``
by default), which can fail on undersampled stamps. `MomentRhoStats` overrides the
per-star measurement hook (``Stats._measure_size_shape``) to use one of Piff's algorithms
directly, so the global method is never changed. Everything else is Piff's: model stars drawn
at the fitted flux and center, T = trace in arcsec², reduced shear g, positions (u, v) in
arcsec, stars flagged on either side dropped, TreeCorr ``GGCorrelation`` (needs ``treecorr``:
``pip install -e ".[stats]"``).

    ρ1 = ⟨δg δg⟩, ρ2 = ⟨g δg⟩, ρ3 = ⟨(g δT/T)(g δT/T)⟩, ρ4 = ⟨δg (g δT/T)⟩, ρ5 = ⟨g (g δT/T)⟩,

with δg = g_data − g_model, δT = T_data − T_model. Piff measures data and model stars about the
same fitted position (``star.fit.center``; Piff ≥ 280064f), which wmom needs since it has
no centroid fit. One SCA spans
≈ 7.5′, so the default separations are 0.1′ … 10′. τ statistics need a galaxy catalog and
are not included.
"""
import numpy as np
import piff
from astropy.table import Table

from ..moments.measure import ALGORITHMS, DEFAULT_WMOM_FWHM, algorithm_label

RHO_NAMES = ("rho1", "rho2", "rho3", "rho4", "rho5")


class MomentRhoStats(piff.RhoStats):
    """Piff's ``RhoStats`` measuring with a fixed algorithm instead of the global method.

    Parameters
    ----------
    algorithm : {'hsm', 'am', 'gauss', 'wmom'}
        Measurement algorithm. Default ``'wmom'``.
    weight_fwhm : float
        wmom weight FWHM, arcsec. Default 0.4.
    **kwargs
        Passed to ``piff.RhoStats`` (``min_sep``, ``max_sep`` [arcmin], ``bin_size``, …).
    """

    _type_name = None  # not a config type; avoids Piff's duplicate-registration check

    def __init__(self, algorithm="wmom", weight_fwhm=DEFAULT_WMOM_FWHM, **kwargs):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.weight_fwhm = float(weight_fwhm)
        self.n_used = 0
        self.n_input = 0

    def _measure_size_shape(self, star):
        if self.algorithm == "wmom":
            return star.wmom(self.weight_fwhm)
        return getattr(star, self.algorithm)

    def measureShapes(self, psf, stars, model_properties=None, fourth_order=False,
                      raw_moments=False, include_flagged=False, logger=None):
        # Keywords: Piff has inserted arguments before ``logger`` in the past.
        positions, data, model = super().measureShapes(
            psf, stars, model_properties=model_properties, fourth_order=fourth_order,
            raw_moments=raw_moments, include_flagged=include_flagged, logger=logger)
        # Record how many stars survive Piff's flag cuts (flagged stars are dropped before
        # measuring, failed measurements in compute()).
        self.n_input = len(stars)
        self.n_used = int(np.sum((data[:, 6] == 0) & (model[:, 6] == 0)))
        return positions, data, model


def rho_stats(psf, stars, algorithms=("wmom",), wmom_fwhm=(DEFAULT_WMOM_FWHM,),
              min_sep=0.1, max_sep=10.0, bin_size=0.5):
    """ρ1 … ρ5 of a fitted PSF on a set of stars, for one or more algorithms.

    Parameters
    ----------
    psf : piff.PSF
        Fitted PSF.
    stars : list of piff.Star
        Stars with data images (e.g. the reserve stars of a fit).
    algorithms : sequence of str
        Piff algorithms (``'hsm'``, ``'am'``, ``'gauss'``, ``'wmom'``). Default ('wmom',).
    wmom_fwhm : float or sequence of float
        wmom weight FWHM(s), arcsec. Default (0.4,).
    min_sep, max_sep : float
        Separation range, arcmin. Default 0.1, 10.
    bin_size : float
        Bin width in ln(r). Default 0.5.

    Returns
    -------
    astropy.table.Table
        One row per algorithm × statistic × separation bin: ``algorithm``, ``rho``
        (``'rho1'`` …), ``r`` (mean separation, arcmin), ``xip``, ``xip_err``, ``xim``,
        ``npairs``, ``n_used`` (stars with good data and model measurements), ``n_input``.
    """
    rows = []
    fwhms = [float(f) for f in np.atleast_1d(wmom_fwhm)]
    for alg in algorithms:
        for fwhm in (fwhms if alg == "wmom" else [None]):
            stats = MomentRhoStats(alg, fwhm if fwhm is not None else DEFAULT_WMOM_FWHM,
                                   min_sep=min_sep, max_sep=max_sep, bin_size=bin_size)
            stats.compute(psf, stars)
            label = algorithm_label(alg, fwhm)
            if stats.skip:  # every star flagged
                continue
            for name in RHO_NAMES:
                corr = getattr(stats, name)
                for k in range(len(corr.meanr)):
                    rows.append({"algorithm": label, "rho": name,
                                 "r": float(corr.meanr[k]), "xip": float(corr.xip[k]),
                                 "xip_err": float(np.sqrt(corr.varxip[k])),
                                 "xim": float(corr.xim[k]), "npairs": float(corr.npairs[k]),
                                 "n_used": stats.n_used, "n_input": stats.n_input})
    return Table(rows=rows)
