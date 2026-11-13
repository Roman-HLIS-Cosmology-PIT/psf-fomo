"""Running Piff fits on a simulation (`PiffValidation`), results (`FitResult`), `load_fit`.

Each fit covers one SCA; observations of that SCA are pooled (they share the WCS in the
current simulations). Stars come from `Simulation.stars`: ``stamp_size`` stamps, weight
1 / ``bkg_var``, properties ``sca`` and ``chipnum`` = SCA.

Example
-------
>>> from psf_fomo.sim import RomanSimulation
>>> from psf_fomo.piffval import FitConfig, PiffValidation
>>> sim = RomanSimulation(n_stars=30, seed=1).run()
>>> cfg = FitConfig(); cfg.set_optics("pinned")
>>> pv = PiffValidation(sim, cfg, sca=1)
>>> pv.set_reserve(0.2, seed=3)
>>> res = pv.run(save_to="out/fit01")
>>> res.fit_summary["chisq_dof"], res.summary(), res.moment_summary()
"""
import copy
import time
from pathlib import Path

import numpy as np
import piff
import piff.config
from astropy.table import Table, vstack

from .. import models  # noqa: F401  (registers our Piff models before any config is parsed)
from ..io import provenance, read_yaml, write_yaml
from ..moments.measure import DEFAULT_WMOM_FWHM
from ..sim import load_simulation
from ..sim import noise as noise_mod
from ..sim import storage
from . import components as components_mod
from . import diagnostics, plotting
from .config import FitConfig
# Shared with real-data fits; imported here so `psf_fomo.piffval.run.<name>` keeps working.
from .fitting import NOT_CONVERGED, choose_reserve, make_logger, piff_shape_method  # noqa: F401
from .residuals import (COMPARISONS, MODEL_OS_SIZE, model_moments, moment_residuals,
                        moment_summary, sca_grid)

DEFAULT_DIAGNOSTICS = {"moments": True, "native": ["wmom", "gauss"],
                       "wmom_fwhm": [DEFAULT_WMOM_FWHM], "oversampled": ["hsm"],
                       "oversample": 5, "include_pixel": True,
                       "comparisons": list(COMPARISONS)}


class PiffValidation:
    """Builder for a Piff fit of one SCA of a simulation.

    Parameters
    ----------
    simulation : Simulation or str or Path
        A `Simulation`, or the folder of a saved one (then its path is recorded in the
        outputs, and `load_fit` can reload it).
    fit_config : FitConfig
        Model and fit settings.
    sca : int
        SCA to fit. Default: the simulation's first SCA.
    obs : int, sequence of int, or None
        Observations to pool. Default None = all.
    name : str, optional
        Label stored with the result.
    """

    def __init__(self, simulation, fit_config, sca=None, obs=None, name=None):
        self.sim_path = None
        if isinstance(simulation, (str, Path)):
            self.sim_path = str(simulation)
            simulation = load_simulation(simulation)
        self.sim = simulation
        if not isinstance(fit_config, FitConfig):
            raise TypeError("fit_config must be a FitConfig")
        self.fit_config = fit_config
        scas = self.sim.config["scas"]
        self.sca = scas[0] if sca is None else int(sca)
        if self.sca not in scas:
            raise ValueError(f"SCA {self.sca} is not in the simulation (SCAs {scas})")
        n_obs = self.sim.config["n_obs"]
        self.obs = list(range(n_obs)) if obs is None else [int(o) for o in np.atleast_1d(obs)]
        if any(not 0 <= o < n_obs for o in self.obs):
            raise ValueError(f"observations must be in 0..{n_obs - 1}")
        self.name = name
        self.reserve = {"frac": 0.0, "seed": 0}
        self.diagnostics = copy.deepcopy(DEFAULT_DIAGNOSTICS)
        self.verbose = 0

    def set_reserve(self, frac=0.2, seed=0):
        """Hold out a fraction of the stars from the fit (Piff's ``is_reserve``).

        As Piff's ``select.reserve_frac``: ``int(frac × N)`` stars (rounded down), chosen at
        random without replacement. Reserve stars are not used to fit the model, but are
        interpolated, refluxed and diagnosed like the others.

        Parameters
        ----------
        frac : float
            Fraction in [0, 1). Default 0.2.
        seed : int
            Seed of the choice. Default 0.
        """
        if not 0 <= frac < 1:
            raise ValueError("frac must be in [0, 1)")
        self.reserve = {"frac": float(frac), "seed": int(seed)}

    def set_diagnostics(self, moments=True, native=("wmom", "gauss"),
                        wmom_fwhm=(DEFAULT_WMOM_FWHM,), oversampled=("hsm",), oversample=5,
                        include_pixel=True, comparisons=COMPARISONS):
        """Choose the moment-residual diagnostics computed by `run`.

        Parameters
        ----------
        moments : bool
            Compute moment residuals at all. Default True (costs two truth renderings and
            three model drawings per star).
        native : sequence of str
            Algorithms (``'hsm'``, ``'am'``, ``'gauss'``, ``'wmom'``) for the native-pixel
            comparisons. Default ('wmom', 'gauss').
        wmom_fwhm : float or sequence of float
            wmom weight FWHM(s) in arcsec. Default (0.4,).
        oversampled : sequence of str
            Algorithms for the oversampled model − truth comparison. Default ('hsm',).
        oversample : int
            Refinement of the oversampled grid. Default 5.
        include_pixel : bool
            Oversampled images are the pixel-convolved effective PSF. Default True.
        comparisons : sequence of str
            Subset of ``('data', 'truth_native', 'truth_os', 'model_os')``. Default the
            first three; ``'model_os'`` (the model alone, oversampled) is opt-in.

        See `psf_fomo.piffval.residuals` for the definitions.
        """
        self.diagnostics = {"moments": bool(moments), "native": list(native),
                            "wmom_fwhm": [float(f) for f in np.atleast_1d(wmom_fwhm)],
                            "oversampled": list(oversampled), "oversample": int(oversample),
                            "include_pixel": bool(include_pixel),
                            "comparisons": list(comparisons)}

    def set_verbose(self, verbose=1):
        """Piff log level: 0 errors only (default), 1 info, 2 verbose, 3 debug."""
        if verbose not in (0, 1, 2, 3):
            raise ValueError("verbose must be 0, 1, 2 or 3")
        self.verbose = verbose

    def config(self):
        """Settings of this validation run as a plain dict (saved with the result)."""
        return {"name": self.name, "simulation": self.sim_path, "sca": self.sca,
                "obs": self.obs, "reserve": dict(self.reserve),
                "diagnostics": copy.deepcopy(self.diagnostics),
                "fit": copy.deepcopy(self.fit_config.config),
                "piff_psf": self.fit_config.piff_config(self.sim.config["filter"])}

    def stars(self):
        """Piff stars of the selected SCA and observations, reserve marked, and their rows."""
        return simulation_stars(self.sim, self.sca, self.obs, **self.reserve)

    def _make_psf(self, logger):
        truth = self.sim.truth
        piff_dict = self.fit_config.piff_config(self.sim.config["filter"])
        psf = piff.PSF.process(copy.deepcopy(piff_dict), logger=logger)
        psf.set_context({self.sca: truth.wcs[self.sca]}, truth.pointing, truth.bandpass)
        return psf, piff_dict

    def initialize(self):
        """Build the Piff model and initialise it on the stars, without fitting.

        Runs Piff's own ``initialize_flux_center`` and ``initialize_params`` (with the
        configured shape method), as at the start of ``psf.fit``; used by the reference solver.

        Returns
        -------
        psf : piff.PSF
            The initialised model (``psf.stars`` set).
        stars : list of piff.Star
            Initialised stars (fit parameters set, reserve marked).
        rows : astropy.table.Table
            Their simulation catalog rows.
        """
        stars, rows = self.stars()
        logger, _ = make_logger(self.verbose)
        psf, _ = self._make_psf(logger)
        shape = self.fit_config.config["shape_method"]
        with piff_shape_method(shape["method"], shape["weight_fwhm"]):
            stars = psf.initialize_flux_center(stars, logger=logger)
            stars, _ = psf.initialize_params(stars, logger=logger)
        psf.stars = stars
        return psf, stars, rows

    def piff_file_config(self, sim_dir, obs, out_dir):
        """Piff config (input, select, psf, output) for a file-based run of one observation.

        Parameters
        ----------
        sim_dir : str or Path
            Folder of the saved simulation.
        obs : int
            Observation to fit.
        out_dir : str or Path
            Folder for Piff's output file.

        Returns
        -------
        dict
            Usable with ``piff.config.process`` or the ``piffify`` command (the
            ``modules`` entry imports `psf_fomo.models`).

        Notes
        -----
        - Image HDU = position of the SCA in the simulation's SCA list (+1). Catalog: with the
          simple WCS the per-SCA HDU and ``x``/``y``; with the Roman WCS the full table and
          ``ra``/``dec`` (Piff drops off-image stars). ``properties`` set ``sca`` and ``obs``;
          ``property_cols`` carries ``star_id``; ``chipnum`` = SCA as in memory.
        - Weight = 1 / ``BKGVAR`` (header: the SCA's median background variance; in memory
          each star uses its own ``bkg_var`` — identical for Gaussian noise). ``gain`` (header
          ``GAIN``) only when star shot noise is simulated, as in memory.
        - ``select.max_snr_weight: 0`` turns off Piff's default down-weighting of stars with
          S/N > 100, so both modes fit the same χ². Reserve: Piff's ``reserve_frac`` + ``seed``
          (a different random choice than in memory).
        - Pointing for the Roman WCS: the simulation's (``ra`` given to Piff in hours).
        """
        sim = self.sim
        scas = sim.config["scas"]
        inp = {"dir": str(sim_dir),
               "image_file_name": storage.image_file_name(obs),
               "image_hdu": scas.index(self.sca) + 1,
               "cat_file_name": storage.catalog_file_name(obs),
               "chipnum": self.sca,
               "bandpass": {"type": "RomanBandpass", "name": sim.config["filter"]},
               "noise": "BKGVAR",
               "stamp_size": sim.config["stamp_size"],
               "properties": {"sca": self.sca, "obs": int(obs)},
               "property_cols": ["star_id"]}
        wcs = sim.config["wcs"]
        if wcs["kind"] == "simple":
            # HDU 0 primary, 1 STARS, then one HDU per SCA in sorted order (sim.storage).
            inp.update(cat_hdu=2 + sorted(scas).index(self.sca), x_col="x", y_col="y")
        else:
            inp.update(cat_hdu=1, ra_col="ra", dec_col="dec", ra_units="deg",
                       dec_units="deg", ra=wcs["ra"] / 15.0, dec=wcs["dec"])
        if noise_mod.star_gain(sim.config["noise"]) is not None:
            inp["gain"] = "GAIN"
        shape = self.fit_config.config["shape_method"]
        config = {"modules": ["psf_fomo.models"], "input": inp,
                  "select": {"max_snr_weight": 0.0, "reserve_frac": self.reserve["frac"],
                             "seed": self.reserve["seed"]},
                  "psf": self.fit_config.piff_config(sim.config["filter"]),
                  "output": {"dir": str(out_dir), "file_name": "psf.fits"},
                  "shape_method": shape["method"]}
        if shape["weight_fwhm"] is not None:
            config["shape_weight_fwhm"] = shape["weight_fwhm"]
        return config

    def run(self, save_to=None, mode="memory"):
        """Fit the PSF and compute the diagnostics.

        Parameters
        ----------
        save_to : str or Path, optional
            Folder for the outputs (see `FitResult.save`). Default None; required for
            ``mode='files'``.
        mode : {'memory', 'files'}
            ``'memory'``: stars cut from the simulation in memory. ``'files'``: through Piff's
            own file input, from the saved simulation (see `piff_file_config`); one
            observation only. The simulation is saved to ``save_to/simulation`` if it was not
            given as a folder; the Piff config is written to ``save_to/piff.yaml``.

        Returns
        -------
        FitResult
        """
        if mode not in ("memory", "files"):
            raise ValueError("mode must be 'memory' or 'files'")
        logger, watch = make_logger(self.verbose)
        shape = self.fit_config.config["shape_method"]
        piff_dict = self.fit_config.piff_config(self.sim.config["filter"])
        t0 = time.time()
        if mode == "memory":
            stars, rows = self.stars()
            psf, _ = self._make_psf(logger)
            with piff_shape_method(shape["method"], shape["weight_fwhm"]):
                psf.fit(stars, logger=logger)
        else:
            psf, rows = self._run_files(save_to, logger)
        seconds = time.time() - t0
        table = diagnostics.star_table(psf, psf.stars, rows)
        max_iter = piff_dict.get("max_iter", 30)
        summary = diagnostics.fit_summary(psf, not watch.triggered, max_iter, table, seconds)
        config = self.config()
        config["mode"] = mode
        result = FitResult(psf, table, summary, config, provenance(), sim=self.sim,
                           stars=psf.stars, rows=rows)
        if self.diagnostics["moments"]:
            settings = {k: v for k, v in self.diagnostics.items() if k != "moments"}
            result.compute_moments(**settings)
        if save_to is not None:
            result.save(save_to)
        return result

    def _run_files(self, save_to, logger):
        """File-based fit: write the Piff config, run Piff's input/select/fit, match rows."""
        if save_to is None:
            raise ValueError("mode='files' needs save_to (Piff reads and writes files)")
        if len(self.obs) != 1:
            raise ValueError("mode='files' fits one observation: give obs=<index>")
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        if self.sim_path is None:
            self.sim_path = str(save_to / "simulation")
            self.sim.save(self.sim_path)
        obs = self.obs[0]
        piff_config = self.piff_file_config(self.sim_path, obs, save_to)
        write_yaml(save_to / "piff.yaml", piff_config)
        shape = self.fit_config.config["shape_method"]
        # piff.config.process sets the global shape method from the config; restore it after.
        with piff_shape_method(shape["method"], shape["weight_fwhm"]):
            psf = piff.config.process(copy.deepcopy(piff_config), logger)
        catalog = self.sim.catalog(obs, self.sca)
        index = {int(r["star_id"]): i for i, r in enumerate(catalog)}
        rows = catalog[[index[int(s.data.properties["star_id"])] for s in psf.stars]]
        return psf, rows


class FitResult:
    """Result of `PiffValidation.run` (or `load_fit`).

    Attributes
    ----------
    psf : piff.PSF
        The fitted PSF.
    table : astropy.table.Table
        One row per star, see `diagnostics.star_table`.
    fit_summary : dict
        Whole-fit numbers, see `diagnostics.fit_summary`. ``converged`` is False when Piff
        stopped at ``max_iter``.
    moments : astropy.table.Table or None
        Moment residuals (see `psf_fomo.piffval.residuals`), one row per star × comparison ×
        algorithm; None if not computed.
    config : dict
        Settings of the run (`PiffValidation.config`), incl. the Piff ``psf`` dict.
    provenance : dict
        Software versions and creation time.
    sim : Simulation or None
        The simulation, if available (after `load_fit` only when its folder was recorded,
        or when passed to `load_fit`). Needed for moments, components and image plots.
    """

    def __init__(self, psf, table, fit_summary, config, provenance, sim=None, stars=None,
                 rows=None, moments=None):
        self.psf = psf
        self.table = table
        self.fit_summary = fit_summary
        self.config = config
        self.provenance = provenance
        self.sim = sim
        self.moments = moments
        self._stars = stars
        self._rows = rows

    def fitted_stars(self):
        """Fitted stars with their data images, and the matching simulation catalog rows.

        After `load_fit` the images are re-cut from the simulation and the saved fits
        (parameters, flux, centre, reserve and flag status) are attached.
        """
        if self._stars is None:
            if self.sim is None:
                raise ValueError("needs the simulation: load_fit(path, sim=...)")
            data, rows = simulation_stars(self.sim, self.config["sca"], self.config["obs"])
            index = {(int(r["obs"]), int(r["star_id"])): i for i, r in enumerate(rows)}
            stars, keep = [], []
            for s in self.psf.stars:  # matched by (obs, star_id): Piff's input may drop stars
                i = index[int(s.data.properties["obs"]), int(s.data.properties["star_id"])]
                d = data[i].data
                d.properties["is_reserve"] = bool(s.is_reserve)
                d.properties["is_flagged"] = bool(s.is_flagged)
                stars.append(piff.Star(d, s.fit))
                keep.append(i)
            self._stars, self._rows = stars, rows[keep]
        return self._stars, self._rows

    def selected_stars(self, which="fit"):
        """Unflagged ``'fit'``, ``'reserve'`` or ``'all'`` stars and their catalog rows."""
        if which not in ("fit", "reserve", "all"):
            raise ValueError("which must be 'fit', 'reserve' or 'all'")
        stars, rows = self.fitted_stars()
        keep = [not s.is_flagged and (which == "all" or s.is_reserve == (which == "reserve"))
                for s in stars]
        return [s for s, k in zip(stars, keep) if k], rows[np.array(keep, dtype=bool)]

    def summary(self, columns=("flux_r", "flux_tot_r", "dx", "dy", "chisq_dof")):
        """Mean, std, error on the mean, n and n_flagged, separately for fit and reserve."""
        return diagnostics.summary(self.table, columns)

    def compute_moments(self, **settings):
        """(Re)compute ``self.moments``; keyword arguments as `PiffValidation.set_diagnostics`
        (without ``moments``). Needs the simulation."""
        if self.sim is None:
            raise ValueError("needs the simulation: load_fit(path, sim=...)")
        stars, _ = self.fitted_stars()
        self.moments = moment_residuals(self.psf, stars, self.table, self.sim, **settings)
        return self.moments

    def model_moments(self, n_grid=5, x=None, y=None, algorithms=("hsm",),
                      wmom_fwhm=(DEFAULT_WMOM_FWHM,), oversample=5, include_pixel=True,
                      size=MODEL_OS_SIZE):
        """Oversampled model moments across the SCA (see `residuals.model_moments`).

        Positions: ``x``, ``y`` (native pixels) if given, else the centres of an
        ``n_grid`` × ``n_grid`` grid of cells covering the SCA. Needs no simulation.
        """
        if x is None or y is None:
            x, y = sca_grid(n_grid)
        return model_moments(self.psf, x, y, self.config["sca"], algorithms, wmom_fwhm,
                             oversample, include_pixel, size)

    def moment_summary(self, columns=("dT_T", "de1", "de2")):
        """Mean, std, error on the mean, n, n_flagged per (set, comparison, algorithm)."""
        if self.moments is None:
            raise ValueError("no moments computed (see compute_moments)")
        return moment_summary(self.moments, columns)

    def components(self):
        """Component summaries vs truth (see `psf_fomo.piffval.components.components`)."""
        stars, rows = self.fitted_stars()
        return components_mod.components(self.psf, stars, rows)

    def reference(self, freeze=(), starts=("given",), max_nfev=50, moments="same",
                  report="all", given=True):
        """Joint χ² optimum of this fit's model (see `psf_fomo.reference.reference_from_fit`).

        Returns a dict with ``'given'`` (Piff's solution with true centres and profiled
        fluxes, on the reference's χ² and dof; only if ``given``) and ``'optimum'``
        ({start: ReferenceResult}). ``report='fit'`` reports only the fit stars.
        """
        from ..reference import reference_from_fit  # the reference package imports piffval
        return reference_from_fit(self, freeze=freeze, starts=starts, max_nfev=max_nfev,
                                  moments=moments, report=report, given=given)

    def rho(self, algorithms=("wmom",), wmom_fwhm=(DEFAULT_WMOM_FWHM,), which="all",
            min_sep=0.1, max_sep=10.0, bin_size=0.5):
        """ρ1 … ρ5 (see `psf_fomo.piffval.rho.rho_stats`) for the selected stars.

        ``which`` = ``'all'`` (default), ``'fit'`` or ``'reserve'``; Piff-flagged stars are
        left out. Separations in arcmin. Needs ``treecorr``.
        """
        from .rho import rho_stats  # needs treecorr only when called
        stars, _ = self.selected_stars(which)
        return rho_stats(self.psf, stars, algorithms, wmom_fwhm, min_sep, max_sep, bin_size)

    def plot_rho(self, rho_table=None, **kwargs):
        """Plot ρ1 … ρ5; computes them with ``self.rho(**kwargs)`` if no table is given."""
        return plotting.plot_rho(self.rho(**kwargs) if rho_table is None else rho_table)

    def plot_mean_images(self, which="fit", oversampled=False, oversample=5):
        """See `plotting.plot_mean_images`."""
        return plotting.plot_mean_images(self, which, oversampled, oversample)

    def plot_chi2_radius(self, which="fit", bin_width=1.0):
        """See `plotting.plot_chi2_radius`."""
        return plotting.plot_chi2_radius(self, which, bin_width)

    def plot_moment_residuals(self, comparison="truth_os", algorithms=None):
        """See `plotting.plot_moment_residuals`."""
        return plotting.plot_moment_residuals(self, comparison, algorithms)

    def plot_field(self, column, comparison=None, algorithm=None, which="all"):
        """See `plotting.plot_field`."""
        return plotting.plot_field(self, column, comparison, algorithm, which)

    def plot_components(self, kind="pixelgrid", n_grid=3, oversample=5, size=9):
        """See `plotting.plot_components`."""
        return plotting.plot_components(self, kind, n_grid, oversample, size)

    def plot_zernikes(self):
        """See `plotting.plot_zernikes`."""
        return plotting.plot_zernikes(self)

    def save(self, path):
        """Save to a folder: ``psf.fits`` (Piff), ``stars.fits``, ``moments.fits`` (if
        computed), ``config.yaml`` (run settings, fit summary, provenance)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.psf.write(str(path / "psf.fits"))
        self.table.write(path / "stars.fits", overwrite=True)
        if self.moments is not None:
            self.moments.write(path / "moments.fits", overwrite=True)
        write_yaml(path / "config.yaml", {"config": self.config,
                                          "fit_summary": self.fit_summary,
                                          "provenance": self.provenance})


def load_fit(path, sim=None, load_sim=True):
    """Load a `FitResult` saved with `FitResult.save` (or ``run(save_to=...)``).

    Parameters
    ----------
    path : str or Path
        Folder of the saved fit.
    sim : Simulation, optional
        The simulation the fit was made from (needed for images, moments and components
        when it was not given as a folder).
    load_sim : bool
        If ``sim`` is not given, load it from the recorded folder (if any). Default True.

    Returns
    -------
    FitResult
    """
    path = Path(path)
    saved = read_yaml(path / "config.yaml")
    psf = piff.read(str(path / "psf.fits"))
    table = Table.read(path / "stars.fits")
    moments = Table.read(path / "moments.fits") if (path / "moments.fits").exists() else None
    sim_path = saved["config"].get("simulation")
    if sim is None and load_sim and sim_path:
        sim = load_simulation(sim_path)
    return FitResult(psf, table, saved["fit_summary"], saved["config"], saved["provenance"],
                     sim=sim, moments=moments)


def simulation_stars(sim, sca, obs, frac=0.0, seed=0):
    """Stars of one SCA and the given observations, in a fixed order, reserve marked.

    Returns (stars, rows); see `PiffValidation.set_reserve` for the reserve rule.
    """
    stars, rows = [], []
    for o in obs:
        s, r = sim.stars(sca=sca, obs=o)
        stars += s
        rows.append(r)
    rows = vstack(rows)
    for star, flag in zip(stars, choose_reserve(len(stars), frac, seed)):
        star.data.properties["is_reserve"] = bool(flag)
    return stars, rows
