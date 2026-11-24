"""Piff fits of one real L2 image (`PiffDataFit`), results (`PiffDataFitResult`), `load_data_fit`.

One exposure of one SCA per fit. The run follows ``piff.config.process`` step by step (Piff's
own Input → Select → PSF → fit), so that the star selection and the fit can use different
shape methods (SizeMag needs ``gauss`` on native stamps) and the outcome of each selection
stage is kept:

1. checks of the image (`L2Image.check_weights`, ``check_orientation``, ``check_finite``);
2. the Piff catalog (`StarCatalog.piff_table`, 1-based positions, per-star gain) is written
   to ``save_to/catalog.fits`` and the full Piff config to ``save_to/piff.yaml``;
3. Piff's Input reads every detection; Piff's Select (selection type, then the cuts, then
   the reserve) picks the stars; Piff fits them.

``piff.yaml`` also runs with Piff's ``piffify`` (one shape method for selection and fit there).

Telescope attitude (`PiffDataFit.set_attitude`, experimental, off by default): with a RomanOptics
model and an image whose pointing is known (`L2Image.attitude`), the boresight and V3
position angle are written into the RomanOptics config (``boresight_ra``, ``boresight_dec``,
``telescope_pa``, degrees), so GalSim orients the optical PSF from the telescope attitude
instead of the WCS. This needs a GalSim whose ``roman.getPSF`` takes ``boresight`` /
``telescope_PA`` and a Piff whose RomanOptics takes these options; neither is in a GalSim or
Piff release yet. `attitude_support` reports what the installed versions lack.

Example
-------
>>> from psf_fomo.piff_simfit import PiffFitConfig
>>> from psf_fomo.piff_datafit import L2Image, StarCatalog, PiffDataFit
>>> img = L2Image("exp_wfi05.fits")
>>> cat = StarCatalog("exp_wfi05_det.fits"); cat.set_cuts(min_snr=50, min_sep=2.0)
>>> cfg = PiffFitConfig(); cfg.set_optics("free"); cfg.add_kernel("signed")
>>> fit = PiffDataFit(img, cat, cfg); fit.set_reserve(0.2, seed=3)
>>> res = fit.run(save_to="out/real01")
>>> res.fit_summary["chisq_dof"], res.summary(), res.moment_summary()
"""
import copy
import inspect
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import galsim
import galsim.roman
import piff
from astropy.table import Table

from .. import models  # noqa: F401  (registers our Piff models before any config is parsed)
from ..io import provenance, read_yaml, write_yaml
from ..moments.measure import DEFAULT_WMOM_FWHM, upgrade_sizes
from ..piff_simfit import plotting
from ..piff_simfit.components import components as component_tables, leaf_psfs
from ..piff_simfit.config import PiffFitConfig, needs_star_seds
from ..sim.storage import (SED_FILE, piff_sed_input, sample_sed, sed_from_setting,
                           sed_setting, write_sed)
from ..piff_simfit.diagnostics import summary as star_summary
from ..piff_simfit.fitting import fit_summary, make_logger, piff_shape_method
from ..piff_simfit.residuals import (MODEL_OS_SIZE, fit_sed, model_moments, moment_residuals,
                                     moment_summary, sca_grid)
from . import diagnostics
from .catalog import PIFF_ID, StarCatalog
from .l2 import FIT_COPY_NAME, L2Image

DATA_COMPARISONS = ("data", "model_os")
DEFAULT_DIAGNOSTICS = {"moments": True, "native": ["wmom", "gauss"],
                       "wmom_fwhm": [DEFAULT_WMOM_FWHM], "oversampled": ["hsm"],
                       "oversample": 5, "include_pixel": True,
                       "comparisons": list(DATA_COMPARISONS)}
SUMMARY_COLUMNS = ("flux_cat_r", "dx", "dy", "chisq_dof", "sky_resid")


class PiffDataFit:
    """Builder for a Piff fit of one real image.

    Parameters
    ----------
    image : L2Image
        The exposure (variant, weights and sky set on it).
    catalog : StarCatalog
        The detections, star selection and cuts.
    fit_config : PiffFitConfig
        Model and fit settings (the same builder as for simulations).
    name : str, optional
        Label stored with the result.
    """

    def __init__(self, image, catalog, fit_config, name=None):
        if not isinstance(image, L2Image):
            raise TypeError("image must be an L2Image")
        if not isinstance(catalog, StarCatalog):
            raise TypeError("catalog must be a StarCatalog")
        if not isinstance(fit_config, PiffFitConfig):
            raise TypeError("fit_config must be a PiffFitConfig")
        self.image, self.catalog, self.fit_config = image, catalog, fit_config
        self.name = name
        self.stamp_size = 25
        self.reserve = {"frac": 0.0, "seed": 0}
        self.diagnostics = copy.deepcopy(DEFAULT_DIAGNOSTICS)
        self.verbose = 0
        self.use_attitude = False
        self.sed = sed_setting("flat")

    def set_sed(self, sed="flat", wave_type="nm", flux_type="fphotons"):
        """The SED given to every star, used by a chromatic RomanOptics model.

        A chromatic RomanOptics (``PiffFitConfig.set_optics(chromatic=True)``; the default is
        monochromatic) needs an SED for each star. Real stars have none in the catalog, so one
        SED is assumed for all of them.

        Parameters
        ----------
        sed : 'flat', str, galsim.SED or None
            ``'flat'`` (default): constant photons per unit wavelength. A file name: read by
            ``galsim.SED`` (and by Piff) with ``wave_type`` and ``flux_type``. A spectral
            ``galsim.SED``: sampled every 0.5 nm over the filter's wavelength range. None: no
            SED (only for a monochromatic model, ``PiffFitConfig.set_optics(chromatic=False)``).
        wave_type, flux_type : str
            Units of an SED file, as for ``galsim.SED``.
        """
        bandpass = galsim.roman.getBandpasses()[self.image.filter_name]
        if sed is not None and not isinstance(sed, galsim.SED) and sed != "flat":
            sed = str(Path(sed).resolve())        # Piff reads it from the fit folder
        self.sed = sed_setting(sed, bandpass, wave_type, flux_type)

    def set_attitude(self, use=False):
        """Whether RomanOptics is oriented by the telescope attitude (see the module docstring).

        Experimental: needs GalSim and Piff changes that are not released yet
        (`attitude_support` lists what the installed versions lack). Off unless set here.

        Parameters
        ----------
        use : {False, 'auto', True}
            False (default): never (GalSim orients the PSF from the WCS). ``'auto'``: use it
            when the model has RomanOptics, the image's attitude is known and the installed
            GalSim and Piff support it; otherwise (warning if only the software is missing) the
            WCS orientation. True: require it (`run` raises if it cannot be used).
        """
        if use != "auto" and not isinstance(use, bool):
            raise ValueError("use must be 'auto', True or False")
        self.use_attitude = use

    def attitude(self, warn=True):
        """The attitude this run passes to RomanOptics, or None (WCS orientation).

        Returns
        -------
        dict or None
            ``boresight_ra``, ``boresight_dec``, ``telescope_pa`` (degrees).

        Raises
        ------
        ValueError, RuntimeError
            With ``set_attitude(True)``: no attitude for the image, or no support in the
            installed GalSim / Piff.
        """
        if self.use_attitude is False or not uses_roman_optics(
                self.fit_config.piff_config(self.image.filter_name)):
            return None
        values = self.image.attitude
        if values is None:
            if self.use_attitude is True:
                raise ValueError(f"set_attitude(True), but {self.image.path.name} has no "
                                 "RA_V1, DEC_V1, PA_V3 in its PRIMARY header; pass "
                                 "L2Image(..., pointing=meta.pointing)")
            return None
        missing = attitude_support()
        if missing:
            message = "orienting RomanOptics by the telescope attitude needs " + " and ".join(
                missing)
            if self.use_attitude is True:
                raise RuntimeError(message)
            if warn:
                warnings.warn(message + "; the PSF is oriented from the WCS instead")
            return None
        return dict(values)

    def set_stamp_size(self, size=25):
        """Stamp side in native pixels (Piff's ``stamp_size``). Default 25."""
        if size < 5:
            raise ValueError("stamp size must be ≥ 5")
        self.stamp_size = int(size)

    def set_reserve(self, frac=0.2, seed=0):
        """Hold out ``int(frac × N)`` selected stars from the fit (Piff's ``reserve_frac``).

        Reserve stars are not fitted but interpolated, refluxed and diagnosed.

        Parameters
        ----------
        frac : float
            Fraction in [0, 1). Default 0.2.
        seed : int
            Seed of Piff's random choice. Default 0.
        """
        if not 0 <= frac < 1:
            raise ValueError("frac must be in [0, 1)")
        self.reserve = {"frac": float(frac), "seed": int(seed)}

    def set_diagnostics(self, moments=True, native=("wmom", "gauss"),
                        wmom_fwhm=(DEFAULT_WMOM_FWHM,), oversampled=("hsm",), oversample=5,
                        include_pixel=True, comparisons=DATA_COMPARISONS):
        """Moment diagnostics computed by `run` (see `psf_fomo.piff_simfit.residuals`).

        Parameters
        ----------
        moments : bool
            Compute them at all. Default True.
        native : sequence of str
            Algorithms for ``'data'`` (data stamp vs model on the same stamp, both about the
            catalog position). Default ('wmom', 'gauss').
        wmom_fwhm : float or sequence of float
            wmom weight FWHM(s), arcsec. Default (0.4,).
        oversampled : sequence of str
            Algorithms for ``'model_os'`` (the model alone, oversampled). Default ('hsm',).
        oversample : int
            Refinement of the oversampled grid. Default 5.
        include_pixel : bool
            Oversampled images are the pixel-convolved effective PSF. Default True.
        comparisons : sequence of str
            Subset of ('data', 'model_os'). Default both.
        """
        bad = set(comparisons) - set(DATA_COMPARISONS)
        if bad:
            raise ValueError(f"real data has no truth: comparisons {sorted(bad)} are not "
                             f"available; use {DATA_COMPARISONS}")
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

    def piff_config(self, catalog_file, piff_catalog, image_file=None):
        """Full Piff config (``modules``, ``input``, ``select``, ``psf``, ``shape_method``).

        Parameters
        ----------
        catalog_file : str or Path
            Where the Piff catalog is (or will be) written.
        piff_catalog : astropy.table.Table
            That catalog (`StarCatalog.piff_table`), for its column names.
        image_file : str or Path, optional
            The image file Piff reads (default the image's own; `run` passes the cleaned
            working copy when there is one).

        Returns
        -------
        dict
        """
        inp = piff_input_config(self.image, self.catalog, catalog_file, piff_catalog,
                                self.stamp_size, image_file)
        select = self.catalog.piff_select()
        select.update(reserve_frac=self.reserve["frac"], seed=self.reserve["seed"])
        shape = self.fit_config.config["shape_method"]
        psf = self.fit_config.piff_config(self.image.filter_name)
        attitude = self.attitude(warn=False)
        if attitude is not None:
            add_attitude(psf, attitude)
        if needs_star_seds(psf):
            if self.sed is None:
                raise ValueError("the Piff model is chromatic (RomanOptics, chromatic=True) "
                                 "and needs an SED for the stars: set_sed('flat' | file | "
                                 "galsim.SED), or PiffFitConfig.set_optics(..., chromatic=False)")
            # 'flat' / 'table': sed.fits, written next to the catalog by `run`.
            inp.update(piff_sed_input(self.sed, user_file=self.sed["sed"]))
        config = {"modules": ["psf_fomo.models"], "input": inp, "select": select,
                  "psf": psf, "shape_method": shape["method"]}
        if shape["weight_fwhm"] is not None:
            config["shape_weight_fwhm"] = shape["weight_fwhm"]
        return config

    def config(self):
        """Settings of this run as a plain dict (saved with the result)."""
        return {"name": self.name, "sca": self.image.sca, "image": self.image.config(),
                "catalog": self.catalog.config(), "stamp_size": self.stamp_size,
                "reserve": dict(self.reserve), "diagnostics": copy.deepcopy(self.diagnostics),
                "fit": copy.deepcopy(self.fit_config.config), "attitude": self.use_attitude,
                "sed": copy.deepcopy(self.sed)}

    def run(self, save_to=None):
        """Check the image, select stars, fit, compute the diagnostics, and save if asked.

        Parameters
        ----------
        save_to : str or Path, optional
            Output folder (Piff reads the catalog from it): ``catalog.fits``, ``piff.yaml``,
            ``image_cleaned.fits`` (only if the image has non-finite pixels: a copy with them
            cleaned; the original is not changed), ``detections.fits``
            (only for a catalog given as a table: the full table), and the result files (see
            `PiffDataFitResult.save`); reload with `load_data_fit`. Default None: nothing is kept
            on disk (Piff's inputs go to a temporary folder, removed after the fit). The
            result then works in memory as usual, but cannot be reloaded with
            `load_data_fit`.

        Returns
        -------
        PiffDataFitResult
        """
        if save_to is None:
            with tempfile.TemporaryDirectory(prefix="psf_fomo_fit_") as folder:
                return self._run(Path(folder), save=False)
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        return self._run(save_to, save=True)

    def _run(self, folder, save):
        save_to = folder
        inputs = self._write_inputs(save_to)
        config = inputs["config"]
        logger, watch = make_logger(self.verbose)
        t0 = time.time()
        psf, objects, candidates, stars = self._read_and_select(config, logger)
        shape = self.fit_config.config["shape_method"]
        with piff_shape_method(shape["method"], shape["weight_fwhm"]):
            psf.fit(stars, logger=logger)
        seconds = time.time() - t0

        table = diagnostics.star_table(psf, psf.stars, inputs["piff_catalog"])
        max_iter = config["psf"].get("max_iter", 30)
        summary = fit_summary(psf, not watch.triggered, max_iter, table, seconds)
        result = self._result(psf, table, summary, inputs, objects, candidates, stars,
                              save_to if save else None)
        if self.diagnostics["moments"]:
            settings = {k: v for k, v in self.diagnostics.items() if k != "moments"}
            result.compute_moments(**settings)
        if save:
            result.save(save_to)
        return result

    def initialize(self, save_to=None):
        """Piff's initialization only, no fit: the start of a reference optimum without a
        Piff run (``res.reference(starts=('zero',), given=False, refresh=...)``). Experimental.

        The same steps as `run` up to the fit (checks, working copy, catalog, Piff's Input and
        Select, the PSF), then what Piff's ``fit`` does before its first iteration
        (``initialize_flux_center``, ``initialize_params``, one reflux), followed by the
        Poisson update of its iterations: with model Poisson weights (a ``gain`` per star),
        the variance of the current model (fitted flux) is added to the background weights
        (``Star.addPoisson``) and the stars are refluxed (flux and center) with those weights.
        This is Piff's state after an iteration that left the model at its initial values.
        The reference solver then holds weights and centers fixed while it optimizes; its
        ``refresh`` rounds repeat this update at the optimum (see
        `psf_fomo.piff_datafit.reference.reference_from_data_fit`).

        No moments are computed (call ``compute_moments`` if wanted).

        Parameters
        ----------
        save_to : str or Path, optional
            Folder for the result, as `run` (reload with `load_data_fit`). Default None:
            nothing is kept on disk (Piff's inputs go to a temporary folder, removed after
            the stars are read; the result then cannot be reloaded).

        Returns
        -------
        PiffDataFitResult
            ``fit_summary['initialized_only']`` True, ``niter`` 0, ``converged`` False; χ² and
            dof are Piff's (pixels − 3 per star) of the initialized model.
        """
        if save_to is None:
            with tempfile.TemporaryDirectory() as folder:
                return self._initialize(Path(folder), save=False)
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        return self._initialize(save_to, save=True)

    def _initialize(self, folder, save):
        inputs = self._write_inputs(folder)
        logger, _ = make_logger(self.verbose)
        t0 = time.time()
        psf, objects, candidates, stars = self._read_and_select(inputs["config"], logger)
        shape = self.fit_config.config["shape_method"]
        with piff_shape_method(shape["method"], shape["weight_fwhm"]):
            initialize_stars(psf, stars, logger)
        seconds = time.time() - t0
        table = diagnostics.star_table(psf, psf.stars, inputs["piff_catalog"])
        summary = fit_summary(psf, False, inputs["config"]["psf"].get("max_iter", 30), table,
                              seconds)
        summary["initialized_only"] = True
        result = self._result(psf, table, summary, inputs, objects, candidates, stars,
                              folder if save else None)
        if save:
            result.save(folder)
        return result

    def _write_inputs(self, folder):
        """Checks, the working copy, the Piff catalog and config, written to ``folder``."""
        psf_config = self.fit_config.piff_config(self.image.filter_name)
        # A frame mismatch is a warning, not an error: it matters only for RomanOptics
        # (GalSim's aberration field at the star's pixel position), and with a GalSim from
        # before its science-frame update every L2 file is mirrored. The result is kept in
        # the checks.
        attitude = self.attitude()      # first: set_attitude(True) fails before any work
        checks = {"weights_ratio": self.image.check_weights(),
                  "orientation": self.image.check_orientation(strict=False),
                  "nonfinite": self.image.check_finite(),
                  "attitude": {"requested": self.use_attitude, "used": attitude is not None,
                               "values": attitude}}
        image_file = self.image.write_fit_copy(folder / FIT_COPY_NAME)
        piff_catalog = self.catalog.piff_table(self.image, self.stamp_size)
        catalog_file = folder / "catalog.fits"
        piff_catalog.write(catalog_file, overwrite=True)
        config = self.piff_config(catalog_file, piff_catalog, image_file)
        if config["input"].get("sed_file_name") == SED_FILE:
            bandpass = galsim.roman.getBandpasses()[self.image.filter_name]
            write_sed(folder / SED_FILE, *sample_sed(sed_from_setting(self.sed), bandpass))
        write_yaml(folder / "piff.yaml", config)
        return {"config": config, "piff_catalog": piff_catalog, "checks": checks}

    def _read_and_select(self, config, logger):
        """Piff's Input and Select, and the PSF with its context (not yet fitted)."""
        objects, wcs, pointing, bandpass = piff.Input.process(
            copy.deepcopy(config["input"]), copy.deepcopy(config), logger=logger)
        with piff_shape_method(self.catalog.selection["shape_method"]):
            candidates, stars = select_stars(config["select"], objects, config, logger)
        psf = piff.PSF.process(copy.deepcopy(config["psf"]), logger=logger)
        psf.set_context(wcs, pointing, bandpass)
        return psf, objects, candidates, stars

    def _result(self, psf, table, summary, inputs, objects, candidates, stars, folder):
        """The `PiffDataFitResult` of a run or an initialization (``folder`` None: not saved)."""
        piff_catalog = inputs["piff_catalog"]
        summary.update(n_catalog=len(piff_catalog), n_objects=len(objects),
                       n_candidates=len(candidates))
        run_config = self.config()
        run_config["piff"] = inputs["config"]
        if self.catalog.path is None and folder is not None:
            # A catalog given as a table is kept with the run, so the config can rebuild it.
            detections = folder / "detections.fits"
            self.catalog.table.write(detections, overwrite=True)
            run_config["catalog"]["path"] = str(detections.resolve())
        return PiffDataFitResult(psf, table, summary, run_config, provenance(), inputs["checks"],
                                 selection_table(piff_catalog, objects, candidates, stars),
                                 stars=psf.stars)


def piff_input_config(image, catalog, catalog_file, piff_catalog, stamp_size=25,
                      image_file=None):
    """Piff's ``input`` field: the image part (`L2Image.piff_input`) plus the catalog.

    Parameters
    ----------
    image : L2Image
    catalog : StarCatalog
    catalog_file : str or Path
        Where the Piff catalog (``piff_catalog``) is written.
    piff_catalog : astropy.table.Table
        `StarCatalog.piff_table` output, for its column names.
    stamp_size : int
        Stamp side, pixels. Default 25.
    image_file : str or Path, optional
        File Piff reads (default the image's own file; `PiffDataFit` passes the cleaned working
        copy when there is one).

    Returns
    -------
    dict
    """
    inp = image.piff_input(image_file)
    inp.update(cat_file_name=str(Path(catalog_file).resolve()), cat_hdu=1, x_col="x",
               y_col="y", stamp_size=int(stamp_size),
               property_cols=catalog.property_cols(piff_catalog))
    if "gain" in piff_catalog.colnames:
        inp["gain_col"] = "gain"
    return inp


def initialize_stars(psf, stars, logger):
    """Piff's ``fit`` up to its first iteration plus its Poisson update
    (`PiffDataFit.initialize`).

    Sets ``psf.stars`` and the attributes Piff's ``fit`` sets (``chisq``, ``dof`` over fit
    stars, ``niter`` 0, ``nremoved``, ``last_delta_chisq`` 0), so the result can be summarized
    and written like a fitted PSF.
    """
    stars = psf.initialize_flux_center(stars, logger=logger)
    stars, nremoved = psf.initialize_params(stars, logger=logger)
    stars = solve_mean_interps(psf, stars)
    stars, n = psf.reflux_stars(stars, logger)
    nremoved += n
    # As at the end of each Piff iteration: Poisson variance of the model at the fitted flux
    # (a no-op without a 'gain' property), then reflux with the new weights.
    stars = [s.addPoisson(m) for s, m in zip(stars, psf.drawStarList(stars))]
    stars, n = psf.reflux_stars(stars, logger)
    fit = [s for s in stars if not s.is_reserve and not s.is_flagged]
    psf.stars = stars
    psf.chisq = float(sum(s.fit.chisq for s in fit))
    psf.dof = int(sum(s.fit.dof for s in fit))
    psf.niter = 0
    psf.nremoved = nremoved + n
    psf.last_delta_chisq = 0.0
    return stars


def solve_mean_interps(psf, stars):
    """Solve unsolved ``Mean`` interpolators on the initialized fit stars and interpolate.

    Piff's ``Polynomial`` sets its coefficients at initialization and interpolates the stars;
    ``Mean`` stays unsolved (``mean`` None) until the first iteration, so the PSF could not
    be written or drawn at new positions. Solving it on the initial parameters of the fit
    stars (as an iteration does) gives the same kind of initial state as ``Polynomial``.
    """
    fit = [s for s in stars if not s.is_reserve and not s.is_flagged]
    for leaf in leaf_psfs(psf):
        interp = leaf.interp
        if isinstance(interp, piff.Mean) and interp.mean is None:
            interp.solve(fit, transform_params=leaf.model.transform_params)
            stars = interp.interpolateList(stars,
                                           reconstruct_params=leaf.model.reconstruct_params)
            fit = [s for s in stars if not s.is_reserve and not s.is_flagged]
    return stars


def uses_roman_optics(psf_config):
    """True if a Piff ``psf`` config dict contains a ``RomanOptics`` component."""
    if isinstance(psf_config, dict):
        return (psf_config.get("type") == "RomanOptics"
                or any(uses_roman_optics(v) for v in psf_config.values()))
    if isinstance(psf_config, (list, tuple)):
        return any(uses_roman_optics(v) for v in psf_config)
    return False


def add_attitude(psf_config, attitude):
    """Add the attitude options to every ``RomanOptics`` block of a Piff ``psf`` config, in place."""
    if isinstance(psf_config, dict):
        if psf_config.get("type") == "RomanOptics":
            psf_config.update(attitude)
        for value in psf_config.values():
            add_attitude(value, attitude)
    elif isinstance(psf_config, (list, tuple)):
        for value in psf_config:
            add_attitude(value, attitude)


def attitude_support():
    """What the installed software lacks for attitude-oriented RomanOptics (empty: nothing).

    GalSim: ``galsim.roman.getPSF`` must accept ``boresight`` and ``telescope_PA``. Piff:
    RomanOptics must accept ``boresight_ra``, ``boresight_dec``, ``telescope_pa``.

    Returns
    -------
    list of str
    """
    import galsim.roman
    from piff.roman.roman_psf import RomanOpticalModel

    missing = []
    if not {"boresight", "telescope_PA"} <= set(inspect.signature(galsim.roman.getPSF).parameters):
        missing.append("a GalSim whose roman.getPSF takes boresight and telescope_PA")
    piff_options = set(inspect.signature(RomanOpticalModel.__init__).parameters)
    if not {"boresight_ra", "boresight_dec", "telescope_pa"} <= piff_options:
        missing.append("a Piff whose RomanOptics takes boresight_ra, boresight_dec, "
                       "telescope_pa")
    return missing


def select_stars(select_config, objects, base, logger, allow_empty=False):
    """Piff's ``Select.process`` (piff/select.py), keeping the stars of the first stage.

    Returns (candidates, stars): the objects the selection type picked (e.g. SizeMag's
    stellar locus) and the final stars after the rejection cuts, with reserve marked.
    Like Piff, raises RuntimeError when a stage leaves no stars, unless ``allow_empty``
    (for the selection preview, which reports each stage as it is).
    """
    select_config = copy.deepcopy(select_config)
    handler = piff.Select.valid_select_types[select_config.get("type", "Flag")](
        select_config, base, logger=logger)
    candidates = handler.selectStars(objects, logger)
    if len(candidates) == 0:
        if allow_empty:
            return [], []
        raise RuntimeError("No stars were selected.")
    stars = handler.rejectStars(candidates, logger, objects=objects)
    if len(stars) == 0:
        if allow_empty:
            return candidates, []
        raise RuntimeError("All stars were rejected.")
    handler.reserveStars(stars, logger)
    return candidates, stars


def selection_table(piff_catalog, objects, candidates, stars):
    """The Piff catalog with the selection outcome of every detection.

    Adds ``on_image`` (Piff made a stamp: inside the image, not fully masked), ``candidate``
    (picked by the selection type, e.g. SizeMag) and ``star`` (passed the cuts; fitted or
    reserve).
    """
    out = piff_catalog.copy()
    ids = np.asarray(out[PIFF_ID])
    for name, group in (("on_image", objects), ("candidate", candidates), ("star", stars)):
        chosen = {int(s.data.properties[PIFF_ID]) for s in group}
        out[name] = [int(i) in chosen for i in ids]
    return out


class PiffDataFitResult:
    """Result of `PiffDataFit.run` (or `load_data_fit`).

    Attributes
    ----------
    psf : piff.PSF
        The fitted PSF.
    table : astropy.table.Table
        One row per star (fit and reserve), see `psf_fomo.piff_datafit.diagnostics.star_table`.
    fit_summary : dict
        `psf_fomo.piff_simfit.fitting.fit_summary` plus ``n_catalog`` (detections),
        ``n_objects`` (with a stamp), ``n_candidates`` (picked by the selection type).
        ``converged`` is False when Piff stopped at ``max_iter``.
    config : dict
        `PiffDataFit.config` plus ``piff`` (the full Piff config).
    provenance : dict
        Software versions and creation time.
    checks : dict
        Image checks: ``weights_ratio``, ``orientation``, ``nonfinite`` (``total``:
        non-finite pixels, cleaned in the working copy; ``unmasked``: of those, not masked),
        ``attitude`` (``requested``: `PiffDataFit.set_attitude` setting; ``used``: RomanOptics was
        oriented by the attitude; ``values``: the values passed, degrees, or None).
    selection : astropy.table.Table
        Every detection with ``on_image``, ``candidate``, ``star`` (see `selection_table`).
    moments : astropy.table.Table or None
        Moment table (``'data'``, ``'model_os'``), see `psf_fomo.piff_simfit.residuals`.
    """

    def __init__(self, psf, table, fit_summary, config, provenance, checks, selection,
                 stars=None, moments=None):
        self.psf = psf
        self.table = table
        self.fit_summary = fit_summary
        self.config = config
        self.provenance = provenance
        self.checks = checks
        self.selection = selection
        self.moments = moments
        self._stars = stars

    def fitted_stars(self):
        """Fitted stars with their data images, and the star table rows (same order).

        After `load_data_fit` the stamps are re-cut with Piff's Input from the saved config,
        the saved fits attached (parameters, flux, center, reserve and flag status), Piff's
        ``max_snr_weight`` scaling re-applied from the saved ``snr`` / ``raw_snr``, and,
        with model Poisson weights, the Poisson term re-added from the fitted model (as in
        Piff's last iteration, up to its final reflux).
        """
        if self._stars is None:
            self._stars = recut_stars(self.psf, self.config["piff"])
        return self._stars, self.table

    def selected_stars(self, which="fit"):
        """Unflagged ``'fit'``, ``'reserve'`` or ``'all'`` stars and their table rows."""
        if which not in ("fit", "reserve", "all"):
            raise ValueError("which must be 'fit', 'reserve' or 'all'")
        stars, rows = self.fitted_stars()
        keep = [not s.is_flagged and (which == "all" or s.is_reserve == (which == "reserve"))
                for s in stars]
        return [s for s, k in zip(stars, keep) if k], rows[np.array(keep, dtype=bool)]

    def summary(self, columns=SUMMARY_COLUMNS):
        """Mean, std, error on the mean, n and n_flagged, separately for fit and reserve."""
        return star_summary(self.table, columns)

    def compute_moments(self, **settings):
        """(Re)compute ``self.moments``; keyword arguments as `PiffDataFit.set_diagnostics`
        (without ``moments``)."""
        settings.setdefault("comparisons", DATA_COMPARISONS)
        stars, table = self.fitted_stars()
        self.moments = moment_residuals(self.psf, stars, table, sim=None, **settings)
        return self.moments

    def moment_summary(self, columns=("dT_T", "de1", "de2")):
        """Mean, std, error on the mean, n, n_flagged per (set, comparison, algorithm).

        Residuals exist for ``'data'`` only; for ``'model_os'`` summarize the model moments,
        e.g. ``columns=('T_model', 'e1_model', 'e2_model')``.
        """
        if self.moments is None:
            raise ValueError("no moments computed (see compute_moments)")
        return moment_summary(self.moments, columns)

    def model_moments(self, n_grid=5, x=None, y=None, algorithms=("hsm",),
                      wmom_fwhm=(DEFAULT_WMOM_FWHM,), oversample=5, include_pixel=True,
                      size=MODEL_OS_SIZE):
        """Oversampled model moments across the SCA (see `residuals.model_moments`).

        Positions: ``x``, ``y`` (native pixels) if given, else the centers of an
        ``n_grid`` × ``n_grid`` grid of cells covering the SCA.
        """
        if x is None or y is None:
            x, y = sca_grid(n_grid)
        stars, _ = self.fitted_stars()
        return model_moments(self.psf, x, y, self.config["sca"], algorithms, wmom_fwhm,
                             oversample, include_pixel, size, sed_eff=fit_sed(self.psf, stars))

    def components(self):
        """Fitted component summaries, no truth (see `psf_fomo.piff_simfit.components`)."""
        stars, table = self.fitted_stars()
        return component_tables(self.psf, stars, table, truth=False)

    def reference(self, freeze=(), starts=("given",), max_nfev=50, moments="same",
                  centers="fitted", report="all", given=True, refresh=None):
        """Joint χ² optimum of this fit's model with centers held at Piff's fitted ones
        (see `psf_fomo.piff_datafit.reference.reference_from_data_fit`).

        Returns a dict with ``'given'`` (Piff's solution, fluxes profiled, on the reference's
        χ² and dof; only if ``given``) and ``'optimum'`` ({start: ReferenceResult}).
        ``report='fit'`` reports only the fit stars (faster with many reserve stars).
        ``refresh``: rounds of Poisson-weight and center update at the optimum (default 2
        after `PiffDataFit.initialize`, 0 after a Piff fit).
        """
        from .reference import reference_from_data_fit
        return reference_from_data_fit(self, freeze=freeze, starts=starts, max_nfev=max_nfev,
                                       moments=moments, centers=centers, report=report,
                                       given=given, refresh=refresh)

    def rho(self, algorithms=("wmom",), wmom_fwhm=(DEFAULT_WMOM_FWHM,), which="all",
            min_sep=0.1, max_sep=10.0, bin_size=0.5):
        """ρ1 … ρ5 (see `psf_fomo.piff_simfit.rho.rho_stats`); separations in arcmin."""
        from ..piff_simfit.rho import rho_stats  # needs treecorr only when called
        stars, _ = self.selected_stars(which)
        return rho_stats(self.psf, stars, algorithms, wmom_fwhm, min_sep, max_sep, bin_size)

    def plot_rho(self, rho_table=None, **kwargs):
        """Plot ρ1 … ρ5; computes them with ``self.rho(**kwargs)`` if no table is given."""
        return plotting.plot_rho(self.rho(**kwargs) if rho_table is None else rho_table)

    def plot_mean_images(self, which="fit"):
        """Mean data / model / residual stamps (see `piff_simfit.plotting.plot_mean_images`)."""
        return plotting.plot_mean_images(self, which, oversampled=False)

    def plot_chi2_radius(self, which="fit", bin_width=1.0):
        """See `piff_simfit.plotting.plot_chi2_radius`."""
        return plotting.plot_chi2_radius(self, which, bin_width)

    def plot_moment_residuals(self, algorithms=None):
        """Model − data moment histograms (see `piff_simfit.plotting.plot_moment_residuals`)."""
        return plotting.plot_moment_residuals(self, "data", algorithms)

    def plot_field(self, column, comparison=None, algorithm=None, which="all"):
        """A star-table or moment column across the SCA (see `piff_simfit.plotting.plot_field`);
        e.g. ``plot_field('sky_resid')`` or ``plot_field('T_model', 'model_os', 'hsm')``."""
        return plotting.plot_field(self, column, comparison, algorithm, which)

    def plot_components(self, kind="pixelgrid", n_grid=3, oversample=5, size=9,
                        include_pixel=True):
        """See `piff_simfit.plotting.plot_components`."""
        return plotting.plot_components(self, kind, n_grid, oversample, size, include_pixel)

    def plot_component_images(self, x=None, y=None, size=9, oversample=5, include_pixel=True):
        """Every model component alone at one position, native and oversampled (see
        `piff_simfit.plotting.plot_component_images`)."""
        return plotting.plot_component_images(self, x, y, size, oversample, include_pixel)

    def plot_zernikes(self):
        """Fitted extra Zernikes with Piff's σ (see `piff_simfit.plotting.plot_zernikes`)."""
        return plotting.plot_zernikes(self)

    def plot_size_mag(self, ax=None):
        """Catalog size vs magnitude of every detection, marked by selection outcome.

        For measured sizes (gauss, wmom, ...) use `StarCatalog.measure` and
        `psf_fomo.piff_datafit.plotting.plot_size_mag`."""
        from .plotting import plot_size_mag
        return plot_size_mag(self.selection, ax=ax)

    def save(self, path):
        """Save to a folder: ``psf.fits`` (Piff), ``stars.fits``, ``selection.fits``,
        ``moments.fits`` (if computed), ``config.yaml`` (run settings incl. the Piff config,
        fit summary, checks, provenance). `PiffDataFit.run` also writes ``catalog.fits``,
        ``piff.yaml``, the cleaned working copy ``image_cleaned.fits`` (if needed)
        and, for an in-memory catalog, ``detections.fits`` there."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.psf.write(str(path / "psf.fits"))
        self.table.write(path / "stars.fits", overwrite=True)
        self.selection.write(path / "selection.fits", overwrite=True)
        if self.moments is not None:
            self.moments.write(path / "moments.fits", overwrite=True)
        write_yaml(path / "config.yaml", {"config": self.config,
                                          "fit_summary": self.fit_summary,
                                          "checks": self.checks,
                                          "provenance": self.provenance})


def load_data_fit(path):
    """Load a `PiffDataFitResult` saved by `PiffDataFit.run`.

    The Piff catalog and, if there is one, the cleaned working copy are read from the folder;
    otherwise the image must still be at
    its recorded path.

    Parameters
    ----------
    path : str or Path
        Folder of the saved fit.

    Returns
    -------
    PiffDataFitResult
    """
    path = Path(path)
    saved = read_yaml(path / "config.yaml")
    config = saved["config"]
    if config["image"].get("flip"):
        raise ValueError(f"{path}: this fit was made on a copy flipped into the old "
                         "galsim.roman frame; it does not match GalSim's science frame now in "
                         "use. Refit the image.")
    config["piff"]["input"]["cat_file_name"] = str((path / "catalog.fits").resolve())
    if (path / FIT_COPY_NAME).exists():
        config["piff"]["input"]["image_file_name"] = str((path / FIT_COPY_NAME).resolve())
    psf = piff.read(str(path / "psf.fits"))
    table = Table.read(path / "stars.fits")
    selection = Table.read(path / "selection.fits")
    moments = (upgrade_sizes(Table.read(path / "moments.fits"))
               if (path / "moments.fits").exists() else None)
    return PiffDataFitResult(psf, table, saved["fit_summary"], config, saved["provenance"],
                             saved["checks"], selection, moments=moments)


def recut_stars(psf, piff_config):
    """Stamps of a saved fit's stars, re-cut with Piff's Input, fits attached, same order."""
    logger, _ = make_logger(0)
    objects, _, _, _ = piff.Input.process(copy.deepcopy(piff_config["input"]),
                                          copy.deepcopy(piff_config), logger=logger)
    by_id = {int(o.data.properties[PIFF_ID]): o for o in objects}
    stars = []
    for s in psf.stars:
        data = by_id[int(s.data.properties[PIFF_ID])].data
        data.properties["is_reserve"] = bool(s.is_reserve)
        data.properties["is_flagged"] = bool(s.is_flagged)
        data.weight *= snr_cap_factor(s.data.properties)  # in place, as Piff's Select
        star = piff.Star(data, s.fit)
        if "gain" in data.properties:  # model Poisson weights, as in Piff's fit loop
            star = star.addPoisson(psf.drawStar(star))
        stars.append(star)
    return stars


def snr_cap_factor(properties):
    """Weight factor Piff's Select applied for ``max_snr_weight``: (snr / raw_snr)².

    Piff caps a star's effective S/N by multiplying its weight (in place, so also the
    original weight used for the Poisson term) by (max_snr_weight / raw_snr)² and stores
    ``snr`` (capped) and ``raw_snr``; 1 when no cap applied.
    """
    snr, raw = properties.get("snr"), properties.get("raw_snr")
    if snr is None or raw is None or raw <= 0 or snr >= raw:
        return 1.0
    return float(snr / raw) ** 2
