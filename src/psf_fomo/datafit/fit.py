"""Piff fits of one real L2 image (`DataFit`), results (`DataFitResult`), `load_data_fit`.

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

Telescope attitude (`DataFit.set_attitude`): with a RomanOptics model and an image whose
pointing is known (`L2Image.attitude`), the boresight and V3 position angle are written into
the RomanOptics config (``boresight_ra``, ``boresight_dec``, ``telescope_pa``, degrees), so
GalSim orients the optical PSF from the telescope attitude instead of the WCS. This needs a
GalSim whose ``roman.getPSF`` takes ``boresight`` / ``telescope_PA`` and a Piff whose
RomanOptics takes these options (`attitude_support`).

Example
-------
>>> from psf_fomo.piffval import FitConfig
>>> from psf_fomo.datafit import L2Image, StarCatalog, DataFit
>>> img = L2Image("exp_wfi05.fits")
>>> cat = StarCatalog("exp_wfi05_det.fits"); cat.set_cuts(min_snr=50, min_sep=2.0)
>>> cfg = FitConfig(); cfg.set_optics("free"); cfg.add_kernel("signed")
>>> fit = DataFit(img, cat, cfg); fit.set_reserve(0.2, seed=3)
>>> res = fit.run(save_to="out/real01")
>>> res.fit_summary["chisq_dof"], res.summary(), res.moment_summary()
"""
import copy
import inspect
import time
import warnings
from pathlib import Path

import numpy as np
import piff
from astropy.table import Table

from .. import models  # noqa: F401  (registers our Piff models before any config is parsed)
from ..io import provenance, read_yaml, write_yaml
from ..moments.measure import DEFAULT_WMOM_FWHM
from ..piffval import plotting
from ..piffval.components import components as component_tables
from ..piffval.config import FitConfig
from ..piffval.diagnostics import summary as star_summary
from ..piffval.fitting import fit_summary, make_logger, piff_shape_method
from ..piffval.residuals import (MODEL_OS_SIZE, model_moments, moment_residuals,
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


class DataFit:
    """Builder for a Piff fit of one real image.

    Parameters
    ----------
    image : L2Image
        The exposure (variant, weights and sky set on it).
    catalog : StarCatalog
        The detections, star selection and cuts.
    fit_config : FitConfig
        Model and fit settings (the same builder as for simulations).
    name : str, optional
        Label stored with the result.
    """

    def __init__(self, image, catalog, fit_config, name=None):
        if not isinstance(image, L2Image):
            raise TypeError("image must be an L2Image")
        if not isinstance(catalog, StarCatalog):
            raise TypeError("catalog must be a StarCatalog")
        if not isinstance(fit_config, FitConfig):
            raise TypeError("fit_config must be a FitConfig")
        self.image, self.catalog, self.fit_config = image, catalog, fit_config
        self.name = name
        self.stamp_size = 25
        self.reserve = {"frac": 0.0, "seed": 0}
        self.diagnostics = copy.deepcopy(DEFAULT_DIAGNOSTICS)
        self.verbose = 0
        self.use_attitude = "auto"

    def set_attitude(self, use="auto"):
        """Whether RomanOptics is oriented by the telescope attitude (see the module docstring).

        Parameters
        ----------
        use : {'auto', True, False}
            ``'auto'`` (default): use it when the model has RomanOptics, the image's attitude
            is known and the installed GalSim and Piff support it; otherwise (warning if only
            the software is missing) GalSim orients the PSF from the WCS. True: require it
            (`run` raises if it cannot be used). False: never use it (WCS orientation).
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
        """Moment diagnostics computed by `run` (see `psf_fomo.piffval.residuals`).

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
            The image file Piff reads (default the image's own; `run` passes the working copy
            in GalSim's frame).

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
                "fit": copy.deepcopy(self.fit_config.config), "attitude": self.use_attitude}

    def run(self, save_to):
        """Check the image, select stars, fit, compute the diagnostics, save.

        Parameters
        ----------
        save_to : str or Path
            Output folder (Piff reads the catalog from it): ``catalog.fits``, ``piff.yaml``,
            ``image_galsim_frame.fits`` (the image flipped into GalSim's frame, for
            ``frame='science'`` files; the original is not changed), ``detections.fits``
            (only for a catalog given as a table: the full table), and the result files (see
            `DataFitResult.save`).

        Returns
        -------
        DataFitResult
        """
        save_to = Path(save_to)
        save_to.mkdir(parents=True, exist_ok=True)
        psf_config = self.fit_config.piff_config(self.image.filter_name)
        # Only RomanOptics depends on the pixel frame (GalSim's aberration field at the star's
        # pixel position); other models are built on the sky through the WCS.
        strict = uses_roman_optics(psf_config)
        attitude = self.attitude()      # first: set_attitude(True) fails before any work
        checks = {"weights_ratio": self.image.check_weights(),
                  "orientation": self.image.check_orientation(strict=strict),
                  "nonfinite": self.image.check_finite(),
                  "attitude": {"requested": self.use_attitude, "used": attitude is not None,
                               "values": attitude}}
        image_file = self.image.write_fit_copy(save_to / FIT_COPY_NAME)
        piff_catalog = self.catalog.piff_table(self.image, self.stamp_size)
        catalog_file = save_to / "catalog.fits"
        piff_catalog.write(catalog_file, overwrite=True)
        config = self.piff_config(catalog_file, piff_catalog, image_file)
        write_yaml(save_to / "piff.yaml", config)

        logger, watch = make_logger(self.verbose)
        shape = self.fit_config.config["shape_method"]
        t0 = time.time()
        objects, wcs, pointing, bandpass = piff.Input.process(
            copy.deepcopy(config["input"]), copy.deepcopy(config), logger=logger)
        with piff_shape_method(self.catalog.selection["shape_method"]):
            candidates, stars = select_stars(config["select"], objects, config, logger)
        psf = piff.PSF.process(copy.deepcopy(config["psf"]), logger=logger)
        psf.set_context(wcs, pointing, bandpass)
        with piff_shape_method(shape["method"], shape["weight_fwhm"]):
            psf.fit(stars, logger=logger)
        seconds = time.time() - t0

        table = diagnostics.star_table(psf, psf.stars, piff_catalog)
        max_iter = config["psf"].get("max_iter", 30)
        summary = fit_summary(psf, not watch.triggered, max_iter, table, seconds)
        summary.update(n_catalog=len(piff_catalog), n_objects=len(objects),
                       n_candidates=len(candidates))
        run_config = self.config()
        run_config["piff"] = config
        if self.catalog.path is None:
            # A catalog given as a table is kept with the run, so the config can rebuild it.
            detections = save_to / "detections.fits"
            self.catalog.table.write(detections, overwrite=True)
            run_config["catalog"]["path"] = str(detections.resolve())
        result = DataFitResult(psf, table, summary, run_config, provenance(), checks,
                               selection_table(piff_catalog, objects, candidates, stars),
                               stars=psf.stars)
        if self.diagnostics["moments"]:
            settings = {k: v for k, v in self.diagnostics.items() if k != "moments"}
            result.compute_moments(**settings)
        result.save(save_to)
        return result


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
        `StarCatalog.piff_table` output, for its column names; its positions must be in the
        frame of the file Piff reads.
    stamp_size : int
        Stamp side, pixels. Default 25.
    image_file : str or Path, optional
        File Piff reads (default the image's own file; `DataFit` passes the working copy in
        GalSim's frame).

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


class DataFitResult:
    """Result of `DataFit.run` (or `load_data_fit`).

    Attributes
    ----------
    psf : piff.PSF
        The fitted PSF.
    table : astropy.table.Table
        One row per star (fit and reserve), see `psf_fomo.datafit.diagnostics.star_table`.
    fit_summary : dict
        `psf_fomo.piffval.fitting.fit_summary` plus ``n_catalog`` (detections),
        ``n_objects`` (with a stamp), ``n_candidates`` (picked by the selection type).
        ``converged`` is False when Piff stopped at ``max_iter``.
    config : dict
        `DataFit.config` plus ``piff`` (the full Piff config).
    provenance : dict
        Software versions and creation time.
    checks : dict
        Image checks: ``weights_ratio``, ``orientation``, ``nonfinite`` (``total``:
        non-finite pixels, cleaned in the working copy; ``unmasked``: of those, not masked),
        ``attitude`` (``requested``: `DataFit.set_attitude` setting; ``used``: RomanOptics was
        oriented by the attitude; ``values``: the values passed, degrees, or None).
    selection : astropy.table.Table
        Every detection with ``on_image``, ``candidate``, ``star`` (see `selection_table`).
    moments : astropy.table.Table or None
        Moment table (``'data'``, ``'model_os'``), see `psf_fomo.piffval.residuals`.
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
        the saved fits attached (parameters, flux, centre, reserve and flag status), Piff's
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
        """(Re)compute ``self.moments``; keyword arguments as `DataFit.set_diagnostics`
        (without ``moments``)."""
        settings.setdefault("comparisons", DATA_COMPARISONS)
        stars, table = self.fitted_stars()
        self.moments = moment_residuals(self.psf, stars, table, sim=None, **settings)
        return self.moments

    def moment_summary(self, columns=("dT_T", "de1", "de2")):
        """Mean, std, error on the mean, n, n_flagged per (set, comparison, algorithm).

        Residuals exist for ``'data'`` only; for ``'model_os'`` summarise the model moments,
        e.g. ``columns=('T_model', 'e1_model', 'e2_model')``.
        """
        if self.moments is None:
            raise ValueError("no moments computed (see compute_moments)")
        return moment_summary(self.moments, columns)

    def model_moments(self, n_grid=5, x=None, y=None, algorithms=("hsm",),
                      wmom_fwhm=(DEFAULT_WMOM_FWHM,), oversample=5, include_pixel=True,
                      size=MODEL_OS_SIZE):
        """Oversampled model moments across the SCA (see `residuals.model_moments`).

        Positions: ``x``, ``y`` (native pixels) if given, else the centres of an
        ``n_grid`` × ``n_grid`` grid of cells covering the SCA.
        """
        if x is None or y is None:
            x, y = sca_grid(n_grid)
        return model_moments(self.psf, x, y, self.config["sca"], algorithms, wmom_fwhm,
                             oversample, include_pixel, size)

    def components(self):
        """Fitted component summaries, no truth (see `psf_fomo.piffval.components`)."""
        stars, table = self.fitted_stars()
        return component_tables(self.psf, stars, table, truth=False)

    def reference(self, freeze=(), starts=("given",), max_nfev=50, moments="same",
                  centers="fitted"):
        """Joint χ² optimum of this fit's model with centres held at Piff's fitted ones
        (see `psf_fomo.datafit.reference.reference_from_data_fit`).

        Returns a dict with ``'given'`` (Piff's solution, fluxes profiled, on the reference's
        χ² and dof) and ``'optimum'`` ({start: ReferenceResult}).
        """
        from .reference import reference_from_data_fit
        return reference_from_data_fit(self, freeze=freeze, starts=starts, max_nfev=max_nfev,
                                       moments=moments, centers=centers)

    def rho(self, algorithms=("wmom",), wmom_fwhm=(DEFAULT_WMOM_FWHM,), which="all",
            min_sep=0.1, max_sep=10.0, bin_size=0.5):
        """ρ1 … ρ5 (see `psf_fomo.piffval.rho.rho_stats`); separations in arcmin."""
        from ..piffval.rho import rho_stats  # needs treecorr only when called
        stars, _ = self.selected_stars(which)
        return rho_stats(self.psf, stars, algorithms, wmom_fwhm, min_sep, max_sep, bin_size)

    def plot_rho(self, rho_table=None, **kwargs):
        """Plot ρ1 … ρ5; computes them with ``self.rho(**kwargs)`` if no table is given."""
        return plotting.plot_rho(self.rho(**kwargs) if rho_table is None else rho_table)

    def plot_mean_images(self, which="fit"):
        """Mean data / model / residual stamps (see `piffval.plotting.plot_mean_images`)."""
        return plotting.plot_mean_images(self, which, oversampled=False)

    def plot_chi2_radius(self, which="fit", bin_width=1.0):
        """See `piffval.plotting.plot_chi2_radius`."""
        return plotting.plot_chi2_radius(self, which, bin_width)

    def plot_moment_residuals(self, algorithms=None):
        """Model − data moment histograms (see `piffval.plotting.plot_moment_residuals`)."""
        return plotting.plot_moment_residuals(self, "data", algorithms)

    def plot_field(self, column, comparison=None, algorithm=None, which="all"):
        """A star-table or moment column across the SCA (see `piffval.plotting.plot_field`);
        e.g. ``plot_field('sky_resid')`` or ``plot_field('T_model', 'model_os', 'hsm')``."""
        return plotting.plot_field(self, column, comparison, algorithm, which)

    def plot_components(self, kind="pixelgrid", n_grid=3, oversample=5, size=9):
        """See `piffval.plotting.plot_components`."""
        return plotting.plot_components(self, kind, n_grid, oversample, size)

    def plot_zernikes(self):
        """Fitted extra Zernikes with Piff's σ (see `piffval.plotting.plot_zernikes`)."""
        return plotting.plot_zernikes(self)

    def plot_size_mag(self, ax=None):
        """Catalog size vs magnitude of every detection, marked by selection outcome.

        For measured sizes (gauss, wmom, ...) use `StarCatalog.measure` and
        `psf_fomo.datafit.plotting.plot_size_mag`."""
        from .plotting import plot_size_mag
        return plot_size_mag(self.selection, ax=ax)

    def save(self, path):
        """Save to a folder: ``psf.fits`` (Piff), ``stars.fits``, ``selection.fits``,
        ``moments.fits`` (if computed), ``config.yaml`` (run settings incl. the Piff config,
        fit summary, checks, provenance). `DataFit.run` also writes ``catalog.fits``,
        ``piff.yaml``, the working copy ``image_galsim_frame.fits`` (``frame='science'``)
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
    """Load a `DataFitResult` saved by `DataFit.run`.

    The Piff catalog and, for ``frame='science'`` images, the flipped working copy are read
    from the folder (so the folder can be moved); a ``frame='galsim'`` image must still be at
    its recorded path.

    Parameters
    ----------
    path : str or Path
        Folder of the saved fit.

    Returns
    -------
    DataFitResult
    """
    path = Path(path)
    saved = read_yaml(path / "config.yaml")
    config = saved["config"]
    config["piff"]["input"]["cat_file_name"] = str((path / "catalog.fits").resolve())
    if (path / FIT_COPY_NAME).exists():
        config["piff"]["input"]["image_file_name"] = str((path / FIT_COPY_NAME).resolve())
    psf = piff.read(str(path / "psf.fits"))
    table = Table.read(path / "stars.fits")
    selection = Table.read(path / "selection.fits")
    moments = Table.read(path / "moments.fits") if (path / "moments.fits").exists() else None
    return DataFitResult(psf, table, saved["fit_summary"], config, saved["provenance"],
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
