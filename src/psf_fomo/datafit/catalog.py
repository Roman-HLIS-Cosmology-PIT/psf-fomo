"""Detection catalogs and star selection for real-data fits (`StarCatalog`).

The user's SEP detection catalog (FITS table, HDU 1): ``id`` (1…N), ``x``, ``y`` (0-based:
the first pixel centre is (0, 0)), ``ra_deg``, ``dec_deg``, ``flux_auto`` [DN/s] (the flux to
use), ``flux_auto_err_bkg``, ``snr_auto_bkg``, ``r50`` [pix], shape columns, flags and the
boolean ``clean``; header ``SKY``, ``DETCFG``, ``MATCHRAD``, ``OBSTIME``, ``NMATCH``, ``ABZP``.

All detections are given to Piff as input objects, so that Piff's ``min_sep`` sees every
neighbour (galaxies and unclean objects included). Which objects are stars is a Piff
``select`` step on top:

- ``'sizemag'`` (default): Piff's ``SizeMag`` star finder (stellar locus in size vs
  magnitude). It measures every object with Piff's global shape method; HSM fails on most
  native H158 stamps (CLAUDE.md §9.8), so the selection runs with ``gauss`` by default.
- ``'all'``: every detection.
- ``'column'``: a flag column of the catalog (Piff ``Flag``).

Then Piff's cuts (`StarCatalog.set_cuts`), applied after the star selection; ``clean_only``
adds ``clean == 0`` to Piff's ``reject_where``.
"""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

PIFF_ID = "id"          # star id property in Piff (the catalog's id column, renamed)
PIFF_CLEAN = "clean"    # clean property in Piff (0/1)
SELECT_METHODS = ("sizemag", "all", "column")
HEADER_KEYS = ("SKY", "DETCFG", "MATCHRAD", "OBSTIME", "NMATCH", "ABZP")
SHAPE_METHODS = ("hsm", "gauss", "am", "wmom")
DEFAULT_CUTS = {"clean_only": True, "min_snr": None, "min_sep": None, "max_mask_pixels": None,
                "max_edge_frac": None, "stamp_center_size": None, "max_pixel_cut": None,
                "reject_where": None, "nstars": None, "max_snr_weight": None}


class StarCatalog:
    """A detection catalog and the rule that picks the PSF stars from it.

    Parameters
    ----------
    catalog : str, Path or astropy.table.Table
        The detection catalog: a FITS table file (the SEP command-line output), or the
        table itself (e.g. what ``detect_sources`` returns in Python; a copy is kept). The
        header values (``ABZP``, ``SKY``, ``OBSTIME``, ...) come from the FITS header, or
        from the table's ``meta`` (keys matched case-insensitively).
    hdu : int
        Table HDU of a file. Default 1.
    id_col, x_col, y_col : str
        Unique object id and pixel positions. Defaults ``'id'``, ``'x'``, ``'y'``.
    zero_based : bool
        Positions in numpy convention (first pixel centre (0, 0)), as SEP writes them. Default
        True; they are shifted by +1 to GalSim / Piff's convention when the Piff catalog is
        written.
    flux_col : str or None
        Catalog flux in DN/s (for ``flux_cat_r`` and magnitudes). Default ``'flux_auto'``.
    size_col : str or None
        Size column for plots, pixels. Default ``'r50'``.
    clean_col : str or None
        Boolean quality column used by ``set_cuts(clean_only=True)``. Default ``'clean'``.
    extra_cols : sequence of str
        Further columns passed to Piff as star properties (usable in ``reject_where``).

    Example
    -------
    >>> cat = StarCatalog("exp_wfi05_det.fits")      # or StarCatalog(detections_table)
    >>> cat.select_stars("sizemag")
    >>> cat.set_cuts(clean_only=True, min_snr=50, min_sep=2.0)
    """

    def __init__(self, catalog, hdu=1, id_col="id", x_col="x", y_col="y", zero_based=True,
                 flux_col="flux_auto", size_col="r50", clean_col="clean", extra_cols=()):
        self.hdu = int(hdu)
        if isinstance(catalog, Table):
            # In memory: copied, so later edits of the caller's table do not change a run.
            self.path = None
            self.table = catalog.copy()
            meta = {str(k).upper(): v for k, v in catalog.meta.items()}
        else:
            self.path = Path(catalog).resolve()
            self.table = Table.read(self.path, hdu=self.hdu)
            meta = fits.getheader(self.path, self.hdu)
        self.header = {k: meta[k] for k in HEADER_KEYS if k in meta}
        self.columns = {"id": id_col, "x": x_col, "y": y_col, "flux": flux_col,
                        "size": size_col, "clean": clean_col}
        self.extra_cols = list(extra_cols)
        needed = [c for c in self.columns.values() if c is not None] + self.extra_cols
        missing = [c for c in needed if c not in self.table.colnames]
        if missing:
            source = self.path.name if self.path else "the table"
            raise ValueError(f"{source}: columns {missing} not found; present: "
                             f"{self.table.colnames}")
        ids = np.asarray(self.table[id_col])
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"column {id_col!r} is not unique")
        self.zero_based = bool(zero_based)
        self.selection = {"method": "sizemag", "shape_method": "gauss", "options": {}}
        self.cuts = dict(DEFAULT_CUTS)

    def select_stars(self, method="sizemag", shape_method="gauss", column=None, use_flag=None,
                     skip_flag=None, **options):
        """Choose how stars are picked among the detections.

        Parameters
        ----------
        method : {'sizemag', 'all', 'column'}
            ``'sizemag'``: Piff ``SizeMag``; ``options`` are its parameters (``impurity``,
            ``fit_order``, ``num_iter``, ``initial_select``). ``'all'``: every detection.
            ``'column'``: Piff ``Flag`` on ``column`` (with ``use_flag`` / ``skip_flag`` as
            Piff: by default objects with ``column == 0`` are used; ``use_flag=1`` uses those
            with bit 1 set).
        shape_method : {'hsm', 'gauss', 'am', 'wmom'}
            Piff's global shape method during the selection (SizeMag sizes, and the
            shape-based cuts). Default ``'gauss'`` (HSM fails on most native stamps). The fit
            has its own (`FitConfig.set_piff_shape_method`).
        """
        if method not in SELECT_METHODS:
            raise ValueError(f"method must be one of {SELECT_METHODS}")
        if shape_method not in SHAPE_METHODS:
            raise ValueError(f"shape_method must be one of {SHAPE_METHODS}")
        if method == "column":
            if column is None or column not in self.table.colnames:
                raise ValueError("method='column' needs column=<a catalog column>")
            options = {"flag_name": column}
            if use_flag is not None:
                options["use_flag"] = int(use_flag)
            if skip_flag is not None:
                options["skip_flag"] = int(skip_flag)
        elif method == "all" and options:
            raise ValueError("method='all' takes no options")
        self.selection = {"method": method, "shape_method": shape_method,
                          "options": dict(options)}

    def set_cuts(self, clean_only=True, min_snr=None, min_sep=None, max_mask_pixels=None,
                 max_edge_frac=None, stamp_center_size=None, max_pixel_cut=None,
                 reject_where=None, nstars=None, max_snr_weight=None):
        """Piff's rejection cuts, applied after the star selection.

        Parameters
        ----------
        clean_only : bool
            Reject objects whose ``clean`` column is false. Default True.
        min_snr : float, optional
            Minimum S/N (Piff's own estimate from the stamp and weight).
        min_sep : float, optional
            Reject stars with any other detection within this distance, arcsec.
        max_mask_pixels : int, optional
            Reject stars with more masked (zero-weight) pixels in the stamp.
        max_edge_frac, stamp_center_size : float, optional
            Reject stars with more than this flux fraction outside a radius of
            ``stamp_center_size`` pixels (Piff default 13, meant for 32-pixel stamps).
        max_pixel_cut : float, optional
            Reject stars brighter than the flux whose typical peak is this value, DN/s.
        reject_where : str, optional
            Extra Piff ``reject_where`` expression on star properties (catalog columns passed
            as properties, e.g. ``'r50_flag != 0'`` with ``extra_cols=['r50_flag']``).
        nstars : int, optional
            Keep at most this many stars (highest S/N).
        max_snr_weight : float, optional
            Piff caps each star's effective S/N at this value by inflating its variance.
            Default None = Piff's default (100): bright stars do not dominate the fit, but
            χ²/dof then falls below 1 for them. 0 turns the cap off (as the simulation
            validation does).
        """
        if clean_only and self.columns["clean"] is None:
            raise ValueError("clean_only needs clean_col")
        self.cuts = {"clean_only": bool(clean_only), "min_snr": min_snr, "min_sep": min_sep,
                     "max_mask_pixels": max_mask_pixels, "max_edge_frac": max_edge_frac,
                     "stamp_center_size": stamp_center_size, "max_pixel_cut": max_pixel_cut,
                     "reject_where": reject_where, "nstars": nstars,
                     "max_snr_weight": None if max_snr_weight is None
                     else float(max_snr_weight)}

    def piff_select(self):
        """Piff's ``select`` field for this selection and these cuts (without reserve).

        Returns
        -------
        dict
        """
        method = self.selection["method"]
        out = {"type": {"sizemag": "SizeMag", "all": "Flag", "column": "Flag"}[method]}
        out.update(self.selection["options"])
        rejects = []
        if self.cuts["clean_only"]:
            rejects.append(f"{PIFF_CLEAN} == 0")
        if self.cuts["reject_where"]:
            rejects.append(f"({self.cuts['reject_where']})")
        if rejects:
            out["reject_where"] = " or ".join(rejects)
        for key in ("min_snr", "min_sep", "max_mask_pixels", "max_edge_frac",
                    "stamp_center_size", "max_pixel_cut", "nstars", "max_snr_weight"):
            if self.cuts[key] is not None:
                out[key] = self.cuts[key]
        return out

    def piff_table(self, image=None, stamp_size=25):
        """The catalog as Piff reads it.

        Columns: ``x``, ``y`` (GalSim / Piff convention, 1-based), ``id``, and when available
        ``flux_cat`` [DN/s], ``size_cat`` [pix], ``clean`` (0/1), ``mag`` (``ABZP`` −
        2.5 log10 ``flux_cat``; NaN for flux ≤ 0), the selection column and ``extra_cols``;
        with an `L2Image` using ``'model_poisson'`` weights also ``gain`` (median in the stamp,
        e⁻ per DN/s; where undefined, the image median). Rows with non-finite positions are
        dropped.

        Parameters
        ----------
        image : L2Image, optional
            Needed for the ``gain`` column.
        stamp_size : int
            Stamp size for the gain median, pixels. Default 25.

        Returns
        -------
        astropy.table.Table
        """
        t, c = self.table, self.columns
        shift = 1.0 if self.zero_based else 0.0
        x = np.asarray(t[c["x"]], dtype=float) + shift
        y = np.asarray(t[c["y"]], dtype=float) + shift
        keep = np.isfinite(x) & np.isfinite(y)
        out = Table({"x": x[keep], "y": y[keep], PIFF_ID: np.asarray(t[c["id"]])[keep]})
        if c["flux"] is not None:
            flux = np.asarray(t[c["flux"]], dtype=float)[keep]
            out["flux_cat"] = flux
            if "ABZP" in self.header:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["mag"] = np.where(flux > 0, self.header["ABZP"] - 2.5 * np.log10(flux),
                                          np.nan)
        if c["size"] is not None:
            out["size_cat"] = np.asarray(t[c["size"]], dtype=float)[keep]
        if c["clean"] is not None:
            out[PIFF_CLEAN] = np.asarray(t[c["clean"]]).astype(int)[keep]
        selection_col = self.selection["options"].get("flag_name")
        for name in self.extra_cols + ([selection_col] if selection_col else []):
            if name not in out.colnames:
                out[name] = np.asarray(t[name])[keep]
        if image is not None and image.weights == "model_poisson":
            gain = image.gain_at(out["x"], out["y"], stamp_size)
            valid = image.array("gain")
            fallback = float(np.median(valid[np.isfinite(valid) & (valid > 0)]))
            out["gain"] = np.where(np.isfinite(gain), gain, fallback)
        return out

    def measure(self, image, algorithms=("gauss",), wmom_fwhm=(0.4,), stamp_size=25,
                mag_range=None, max_objects=None, preview_selection=False, verbose=0):
        """Moments of every detection on ``image``, before any fit (one row per detection).

        See `psf_fomo.datafit.measure.measure_objects` for the parameters and columns
        (``T_<alg>`` [pix²], ``sigma_<alg>``, ``e1_<alg>``, ``e2_<alg>``, ``flux_<alg>``,
        ``mag_<alg>``, ``flag_<alg>``; ``candidate`` / ``star`` with ``preview_selection``).

        Example
        -------
        >>> t = cat.measure(img, algorithms=("gauss", "wmom"), preview_selection=True)
        >>> plot_size_mag(t, algorithm="gauss")
        """
        from .measure import measure_objects
        return measure_objects(image, self, algorithms, wmom_fwhm, stamp_size, mag_range,
                               max_objects, preview_selection, verbose)

    def property_cols(self, table):
        """Columns of `piff_table` that Piff should carry as star properties."""
        return [name for name in table.colnames if name not in ("x", "y", "gain")]

    def config(self):
        """Settings as a plain dict (`from_config` rebuilds). ``path`` is None for a catalog
        given as a table (`DataFit.run` saves it as ``detections.fits`` and records that)."""
        return {"path": None if self.path is None else str(self.path), "hdu": self.hdu,
                "columns": dict(self.columns),
                "zero_based": self.zero_based, "extra_cols": list(self.extra_cols),
                "header": dict(self.header), "selection": dict(self.selection),
                "cuts": dict(self.cuts)}

    @classmethod
    def from_config(cls, config, table=None):
        """Rebuild from `config` output; ``table`` is needed when it has no ``path``."""
        c = config["columns"]
        source = config["path"] if table is None else table
        if source is None:
            raise ValueError("this catalog was given as a table: pass it as table=...")
        cat = cls(source, config["hdu"], c["id"], c["x"], c["y"],
                  config["zero_based"], c["flux"], c["size"], c["clean"], config["extra_cols"])
        cat.selection = dict(config["selection"])
        cat.cuts = dict(config["cuts"])
        return cat
