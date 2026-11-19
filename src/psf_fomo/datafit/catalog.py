"""Detection catalogs and star selection for real-data fits (`StarCatalog`).

A SEP detection catalog (FITS table, HDU 1): ``id`` (1…N), ``x``, ``y`` (0-based:
the first pixel centre is (0, 0)), ``ra_deg``, ``dec_deg``, ``flux_auto`` [DN/s] (the flux to
use), ``flux_auto_err_bkg``, ``snr_auto_bkg``, ``r50`` [pix], shape columns, flags and the
boolean ``clean``; header ``SKY``, ``DETCFG``, ``MATCHRAD``, ``OBSTIME``, ``NMATCH``, ``ABZP``.

All detections are given to Piff as input objects, so that Piff's ``min_sep`` sees every
neighbour (galaxies and unclean objects included). Which objects are stars is a Piff
``select`` step on top:

- ``'sizemag'`` (default): Piff's ``SizeMag`` star finder (stellar locus in size vs
  magnitude). It measures every object with Piff's global shape method; the selection runs
  with ``gauss`` by default.
- ``'all'``: every detection.
- ``'column'``: a flag column of the catalog (Piff ``Flag``).
- ``'locus'``: our own stellar-locus selection (`StarCatalog.select_locus`, module
  `psf_fomo.datafit.locus`) on a pre-fit measurement table; the chosen stars become a 0/1
  column ``psf_locus`` that Piff's ``Flag`` selection reads. For catalogs that are already
  star lists (e.g. Gaia matches), where SizeMag (built to separate stars from galaxies) does
  not apply.

Then Piff's cuts (`StarCatalog.set_cuts`), applied after the star selection; ``clean_only``
adds ``clean == 0`` to Piff's ``reject_where``.
"""
import copy
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from ..io import to_plain
from ..piffval.fitting import DEFAULT_MIN_SEP, make_cuts, select_config

PIFF_ID = "id"          # star id property in Piff (the catalog's id column, renamed)
PIFF_CLEAN = "clean"    # clean property in Piff (0/1)
SELECT_METHODS = ("sizemag", "all", "column")   # 'locus' is set by select_locus
LOCUS_COLUMN = "psf_locus"   # 0/1 column of the stars chosen by select_locus
HEADER_KEYS = ("SKY", "DETCFG", "MATCHRAD", "OBSTIME", "NMATCH", "ABZP")
SHAPE_METHODS = ("hsm", "gauss", "am", "wmom")
DEFAULT_CUTS = {"clean_only": True, **make_cuts()}   # min_sep 2.5″, nothing else


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
        # Plain Python values: a table's meta may hold NumPy scalars (they would not save).
        self.header = {k: to_plain(meta[k]) for k in HEADER_KEYS if k in meta}
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
            shape-based cuts). Default ``'gauss'``. The fit
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

    def select_locus(self, measured, **settings):
        """Select the PSF stars with our stellar-locus selection, replacing Piff's star finder.

        Experimental.

        Runs `psf_fomo.datafit.locus.select_locus` on a `measure` table of this catalog and
        makes the chosen stars the selection: a 0/1 column ``psf_locus`` read by Piff's
        ``Flag`` selection (Piff's cuts and reserve still apply). Stars not in ``measured``
        (no stamp) are not selected. The settings, the summary and the selected ids are kept
        in the selection (`config`), so `from_config` rebuilds the same selection.

        Parameters
        ----------
        measured : astropy.table.Table
            `StarCatalog.measure` output for this catalog (e.g. ``algorithms=("wmom",)``).
        **settings
            `select_locus` parameters (``size``, ``order``, ``k``, ``mag``, ``mag_max``,
            ``bright_limit``, ``mag_bin``, ``sat_sigma``, ``sat_bins``, ``min_bin_count``).

        Returns
        -------
        LocusResult
            ``.table`` (per-star reason), ``.summary``, ``.bins``, ``.plot()``.

        Example
        -------
        >>> t = cat.measure(img, algorithms=("wmom",))
        >>> res = cat.select_locus(t)          # defaults: T_wmom_0.4, order 1, k 3
        >>> res.summary; res.plot()
        >>> cat.set_cuts(min_snr=50, min_sep=2.5)
        """
        from .locus import select_locus
        ids = np.asarray(self.table[self.columns["id"]])
        unknown = set(np.asarray(measured[PIFF_ID]).tolist()) - set(ids.tolist())
        if unknown:
            raise ValueError(f"{len(unknown)} measured ids are not in this catalog: was the "
                             "table measured from another catalog?")
        result = select_locus(measured, **settings)
        chosen = result.selected_ids
        self.table[LOCUS_COLUMN] = np.isin(ids, chosen).astype(int)
        self.selection = {"method": "locus",
                          "shape_method": self.selection.get("shape_method", "gauss"),
                          "options": {"flag_name": LOCUS_COLUMN, "use_flag": 1},
                          "locus": {"settings": dict(settings), "summary": result.summary,
                                    "ids": chosen}}
        return result

    def set_cuts(self, clean_only=True, min_snr=None, min_sep=DEFAULT_MIN_SEP,
                 max_mask_pixels=None, max_edge_frac=None, stamp_center_size=None,
                 max_pixel_cut=None, reject_where=None, nstars=None, max_snr_weight=None):
        """Piff's rejection cuts, applied after the star selection.

        The same cuts as `psf_fomo.piffval.PiffValidation.set_cuts` (built by
        `psf_fomo.piffval.fitting.make_cuts`), plus ``clean_only``. Without a call, the
        defaults apply (``clean_only``, ``min_sep`` 2.5″).

        Parameters
        ----------
        clean_only : bool
            Reject objects whose ``clean`` column is false. Default True.
        min_snr : float, optional
            Minimum S/N (Piff's own estimate from the stamp and weight).
        min_sep : float or None
            Reject stars with any other detection within this distance, arcsec. Default 2.5
            (a 25-pixel stamp reaches 1.9″ at its corners, plus the PSF wings). None: no
            separation cut.
        max_mask_pixels : int, optional
            Reject stars with at least this many masked (zero-weight) pixels in the stamp.
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
        cuts = make_cuts(min_sep=min_sep, min_snr=min_snr, max_snr_weight=max_snr_weight,
                         max_edge_frac=max_edge_frac, stamp_center_size=stamp_center_size,
                         max_mask_pixels=max_mask_pixels, max_pixel_cut=max_pixel_cut,
                         reject_where=reject_where, nstars=nstars)
        self.cuts = {"clean_only": bool(clean_only), **cuts}

    def piff_select(self):
        """Piff's ``select`` field for this selection and these cuts (without reserve).

        Returns
        -------
        dict
        """
        method = self.selection["method"]
        out = {"type": {"sizemag": "SizeMag", "all": "Flag", "column": "Flag",
                        "locus": "Flag"}[method]}
        out.update(self.selection["options"])
        clean = [f"{PIFF_CLEAN} == 0"] if self.cuts["clean_only"] else []
        out.update(select_config(self.cuts, extra_reject=clean))
        return out

    def piff_table(self, image=None, stamp_size=25, fit_frame=True):
        """The catalog as Piff reads it.

        Columns: ``x``, ``y`` (GalSim / Piff convention, 1-based), ``id``, and when available
        ``flux_cat`` [DN/s], ``size_cat`` [pix], ``clean`` (0/1), ``mag`` (``ABZP`` −
        2.5 log10 ``flux_cat``; NaN for flux ≤ 0), the selection column and ``extra_cols``;
        with an `L2Image` using ``'model_poisson'`` weights also ``gain`` (median in the stamp,
        e⁻ per DN/s; where undefined, the image median). Rows with non-finite positions are
        dropped.

        With an `L2Image` whose frame needs a flip (real L2 files, ``frame='science'``) and
        ``fit_frame``, ``x``, ``y`` are in the fit (GalSim) frame, matching the flipped
        working copy, and ``x_l2``, ``y_l2`` keep the positions in the file (1-based).

        Parameters
        ----------
        image : L2Image, optional
            Needed for the ``gain`` column and the frame.
        stamp_size : int
            Stamp size for the gain median, pixels. Default 25.
        fit_frame : bool
            Positions in the fit frame (default; for `DataFit`). False: in the file's own frame
            (for measurements on the original file).

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
            # The gain plane is read from the original file: positions in its own frame.
            gain = image.gain_at(out["x"], out["y"], stamp_size)
            valid = image.array("gain")
            fallback = float(np.median(valid[np.isfinite(valid) & (valid > 0)]))
            out["gain"] = np.where(np.isfinite(gain), gain, fallback)
        if image is not None and fit_frame and image.flip is not None:
            out["x_l2"], out["y_l2"] = out["x"].copy(), out["y"].copy()
            out["x"], out["y"] = image.to_fit_frame(out["x_l2"], out["y_l2"])
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
        """Columns of `piff_table` that Piff should carry as star properties (incl. ``x_l2``,
        ``y_l2`` when present)."""
        return [name for name in table.colnames if name not in ("x", "y", "gain")]

    def config(self):
        """Settings as a plain dict (`from_config` rebuilds). ``path`` is None for a catalog
        given as a table (`DataFit.run` saves it as ``detections.fits`` and records that)."""
        return {"path": None if self.path is None else str(self.path), "hdu": self.hdu,
                "columns": dict(self.columns),
                "zero_based": self.zero_based, "extra_cols": list(self.extra_cols),
                "header": dict(self.header), "selection": copy.deepcopy(self.selection),
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
        if cat.selection["method"] == "locus":
            ids = np.asarray(cat.table[c["id"]])
            cat.table[LOCUS_COLUMN] = np.isin(ids, cat.selection["locus"]["ids"]).astype(int)
        cat.cuts = dict(config["cuts"])
        return cat
