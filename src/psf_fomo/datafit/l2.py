"""Roman L2 images in the FITS layout written by the user's ASDF → FITS converter.

Layout (HDUs found by EXTNAME; see ``docs/datafit.md``):

- ``PRIMARY``: metadata (detector, filter, exposure time).
- ``SCI``: optimal-weight slope [DN/s] with the TAN-SIP WCS. ``SCI_UNIF`` = ``SCI`` + ``DUMO``
  (uniform-weight slope).
- ``MASK`` (1 = bad), ``IVAR`` (1/variance, (DN/s)⁻²), ``GAIN`` (= ``SCI / VAR_POISSON``,
  electrons per DN/s); with ``--uniform`` also ``MASK_UNIF``, ``IVAR_UNIF``, ``GAIN_UNIF``.
- ``VAR_RNOISE``, ``VAR_POISSON``, ``DQ``, ``ERR``, ... (not used by the fit).
- ``SKYBKG`` in every header: one sky level per detector, DN/s per pixel (SEP, on ``SCI``).

Two variants: ``'unif'`` (default; ``SCI_UNIF`` with the ``*_UNIF`` mask, weight and gain) and
``'standard'`` (``SCI`` with ``MASK``, ``IVAR``, ``GAIN``). The planes of one variant are never
mixed with the other: the uniform slope has its own noise model.

Weights (`L2Image.set_weights`):

- ``'model_poisson'`` (default): the weight plane holds read + background noise only, and Piff
  adds the star's Poisson variance from the model at each iteration (Piff's ``gain``, one value
  per star = `gain_at`). ``IVAR_UNIF`` is always of that form; ``IVAR`` only when the converter
  ran with ``--background-weights``, which `check_weights` verifies.
- ``'ivar'``: the weight plane as it is, no Poisson term added by Piff.

Coordinates: GalSim / FITS convention, pixel centres at 1 … 4088.

Pixel frames. Roman's L2 products are in the "science" frame (Roman documentation, RDox
"Coordinate Systems"): each detector's readout frame flipped along one axis so that all SCAs
look alike. ``galsim.roman`` (and so Piff's RomanOptics) uses a frame mirrored relative to it,
which RomanOptics needs (its aberration field is evaluated at the star's pixel position).
Measured on real files with the exposure's own pointing (`compare_frames`; SCA 11: y flip,
1.0″ rms; SCA 3: x flip, 1.3″), and as the RDox figure shows for all SCAs:

- y flip (y_GalSim = n + 1 − y_L2) for SCAs 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17;
- x flip (x_GalSim = n + 1 − x_L2) for SCAs 3, 6, 9, 12, 15, 18.

``L2Image(frame='science')`` (default, real L2 files) fits on a flipped working copy
(`write_fit_copy`; the original file is never changed); ``frame='galsim'`` for files already
in GalSim's frame (e.g. our simulations). The working copy's WCS header is flipped
algebraically (`flip_wcs_header`), not refitted, so every flipped pixel keeps its sky position.
"""
import warnings
from pathlib import Path

import numpy as np
import galsim
import galsim.roman
from astropy.io import fits
from astropy.table import Table

from ..sim.wcs import DEFAULT_DEC, DEFAULT_RA

VARIANTS = {
    "unif": {"image": "SCI_UNIF", "mask": "MASK_UNIF", "ivar": "IVAR_UNIF", "gain": "GAIN_UNIF"},
    "standard": {"image": "SCI", "mask": "MASK", "ivar": "IVAR", "gain": "GAIN"},
}
WEIGHT_MODES = ("model_poisson", "ivar")
# Roman F names → GalSim bandpass names (as Piff's RomanOptics aliases).
FILTER_NAMES = {"F062": "R062", "F087": "Z087", "F106": "Y106", "F129": "J129",
                "F146": "W146", "F158": "H158", "F184": "F184", "F213": "K213"}
SCA_KEYS = ("DETECTOR", "SCA", "SCA_NUM")       # PRIMARY keys tried in order
FILTER_KEYS = ("FILTER", "OPT_ELEM")
EXPTIME_KEYS = ("EXPTIME", "EFFEXPTM", "EXPOSURE_TIME")
SKY_KEY = "SKYBKG"
N_CHECK_PIXELS = 2000   # at most this many of the brightest source pixels in check_weights
MIN_CHECK_PIXELS = 20   # fewer source pixels than this: check_weights cannot decide
MIN_CHECK_SIGMA = 10.0  # a source pixel is this many background σ above the sky
FRAMES = ("science", "galsim")
X_FLIP_SCAS = (3, 6, 9, 12, 15, 18)   # the bottom detector row; all other SCAs flip y
FIT_COPY_NAME = "image_galsim_frame.fits"


def flip_axis(sca):
    """Axis (``'x'`` or ``'y'``) flipped between Roman's science frame and GalSim's, for an SCA."""
    return "x" if int(sca) in X_FLIP_SCAS else "y"


class L2Image:
    """One Roman L2 exposure of one SCA (FITS layout of the user's converter).

    Parameters
    ----------
    path : str or Path
        The FITS file.
    variant : {'unif', 'standard'}
        ``'unif'`` (default): ``SCI_UNIF``, ``MASK_UNIF``, ``IVAR_UNIF``, ``GAIN_UNIF`` (the
        converter's ``--uniform``). ``'standard'``: ``SCI``, ``MASK``, ``IVAR``, ``GAIN``.
    sca : int, optional
        SCA number 1–18. Default: from the PRIMARY header (``DETECTOR`` such as ``'WFI05'``,
        or ``SCA`` / ``SCA_NUM``).
    filter_name : str, optional
        Roman F name (``'F158'``) or GalSim name (``'H158'``). Default: from the PRIMARY header
        (``FILTER`` or ``OPT_ELEM``).
    frame : {'science', 'galsim'}
        Pixel frame of this file. ``'science'`` (default): Roman's L2 science frame; fits run
        on a copy flipped into GalSim's frame (`flip`, `write_fit_copy`). ``'galsim'``: the
        file is already in GalSim's frame (e.g. a simulation); nothing is flipped.

    Attributes
    ----------
    sca : int
    flip : {'x', 'y'} or None
        Axis flipped between this file and the fit frame (None for ``frame='galsim'``).
    filter_name : str
        GalSim bandpass name (e.g. ``'H158'``).
    hdus : dict
        Role (``'image'``, ``'mask'``, ``'ivar'``, ``'gain'``) → HDU index.
    sky_header : float
        ``SKYBKG`` of the image HDU, DN/s per pixel.
    exptime : float or None
        Exposure time from the PRIMARY header, if found (s).
    wcs_err : float or None
        ``WCSERR`` of the image header, if present (recorded only).

    Example
    -------
    >>> img = L2Image("r0003201001001001004_0001_wfi05_cal.fits")
    >>> img.check_weights(); img.check_orientation()
    >>> img.piff_input()["image_hdu"]
    """

    def __init__(self, path, variant="unif", sca=None, filter_name=None, frame="science"):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}")
        if frame not in FRAMES:
            raise ValueError(f"frame must be one of {FRAMES}")
        self.frame = frame
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.variant = variant
        with fits.open(self.path) as hdul:
            names = [h.name.upper() for h in hdul]
            missing = [n for n in VARIANTS[variant].values() if n not in names]
            if missing:
                hint = (" (the uniform planes are written only with the converter's "
                        "--uniform; use variant='standard' otherwise)" if variant == "unif"
                        else "")
                raise ValueError(f"{self.path.name}: HDUs {missing} not found; present: "
                                 f"{names}{hint}")
            self.hdus = {role: names.index(name) for role, name in VARIANTS[variant].items()}
            for extra in ("VAR_RNOISE",):
                if extra in names:
                    self.hdus[extra.lower()] = names.index(extra)
            primary = hdul[0].header
            image_header = hdul[self.hdus["image"]].header
            self.shape = (int(image_header["NAXIS2"]), int(image_header["NAXIS1"]))
        self.sca = int(sca) if sca is not None else _header_sca(primary)
        if not 1 <= self.sca <= 18:
            raise ValueError(f"SCA must be in 1..18, got {self.sca}")
        self.filter_name = galsim_filter_name(filter_name if filter_name is not None
                                              else _header_value(primary, FILTER_KEYS, "filter"))
        exptime = _find(primary, EXPTIME_KEYS)
        self.exptime = None if exptime is None else float(exptime)
        if SKY_KEY not in image_header:
            raise ValueError(f"{SKY_KEY} not in the {VARIANTS[variant]['image']} header")
        self.sky_header = float(image_header[SKY_KEY])
        self.wcs_err = float(image_header["WCSERR"]) if "WCSERR" in image_header else None
        self.wcs = galsim.GSFitsWCS(str(self.path), hdu=self.hdus["image"])
        self.weights = "model_poisson"
        self.sky = self.sky_header
        self._arrays = {}

    @property
    def flip(self):
        """Axis flipped between this file and GalSim's frame, or None."""
        return None if self.frame == "galsim" else flip_axis(self.sca)

    def to_fit_frame(self, x, y):
        """Pixel positions (1-based) of this file in the fit (GalSim) frame.

        The flip is its own inverse, so this also maps fit-frame positions back to the file.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        ny, nx = self.shape
        if self.flip == "x":
            return nx + 1 - x, y
        if self.flip == "y":
            return x, ny + 1 - y
        return x, y

    def write_fit_copy(self, path):
        """Write the file Piff fits: flipped into GalSim's frame, non-finite pixels cleaned.

        Every 2-d plane of the image's shape is flipped along `flip`, and every header with a
        WCS is transformed with `flip_wcs_header`. Pixels whose image or weight is not finite
        (`nonfinite`) get image 0, weight 0 and mask 1: they had (or should have had) zero
        weight anyway, but a NaN would still poison Piff's sums (see `check_finite`). Other
        planes and HDUs are copied as they are. The original file is not changed.

        Parameters
        ----------
        path : str or Path
            Output file (overwritten).

        Returns
        -------
        Path
            ``path``, or this file's own path when neither a flip nor a clean-up is needed.
        """
        bad = self.nonfinite()
        if self.flip is None and not bad.any():
            return self.path
        path = Path(path)
        numpy_axis = 1 if self.flip == "x" else 0   # arrays are indexed [y, x]
        cleaned = {self.hdus["image"]: 0, self.hdus["ivar"]: 0, self.hdus["mask"]: 1}
        with fits.open(self.path) as hdul:
            out = []
            for i, hdu in enumerate(hdul):
                if i > 0 and not hdu.is_image:
                    out.append(hdu.copy())  # e.g. a table: no pixel axes
                    continue
                header = hdu.header.copy()
                # astropy has applied any integer scaling to the data already; it picks its
                # own when writing (keeping BZERO / BSCALE would scale twice).
                for key in ("BZERO", "BSCALE"):
                    header.remove(key, ignore_missing=True)
                if "CTYPE1" in header and self.flip is not None:
                    header = flip_wcs_header(header, self.flip, self.shape)
                data = hdu.data
                if data is not None and data.ndim == 2 and data.shape == self.shape:
                    if i in cleaned and bad.any():
                        data = np.array(data)          # a writable copy
                        data[bad] = cleaned[i]
                    if self.flip is not None:
                        data = np.ascontiguousarray(np.flip(data, axis=numpy_axis))
                if i == 0:
                    out.append(fits.PrimaryHDU(data=data, header=header))
                else:  # compressed planes are written uncompressed
                    out.append(fits.ImageHDU(data=data, header=header, name=hdu.name))
            fits.HDUList(out).writeto(path, overwrite=True)
        return path

    def set_weights(self, mode="model_poisson"):
        """Noise model of the fit.

        Parameters
        ----------
        mode : {'model_poisson', 'ivar'}
            ``'model_poisson'`` (default): read + background weight plane, star Poisson noise
            added by Piff from the model (per-star gain, `gain_at`). ``'ivar'``: the weight
            plane as it is, no Poisson term added.
        """
        if mode not in WEIGHT_MODES:
            raise ValueError(f"mode must be one of {WEIGHT_MODES}")
        self.weights = mode

    def set_sky(self, value=None):
        """Sky level subtracted from the image before the fit, DN/s per pixel.

        Parameters
        ----------
        value : float, optional
            Default None = the header ``SKYBKG``. 0 subtracts nothing.
        """
        self.sky = self.sky_header if value is None else float(value)

    def array(self, role):
        """Data of one plane (``'image'``, ``'mask'``, ``'ivar'``, ``'gain'``,
        ``'var_rnoise'``) as a numpy array indexed [y − 1, x − 1]; read once and cached."""
        if role not in self.hdus:
            raise ValueError(f"no {role!r} plane in {self.path.name}")
        if role not in self._arrays:
            # Native dtype (float32 planes, integer mask): a 4088² plane in float64 is 134 MB.
            self._arrays[role] = np.asarray(fits.getdata(self.path, self.hdus[role]))
        return self._arrays[role]

    def gain_at(self, x, y, stamp_size=25):
        """Median gain (electrons per DN/s) in the stamp around each position.

        Uses unmasked pixels with a finite, positive gain (``GAIN = SCI / VAR_POISSON`` is
        undefined where the slope is ≤ 0).

        Parameters
        ----------
        x, y : float or array of float
            Positions, GalSim image coordinates (1-based).
        stamp_size : int
            Stamp side in pixels. Default 25.

        Returns
        -------
        numpy.ndarray
            One value per position; NaN where no pixel qualifies.
        """
        gain, mask = self.array("gain"), self.array("mask")
        good = (mask == 0) & np.isfinite(gain) & (gain > 0)
        ny, nx = gain.shape
        half = stamp_size // 2
        out = []
        for xi, yi in zip(np.atleast_1d(x), np.atleast_1d(y)):
            ix, iy = int(np.floor(xi + 0.5)) - 1, int(np.floor(yi + 0.5)) - 1  # 0-based
            sl = (slice(max(iy - half, 0), min(iy + half + 1, ny)),
                  slice(max(ix - half, 0), min(ix + half + 1, nx)))
            values = gain[sl][good[sl]]
            out.append(float(np.median(values)) if values.size else np.nan)
        return np.array(out)

    def check_weights(self, tolerance=0.1):
        """Check that the weight plane fits the chosen noise model; return the median ratio.

        Only ``'model_poisson'`` with the standard variant needs it: ``IVAR`` must then be the
        converter's ``--background-weights`` form 1/(``VAR_RNOISE`` + ``SKYBKG``/``GAIN``).
        A data-based ``IVAR`` (1/``ERR``², which contains the star's Poisson noise) would make
        Piff count that noise twice. Tested on the brightest unmasked pixels more than 10σ
        above the sky (at most 2000, at least 20), where the two forms differ most.

        Parameters
        ----------
        tolerance : float
            Allowed |median ratio − 1|. Default 0.1.

        Returns
        -------
        float or None
            Median of IVAR × (VAR_RNOISE + SKYBKG/GAIN) on those pixels; None when no check
            applies (uniform variant or ``'ivar'`` mode).

        Raises
        ------
        ValueError
            If the ratio is outside the tolerance, or ``VAR_RNOISE`` is missing.
        """
        if self.variant == "unif" or self.weights != "model_poisson":
            return None  # IVAR_UNIF has no data term; 'ivar' uses the plane as it is
        if "var_rnoise" not in self.hdus:
            raise ValueError("VAR_RNOISE not in the file: cannot verify that IVAR is the "
                             "background-only weight that 'model_poisson' needs")
        sci, ivar, gain = self.array("image"), self.array("ivar"), self.array("gain")
        rnoise, mask = self.array("var_rnoise"), self.array("mask")
        good = ((mask == 0) & np.isfinite(sci) & np.isfinite(ivar) & (ivar > 0)
                & np.isfinite(gain) & (gain > 0) & np.isfinite(rnoise))
        background_var = np.where(good, rnoise + self.sky_header / np.where(good, gain, 1), 1)
        # Only source pixels discriminate: on sky pixels both forms agree.
        source = good & (sci > self.sky_header + MIN_CHECK_SIGMA * np.sqrt(background_var))
        if source.sum() < MIN_CHECK_PIXELS:
            raise ValueError(f"only {int(source.sum())} unmasked pixels above "
                             f"{MIN_CHECK_SIGMA}σ of the sky: cannot verify IVAR")
        values = sci[source]
        bright = np.argsort(values)[-N_CHECK_PIXELS:]
        ratio = float(np.median(ivar[source][bright] * background_var[source][bright]))
        if abs(ratio - 1) > tolerance:
            raise ValueError(
                f"IVAR is not 1/(VAR_RNOISE + SKYBKG/GAIN) on the brightest pixels (median "
                f"ratio {ratio:.3f}): it probably contains the stars' Poisson noise (converter "
                "run without --background-weights). Re-run the converter with "
                "--background-weights, or use set_weights('ivar').")
        return ratio

    def check_orientation(self, pa_deg=None, pa_is_fpa=False, strict=True):
        """Compare the fit frame (this file after `flip`) with ``galsim.roman``'s SCA frame.

        Piff's RomanOptics evaluates GalSim's field-dependent aberrations at the star's pixel
        position, so the fit must run in the SCA frame of ``galsim.roman``. Compares the local
        WCS Jacobian at the image centre (with the frame flip applied) with that of
        ``galsim.roman.getWCS`` at the SCA centre. GalSim's WCS is built at a fixed observable
        pointing (``psf_fomo.sim.wcs`` default; ``getWCS`` may refuse pointings it deems
        unobservable on its default date): parity does not depend on the pointing, the
        rotation only weakly (< 1° away from the celestial poles). For the full mapping with
        the exposure's own pointing use `compare_frames`.

        Parameters
        ----------
        pa_deg : float, optional
            Position angle (degrees) of the observation, as ``galsim.roman.getWCS`` expects.
            Without it only the parity (a mirrored axis) and the pixel scale are checked; with
            it also the rotation is reported.
        pa_is_fpa : bool
            ``pa_deg`` is the focal-plane PA (GalSim's ``PA_is_FPA``). Default False.
        strict : bool
            Raise on a failed check (default). False: warn and return the result (for fits
            that do not use RomanOptics, where the frame does not matter).

        Returns
        -------
        dict
            ``parity_ok``, ``scale`` (arcsec/pix, √|det J|), ``rotation_deg`` (fit frame −
            GalSim, None without ``pa_deg``), ``frame``, ``flip``.

        Raises
        ------
        ValueError
            With ``strict``: if the parity differs (one axis mirrored) or the pixel scale is
            off by > 5 %.

        Notes
        -----
        A pure 180° rotation of the pixel frame keeps the parity; it is visible only through
        ``rotation_deg`` with a known PA.
        """
        ny, nx = self.shape
        centre = galsim.PositionD((nx + 1) / 2, (ny + 1) / 2)  # its own mirror image
        j_file = self.wcs.local(centre).jacobian().getMatrix()
        # The fit frame: a flipped axis reverses that column of the Jacobian.
        if self.flip == "x":
            j_file = j_file @ np.diag([-1.0, 1.0])
        elif self.flip == "y":
            j_file = j_file @ np.diag([1.0, -1.0])
        pa = (pa_deg if pa_deg is not None else 0.0) * galsim.degrees
        pointing = galsim.CelestialCoord(DEFAULT_RA * galsim.degrees,
                                         DEFAULT_DEC * galsim.degrees)
        roman = galsim.roman.getWCS(world_pos=pointing, PA=pa, SCAs=[self.sca],
                                    PA_is_FPA=pa_is_fpa)[self.sca]
        n = galsim.roman.n_pix
        j_roman = roman.local(galsim.PositionD((n + 1) / 2, (n + 1) / 2)).jacobian().getMatrix()
        scale = float(np.sqrt(abs(np.linalg.det(j_file))))
        out = {"parity_ok": bool(np.sign(np.linalg.det(j_file))
                                 == np.sign(np.linalg.det(j_roman))),
               "scale": scale, "rotation_deg": None, "frame": self.frame, "flip": self.flip}
        if pa_deg is not None:
            out["rotation_deg"] = float(np.degrees(_angle(j_file) - _angle(j_roman)) + 180) % 360 - 180
        problems = []
        if not out["parity_ok"]:
            problems.append(
                f"the fit frame (file frame {self.frame!r}, flip {self.flip}) is mirrored "
                f"relative to galsim.roman's SCA {self.sca} frame: RomanOptics' field "
                "dependence would be wrong. Is the file's frame set correctly (real L2: "
                "'science'; files already in GalSim's frame: 'galsim')?")
        if abs(scale / galsim.roman.pixel_scale - 1) > 0.05:
            problems.append(f"pixel scale {scale:.4f}″ differs from "
                            f"galsim.roman.pixel_scale = {galsim.roman.pixel_scale}″")
        for problem in problems:
            if strict:
                raise ValueError(problem)
            warnings.warn(problem)
        return out

    def compare_frames(self, pointing, date=None):
        """Map this file's pixels onto ``galsim.roman``'s SCA frame using the exposure's pointing.

        See `compare_frames` (module function). Needs a full-SCA image (4088 × 4088).
        """
        return compare_frames(self.wcs, self.sca, self.shape, pointing, date)

    def nonfinite(self):
        """Boolean map (file frame) of pixels whose image or weight is not finite."""
        return ~(np.isfinite(self.array("image")) & np.isfinite(self.array("ivar")))

    def check_finite(self):
        """Count pixels whose image or weight is not finite.

        Piff gives masked pixels zero weight, but its sums still multiply weight × image, and
        0 × NaN is NaN: one such pixel in a stamp makes that star's flux and centre NaN, and
        Piff excludes the star. The working copy of a fit (`write_fit_copy`) therefore sets
        these pixels to image 0, weight 0, mask 1. Warns for unmasked ones (those are
        unexpected: real data were never meant to be fitted there).

        Returns
        -------
        dict
            ``total`` (non-finite pixels, all cleaned in the working copy) and ``unmasked``
            (of those, not flagged in the mask plane).
        """
        bad = self.nonfinite()
        unmasked = int((bad & (self.array("mask") == 0)).sum())
        if unmasked:
            warnings.warn(f"{unmasked} unmasked pixels have a non-finite image or weight")
        return {"total": int(bad.sum()), "unmasked": unmasked}

    def piff_input(self, image_file=None):
        """Image part of Piff's ``input`` field.

        Parameters
        ----------
        image_file : str or Path, optional
            File Piff reads: default this file; `DataFit` passes the flipped working copy
            (`write_fit_copy`, same HDU layout).

        Returns
        -------
        dict
            ``image_file_name`` (absolute), ``image_hdu``, ``weight_hdu``, ``badpix_hdu``,
            ``sky`` (only if ≠ 0), ``chipnum`` = SCA, ``properties`` {``sca``}, ``bandpass``.
            The catalog part and the per-star gain (``gain_col``) are added by `DataFit`.
        """
        path = self.path if image_file is None else Path(image_file).resolve()
        out = {"image_file_name": str(path), "image_hdu": self.hdus["image"],
               "weight_hdu": self.hdus["ivar"], "badpix_hdu": self.hdus["mask"],
               "chipnum": self.sca, "properties": {"sca": self.sca},
               "bandpass": {"type": "RomanBandpass", "name": self.filter_name}}
        if self.sky != 0:
            out["sky"] = self.sky
        return out

    def config(self):
        """Settings as a plain dict (saved with fit results; `from_config` rebuilds)."""
        return {"path": str(self.path), "variant": self.variant, "sca": self.sca,
                "filter_name": self.filter_name, "frame": self.frame, "flip": self.flip,
                "weights": self.weights, "sky": self.sky,
                "sky_header": self.sky_header, "exptime": self.exptime,
                "wcs_err": self.wcs_err, "hdus": dict(self.hdus)}

    @classmethod
    def from_config(cls, config):
        """Rebuild from `config` output."""
        img = cls(config["path"], config["variant"], config["sca"], config["filter_name"],
                  config.get("frame", "science"))
        img.set_weights(config["weights"])
        img.set_sky(config["sky"])
        return img


def flip_wcs_header(header, axis, shape):
    """A copy of a FITS WCS header for the image flipped along one pixel axis.

    Pixel x' = n + 1 − x (axis ``'x'``) or y' = n + 1 − y (axis ``'y'``), n the image size
    along that axis; every flipped pixel keeps its sky position. Exact (no refit): with
    offsets u = x − CRPIX1, v = y − CRPIX2 (and U, V after the SIP distortion), flipping x
    maps u → −u, U → −U, so

    - CRPIX on the flipped axis → n + 1 − CRPIX;
    - that column of the linear matrix (CD, or PC; CDELT when neither exists) changes sign;
    - SIP coefficients with power k of the flipped coordinate get (−1)^k, and the polynomial
      of the flipped coordinate itself (A, AP for x; B, BP for y) one more factor −1.

    Parameters
    ----------
    header : astropy.io.fits.Header
    axis : {'x', 'y'}
    shape : (ny, nx)
        Image shape.

    Returns
    -------
    astropy.io.fits.Header
    """
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    h = header.copy()
    i = 1 if axis == "x" else 2
    n = shape[1] if axis == "x" else shape[0]
    h[f"CRPIX{i}"] = n + 1 - float(h[f"CRPIX{i}"])
    if any(f"CD{j}_{k}" in h for j in (1, 2) for k in (1, 2)):
        for j in (1, 2):
            h[f"CD{j}_{i}"] = -float(h.get(f"CD{j}_{i}", 0.0))
    elif any(f"PC{j}_{k}" in h for j in (1, 2) for k in (1, 2)):
        for j in (1, 2):
            h[f"PC{j}_{i}"] = -float(h.get(f"PC{j}_{i}", 1.0 if j == i else 0.0))
    else:
        h[f"CDELT{i}"] = -float(h.get(f"CDELT{i}", 1.0))
    own = ("A", "AP") if axis == "x" else ("B", "BP")
    for prefix in ("A", "B", "AP", "BP"):
        order = int(h.get(f"{prefix}_ORDER", -1))
        for p in range(order + 1):
            for q in range(order + 1 - p):
                key = f"{prefix}_{p}_{q}"
                if key in h:
                    sign = (-1) ** (p if axis == "x" else q)
                    h[key] = float(h[key]) * (-sign if prefix in own else sign)
    return h


def compare_frames(wcs, sca, shape, pointing, date=None, n_grid=7):
    """Which pixel mapping takes a full-SCA image onto ``galsim.roman``'s SCA frame.

    A grid of pixels goes to the sky through ``wcs`` and, after each candidate mapping (as-is,
    flip x, flip y, rotate 180), through ``galsim.roman.getWCS`` for the exposure's own
    pointing. Both are measured relative to their own SCA centre (a pointing offset cancels).
    The pointing is tried in four interpretations: ``world_pos`` = the V1 axis or the target,
    and ``PA`` = ``pa_v3`` (``PA_is_FPA=False``) or ``pa_aperture`` (``PA_is_FPA=True``). On
    real L2 files (SCAs 3 and 11) ``world_pos`` = target with ``PA`` = ``pa_v3`` matched, with
    SCA centres within ≈ 1″.

    How to read it: (1) the right pointing interpretation has a small ``centre_offset``
    (arcsec; independent of the mapping); (2) ``rms_rotation_free`` (best rotation removed,
    PA-independent) is small (≈ 1″) only for the right parity, the same pair in every
    interpretation; (3) in the right interpretation the right mapping has ``rms`` ≈
    ``rms_rotation_free`` and ``extra_rotation`` ≈ 0°.

    Parameters
    ----------
    wcs : galsim.BaseWCS
        The image's (celestial) WCS.
    sca : int
    shape : (ny, nx)
        Image shape; must be the full SCA (``galsim.roman.n_pix`` square).
    pointing : dict
        The exposure's ``meta.pointing``: ``ra_v1``, ``dec_v1``, ``pa_v3``, ``pa_aperture``,
        ``target_ra``, ``target_dec`` in degrees.
    date : datetime.datetime, optional
        Observation date for ``getWCS`` (it may refuse a pointing on its default date).
    n_grid : int
        Grid points per axis. Default 7.

    Returns
    -------
    astropy.table.Table
        One row per interpretation × mapping: ``world_pos``, ``pa``, ``pa_is_fpa``,
        ``mapping``, ``centre_offset``, ``rms``, ``rms_rotation_free`` (arcsec),
        ``extra_rotation`` (deg), ``error`` (getWCS message if it failed). ``meta['best']``:
        the mapping with the smallest ``rms``.
    """
    n = galsim.roman.n_pix
    if tuple(shape) != (n, n):
        raise ValueError(f"compare_frames needs a full SCA image ({n} × {n}), got {shape}")
    g = np.linspace(1, n, n_grid)
    xs, ys = (a.ravel() for a in np.meshgrid(g, g))
    centre = galsim.PositionD((n + 1) / 2, (n + 1) / 2)
    mappings = {"as-is": (xs, ys), "flip x": (n + 1 - xs, ys),
                "flip y": (xs, n + 1 - ys), "rotate 180": (n + 1 - xs, n + 1 - ys)}
    c_file = wcs.toWorld(centre)
    p_file = _tangent(wcs, xs, ys, c_file)
    rows = []
    for pos_name, ra, dec in (("V1", "ra_v1", "dec_v1"), ("target", "target_ra", "target_dec")):
        world = galsim.CelestialCoord(float(pointing[ra]) * galsim.degrees,
                                      float(pointing[dec]) * galsim.degrees)
        for pa_name, is_fpa in (("pa_v3", False), ("pa_aperture", True)):
            base = {"world_pos": pos_name, "pa": pa_name, "pa_is_fpa": is_fpa}
            try:
                gs = galsim.roman.getWCS(world_pos=world,
                                         PA=float(pointing[pa_name]) * galsim.degrees,
                                         date=date, SCAs=[int(sca)], PA_is_FPA=is_fpa)[int(sca)]
            except Exception as err:  # e.g. a pointing GalSim deems unobservable on its date
                for name in mappings:
                    rows.append(dict(base, mapping=name, centre_offset=np.nan, rms=np.nan,
                                     rms_rotation_free=np.nan, extra_rotation=np.nan,
                                     error=str(err)))
                continue
            c_gs = gs.toWorld(centre)
            offset = c_file.distanceTo(c_gs) / galsim.arcsec
            for name, (tx, ty) in mappings.items():
                p_gs = _tangent(gs, tx, ty, c_gs)
                rms = float(np.sqrt(np.mean(np.sum((p_file - p_gs) ** 2, axis=1))))
                free, angle = _rms_after_rotation(p_file, p_gs)
                rows.append(dict(base, mapping=name, centre_offset=float(offset), rms=rms,
                                 rms_rotation_free=free, extra_rotation=angle, error=""))
    out = Table(rows=rows)
    finite = np.isfinite(out["rms"])
    out.meta["best"] = (str(out["mapping"][np.nanargmin(out["rms"])]) if finite.any()
                        else None)
    return out


def _tangent(wcs, x, y, about):
    """Sky positions of pixels (x, y) projected about ``about``, arcsec, shape (n, 2)."""
    uv = [about.project(wcs.toWorld(galsim.PositionD(a, b))) for a, b in zip(x, y)]
    return np.array([(u / galsim.arcsec, v / galsim.arcsec) for u, v in uv])


def _rms_after_rotation(p, q):
    """rms |p − R q| for the best rotation R (absorbs a position-angle error, not a mirror)."""
    angle = np.arctan2(np.sum(q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0]), np.sum(p * q))
    c, s = np.cos(angle), np.sin(angle)
    rq = q @ np.array([[c, s], [-s, c]])
    return float(np.sqrt(np.mean(np.sum((p - rq) ** 2, axis=1)))), float(np.degrees(angle))


def galsim_filter_name(name):
    """GalSim Roman bandpass name for a Roman F name or a GalSim name."""
    name = str(name).strip().upper()
    if name in FILTER_NAMES:
        return FILTER_NAMES[name]
    if name in FILTER_NAMES.values():
        return name
    raise ValueError(f"unknown Roman filter {name!r}; expected one of {list(FILTER_NAMES)} "
                     f"or {list(FILTER_NAMES.values())}")


def _find(header, keys):
    for key in keys:
        if key in header:
            return header[key]
    return None


def _header_value(header, keys, what):
    value = _find(header, keys)
    if value is None:
        raise ValueError(f"no {what} in the PRIMARY header (tried {list(keys)}); pass it "
                         "explicitly")
    return value


def _header_sca(header):
    """SCA number from e.g. DETECTOR = 'WFI05' or SCA = 5."""
    value = _header_value(header, SCA_KEYS, "SCA")
    digits = "".join(c for c in str(value) if c.isdigit())
    if not digits:
        raise ValueError(f"cannot read an SCA number from {value!r}; pass sca=...")
    return int(digits)


def _angle(jacobian):
    """Direction of the pixel +x axis on the sky, radians (from the WCS Jacobian)."""
    return np.arctan2(jacobian[1, 0], jacobian[0, 0])
