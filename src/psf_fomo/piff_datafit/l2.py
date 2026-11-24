"""Roman L2 images converted from ASDF to FITS (`psf_fomo.utils.convert_asdf_to_fits`).

Layout (HDUs found by EXTNAME; see ``docs/piff_datafit.md``):

- ``PRIMARY``: metadata (detector, filter, exposure time) and, when the converter copies the
  ASDF ``meta.pointing``, the pointing in degrees: ``RA_V1``, ``DEC_V1``, ``PA_V3`` (telescope
  boresight and V3 position angle) and optionally ``TARG_RA``, ``TARG_DEC``, ``PA_APER``.
- ``SCI``: optimal-weight slope [DN/s] with the TAN-SIP WCS. ``SCI_UNIF`` = ``SCI`` + ``DUMO``
  (uniform-weight slope).
- ``MASK`` (1 = bad), ``IVAR`` (1/variance, (DN/s)⁻²), ``GAIN`` (= ``SCI / VAR_POISSON``,
  electrons per DN/s); with ``uniform_weights=True`` also ``MASK_UNIF``, ``IVAR_UNIF``,
  ``GAIN_UNIF``.
- ``VAR_RNOISE``, ``VAR_POISSON``, ``DQ``, ``ERR``, ... (not used by the fit).
- ``SKYBKG`` in every header: one sky level per detector, DN/s per pixel (SEP, on ``SCI``).

Two variants: ``'unif'`` (default; ``SCI_UNIF`` with the ``*_UNIF`` mask, weight and gain) and
``'standard'`` (``SCI`` with ``MASK``, ``IVAR``, ``GAIN``). The planes of one variant are never
mixed with the other: the uniform slope has its own noise model.

Weights (`L2Image.set_weights`):

- ``'model_poisson'`` (default): the weight plane holds read + background noise only, and Piff
  adds the star's Poisson variance from the model at each iteration (Piff's ``gain``, one value
  per star = `gain_at`). ``IVAR_UNIF`` is always of that form; ``IVAR`` only when the converter
  ran with ``background_weights=True``, which `check_weights` verifies.
- ``'ivar'``: the weight plane as it is, no Poisson term added by Piff.

Coordinates: GalSim / FITS convention, pixel centers at 1 … 4088.

Pixel frame. L2 products are in Roman's "science" frame (Roman documentation, RDox
"Coordinate Systems"), and so are ``galsim.roman``'s WCS and PSFs (GalSim ``main`` since the
science-frame update; no release has it yet). Files are fitted as they are; `check_orientation`
compares a file's WCS with ``galsim.roman.getWCS`` and catches a mirrored axis, e.g. from an
older GalSim.
"""
import warnings
from pathlib import Path

import numpy as np
import galsim
import galsim.roman
from astropy.io import fits

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
# PRIMARY keys (degrees, copied by the converter from ASDF meta.pointing) → meta.pointing names.
POINTING_KEYS = {"RA_V1": "ra_v1", "DEC_V1": "dec_v1", "PA_V3": "pa_v3",
                 "TARG_RA": "target_ra", "TARG_DEC": "target_dec", "PA_APER": "pa_aperture"}
N_CHECK_PIXELS = 2000   # at most this many of the brightest source pixels in check_weights
MIN_CHECK_PIXELS = 20   # fewer source pixels than this: check_weights cannot decide
MIN_CHECK_SIGMA = 10.0  # a source pixel is this many background σ above the sky
FIT_COPY_NAME = "image_cleaned.fits"


class L2Image:
    """One Roman L2 exposure of one SCA (FITS layout described in the module docstring).

    Parameters
    ----------
    path : str or Path
        The FITS file.
    variant : {'unif', 'standard'}
        ``'unif'`` (default): ``SCI_UNIF``, ``MASK_UNIF``, ``IVAR_UNIF``, ``GAIN_UNIF`` (the
        converter's ``uniform_weights=True``). ``'standard'``: ``SCI``, ``MASK``, ``IVAR``,
        ``GAIN``.
    sca : int, optional
        SCA number 1–18. Default: from the PRIMARY header (``DETECTOR`` such as ``'WFI05'``,
        or ``SCA`` / ``SCA_NUM``).
    filter_name : str, optional
        Roman F name (``'F158'``) or GalSim name (``'H158'``). Default: from the PRIMARY header
        (``FILTER`` or ``OPT_ELEM``).
    pointing : dict, optional
        The exposure's pointing in degrees, as ASDF ``meta.pointing`` (keys ``ra_v1``,
        ``dec_v1``, ``pa_v3``, ``target_ra``, ``target_dec``, ``pa_aperture``; others are
        ignored). Values given here replace those read from the PRIMARY header
        (``RA_V1``, ``DEC_V1``, ``PA_V3``, ``TARG_RA``, ``TARG_DEC``, ``PA_APER``); use it for
        files converted without them.

    Attributes
    ----------
    sca : int
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
    pointing : dict
        Pointing values found (header, then the ``pointing`` argument), degrees; may be empty.
    attitude : dict or None
        Telescope attitude for ``galsim.roman.getPSF`` (see the property).

    Example
    -------
    >>> img = L2Image("r0003201001001001004_0001_wfi05_cal.fits")
    >>> img.check_weights(); img.check_orientation()
    >>> img.piff_input()["image_hdu"]
    """

    def __init__(self, path, variant="unif", sca=None, filter_name=None, pointing=None):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}")
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.variant = variant
        with fits.open(self.path) as hdul:
            names = [h.name.upper() for h in hdul]
            missing = [n for n in VARIANTS[variant].values() if n not in names]
            if missing:
                hint = (" (the uniform planes are written only with the converter's "
                        "uniform_weights=True; use variant='standard' otherwise)"
                        if variant == "unif" else "")
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
        self.pointing = {name: float(primary[key]) for key, name in POINTING_KEYS.items()
                         if key in primary}
        if pointing is not None:
            self.pointing.update({name: float(pointing[name]) for name in POINTING_KEYS.values()
                                  if name in pointing})
        self.weights = "model_poisson"
        self.sky = self.sky_header
        self._arrays = {}

    @property
    def attitude(self):
        """Telescope attitude for orienting RomanOptics PSFs, or None if not known.

        ``{'boresight_ra', 'boresight_dec', 'telescope_pa'}`` in degrees, from ``ra_v1``,
        ``dec_v1``, ``pa_v3`` of `pointing` (the names of Piff's RomanOptics options; GalSim's
        ``getPSF(boresight=..., telescope_PA=...)``).
        """
        p = self.pointing
        if not all(k in p for k in ("ra_v1", "dec_v1", "pa_v3")):
            return None
        return {"boresight_ra": p["ra_v1"], "boresight_dec": p["dec_v1"],
                "telescope_pa": p["pa_v3"]}

    def write_fit_copy(self, path):
        """Write the file Piff fits when this one has non-finite pixels: those cleaned.

        Pixels whose image or weight is not finite (`nonfinite`) get image 0, weight 0 and
        mask 1: they had (or should have had) zero weight anyway, but a NaN would still poison
        Piff's sums (see `check_finite`). Everything else is copied as it is. The original
        file is not changed.

        Parameters
        ----------
        path : str or Path
            Output file (overwritten).

        Returns
        -------
        Path
            ``path``, or this file's own path when nothing needs cleaning.
        """
        bad = self.nonfinite()
        if not bad.any():
            return self.path
        path = Path(path)
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
                data = hdu.data
                if (data is not None and data.ndim == 2 and data.shape == self.shape
                        and i in cleaned):
                    data = np.array(data)          # a writable copy
                    data[bad] = cleaned[i]
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
        converter's ``background_weights=True`` form 1/(``VAR_RNOISE`` + ``SKYBKG``/``GAIN``).
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
                "run without background_weights=True). Re-run the converter with "
                "background_weights=True, or use set_weights('ivar').")
        return ratio

    def check_orientation(self, pa_deg=None, pa_is_fpa=False, strict=True):
        """Compare this file's pixel frame with ``galsim.roman``'s (Roman's science frame).

        Piff's RomanOptics evaluates GalSim's field-dependent aberrations at the star's pixel
        position, so the file and GalSim must share the pixel frame. Compares the local WCS
        Jacobian at the image center with that of ``galsim.roman.getWCS`` at the SCA center.
        GalSim's WCS is built at a fixed observable pointing (``psf_fomo.sim.wcs`` default;
        ``getWCS`` may refuse pointings it deems unobservable on its default date): parity does
        not depend on the pointing, the rotation only weakly (< 1° away from the celestial
        poles). A GalSim from before its science-frame update fails the parity check.

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
            ``parity_ok``, ``scale`` (arcsec/pix, √|det J|), ``rotation_deg`` (file − GalSim,
            None without ``pa_deg``).

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
        center = galsim.PositionD((nx + 1) / 2, (ny + 1) / 2)
        j_file = self.wcs.local(center).jacobian().getMatrix()
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
               "scale": scale, "rotation_deg": None}
        if pa_deg is not None:
            out["rotation_deg"] = float(np.degrees(_angle(j_file) - _angle(j_roman)) + 180) % 360 - 180
        problems = []
        if not out["parity_ok"]:
            problems.append(
                f"this file's pixel frame is mirrored relative to galsim.roman's SCA {self.sca} "
                "frame: RomanOptics' field dependence would be wrong. L2 files and galsim.roman "
                "share Roman's science frame only with a GalSim that has its science-frame "
                "update (GalSim main); is the installed GalSim older?")
        if abs(scale / galsim.roman.pixel_scale - 1) > 0.05:
            problems.append(f"pixel scale {scale:.4f}″ differs from "
                            f"galsim.roman.pixel_scale = {galsim.roman.pixel_scale}″")
        for problem in problems:
            if strict:
                raise ValueError(problem)
            warnings.warn(problem)
        return out

    def nonfinite(self):
        """Boolean map of pixels whose image or weight is not finite."""
        return ~(np.isfinite(self.array("image")) & np.isfinite(self.array("ivar")))

    def check_finite(self):
        """Count pixels whose image or weight is not finite.

        Piff gives masked pixels zero weight, but its sums still multiply weight × image, and
        0 × NaN is NaN: one such pixel in a stamp makes that star's flux and center NaN, and
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
            File Piff reads: default this file; `PiffDataFit` passes the cleaned working copy
            when there is one (`write_fit_copy`, same HDU layout).

        Returns
        -------
        dict
            ``image_file_name`` (absolute), ``image_hdu``, ``weight_hdu``, ``badpix_hdu``,
            ``sky`` (only if ≠ 0), ``chipnum`` = SCA, ``properties`` {``sca``}, ``bandpass``.
            The catalog part and the per-star gain (``gain_col``) are added by `PiffDataFit`.
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
                "filter_name": self.filter_name,
                "weights": self.weights, "sky": self.sky,
                "sky_header": self.sky_header, "exptime": self.exptime,
                "wcs_err": self.wcs_err, "hdus": dict(self.hdus),
                "pointing": dict(self.pointing)}

    @classmethod
    def from_config(cls, config):
        """Rebuild from `config` output.

        Settings saved with a flipped working copy (``flip`` set: fits made for GalSim's
        frame before its science-frame update) are refused: such fits do not match the
        GalSim this package now uses.
        """
        if config.get("flip"):
            raise ValueError("this fit was made on a copy flipped into the old galsim.roman "
                             "frame; it does not match GalSim's science frame now in use. "
                             "Refit the image.")
        img = cls(config["path"], config["variant"], config["sca"], config["filter_name"],
                  pointing=config.get("pointing"))
        img.set_weights(config["weights"])
        img.set_sky(config["sky"])
        return img


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
