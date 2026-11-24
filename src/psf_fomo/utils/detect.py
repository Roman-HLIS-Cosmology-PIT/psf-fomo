"""SEP detection catalog of one converted Roman exposure, optionally only Gaia stars.

The input is a FITS file written by `psf_fomo.utils.convert_asdf` with
``uniform_weights=True`` (HDUs ``SCI``, ``MASK``, ``SCI_UNIF``, ``MASK_UNIF``, ``IVAR_UNIF``).
Detection and photometry run on the sky-subtracted ``SCI_UNIF``, with ``MASK_UNIF`` as bad
pixels and the background-only ``IVAR_UNIF`` as the noise. The catalog has the columns
`psf_fomo.piff_datafit.StarCatalog` reads.

Usage::

    from psf_fomo.utils import detect_sources

    catalog = detect_sources("fits_out/r0001_wfi01_f158_cal.fits")              # everything
    stars = detect_sources("fits_out/r0001_wfi01_f158_cal.fits", gaia=True,     # Gaia only
                           metadata="r0001_wfi01_f158_cal.asdf")

Gaia matching
-------------
With ``gaia=True`` only detections matched to a Gaia source are returned. The Gaia catalog
is given (``gaia_catalog``: a file or Table with ``source_id``, ``ra``, ``dec`` and optionally
``pmra``, ``pmdec``, ``ref_epoch``, …) or queried from the Gaia archive over the image
footprint (needs astroquery and network access).

Gaia positions are moved with their proper motion to the observation time, taken from
``obs_time``, else from ``metadata`` (a roman_datamodels model, AsdfFile or ASDF path), else
from ``DATE-OBS`` / ``MJD-OBS`` in the FITS header. Without any of these the match is done at
the Gaia reference epoch, which can lose fast-moving stars.

Matching uses mutual nearest neighbors within ``match_radius_arcsec``: each Gaia source
matches at most one detection and vice versa. ``clean_gaia_star`` also requires clean SEP
photometry and no competing candidate within the radius. A positional match is not a
PSF-star selection: no magnitude, saturation, RUWE or morphology cuts are applied.

Units and noise
---------------
Fluxes are sums of ``SCI_UNIF`` pixels (DN/s for L2 data). ``flux_auto_err_bkg`` and
``snr_auto_bkg`` include background and read noise only, not the source's Poisson noise. If
``metadata`` has the photometric calibration, ``flux_auto_jy`` and ``mag_auto_ab`` are added
(``conversion_megajanskys`` and the nominal ``pixel_area``; no pixel-area map or aperture
correction), and the header gets ``ABZP``.

Dependencies (``pip install ".[utils]"``): sep ≥ 1.4; asdf / roman_datamodels for metadata
from files; astroquery for Gaia queries.
"""
from contextlib import nullcontext
from pathlib import Path
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.io import fits
from astropy.table import MaskedColumn, Table
from astropy.time import Time
from astropy.wcs import WCS
from erfa import ErfaWarning

# Same estimator as the converter's SKYBKG (SEP defaults with an explicit mask).
DEFAULT_BACKGROUND_CONFIG = dict(bw=64, bh=64, fw=3, fh=3, fthresh=0.0, maskthresh=0.0)
DEFAULT_DETECTION_CONFIG = dict(thresh=3.0, minarea=3, deblend_nthresh=64, deblend_cont=0.001,
                                filter_type="conv")
GAIA_QUERY_COLUMNS = ("source_id", "ra", "dec", "pmra", "pmdec", "ref_epoch", "parallax",
                      "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "ruwe")
REQUIRED_HDUS = ("SCI", "MASK", "SCI_UNIF", "MASK_UNIF", "IVAR_UNIF")


def detect_sources(exposure, *, gaia=False, gaia_catalog=None, metadata=None, obs_time=None,
                   match_radius_arcsec=0.11, detection_config=None, background_config=None,
                   kron_factor=2.5, min_radius=1.75, gaia_margin_arcsec=10.0, verbose=True):
    """Detection catalog of one converted exposure, optionally only the Gaia-matched stars.

    Parameters
    ----------
    exposure : str, pathlib.Path or astropy.io.fits.HDUList
        FITS file from `convert_asdf` with ``uniform_weights=True``.
    gaia : bool
        Cross-match with Gaia and return only the matched detections. Default False.
    gaia_catalog : str, pathlib.Path or astropy.table.Table, optional
        Gaia catalog to match against. Default (with ``gaia=True``): query the Gaia archive
        over the image footprint.
    metadata : roman_datamodels model, asdf.AsdfFile, str or pathlib.Path, optional
        The original exposure: observation time for the proper motion, and the calibration
        for ``flux_auto_jy`` / ``mag_auto_ab``.
    obs_time : astropy.time.Time or str, optional
        Observation time (UTC); overrides ``metadata`` and the FITS header.
    match_radius_arcsec : float
        Largest detection–Gaia separation of a match, arcsec. Default 0.11.
    detection_config, background_config : dict, optional
        Overrides of `DEFAULT_DETECTION_CONFIG` (``sep.extract``) and
        `DEFAULT_BACKGROUND_CONFIG` (``sep.Background``, for the sky consistency check).
    kron_factor : float
        Kron-radius multiplier of the elliptical AUTO aperture. Default 2.5.
    min_radius : float
        Circular fallback aperture radius, pixels, for small or invalid Kron apertures.
        Default 1.75.
    gaia_margin_arcsec : float
        Radius added to the image footprint for the Gaia query, arcsec. Default 10.
    verbose : bool
        Print a one-line summary. Default True.

    Returns
    -------
    astropy.table.Table
        Every SEP detection, or with ``gaia=True`` the matched ones. Columns: see
        `measure_sources` and `match_gaia`.
    """
    loaded = load_exposure(exposure, background_config)
    catalog, _ = measure_sources(loaded, detection_config, kron_factor, min_radius)

    info = read_metadata(metadata) if metadata is not None else {}
    if info.get("conversion_megajanskys") and info.get("pixel_area_sr"):
        catalog = add_ab_magnitudes(catalog, info["conversion_megajanskys"],
                                    info["pixel_area_sr"])
    elif metadata is not None:
        warnings.warn("the metadata have no valid photometric calibration; "
                      "AB magnitudes not added")

    name = Path(loaded["name"]).name
    if not gaia:
        if verbose:
            print(f"{name}: {len(catalog):,} detections "
                  f"({int(np.sum(catalog['clean'])):,} clean)")
        return catalog

    when = obs_time
    if when is None:
        when = info.get("time")
    if when is None:
        when = _header_time(loaded["header"])
    if when is not None and not isinstance(when, Time):
        when = Time(when, scale="utc")
    if when is None:
        warnings.warn("no observation time (pass obs_time or metadata): Gaia positions are "
                      "not moved with their proper motion, so fast-moving stars may not match")
    if gaia_catalog is None:
        gaia_table = query_gaia(loaded["header"], loaded["image"].shape, gaia_margin_arcsec)
        source = "Gaia archive (gaiadr3.gaia_source cone over the footprint)"
    else:
        gaia_table = read_gaia(gaia_catalog)
        source = "user Table" if isinstance(gaia_catalog, Table) else str(gaia_catalog)
    if not len(gaia_table):
        warnings.warn("the Gaia catalog has no valid sources; the result is empty")

    matched = match_gaia(catalog, gaia_table, when, match_radius_arcsec)
    matched.meta["GAIASRC"] = source
    stars = matched[matched["gaia_match"]]
    if verbose:
        print(f"{name}: {len(catalog):,} detections; {len(gaia_table):,} Gaia sources; "
              f"{len(stars):,} matched ({int(np.sum(stars['clean_gaia_star'])):,} "
              "clean_gaia_star)")
    return stars


def load_exposure(exposure, background_config=None):
    """Read a converted exposure: the sky-subtracted uniform-weight image and its noise.

    Parameters
    ----------
    exposure : str, pathlib.Path or astropy.io.fits.HDUList
        FITS file from `convert_asdf` with ``uniform_weights=True``.
    background_config : dict, optional
        Overrides of `DEFAULT_BACKGROUND_CONFIG`, for re-deriving the sky as a check.

    Returns
    -------
    dict
        ``image`` (``SCI_UNIF`` − ``SKYBKG``, DN/s, masked pixels 0), ``variance``
        (1/``IVAR_UNIF``, masked pixels 1), ``mask`` (bool, True = bad), ``header`` (primary
        + ``SCI_UNIF``), ``name``, ``sky`` (stored), ``sky_recomputed``,
        ``background_config``.

    Notes
    -----
    ``SKYBKG`` was measured on the optimal ``SCI`` image and is subtracted from ``SCI_UNIF``,
    as in the converter. The sky is re-derived from ``SCI`` and ``MASK``; a mismatch only
    warns, because the stored value is the one the weights used.
    """
    import sep
    background_config = dict(DEFAULT_BACKGROUND_CONFIG, **(background_config or {}))
    opener = (nullcontext(exposure) if isinstance(exposure, fits.HDUList)
              else fits.open(exposure, memmap=False))
    with opener as hdus:
        missing = [name for name in REQUIRED_HDUS if name not in hdus]
        if missing:
            raise ValueError(f"missing extensions {missing}; convert with "
                             "convert_asdf(..., uniform_weights=True)")
        original = _native_float(hdus["SCI"].data)
        original_mask = np.ascontiguousarray(hdus["MASK"].data != 0)
        image = _native_float(hdus["SCI_UNIF"].data)
        ivar = _native_float(hdus["IVAR_UNIF"].data)
        bad = np.ascontiguousarray(hdus["MASK_UNIF"].data != 0)
        header = hdus[0].header.copy()
        header.update(hdus["SCI_UNIF"].header)
        name = hdus.filename() or "<in-memory HDUList>"

    if any(a.shape != image.shape for a in (original, original_mask, ivar, bad)):
        raise ValueError("image, noise and mask shapes differ")
    if header.get("WGTTYPE") != "BACKGROUND":
        raise ValueError("expected background-only IVAR_UNIF weights (WGTTYPE = BACKGROUND)")
    stored_sky = float(header["SKYBKG"])
    if not np.isfinite(stored_sky):
        raise ValueError("SKYBKG must be finite")

    original_mask |= ~np.isfinite(original)
    original[original_mask] = 0.0
    recomputed_sky = float(sep.Background(original, mask=original_mask,
                                          **background_config).globalback)
    if not np.isclose(recomputed_sky, stored_sky, rtol=1e-5, atol=1e-7):
        warnings.warn(f"recomputed sky {recomputed_sky:.9g} differs from SKYBKG "
                      f"{stored_sky:.9g}; the converter may have used another version or "
                      "mask. Using SKYBKG.")

    bad |= ~np.isfinite(image) | ~np.isfinite(ivar) | (ivar <= 0)
    if np.all(bad):
        raise ValueError("no valid pixels remain")
    variance = np.zeros_like(image)
    np.divide(1.0, ivar, out=variance, where=~bad)
    bad |= ~np.isfinite(variance) | (variance <= 0)
    variance[bad] = 1.0   # finite placeholder; these pixels stay masked
    image -= stored_sky
    image[bad] = 0.0
    return dict(image=image, variance=variance, mask=bad, header=header, name=str(name),
                sky=stored_sky, sky_recomputed=recomputed_sky,
                background_config=background_config)


def measure_sources(exposure, detection_config=None, kron_factor=2.5, min_radius=1.75):
    """Detect sources and measure the masked AUTO (Kron) flux and half-light radius.

    Parameters
    ----------
    exposure : dict
        `load_exposure` output.
    detection_config : dict, optional
        Overrides of `DEFAULT_DETECTION_CONFIG` (``sep.extract``); ``thresh`` is in units of
        the background σ.
    kron_factor : float
        Kron-radius multiplier of the elliptical AUTO aperture. Default 2.5.
    min_radius : float
        Circular fallback aperture radius, pixels. Default 1.75.

    Returns
    -------
    catalog : astropy.table.Table
        Every detection: SEP's columns (``x``, ``y`` zero-based pixels, ``a``, ``b``,
        ``theta``, …; SEP's ``flag`` as ``extract_flag``), ``id`` (1…N), ``kron_radius``,
        ``flux_auto`` [DN/s], ``flux_auto_err_bkg``, ``snr_auto_bkg``,
        ``auto_masked_fraction``, ``r50`` [pix], the flags of each step, ``ra_deg``,
        ``dec_deg`` and the conservative boolean ``clean``.
    segmentation : numpy.ndarray
        SEP segmentation map (0 = background, else the detection ``id``).
    """
    import sep
    detection_config = dict(DEFAULT_DETECTION_CONFIG, **(detection_config or {}))
    image, variance, bad_pixels = exposure["image"], exposure["variance"], exposure["mask"]
    sep.set_extract_pixstack(1_000_000)
    objects, segmentation = sep.extract(image, var=variance, mask=bad_pixels,
                                        segmentation_map=True, **detection_config)
    segmentation = np.ascontiguousarray(segmentation, dtype=np.int32)
    catalog = Table(objects)
    catalog.rename_column("flag", "extract_flag")
    n_sources = len(catalog)
    ids = np.arange(1, n_sources + 1, dtype=np.int32)
    catalog["id"] = ids

    x, y, semi_major, semi_minor, theta = (np.asarray(catalog[k], dtype=float)
                                           for k in ("x", "y", "a", "b", "theta"))
    # SEP's float32 angles can round just beyond ±π/2, which its aperture code rejects.
    theta = np.clip(theta, -np.pi / 2, np.pi / 2)
    catalog["theta"] = theta
    valid_shape = (np.isfinite(x) & np.isfinite(y) & np.isfinite(semi_major)
                   & np.isfinite(semi_minor) & np.isfinite(theta)
                   & (semi_major >= semi_minor) & (semi_minor > 0))
    index = np.flatnonzero(valid_shape)
    kron_radius = np.full(n_sources, np.nan)
    kron_flag = np.zeros(n_sources, dtype=np.int16)
    if len(index):
        kron_radius[index], kron_flag[index] = sep.kron_radius(
            image, x[index], y[index], semi_major[index], semi_minor[index], theta[index], 6.0,
            mask=bad_pixels, segmap=segmentation, seg_id=ids[index])
    kron_invalid = ~valid_shape | ~np.isfinite(kron_radius) | (kron_radius <= 0)
    use_circle = valid_shape & (kron_invalid
                                | (kron_radius * np.sqrt(semi_major * semi_minor) < min_radius))
    aperture_a = np.where(use_circle, min_radius, kron_factor * kron_radius * semi_major)
    aperture_b = np.where(use_circle, min_radius, kron_factor * kron_radius * semi_minor)

    flux = np.full(n_sources, np.nan)
    flux_error = np.full(n_sources, np.nan)
    auto_flag = np.zeros(n_sources, dtype=np.int16)
    masked_fraction = np.full(n_sources, np.nan)
    ones = np.ones_like(image)
    bad_values = _native_float(bad_pixels)
    for circular in (False, True):
        sel = np.flatnonzero(valid_shape & (use_circle == circular))
        if not len(sel):
            continue
        geometry = (x[sel], y[sel], semi_major[sel], semi_minor[sel], theta[sel],
                    kron_factor * kron_radius[sel], circular, min_radius)
        flux[sel], flux_error[sel], auto_flag[sel] = _aperture_sums(
            image, *geometry, var=variance, mask=bad_pixels, segmap=segmentation,
            seg_id=ids[sel], subpix=5)
        # Geometric fraction of the in-image aperture covered by bad pixels.
        area, _, _ = _aperture_sums(ones, *geometry, subpix=5)
        bad_area, _, _ = _aperture_sums(bad_values, *geometry, subpix=5)
        masked_fraction[sel] = np.divide(bad_area, area, out=np.full(len(sel), np.nan),
                                         where=area > 0)

    r50 = np.full(n_sources, np.nan)
    r50_flag = np.zeros(n_sources, dtype=np.int16)
    rmax = np.maximum(6.0 * semi_major, aperture_a)
    sel = np.flatnonzero(valid_shape & np.isfinite(flux) & (flux > 0))
    if len(sel):
        r50[sel], r50_flag[sel] = sep.flux_radius(
            image, x[sel], y[sel], rmax[sel], 0.5, normflux=flux[sel], mask=bad_pixels,
            segmap=segmentation, seg_id=ids[sel], subpix=5)

    for name, values in dict(
            shape_valid=valid_shape, kron_radius=kron_radius, kron_flag=kron_flag,
            kron_invalid=kron_invalid, auto_is_circle=use_circle, auto_a_pix=aperture_a,
            auto_b_pix=aperture_b, flux_auto=flux, flux_auto_err_bkg=flux_error,
            auto_flag=auto_flag, auto_masked_fraction=masked_fraction, r50=r50,
            r50_flag=r50_flag, r50_rmax_pix=rmax).items():
        catalog[name] = values
    catalog["snr_auto_bkg"] = np.divide(flux, flux_error, out=np.full(n_sources, np.nan),
                                        where=flux_error > 0)
    ra, dec = WCS(exposure["header"]).all_pix2world(x, y, 0)
    catalog["ra_deg"] = ra
    catalog["ra_deg"].unit = "deg"
    catalog["dec_deg"] = dec
    catalog["dec_deg"].unit = "deg"
    catalog["clean"] = (
        valid_shape & ~kron_invalid & (catalog["extract_flag"] == 0) & (kron_flag == 0)
        & (auto_flag == 0) & (r50_flag == 0) & (masked_fraction == 0)
        & np.isfinite(flux) & (flux > 0) & np.isfinite(flux_error) & (flux_error > 0)
        & np.isfinite(r50) & (r50 > 0) & (r50 < rmax))
    catalog.meta.update(
        INFILE=exposure["name"], SCIHDU="SCI_UNIF", NOISEHDU="IVAR_UNIF", MASKHDU="MASK_UNIF",
        SKY=exposure["sky"], SKYRECMP=exposure["sky_recomputed"],
        BKGCFG=str(exposure["background_config"]), DETCFG=str(detection_config),
        KRONFACT=kron_factor, MINRAD=min_radius,
        ERRTYPE="Background/read noise only; excludes source Poisson noise",
        SEPVER=sep.__version__)
    return catalog, segmentation


def read_metadata(source):
    """Observation start time and photometric calibration of the original exposure.

    Parameters
    ----------
    source : roman_datamodels model, asdf.AsdfFile, str or pathlib.Path
        The original calibrated exposure (e.g. ``rdm.open(...)``) or its ``*_cal.asdf`` path.

    Returns
    -------
    dict
        ``time`` (astropy Time), ``conversion_megajanskys`` (MJy/sr per DN/s) and
        ``pixel_area_sr``; missing or invalid values are None.
    """
    if hasattr(source, "_instance"):   # roman_datamodels model
        return _metadata(source._instance)
    import asdf
    if isinstance(source, asdf.AsdfFile):
        return _metadata(source["roman"])
    with asdf.open(source, lazy_load=True) as af:
        return _metadata(af["roman"])


def add_ab_magnitudes(catalog, conversion_megajanskys, pixel_area_sr):
    """Add ``flux_auto_jy`` and ``mag_auto_ab`` from ``flux_auto`` [DN/s].

    Parameters
    ----------
    catalog : astropy.table.Table
        Catalog with ``flux_auto``.
    conversion_megajanskys : float
        MJy/sr per DN/s.
    pixel_area_sr : float
        Nominal pixel area, steradians.

    Returns
    -------
    astropy.table.Table
        A copy with the two columns and header ``ABZP`` (``mag_auto_ab`` = ``ABZP`` −
        2.5 log10 ``flux_auto``), ``PHOTMJSR``, ``PIXAREA``, ``MAGFORM``.
    """
    result = catalog.copy()
    jy_per_flux_unit = conversion_megajanskys * pixel_area_sr * 1e6
    flux_jy = _float_column(result, "flux_auto") * jy_per_flux_unit
    magnitude = np.full(len(result), np.nan)
    positive = np.isfinite(flux_jy) & (flux_jy > 0)
    magnitude[positive] = -2.5 * np.log10(flux_jy[positive] / 3631.0)
    result["flux_auto_jy"] = MaskedColumn(flux_jy, unit="Jy")
    result["mag_auto_ab"] = MaskedColumn(magnitude, unit="mag")
    result.meta.update(PHOTMJSR=conversion_megajanskys, PIXAREA=pixel_area_sr,
                       ABZP=-2.5 * np.log10(jy_per_flux_unit / 3631.0),
                       MAGFORM="mag_auto_ab = ABZP - 2.5*log10(flux_auto)")
    return result


def read_gaia(source):
    """Read a Gaia catalog from a file (CSV, ECSV, FITS, …) or a Table.

    Rows without a valid position or ``source_id`` are dropped; ``source_id`` must be unique
    and is stored as a 64-bit integer.
    """
    if isinstance(source, Table):
        return _validate_gaia(source)
    path = Path(source)
    kwargs = {"format": "ascii.csv"} if path.suffix.lower() == ".csv" else {}
    return _validate_gaia(Table.read(path, **kwargs))


def query_gaia(header, shape, margin_arcsec=10.0, table="gaiadr3.gaia_source"):
    """Query the Gaia archive for every source over the image footprint.

    Parameters
    ----------
    header : astropy.io.fits.Header
        Header with the image WCS.
    shape : tuple of int
        Image shape (ny, nx).
    margin_arcsec : float
        Added to the radius of the circle enclosing the image, arcsec (also room for proper
        motion between the Gaia epoch and the observation). Default 10.
    table : str
        Gaia archive table. Default ``'gaiadr3.gaia_source'``.

    Returns
    -------
    astropy.table.Table
        Columns `GAIA_QUERY_COLUMNS`.
    """
    from astroquery.gaia import Gaia
    center, radius = _image_footprint(header, shape)
    adql = _gaia_cone_adql(center, radius + margin_arcsec * u.arcsec, table)
    return _validate_gaia(Gaia.launch_job_async(adql).get_results())


def match_gaia(catalog, gaia, when, radius_arcsec):
    """Add Gaia association columns to every detection (mutual nearest neighbors).

    Parameters
    ----------
    catalog : astropy.table.Table
        `measure_sources` output (``ra_deg``, ``dec_deg``, ``clean``).
    gaia : astropy.table.Table
        `read_gaia` / `query_gaia` output.
    when : astropy.time.Time or None
        Observation time; None matches at the Gaia reference epoch.
    radius_arcsec : float
        Match radius, arcsec.

    Returns
    -------
    astropy.table.Table
        A copy with ``gaia_match``, ``gaia_sep_arcsec``, ``gaia_n_candidates``,
        ``gaia_ambiguous``, ``gaia_pm_applied``, ``gaia_<column>`` (masked when unmatched),
        the moved Gaia position ``gaia_ra_match_deg``, ``gaia_dec_match_deg``, and
        ``clean_gaia_star``.
    """
    if not np.isfinite(radius_arcsec) or radius_arcsec <= 0:
        raise ValueError("the match radius must be finite and positive")
    cat = catalog.copy()
    n = len(cat)
    match = np.zeros(n, dtype=bool)
    nearest = np.zeros(n, dtype=int)
    separation = np.full(n, np.nan)
    n_candidates = np.zeros(n, dtype=int)
    ambiguous = np.zeros(n, dtype=bool)
    pm_applied = np.zeros(len(gaia), dtype=bool)
    if len(gaia):
        gaia_sky, pm_applied = _gaia_at_time(gaia, when)
        ra, dec = _float_column(cat, "ra_deg"), _float_column(cat, "dec_deg")
        rows = np.flatnonzero(np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90))
        if len(rows):
            detections = SkyCoord(ra[rows] * u.deg, dec[rows] * u.deg)
            forward, distance, _ = detections.match_to_catalog_sky(gaia_sky)
            reverse, _, _ = gaia_sky.match_to_catalog_sky(detections)
            match[rows] = ((distance.arcsec <= radius_arcsec)
                           & (reverse[forward] == np.arange(len(rows))))
            nearest[rows] = forward
            separation[rows] = distance.arcsec
            di, gi, _, _ = search_around_sky(detections, gaia_sky, radius_arcsec * u.arcsec)
            det_count = np.bincount(di, minlength=len(rows))
            gaia_count = np.bincount(gi, minlength=len(gaia))
            n_candidates[rows] = det_count
            ambiguous[rows] = (det_count > 0) & ((det_count > 1) | (gaia_count[forward] > 1))

    cat["gaia_match"] = match
    cat["gaia_n_candidates"] = n_candidates
    cat["gaia_ambiguous"] = ambiguous
    cat["gaia_sep_arcsec"] = MaskedColumn(separation, mask=~match, unit="arcsec")
    cat["gaia_pm_applied"] = match & pm_applied[nearest] if len(gaia) else match
    if len(gaia):
        for name in GAIA_QUERY_COLUMNS:
            if name in gaia.colnames:
                values = gaia[name][nearest]
                cat["gaia_" + name] = MaskedColumn(values,
                                                   mask=~match | np.ma.getmaskarray(values))
        cat["gaia_ra_match_deg"] = MaskedColumn(gaia_sky.ra.deg[nearest], mask=~match,
                                                unit="deg")
        cat["gaia_dec_match_deg"] = MaskedColumn(gaia_sky.dec.deg[nearest], mask=~match,
                                                 unit="deg")
    cat["clean_gaia_star"] = match & np.asarray(cat["clean"]) & ~ambiguous
    cat.meta.update(OBSTIME=when.utc.isot if when is not None else "NONE (Gaia reference epoch)",
                    MATCHRAD=float(radius_arcsec), MATCHMTH="mutual nearest neighbors",
                    GAIAPM="proper motion only; no distance/RV/parallax",
                    NGAIA=len(gaia), NDETALL=n, NMATCH=int(match.sum()))
    return cat


def _native_float(array):
    """Native-endian, C-contiguous float32 copy, as SEP needs."""
    return np.array(array, dtype=np.float32, order="C", copy=True)


def _float_column(table, name):
    """A numeric column as a float array, masked entries NaN."""
    return np.asarray(np.ma.asarray(table[name], dtype=float).filled(np.nan))


def _aperture_sums(data, x, y, semi_major, semi_minor, theta, kron_scale, circular, radius,
                   **kwargs):
    """Circular (fixed radius) or elliptical (scaled Kron) aperture sums with SEP."""
    import sep
    if circular:
        return sep.sum_circle(data, x, y, radius, **kwargs)
    return sep.sum_ellipse(data, x, y, semi_major, semi_minor, theta, kron_scale, **kwargs)


def _metadata(roman):
    meta = roman["meta"]
    exposure = meta.get("exposure", {})
    photometry = meta.get("photometry", {}) or {}
    start = exposure.get("start_time")
    result = dict(time=Time(start) if start is not None else None)
    for key, name in (("conversion_megajanskys", "conversion_megajanskys"),
                      ("pixel_area", "pixel_area_sr")):
        try:
            value = float(photometry.get(key))
        except (TypeError, ValueError):
            value = np.nan
        result[name] = value if np.isfinite(value) and value > 0 else None
    return result


def _header_time(header):
    """Observation time from ``DATE-OBS`` or ``MJD-OBS``, or None."""
    if header.get("DATE-OBS"):
        return Time(header["DATE-OBS"], scale="utc")
    if header.get("MJD-OBS") is not None:
        return Time(float(header["MJD-OBS"]), format="mjd", scale="utc")
    return None


def _validate_gaia(gaia):
    gaia = Table(gaia, copy=True)
    for name in list(gaia.colnames):   # archive results can be upper case (SOURCE_ID)
        if name != name.lower() and name.lower() not in gaia.colnames:
            gaia.rename_column(name, name.lower())
    missing = {"source_id", "ra", "dec"} - set(gaia.colnames)
    if missing:
        raise ValueError(f"the Gaia catalog is missing columns {sorted(missing)}")
    ra, dec = _float_column(gaia, "ra"), _float_column(gaia, "dec")
    valid = np.isfinite(ra) & np.isfinite(dec) & (ra >= 0) & (ra < 360) & (np.abs(dec) <= 90)
    valid &= ~np.ma.getmaskarray(gaia["source_id"])
    gaia = gaia[valid]
    # Never store these 19-digit identifiers as floats.
    gaia["source_id"] = np.asarray(gaia["source_id"], dtype=np.int64)
    if len(np.unique(gaia["source_id"])) != len(gaia):
        raise ValueError("Gaia source_id must be unique")
    return gaia


def _image_footprint(header, shape):
    """Image center and the radius of the circle through its farthest corner."""
    ny, nx = shape
    wcs = WCS(header)
    center = SkyCoord(*wcs.all_pix2world((nx - 1) / 2, (ny - 1) / 2, 0), unit="deg")
    corners = SkyCoord(*wcs.all_pix2world([0, nx - 1, 0, nx - 1], [0, 0, ny - 1, ny - 1], 0),
                       unit="deg")
    return center, center.separation(corners).max()


def _gaia_cone_adql(center, radius, table):
    return (f"SELECT {', '.join(GAIA_QUERY_COLUMNS)} FROM {table} "
            f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {center.ra.deg:.10f}, {center.dec.deg:.10f}, "
            f"{radius.to_value(u.deg):.10f}))")


def _gaia_at_time(gaia, when):
    """Gaia positions moved with their proper motion to ``when`` (None: reference epoch).

    Returns the positions and a boolean array of rows whose proper motion was applied.
    ``ref_epoch`` is a Julian year (TCB); ``pmra`` includes cos(dec). Distance, radial
    velocity and parallax are not used.
    """
    ra, dec = _float_column(gaia, "ra"), _float_column(gaia, "dec")
    moved_ra, moved_dec = ra.copy(), dec.copy()
    usable = np.zeros(len(gaia), dtype=bool)
    if when is not None and {"pmra", "pmdec", "ref_epoch"} <= set(gaia.colnames):
        pmra, pmdec, epoch = (_float_column(gaia, k) for k in ("pmra", "pmdec", "ref_epoch"))
        usable = np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(epoch)
        if usable.any():
            reference = SkyCoord(
                ra=ra[usable] * u.deg, dec=dec[usable] * u.deg,
                pm_ra_cosdec=pmra[usable] * u.mas / u.yr, pm_dec=pmdec[usable] * u.mas / u.yr,
                obstime=Time(epoch[usable], format="jyear", scale="tcb"), frame="icrs")
            with warnings.catch_warnings():
                # Expected for an angular-only propagation without a distance.
                warnings.filterwarnings("ignore", category=ErfaWarning,
                                        message=".*pmsafe.*distance overridden.*")
                moved = reference.apply_space_motion(new_obstime=when)
            moved_ra[usable], moved_dec[usable] = moved.ra.deg, moved.dec.deg
    return SkyCoord(moved_ra * u.deg, moved_dec * u.deg), usable
