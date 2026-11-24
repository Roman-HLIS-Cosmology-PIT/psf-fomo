"""Convert calibrated Roman ASDF images (``*_cal.asdf``) to multi-extension FITS.

Adapted from a script by Mike Jarvis. The output is the L2 layout that
`psf_fomo.piff_datafit.L2Image` reads.

Usage::

    from psf_fomo.utils import convert_asdf_to_fits

    convert_asdf_to_fits("r0001_wfi01_f158_cal.asdf", "fits_out", uniform_weights=True)
    convert_asdf_to_fits("https://.../r0001_wfi01_f158_cal.asdf", "fits_out",
                         name="exposure.fits", uniform_weights=True)

Inputs can be local paths, URLs (``http://``, ``https://``, ``s3://``), an open
``asdf.AsdfFile`` or an open ``roman_datamodels`` model. A remote file is downloaded to a
temporary folder (deleted afterwards) unless ``download_dir`` is given. For proprietary MAST
data set the environment variable ``MAST_API_TOKEN``; ``s3://`` access is anonymous unless AWS
credentials are configured.

Output HDUs
-----------
Copied from the ASDF file when present (pixel values not rescaled):

- ``SCI``: the science image (ASDF ``data``), the project's "optimal" ramp slope, with the WCS.
- ``DQ``: data-quality flags.
- ``ERR``: σ of ``SCI`` (read + Poisson noise of sources and background).
- ``VAR_POISSON``, ``VAR_RNOISE``, ``VAR_FLAT``: the project's variance estimates.
- ``CHISQ``: per-pixel ramp-fit χ².
- ``DUMO``: "difference uniform minus optimal"; ``SCI`` + ``DUMO`` is the uniform-weight slope.

Built for Piff:

- ``MASK``: 1 for pixels to mask. DQ bits other than JUMP_DET (4) and UNRELIABLE_SLOPE
  (16777216) are masked (LOW_QE, 8192, is masked: it includes hot-like pixels), as are pixels
  with invalid values (e.g. of the gain).
- ``IVAR``: inverse variance. By default 1/``ERR``², which contains the sources' Poisson noise
  (use Piff's ``remove_signal_from_weight: true``). With ``background_weights=True`` it is
  1/(read variance + ``SKYBKG``/``GAIN``): background only.
- ``GAIN`` (``include_gain=True`` or ``background_weights=True``): ``SCI``/``VAR_POISSON``, the
  effective gain of each pixel (it differs per pixel because of the ramp weighting and jump
  removal).
- ``SCI_UNIF`` (when ``DUMO`` exists): ``SCI`` + ``DUMO``, the uniform-weight slope. On early
  commissioning data with poor guiding the uniform slope works much better than ``SCI``.

With ``uniform_weights=True`` (needs ``DUMO``; implies ``background_weights``):

- ``MASK_UNIF``: keeps most JUMP_DET pixels, but masks isolated JUMP_DET pixels with large
  ``DUMO``.
- ``IVAR_UNIF``: inverse variance of ``SCI_UNIF``: read and background Poisson variance
  recomputed for uniform ramp weights (background only).
- ``GAIN_UNIF``: effective gain of ``SCI_UNIF``.

``SKYBKG`` (``background_weights=True``): one SEP background level per detector, in ``SCI``
units, in every header.

The ASDF WCS (GWCS) is converted to a TAN-SIP FITS WCS with 0.01 pixel accuracy; the largest
residual on a check grid is recorded as ``WCSERR``.

Dependencies (``pip install ".[utils]"``): asdf, asdf-astropy, roman_datamodels, gwcs; sep for
``background_weights`` / ``uniform_weights``; fsspec with aiohttp (http) or s3fs (s3) for
remote files. The ASDF packages must be recent enough to read Roman files.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import parse_qs, urlparse
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, uniform_filter

GOOD_DQ_BITS = np.uint32(4 | 16777216)  # JUMP_DET | UNRELIABLE_SLOPE: not masked
COPIED_PLANES = [("data", "SCI"), ("dq", "DQ"), ("err", "ERR"), ("var_poisson", "VAR_POISSON"),
                 ("var_rnoise", "VAR_RNOISE"), ("var_flat", "VAR_FLAT"), ("chisq", "CHISQ"),
                 ("dumo", "DUMO")]
WCS_TOLERANCE = 0.01  # pixels


def convert_asdf_to_fits(source, output_dir=".", *, name=None, include_wcs=True,
                         include_gain=False, background_weights=False, uniform_weights=False,
                         mesh_size=128, overwrite=False, download_dir=None, verbose=True):
    """Convert one calibrated Roman image to FITS, in a folder.

    Parameters
    ----------
    source : str, pathlib.Path, asdf.AsdfFile or roman_datamodels model
        A ``*_cal.asdf`` path or URL, or an open product (used as is and left open).
    output_dir : str or pathlib.Path
        Output folder, created if needed (relative to the working directory; in Jupyter the
        notebook's folder). Default ``'.'``.
    name : str, optional
        Output file name (its suffix becomes ``.fits``). Default: the input's name
        (``*_cal.asdf`` → ``*_cal.fits``), from the path, URL or ``meta.filename`` of an open
        product.
    include_wcs : bool
        Write the TAN-SIP WCS (checked to 0.01 pixel). Default True.
    include_gain : bool
        Write the ``GAIN`` HDU. Default False (True with ``background_weights``).
    background_weights : bool
        ``IVAR`` without the sources' Poisson noise, and ``SKYBKG``. Needs sep. Default False
        (True with ``uniform_weights``).
    uniform_weights : bool
        Also write ``MASK_UNIF``, ``IVAR_UNIF``, ``GAIN_UNIF`` for ``SCI_UNIF``. Needs sep and
        the ``DUMO`` plane. Default False.
    mesh_size : int
        Tile size in pixels of the uniform noise calibration. Default 128.
    overwrite : bool
        Replace an existing output file. Default False.
    download_dir : str or pathlib.Path, optional
        Keep downloaded files here (an existing copy is reused). Default: a temporary folder.
    verbose : bool
        Print the WCS residual and noise summaries. Default True.

    Returns
    -------
    pathlib.Path
        Absolute path of the FITS file.

    Examples
    --------
    >>> for path in sorted(Path("asdf").glob("*_cal.asdf")):      # doctest: +SKIP
    ...     convert_asdf_to_fits(path, "fits_out", uniform_weights=True)
    """
    input_name = _product_name(source) if _is_open_product(source) else _source_name(source)
    if name is None:
        if input_name is None:
            raise ValueError("this product has no meta.filename; pass name=...")
        name = input_name
    input_name = input_name or Path(name).with_suffix(".asdf").name
    output = Path(output_dir) / Path(name).with_suffix(".fits").name
    background_weights = background_weights or uniform_weights
    include_gain = include_gain or background_weights
    # Fail before a possibly large download if the output cannot be written.
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use overwrite=True")

    with _open_roman(source, download_dir, verbose) as roman:
        if "data" not in roman:
            raise ValueError(f"{input_name} has no science image; use a *_cal.asdf file")
        if uniform_weights and "dumo" not in roman:
            raise ValueError("uniform_weights=True needs the DUMO plane")
        header = _primary_header(roman["meta"], input_name)
        image_header = header.copy()
        if include_wcs:
            wcs_error = _add_fits_wcs(image_header, roman)
            if verbose:
                print(f"{input_name}: maximum sampled WCS error = {wcs_error:.6g} pixel")
        plane_header = image_header.copy()   # header of the copied planes

        error = np.asarray(roman["err"], dtype=np.float32)
        mask, gain = _mask_and_gain(roman, error, include_gain)
        variance = error.astype(float) ** 2
        if background_weights:
            variance, cards = _background_variance(roman, error, mask, gain, input_name, verbose)
            image_header.update(cards)
            header["SKYBKG"] = plane_header["SKYBKG"] = image_header["SKYBKG"]
        image_header["DQGOOD"] = (int(GOOD_DQ_BITS), "DQ bits allowed in MASK/IVAR")
        ivar = np.zeros_like(error)
        np.divide(1.0, variance, out=ivar, where=~mask)

        copied = [_plane_hdu(roman, key, name, plane_header)
                  for key, name in COPIED_PLANES if key in roman]
        # MASK and IVAR follow SCI, DQ, ERR.
        hdus = [fits.PrimaryHDU(header=header), *copied[:3],
                fits.ImageHDU(mask.astype(np.int16), image_header.copy(), name="MASK"),
                fits.ImageHDU(ivar, image_header.copy(), name="IVAR"), *copied[3:]]
        if include_gain:
            hdus.append(_gain_hdu(gain, image_header, roman["meta"]))
        if "dumo" in roman:
            uniform = (np.asarray(roman["data"], dtype=np.float32)
                       + np.asarray(roman["dumo"], dtype=np.float32))
            uniform_header = image_header.copy()
            uniform_header["RAMPFIT"] = ("UNIFORM", "SCI_UNIF = original data + DUMO")
            uniform_header.add_history("ERR, IVAR, GAIN describe the original optimal ramp fit.")
            hdus.append(fits.ImageHDU(uniform, uniform_header, name="SCI_UNIF"))
            if uniform_weights:
                hdus.extend(_uniform_hdus(roman, uniform_header, mask, mesh_size, verbose))
        output.parent.mkdir(parents=True, exist_ok=True)
        with fits.HDUList(hdus) as hdul:
            hdul.writeto(output, overwrite=overwrite, checksum=True)
    return output.resolve()


def _primary_header(meta, input_name):
    header = fits.Header()
    header["TELESCOP"] = meta["telescope"]
    header["INSTRUME"] = meta["instrument"]["name"]
    header["DETECTOR"] = meta["instrument"]["detector"]
    header["FILTER"] = meta["instrument"]["optical_element"]
    header["EXPTIME"] = (meta["exposure"]["exposure_time"], "Exposure time in seconds")
    header["EFFEXPTM"] = meta["exposure"]["effective_exposure_time"]
    pointing = meta.get("pointing")
    if pointing is not None:
        # Telescope attitude, for orienting the PSF (galsim.roman.getPSF boresight /
        # telescope_PA) and for comparing with galsim.roman.getWCS.
        header["RA_V1"] = (float(pointing["ra_v1"]), "[deg] RA of the V1 axis (boresight)")
        header["DEC_V1"] = (float(pointing["dec_v1"]), "[deg] Dec of the V1 axis (boresight)")
        header["PA_V3"] = (float(pointing["pa_v3"]), "[deg] PA of V3 at V1, N through E")
        for key, card, comment in (
                ("target_ra", "TARG_RA", "[deg] RA of the target aperture"),
                ("target_dec", "TARG_DEC", "[deg] Dec of the target aperture"),
                ("pa_aperture", "PA_APER", "[deg] PA of the aperture y axis")):
            if key in pointing:
                header[card] = (float(pointing[key]), comment)
    header["ORIGFILE"] = input_name
    header.add_history("Converted from Roman ASDF; pixel values copied without rescaling.")
    return header


def _add_fits_wcs(header, roman):
    """Add the TAN-SIP approximation of the GWCS to ``header``; return the largest residual."""
    ny, nx = roman["data"].shape
    gwcs = roman["meta"]["wcs"]
    header.update(gwcs.to_fits_sip(bounding_box=((0, nx - 1), (0, ny - 1)),
                                   max_pix_error=WCS_TOLERANCE, max_inv_pix_error=WCS_TOLERANCE))
    # Check on a separate grid that includes the image edges. GWCS uses zero-based pixels,
    # as does astropy with origin=0.
    x, y = np.meshgrid(np.linspace(0, nx - 1, 17), np.linspace(0, ny - 1, 17))
    ra, dec = gwcs(x, y)
    fit_x, fit_y = WCS(header).all_world2pix(ra, dec, 0)
    error = np.max(np.hypot(fit_x - x, fit_y - y))
    if not np.isfinite(error) or error > WCS_TOLERANCE:
        raise ValueError(f"FITS WCS residual {error:.6g} exceeds {WCS_TOLERANCE} pixel")
    header["WCSERR"] = (float(error), "Max pixel residual on 17x17 check grid")
    header.add_history("TAN-SIP approximation to original ASDF GWCS.")
    return error


def _plane_hdu(roman, key, name, header):
    data = np.asarray(roman[key])
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    return fits.ImageHDU(data=data, header=header.copy(), name=name)


def _mask_and_gain(roman, error, include_gain):
    """Boolean pixel mask and (with ``include_gain``) the effective gain SCI/VAR_POISSON."""
    mask = (((np.asarray(roman["dq"]) & ~GOOD_DQ_BITS) != 0)
            | ~np.isfinite(roman["data"]) | ~np.isfinite(error) | (error <= 0))
    if not include_gain:
        return mask, None
    poisson = np.asarray(roman["var_poisson"], dtype=np.float32)
    science = np.asarray(roman["data"])
    gain = np.zeros_like(error)
    valid = np.isfinite(science) & (science > 0) & np.isfinite(poisson) & (poisson > 0)
    # SCI/VAR_POISSON is the optimal-fit effective gain, including the temporal weighting. At
    # jump-affected pixels it can differ greatly from the uniform-ramp gain; do not reuse it
    # for the source noise of SCI_UNIF.
    np.divide(science, poisson, out=gain, where=valid)
    mask |= ~valid | ~np.isfinite(gain) | (gain <= 0)
    return mask, gain


def _background_variance(roman, error, mask, gain, input_name, verbose):
    """Read + background Poisson variance (no source noise); updates ``mask`` in place.

    Returns the variance and the header cards (``SKYBKG``, ``WGTTYPE``, ``RNMETHOD``,
    ``RNVAR``, ``RNREPLAC``).
    """
    import sep
    variance = error.astype(float) ** 2
    poisson = np.asarray(roman["var_poisson"], dtype=np.float32)
    # Same mask and scalar SEP estimator as the detection. SCI and ERR are not changed.
    background_image = np.array(roman["data"], dtype=np.float32, order="C")
    background_image[mask] = 0
    background = float(sep.Background(background_image, mask=mask).globalback)
    if "var_rnoise" in roman:
        read_variance = np.array(roman["var_rnoise"], dtype=float)
        read_source = "VAR_RNOISE"
    else:
        read_variance = np.zeros_like(variance)
        np.subtract(variance, poisson, out=read_variance, where=~mask)
        read_source = "ERR^2-VAR_POISSON"
        if "var_flat" in roman:
            read_variance -= np.asarray(roman["var_flat"])
            read_source = "ERR^2-VAR_POISSON-VAR_FLAT"
    usable = (~mask & (np.asarray(roman["dq"]) == 0)
              & np.isfinite(read_variance) & (read_variance > 0))
    if not usable.any():
        raise ValueError("no usable unflagged pixels to estimate the median read variance")
    median_read = float(np.median(read_variance[usable]))
    # A repair for the optimal-image weights only; the uniform noise model leaves these
    # pixels out of its calibration.
    replace = ~mask & (read_variance < 0)
    read_variance[replace] = median_read
    mask |= ~np.isfinite(read_variance)
    variance = read_variance.copy()
    variance[~mask] += background / gain[~mask]
    if verbose:
        print(f"{input_name}: background={background:.7g}, "
              f"median read variance={median_read:.7g}, "
              f"replaced {replace.sum()} negative read variances")
    # Detection and Piff must subtract the same background these weights use.
    cards = {"SKYBKG": (background, "Scalar SEP background in SCI units"),
             "WGTTYPE": ("BACKGROUND", "IVAR excludes source Poisson noise"),
             "RNMETHOD": (read_source, "Read variance estimate"),
             "RNVAR": (median_read, "Median unflagged read variance"),
             "RNREPLAC": (int(replace.sum()), "Negative read variances replaced")}
    return variance, cards


def _gain_hdu(gain, image_header, meta):
    header = image_header.copy()
    header["GAINMETH"] = ("SCI/VAR_POISSON", "Estimated effective Poisson gain")
    gain_ref = meta.get("ref_file", {}).get("gain")
    if gain_ref is not None:
        header["GAINREF"] = (str(gain_ref), "Detector gain reference")
    return fits.ImageHDU(data=gain, header=header, name="GAIN")


def _uniform_hdus(roman, header, original_mask, mesh_size, verbose):
    """``MASK_UNIF``, ``IVAR_UNIF`` and ``GAIN_UNIF`` for ``SCI_UNIF``."""
    science = np.asarray(roman["data"], dtype=np.float32)
    error = np.asarray(roman["err"], dtype=np.float32)
    poisson = np.asarray(roman["var_poisson"], dtype=np.float32)
    dumo = np.asarray(roman["dumo"], dtype=np.float32)
    uniform = science + dumo
    dq = np.asarray(roman["dq"])
    background = header["SKYBKG"]

    # Poor guiding marks many real stellar cores JUMP_DET, so rejecting that bit would remove
    # the stars. Instead mask a positive DUMO excursion > 10 ERR only when its eight SCI
    # neighbors lack 10σ summed support (the pixel itself and masked neighbors excluded: an
    # isolated spike cannot support itself). An empirical screen, not a calibrated cosmic-ray
    # classifier: it can keep a real jump on a bright source. WARM/LOW_QE stay masked.
    signal = np.where(original_mask, 0, science - background)
    variance = np.where(original_mask, 0, error ** 2)
    neighbor_signal = 9 * uniform_filter(signal, size=3, mode="constant") - signal
    neighbor_variance = 9 * uniform_filter(variance, size=3, mode="constant") - variance
    supported = neighbor_signal > 10 * np.sqrt(np.maximum(neighbor_variance, 0))
    added_mask = ~original_mask & ((dq & 4) != 0) & (dumo > 10 * error) & ~supported
    mask = original_mask | added_mask | ~np.isfinite(uniform)

    sources = _source_mask(science, error, original_mask, background)
    sources |= _source_mask(uniform, error, mask, background)
    sources = binary_dilation(sources, iterations=3)
    # ERR² includes source and background Poisson noise: remove all of it (and the flat
    # variance) to isolate the read variance. Negative estimates are never clipped into the
    # calibration sample: only finite, positive, unflagged background pixels are used, and
    # amplitudes are interpolated elsewhere.
    if "var_rnoise" in roman:
        read = np.asarray(roman["var_rnoise"], dtype=float)
        read_source = "VAR_RNOISE"
    else:
        read = error.astype(float) ** 2 - poisson
        if "var_flat" in roman:
            read -= np.asarray(roman["var_flat"])
        read_source = "ERR^2 minus other variances"
    samples = (~mask & ~sources & (dq == 0) & np.isfinite(science) & (science > 0)
               & np.isfinite(read) & (read > 0) & np.isfinite(poisson) & (poisson > 0))
    exposure = roman["meta"]["exposure"]
    read_times = [np.asarray(t) * exposure["frame_time"] for t in exposure["read_pattern"]]
    read_map, gain, info = _estimate_uniform_noise(science, read, poisson, samples, read_times,
                                                   mesh_size)
    # Add back only the background photon variance: the source noise comes later from the
    # fitted model, not from the noisy pixel values.
    variance = read_map + background / gain
    ivar = np.zeros_like(read_map)
    np.divide(1.0, variance, out=ivar, where=~mask)

    mask_header = header.copy()
    mask_header["DUMOSIG"] = (10.0, "Minimum positive DUMO/ERR for rejection")
    mask_header["SUPPSNR"] = (10.0, "Required SCI neighbor S/N to retain excursion")
    mask_header.add_history("Empirical DUMO screen; no science pixel values replaced.")
    noise_header = header.copy()
    noise_header.update(info)
    noise_header["WGTTYPE"] = ("BACKGROUND", "IVAR excludes source Poisson variance")
    noise_header["RNMETHOD"] = read_source
    noise_header["NOISEMOD"] = "UNFLAGGED_GLS"
    noise_header["SRCDIL"] = (3, "Segmentation exclusion buffer in pixels")
    noise_header.add_history("Uniform noise inferred from clean optimal-fit variances.")
    noise_header.add_history("GAIN_UNIF assumes stationary illumination for source noise.")
    if verbose:
        print(f"Uniform: {added_mask.sum()} additional masked pixels; {info['NSAMP']} "
              f"clean samples; median background variance={np.median(variance[~mask]):.7g}, "
              f"median gain={np.median(gain[~mask]):.7g}", flush=True)
    return [fits.ImageHDU(mask.astype(np.int16), mask_header, name="MASK_UNIF"),
            fits.ImageHDU(ivar, noise_header, name="IVAR_UNIF"),
            fits.ImageHDU(gain, noise_header, name="GAIN_UNIF")]


def _source_mask(science, error, mask, background):
    """Source footprints (SEP segmentation) to keep out of the noise calibration.

    ERR sets a reference threshold here, not the final uniform-ramp noise model. The caller
    uses the union of the optimal and uniform footprints, so jumps and source cores in either
    image stay out of the background sample.
    """
    import sep
    image = np.array(science, dtype=np.float32, order="C")
    noise = np.array(error, dtype=np.float32, order="C")
    image[mask] = 0
    noise[mask] = 1
    image -= background
    sep.set_extract_pixstack(1000000)
    # Without filter_kernel SEP uses its compact built-in 3×3 detection filter.
    _, segmentation = sep.extract(image, 1.5, err=noise, mask=mask, minarea=4,
                                  deblend_nthresh=32, deblend_cont=0.005, filter_type="conv",
                                  segmentation_map=True)
    return segmentation != 0


def _estimate_uniform_noise(science, read_variance, poisson_variance, samples, read_times,
                            mesh_size):
    """Uniform-ramp read variance and effective gain from clean background pixels.

    Parameters
    ----------
    science : numpy.ndarray
        Optimal ramp slope.
    read_variance, poisson_variance : numpy.ndarray
        Variances of the optimal full-ramp fit.
    samples : numpy.ndarray of bool
        Finite, positive, unflagged, source-free pixels used for the calibration.
    read_times : list of numpy.ndarray
        Times [s] of the individual reads, grouped by resultant.
    mesh_size : int
        Tile size in pixels.

    Returns
    -------
    read_map, gain_map : numpy.ndarray
        Uniform-ramp read variance and gain (model / gain = Poisson variance), image shape.
    metadata : dict
        ``MESH``, ``NSAMP`` (calibration pixels), ``UNIRFAC``, ``UNIPFAC`` (temporal factors).

    Notes
    -----
    The covariance of the calibrated resultants is modeled as A R + C P. R is diagonal with
    1/N_i for N_i reads averaged into resultant i; P averages min(t_a, t_b) over read pairs,
    since cumulative Poisson counts share all earlier arrivals. A is the per-read variance,
    C the photon covariance amplitude of the local illumination. The image is a slope, so its
    read and photon variances also carry temporal factors from the slope weights.

    The optimal fitter's separate variance planes identify A and C by inverting the predicted
    variance ratio under full-ramp GLS (this assumes the unflagged optimal fit uses that
    covariance model). For the uniform slope coefficients w (Σw = 0, w · t = 1) the new
    variances are A wᵀRw and C wᵀPw. C is proportional to the illumination, so C/SCI on
    background pixels gives an inverse gain amplitude. The returned gain includes the uniform
    temporal factor: it is not the detector gain in e⁻/DN. Calibration, flat and photometric
    scaling are inherited from the calibrated planes.

    Only clean background pixels are inverted: jump-affected optimal fits can use different
    read subsets, so their variances cannot be propagated this way. Robust tile medians and
    interpolation give the amplitudes at sources and flagged pixels. This is an inferred
    noise model, not a rerun of the calibration.
    """
    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    nreads = np.array([len(t) for t in read_times])
    times = np.array([np.mean(t) for t in read_times])
    read_covariance = np.diag(1.0 / nreads)
    photon_covariance = np.array([[np.minimum(a[:, None], b[None, :]).mean() for b in read_times]
                                  for a in read_times])
    total_reads = nreads.sum()
    nt = nreads @ times
    # Equal per-read weights become N_i weights for averaged resultants. These slope
    # coefficients cancel a constant intercept and give unit response to t.
    uniform = (total_reads * nreads * times - nt * nreads) / (
        total_reads * (nreads @ times ** 2) - nt ** 2)
    uniform_read = uniform @ read_covariance @ uniform
    uniform_photon = uniform @ photon_covariance @ uniform

    # Differencing removes the intercept. At each amplitude ratio q = C/A, GLS predicts
    # VAR_POISSON/VAR_READ = q × photon_factor(q)/read_factor(q). Tabulate this monotonic
    # relation to invert it for every background pixel.
    difference = np.diff(np.eye(len(times)), axis=0) / np.diff(times)[:, None]
    read_diff = difference @ read_covariance @ difference.T
    photon_diff = difference @ photon_covariance @ difference.T
    log_ratio = np.linspace(-8 * np.log(10), 8 * np.log(10), 4097)
    ratio = np.exp(log_ratio)
    covariance = read_diff[None, :, :] + ratio[:, None, None] * photon_diff[None, :, :]
    weights = np.linalg.solve(covariance, np.ones((len(ratio), len(times) - 1, 1)))[..., 0]
    weights /= weights.sum(axis=1)[:, None]
    read_factor = np.einsum("ni,ij,nj->n", weights, read_diff, weights)
    photon_factor = np.einsum("ni,ij,nj->n", weights, photon_diff, weights)
    predicted_ratio = np.log(ratio * photon_factor / read_factor)

    ny, nx = science.shape
    x_edges = list(range(0, nx, mesh_size)) + [nx]
    y_edges = list(range(0, ny, mesh_size)) + [ny]
    x_centers = (np.array(x_edges[:-1]) + np.array(x_edges[1:]) - 1) / 2.0
    y_centers = (np.array(y_edges[:-1]) + np.array(y_edges[1:]) - 1) / 2.0
    mesh = np.empty((len(y_centers), len(x_centers), 2))
    for j, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        for i, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            region = np.s_[y0:y1, x0:x1]
            use = samples[region]
            if not np.any(use):
                raise ValueError(f"no clean background pixels in tile ({i}, {j}); "
                                 "increase mesh_size")
            read = read_variance[region][use]
            photon = poisson_variance[region][use]
            inferred_ratio = np.interp(np.log(photon / read), predicted_ratio, log_ratio)
            read_amplitude = read / np.interp(inferred_ratio, log_ratio, read_factor)
            photon_amplitude = photon / np.interp(inferred_ratio, log_ratio, photon_factor)
            mesh[j, i, 0] = np.median(read_amplitude)
            mesh[j, i, 1] = np.median(photon_amplitude) / np.median(science[region][use])

    # Interpolate the inverse gain, so the interpolation is linear in the noise amplitudes.
    # Clip at the outer tile centers to avoid extrapolating to negative variances.
    interpolate = RegularGridInterpolator((y_centers, x_centers), mesh)
    read_map = np.empty(science.shape, dtype=np.float32)
    gain_map = np.empty(science.shape, dtype=np.float32)
    x = np.clip(np.arange(nx), x_centers[0], x_centers[-1])
    for start in range(0, ny, mesh_size):
        end = min(start + mesh_size, ny)
        y = np.clip(np.arange(start, end), y_centers[0], y_centers[-1])
        xx, yy = np.meshgrid(x, y)
        amplitudes = interpolate(np.stack([yy, xx], axis=-1))
        read_map[start:end] = amplitudes[:, :, 0] * uniform_read
        gain_map[start:end] = 1.0 / (amplitudes[:, :, 1] * uniform_photon)
    metadata = {"MESH": mesh_size, "NSAMP": int(np.count_nonzero(samples)),
                "UNIRFAC": float(uniform_read), "UNIPFAC": float(uniform_photon)}
    return read_map, gain_map, metadata


def _is_remote(source):
    """True for URLs such as https://… or s3://…, False for local paths."""
    scheme = urlparse(str(source)).scheme
    return len(scheme) > 1 and scheme != "file"   # len > 1 keeps Windows 'C:' paths local


def _source_name(source):
    """File name of a local path or URL.

    MAST download links look like …/Download/file?uri=mast:ROMAN/…/name_cal.asdf, so the
    name comes from the ``uri`` query parameter when there is one.
    """
    if not _is_remote(source):
        return Path(source).name
    parsed = urlparse(str(source))
    uri = parse_qs(parsed.query).get("uri")
    path = uri[0] if uri else parsed.path
    name = path.rstrip("/").rsplit("/", 1)[-1]
    if not name:
        raise ValueError(f"cannot determine a file name from {source}")
    return name


def _remote_options(source):
    """fsspec options: the MAST token for stsci.edu, anonymous S3 without credentials."""
    parsed = urlparse(str(source))
    if parsed.scheme in ("http", "https"):
        token = os.environ.get("MAST_API_TOKEN")
        if token and (parsed.hostname or "").endswith("stsci.edu"):
            return {"client_kwargs": {"headers": {"Authorization": f"token {token}"}}}
    elif parsed.scheme == "s3":
        have_credentials = any(key in os.environ for key in (
            "AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_SHARED_CREDENTIALS_FILE"))
        return {"anon": not have_credentials}
    return {}


@contextmanager
def _local_copy(source, download_dir=None, verbose=True):
    """A local path for ``source``, downloading it first if it is a URL.

    Without ``download_dir`` the download goes to a temporary folder deleted afterwards;
    with it, the file is kept and an existing copy is reused.
    """
    if not _is_remote(source):
        yield Path(source)
        return
    import fsspec
    temporary = None
    if download_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="asdf_to_fits_")
        target_dir = Path(temporary.name)
    else:
        target_dir = Path(download_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _source_name(source)
    try:
        if target.exists():
            if verbose:
                print(f"Using existing download {target}", flush=True)
        else:
            if verbose:
                print(f"Downloading {source}", flush=True)
            partial = target.with_name(target.name + ".part")
            with fsspec.open(str(source), "rb", **_remote_options(source)) as remote, \
                    open(partial, "wb") as local:
                shutil.copyfileobj(remote, local, length=16 * 1024 * 1024)
            # Rename only after a complete download, so an interrupted run never leaves a
            # truncated file that a later run would reuse.
            partial.replace(target)
        yield target
    finally:
        if temporary is not None:
            temporary.cleanup()


def _is_open_product(source):
    """True for an open asdf.AsdfFile or roman_datamodels model."""
    if hasattr(source, "_instance"):
        return True
    import asdf
    return isinstance(source, asdf.AsdfFile)


def _product_name(source):
    """File name in an open product's metadata, or None."""
    import asdf
    roman = source["roman"] if isinstance(source, asdf.AsdfFile) else source._instance
    name = roman["meta"].get("filename")
    # roman_datamodels uses '?' for unset strings.
    return str(name) if name and str(name) != "?" else None


@contextmanager
def _open_roman(source, download_dir=None, verbose=True):
    """The ``roman`` tree of a path, URL, open AsdfFile or roman_datamodels model.

    An open product is used as is and left open (the caller owns it). A path or URL is
    opened here (downloaded first if needed) and closed afterwards.
    """
    import asdf
    from asdf.exceptions import AsdfConversionWarning
    if isinstance(source, asdf.AsdfFile):
        yield source["roman"]
    elif hasattr(source, "_instance"):
        # A roman_datamodels model wraps the same node as af['roman'] of a file on disk, so
        # the conversion is identical.
        yield source._instance
    else:
        # Refuse unrecognized tags, but allow package-version provenance warnings when the
        # installed readers can deserialize every object.
        with _local_copy(source, download_dir, verbose) as local_path, \
                warnings.catch_warnings():
            warnings.simplefilter("error", AsdfConversionWarning)
            with asdf.open(local_path) as af:
                yield af["roman"]
