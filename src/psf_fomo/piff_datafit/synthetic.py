"""Synthetic files in the L2 layout, to try the real-data workflow without real data.

- `simulation_to_l2`: one SCA of a `Simulation` as an L2 file (DN/s, sky, Poisson and read
  noise, every plane of the layout) plus a SEP-style detection catalog of its stars, in a
  temporary folder (`SyntheticL2`) or a folder of your choice.
- `write_l2_file`: any image in DN/s written in the L2 layout (``docs/piff_datafit.md``).
- `write_sep_catalog`: a detection catalog shaped like SEP output (0-based x, y; HDU 1).

The noise model is deliberately simple: Poisson noise of (signal + sky) × gain and Gaussian
read noise, one gain and one sky level per detector. No detector effects beyond those of the
simulation itself.

Example
-------
>>> from psf_fomo.sim import RomanSimulation
>>> from psf_fomo.piff_datafit import L2Image, StarCatalog
>>> from psf_fomo.piff_datafit.synthetic import simulation_to_l2
>>> s = RomanSimulation(scas=[5], n_stars=100, seed=1); s.set_wcs("roman")
>>> sim = s.run(save_noiseless=True)
>>> synth = simulation_to_l2(sim, sca=5)          # temporary files, removed automatically
>>> img, cat = L2Image(synth.image_file), StarCatalog(synth.catalog_file)
"""
import tempfile
from pathlib import Path

import numpy as np
import galsim
import galsim.roman
from astropy.io import fits
from astropy.table import Table



class SyntheticL2:
    """An L2 image file and a detection catalog file made by `simulation_to_l2`.

    Attributes
    ----------
    image_file, catalog_file : Path
        Pass them to `L2Image` and `StarCatalog`.
    folder : Path
        Where they are. A temporary folder (no ``folder`` given) is removed with `cleanup`,
        when this object is deleted, or when Python exits; keep the object while the files
        are in use (`L2Image` reads its file lazily).
    """

    def __init__(self, image_file, catalog_file, tmpdir=None):
        self.image_file = Path(image_file)
        self.catalog_file = Path(catalog_file)
        self.folder = self.image_file.parent
        self._tmpdir = tmpdir

    @property
    def temporary(self):
        """True if the files are in a temporary folder."""
        return self._tmpdir is not None

    def cleanup(self):
        """Remove a temporary folder now (nothing happens for a folder you chose)."""
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __repr__(self):
        kind = "temporary" if self.temporary else "kept"
        return f"SyntheticL2({kind}: {self.image_file.name}, {self.catalog_file.name})"


def simulation_to_l2(sim, sca, folder=None, obs=0, sky=0.6, seed=0, filter_name=None):
    """One SCA of a simulation as an L2 file and a SEP-style detection catalog.

    The simulation's noiseless image (``run(save_noiseless=True)``) is converted to DN/s
    (÷ ``galsim.roman.exptime``), the sky is added, and Poisson noise with the gain
    ``galsim.roman.gain × exptime`` (e⁻ per DN/s) and Gaussian read noise
    (``galsim.roman.read_noise``) are drawn. Catalog fluxes are the true fluxes in DN/s.

    `L2Image` and Piff read FITS files, so the data are written to disk; by default to a
    temporary folder that is removed automatically (see `SyntheticL2`).

    Parameters
    ----------
    sim : Simulation
        Simulation run with ``save_noiseless=True``, with the Roman WCS (``set_wcs('roman')``)
        so that the file has a celestial WCS as real L2 files do.
    sca : int
        Which SCA of the simulation.
    folder : str or Path, optional
        Keep the files in this folder (created if needed): ``l2_scaNN.fits``,
        ``det_scaNN.fits``. Default None: a temporary folder.
    obs : int
        Which observation. Default 0.
    sky : float
        Sky level, DN/s per pixel. Default 0.6.
    seed : int
        Seed of the noise. Default 0.
    filter_name : str, optional
        ``FILTER`` header value. Default: the simulation's filter in Roman's naming
        (H158 → F158).

    Returns
    -------
    SyntheticL2
        ``image_file``, ``catalog_file`` (and ``cleanup()`` for a temporary folder).
    """
    if sim.noiseless is None:
        raise ValueError("the simulation has no noiseless images: run(save_noiseless=True)")
    tmpdir = None
    if folder is None:
        tmpdir = tempfile.TemporaryDirectory(prefix="psf_fomo_l2_")
        folder = tmpdir.name
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    exptime = galsim.roman.exptime
    gain = galsim.roman.gain * exptime                     # e⁻ per DN/s
    read_var = (galsim.roman.read_noise / gain) ** 2       # (DN/s)²

    noiseless = sim.noiseless[obs, sca]                    # ADU per exposure
    expected = noiseless.array / exptime + sky             # DN/s
    rng = np.random.default_rng(seed)
    sci = (rng.poisson(np.clip(expected, 0, None) * gain) / gain
           + rng.normal(0, np.sqrt(read_var), expected.shape))
    if filter_name is None:
        filter_name = "F" + sim.config["filter"][1:]

    # galsim.roman's WCS (and so the simulation's image) is in Roman's science frame, like
    # real L2 files: written as it is.
    image_file = folder / f"l2_sca{sca:02d}.fits"
    write_l2_file(image_file, sci, noiseless.wcs, sca=sca, filter_name=filter_name, sky=sky,
                  read_var=read_var, gain=gain)

    truth = sim.catalogs[obs]
    truth = truth[truth["sca"] == sca]
    x, y = np.asarray(truth["x"], dtype=float), np.asarray(truth["y"], dtype=float)
    flux = np.asarray(truth["flux"], dtype=float) / galsim.roman.gain / exptime   # DN/s
    objects = [{"id": int(i) + 1, "x": float(xi), "y": float(yi), "flux": float(f),
                "is_star": True, "r50": 1.3}
               for i, xi, yi, f in zip(truth["star_id"], x, y, flux)]
    catalog_file = write_sep_catalog(folder / f"det_sca{sca:02d}.fits", objects, sky=sky,
                                     flux_scale=1.0)
    return SyntheticL2(image_file, catalog_file, tmpdir)


def write_l2_file(path, sci, wcs, sca=5, filter_name="F158", sky=2.0, read_var=0.01,
                  gain=250.0, mask=None, dumo=None, uniform=True, background_weights=True,
                  primary=None, planes="all"):
    """Write an image (DN/s, sky included) in the L2 FITS layout.

    Parameters
    ----------
    path : str or Path
        Output file (overwritten).
    sci : numpy.ndarray
        Image in DN/s, sky included.
    wcs : galsim.BaseWCS
        WCS written into the image headers.
    sca : int
        Detector number (``DETECTOR = WFInn``). Default 5.
    filter_name : str
        ``FILTER`` header value (Roman naming, e.g. ``'F158'``). Default ``'F158'``.
    sky, read_var, gain : float
        Sky (DN/s per pixel, ``SKYBKG``), read variance ((DN/s)²), gain (e⁻ per DN/s).
        Defaults 2.0, 0.01, 250.
    mask, dumo : numpy.ndarray, optional
        Bad-pixel mask (1 = bad) and the uniform-weight correction ``DUMO``. Default zeros.
    uniform : bool
        Write ``MASK_UNIF``, ``IVAR_UNIF``, ``GAIN_UNIF``. Default True.
    background_weights : bool
        ``IVAR`` = 1/(``VAR_RNOISE`` + ``SKYBKG``/``GAIN``) if True (default), else 1/``ERR``²
        (contains the stars' Poisson noise).
    primary : dict, optional
        PRIMARY header cards replacing the defaults (``DETECTOR``, ``FILTER``, ``EXPTIME``);
        a value None removes the card. Use it for the pointing (``RA_V1``, ``DEC_V1``,
        ``PA_V3``, degrees).
    planes : {'all', 'unif'}
        ``'all'`` (default): every plane of the layout. ``'unif'``: only ``SCI_UNIF``,
        ``MASK_UNIF``, ``IVAR_UNIF``, ``GAIN_UNIF`` (what the default ``variant='unif'``
        reads; smaller files).

    Returns
    -------
    Path
    """
    sci = np.asarray(sci, dtype=np.float32)
    mask = np.zeros(sci.shape, np.uint8) if mask is None else mask.astype(np.uint8)
    dumo = np.zeros_like(sci) if dumo is None else dumo.astype(np.float32)
    gain_map = np.full(sci.shape, gain, np.float32)
    var_rnoise = np.full(sci.shape, read_var, np.float32)
    var_poisson = (np.clip(sci, 0, None) / gain).astype(np.float32)
    err = np.sqrt(var_rnoise + var_poisson)
    ivar = (1 / (var_rnoise + sky / gain_map) if background_weights else 1 / err**2)

    header = galsim.FitsHeader()
    ny, nx = sci.shape
    wcs.writeToFitsHeader(header, galsim.BoundsI(1, nx, 1, ny))
    wcs_cards = header.header

    cards = {"DETECTOR": f"WFI{sca:02d}", "FILTER": filter_name,
             "EXPTIME": galsim.roman.exptime}
    cards.update(primary or {})
    hdus = [fits.PrimaryHDU()]
    for key, value in cards.items():
        if value is not None:
            hdus[0].header[key] = value

    def image_hdu(name, data, with_wcs=False):
        h = fits.ImageHDU(np.asarray(data), name=name)
        if with_wcs:
            h.header.extend(wcs_cards, unique=True)
        return h

    if planes == "all":
        hdus += [image_hdu("SCI", sci, True), image_hdu("DQ", mask.astype(np.int32)),
                 image_hdu("ERR", err), image_hdu("MASK", mask),
                 image_hdu("IVAR", ivar.astype(np.float32)),
                 image_hdu("VAR_POISSON", var_poisson), image_hdu("VAR_RNOISE", var_rnoise),
                 image_hdu("GAIN", gain_map), image_hdu("DUMO", dumo)]
    hdus += [image_hdu("SCI_UNIF", sci + dumo, True)]
    if uniform or planes == "unif":
        hdus += [image_hdu("MASK_UNIF", mask),
                 image_hdu("IVAR_UNIF", (1 / (var_rnoise + sky / gain_map)).astype(np.float32)),
                 image_hdu("GAIN_UNIF", gain_map)]
    for h in hdus:
        h.header["SKYBKG"] = sky
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return Path(path)


def write_sep_catalog(path, objects, abzp=26.0, sky=2.0, flux_scale=0.95, unclean=()):
    """Write a detection catalog shaped like SEP output (0-based x, y; HDU 1).

    Parameters
    ----------
    path : str or Path
        Output file (overwritten).
    objects : list of dict
        ``id`` (1…N), ``x``, ``y`` (1-based, GalSim convention), ``flux`` [DN/s], ``r50``
        [pix], ``is_star`` (bool).
    abzp : float
        AB zeropoint written as ``ABZP`` (magnitudes = ``ABZP`` − 2.5 log10 flux). Default 26.
    sky : float
        ``SKY`` header value, DN/s. Default 2.0.
    flux_scale : float
        ``flux_auto`` = flux × ``flux_scale`` (an aperture flux misses some light). Default
        0.95.
    unclean : sequence of int
        Ids written with ``clean = False``.

    Returns
    -------
    Path
        Columns ``id``, ``x``, ``y`` (0-based), ``flux_auto`` [DN/s], ``snr_auto_bkg``,
        ``r50`` [pix], ``clean``, ``is_star`` (1/0); header ``ABZP``, ``SKY``.
    """
    t = Table({"id": [o["id"] for o in objects],
               "x": [o["x"] - 1 for o in objects], "y": [o["y"] - 1 for o in objects],
               "flux_auto": [o["flux"] * flux_scale for o in objects],
               "snr_auto_bkg": [o["flux"] / 2.0 for o in objects],
               "r50": [o["r50"] for o in objects],
               "clean": [o["id"] not in set(unclean) for o in objects],
               "is_star": [int(o["is_star"]) for o in objects]})
    t.meta.update({"ABZP": abzp, "SKY": sky})
    t.write(path, overwrite=True)
    return Path(path)

