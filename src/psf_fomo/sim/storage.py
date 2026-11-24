"""Saving and loading simulation files.

Layout of a saved simulation folder:

- ``obs_000.fits``: one image HDU per SCA (``EXTNAME = 'SCA01'`` …, in the order of the
  configured SCAs, starting at HDU 1), in ADU, with GalSim WCS keywords, ``GAIN`` (e⁻/ADU)
  and ``BKGVAR`` (median background variance, ADU²). Same layout for the optional
  ``noiseless_000.fits``.
- ``catalog_000.fits``: HDU 1 (``'STARS'``) is the full star table of the observation, with
  an ``sca`` column. With the simple WCS, per-SCA subsets follow as ``'SCA01'`` … HDUs,
  because Piff's file input pairs one catalog with each image and does not filter by chip.
- ``config.yaml``: the simulation configuration and provenance.
- for chromatic simulations, the SED: ``sed.fits`` (FITS table, ``WAVE`` in nm, ``FLUX`` in
  photons per nm, over the filter's wavelength range) for the flat SED or an SED given as a
  `galsim.SED`; a copy of the user's file for an SED given as a file. Piff's file input reads
  it (``sed_file_name``).
"""
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table
import galsim

SED_FILE = "sed.fits"
SED_STEP_NM = 0.5   # wavelength sampling of a tabulated SED


def image_file_name(obs, noiseless=False):
    """File name of the images of one observation."""
    return f"{'noiseless' if noiseless else 'obs'}_{obs:03d}.fits"


def catalog_file_name(obs):
    """File name of the star catalog of one observation."""
    return f"catalog_{obs:03d}.fits"


def sca_extname(sca):
    """HDU name used for an SCA, e.g. ``'SCA05'``."""
    return f"SCA{sca:02d}"


def write_images(path, images, extra_headers):
    """Write one observation's SCA images to a multi-extension FITS file.

    Parameters
    ----------
    path : Path
        Output file name.
    images : dict
        SCA number → galsim.Image, written in insertion order.
    extra_headers : dict
        SCA number → dict of extra header keywords.
    """
    hdus = [fits.PrimaryHDU()]
    for sca, image in images.items():
        header = galsim.FitsHeader()
        image.wcs.writeToFitsHeader(header, image.bounds)
        for key, value in extra_headers[sca].items():
            header[key] = value
        hdus.append(fits.ImageHDU(image.array, header=header.header, name=sca_extname(sca)))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def read_images(path, scas):
    """Read the SCA images written by `write_images`.

    Parameters
    ----------
    path : Path
        Input file name.
    scas : sequence of int
        SCA numbers, in the order they were written.

    Returns
    -------
    dict
        SCA number → galsim.Image (with WCS).
    """
    return {sca: galsim.fits.read(str(path), hdu=i + 1) for i, sca in enumerate(scas)}


def write_catalog(path, table, per_sca_hdus):
    """Write one observation's star table.

    Parameters
    ----------
    path : Path
        Output file name.
    table : astropy.table.Table
        Full star table with an ``sca`` column.
    per_sca_hdus : bool
        If True, also write each SCA's rows as its own HDU.
    """
    hdus = [fits.PrimaryHDU(), fits.table_to_hdu(table)]
    hdus[1].name = "STARS"
    if per_sca_hdus:
        for sca in sorted(set(table["sca"])):
            hdu = fits.table_to_hdu(table[table["sca"] == sca])
            hdu.name = sca_extname(sca)
            hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def read_catalog(path):
    """Read the full star table (HDU ``'STARS'``) of one observation."""
    return Table.read(path, hdu="STARS")


def sed_setting(sed="flat", bandpass=None, wave_type="nm", flux_type="fphotons"):
    """A plain-dict description of an SED (stored in configurations), or None.

    Parameters
    ----------
    sed : 'flat', str, galsim.SED or None
        ``'flat'``: constant photons per unit wavelength. A file name: read by ``galsim.SED``
        with ``wave_type`` and ``flux_type``. A spectral ``galsim.SED``: sampled every 0.5 nm
        over the bandpass (``{'sed': 'table', 'wave_nm': …, 'fphotons': …}``). None: None.
    bandpass : galsim.Bandpass
        Needed for a ``galsim.SED``.
    wave_type, flux_type : str
        Units of an SED file, as for ``galsim.SED``.
    """
    if sed is None:
        return None
    if isinstance(sed, galsim.SED):
        wave, flux = sample_sed(sed, bandpass)
        return {"sed": "table", "wave_nm": wave.tolist(), "fphotons": flux.tolist()}
    return {"sed": str(sed), "wave_type": wave_type, "flux_type": flux_type}


def sed_from_setting(setting):
    """The ``galsim.SED`` described by `sed_setting` output (not normalized)."""
    if setting["sed"] == "flat":
        return galsim.SED("1", wave_type="nm", flux_type="fphotons")
    if setting["sed"] == "table":
        table = galsim.LookupTable(setting["wave_nm"], setting["fphotons"],
                                   interpolant="linear")
        return galsim.SED(table, wave_type="nm", flux_type="fphotons")
    return galsim.SED(setting["sed"], wave_type=setting["wave_type"],
                      flux_type=setting["flux_type"])


def sample_sed(sed, bandpass, step=SED_STEP_NM):
    """An SED sampled over a bandpass's wavelength range.

    Parameters
    ----------
    sed : galsim.SED
        A spectral SED (not a dimensionless one).
    bandpass : galsim.Bandpass
        Its ``blue_limit`` … ``red_limit`` (nm) set the range.
    step : float
        Sampling in nm. Default 0.5.

    Returns
    -------
    wave, flux : numpy.ndarray
        Wavelengths in nm and the SED in photons per nm (``fphotons``).
    """
    if not sed.spectral:
        raise ValueError("the SED must be spectral (a flux density), not dimensionless")
    wave = np.arange(bandpass.blue_limit, bandpass.red_limit, step)
    wave = np.append(wave, bandpass.red_limit)      # both limits included
    # A constant SED may evaluate to a single number: give it the shape of the wavelengths.
    return wave, np.broadcast_to(sed(wave), wave.shape).astype(float)


def write_sed(path, wave, flux):
    """Write an SED as a FITS table: ``WAVE`` [nm], ``FLUX`` [photons per nm]."""
    Table({"WAVE": np.asarray(wave, dtype=float),
           "FLUX": np.asarray(flux, dtype=float)}).write(path, overwrite=True)


def piff_sed_input(setting, user_file=None):
    """Piff ``input`` entries giving every star one SED, or {} for None.

    For ``'flat'`` and ``'table'`` the file is `SED_FILE` (write it with `write_sed`), named
    relative: Piff looks for it next to the catalog.

    Parameters
    ----------
    setting : dict or None
        `sed_setting` output (e.g. a simulation's ``config['chromatic']``).
    user_file : str, optional
        File name to give Piff for an SED read from a file. Default: its name alone (as
        copied next to a saved simulation's catalogs).
    """
    if setting is None:
        return {}
    if setting["sed"] in ("flat", "table"):
        return {"sed_file_name": SED_FILE, "sed_wave_type": "nm", "sed_flux_type": "fphotons"}
    return {"sed_file_name": user_file or Path(setting["sed"]).name,
            "sed_wave_type": setting["wave_type"], "sed_flux_type": setting["flux_type"]}
