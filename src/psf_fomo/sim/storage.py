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
- the SED file, if a user SED was used.
"""
from astropy.io import fits
from astropy.table import Table
import galsim


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
