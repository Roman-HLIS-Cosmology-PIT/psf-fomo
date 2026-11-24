"""Plots of a detection catalog: detections over the image, and size vs magnitude.

Usage::

    from psf_fomo.utils import detect_sources, plot_detections, plot_detection_sizes

    catalog = detect_sources(fits_path, metadata=asdf_path)
    stars = detect_sources(fits_path, gaia=True, metadata=asdf_path)

    plot_detections(fits_path, catalog, stars)                                 # whole SCA
    plot_detections(fits_path, catalog, stars, region=(1000, 1400, 2000, 2400))  # zoom
    plot_detection_sizes(catalog, stars)

Either catalog may be omitted. A catalog from ``detect_sources(..., gaia=True)`` passed alone
as ``catalog`` is drawn as Gaia matches.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from astropy.io import fits
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

DETECTION_COLOR = "royalblue"
GAIA_COLOR = "darkorange"
MASK_COLOR = "mediumseagreen"
APERTURE_MAX_WIDTH = 1000   # draw AUTO ellipses when the shown region is at most this wide


def plot_detections(exposure, catalog=None, stars=None, *, region=None, apertures=None,
                    ax=None, title=None, show_mask=True):
    """The sky-subtracted ``SCI_UNIF`` image with detections and Gaia matches.

    Parameters
    ----------
    exposure : str or pathlib.Path
        Converted FITS file (`convert_asdf` with ``uniform_weights=True``).
    catalog : astropy.table.Table, optional
        Detections from `detect_sources` (blue).
    stars : astropy.table.Table, optional
        Gaia-matched detections from ``detect_sources(..., gaia=True)`` (orange circles).
    region : tuple (xmin, xmax, ymin, ymax), optional
        Zero-based pixel range to show. Default: the whole image. The display stretch is
        computed from this region only.
    apertures : bool, optional
        Draw each detection's AUTO aperture ellipse instead of a circle. Default: True when
        the region is at most 1000 pixels wide.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default: a new figure.
    title : str, optional
        Default: the file name.
    show_mask : bool
        Paint masked (``MASK_UNIF``) pixels in one color. Default True.

    Returns
    -------
    matplotlib.axes.Axes
    """
    catalog, stars = _split(catalog, stars)
    image, name = _sky_subtracted_image(exposure)
    ny, nx = image.shape
    xmin, xmax, ymin, ymax = region if region is not None else (0, nx - 1, 0, ny - 1)
    if apertures is None:
        apertures = (xmax - xmin) <= APERTURE_MAX_WIDTH
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8 * (ymax - ymin + 1) / (xmax - xmin + 1) + 0.9),
                             constrained_layout=True)

    view = image[max(int(ymin), 0):int(ymax) + 1, max(int(xmin), 0):int(xmax) + 1]
    finite = view[np.isfinite(view)]
    norm = ImageNormalize(finite if finite.size else np.zeros(1),
                          interval=ZScaleInterval(), stretch=AsinhStretch(0.1))
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad(MASK_COLOR if show_mask else "black")
    # Pixel i covers [i − 0.5, i + 0.5], so catalog positions land on pixel centers.
    ax.imshow(image, origin="lower", cmap=cmap, norm=norm, interpolation="nearest",
              extent=(-0.5, nx - 0.5, -0.5, ny - 0.5))

    handles = []
    if catalog is not None and len(catalog):
        x, y = _column(catalog, "x"), _column(catalog, "y")
        if apertures:
            a = np.nan_to_num(_column(catalog, "auto_a_pix"), nan=1.5)
            b = np.nan_to_num(_column(catalog, "auto_b_pix"), nan=1.5)
            theta = np.degrees(np.nan_to_num(_column(catalog, "theta")))
            ax.add_collection(EllipseCollection(
                2 * a, 2 * b, theta, units="xy", offsets=np.column_stack([x, y]),
                offset_transform=ax.transData, facecolors="none",
                edgecolors=DETECTION_COLOR, linewidths=1.0))
            handles.append(Line2D([], [], color=DETECTION_COLOR, lw=1.0,
                                  label=f"SEP detections, AUTO apertures ({len(catalog):,})"))
        else:
            ax.scatter(x, y, s=28, facecolors="none", edgecolors=DETECTION_COLOR,
                       linewidths=0.9)
            handles.append(Line2D([], [], color=DETECTION_COLOR, marker="o", ls="",
                                  markersize=5.5, markerfacecolor="none", markeredgewidth=0.9,
                                  label=f"SEP detections ({len(catalog):,})"))
    if stars is not None and len(stars):
        ax.scatter(_column(stars, "x"), _column(stars, "y"), s=90, facecolors="none",
                   edgecolors=GAIA_COLOR, linewidths=1.4)
        handles.append(Line2D([], [], color=GAIA_COLOR, marker="o", ls="", markersize=9,
                              markerfacecolor="none", markeredgewidth=1.4,
                              label=f"Gaia matches ({len(stars):,})"))
    if show_mask and np.isnan(image).any():
        handles.append(Patch(color=MASK_COLOR, label="Masked pixels"))

    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    ax.set_title(title or name.rsplit("/", 1)[-1], fontsize=11)
    if handles:
        # Below the image, so it never hides sources.
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08),
                  ncol=len(handles), fontsize=8, frameon=False)
    return ax


def plot_detection_sizes(catalog=None, stars=None, *, ax=None, title=None, ylim=(0, 10)):
    """Half-light radius ``r50`` vs AUTO magnitude, Gaia matches highlighted.

    Uses ``mag_auto_ab`` when present (`detect_sources` with ``metadata``), otherwise the
    instrumental magnitude −2.5 log10(``flux_auto`` [DN/s]). Gaia matches that pass
    ``clean_gaia_star`` are filled, other matches open. (For moments measured on the
    stamps, see `psf_fomo.piff_datafit.plot_size_mag`.)

    Parameters
    ----------
    catalog : astropy.table.Table, optional
        Detections from `detect_sources`.
    stars : astropy.table.Table, optional
        Gaia-matched detections from ``detect_sources(..., gaia=True)``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Default: a new figure.
    title : str, optional
        Default: ``'Size–magnitude'`` and the input file name.
    ylim : tuple or None
        ``r50`` range, pixels. Default (0, 10); None = automatic.

    Returns
    -------
    matplotlib.axes.Axes
    """
    catalog, stars = _split(catalog, stars)
    reference = catalog if catalog is not None else stars
    if reference is None:
        raise ValueError("pass a detection catalog, a Gaia-matched catalog, or both")
    use_ab = "mag_auto_ab" in reference.colnames

    def magnitude(table):
        if use_ab:
            return _column(table, "mag_auto_ab")
        flux = _column(table, "flux_auto")
        out = np.full(len(table), np.nan)
        out[flux > 0] = -2.5 * np.log10(flux[flux > 0])
        return out

    def usable(table):
        mag, r50 = magnitude(table), _column(table, "r50")
        good = np.isfinite(mag) & np.isfinite(r50) & (r50 > 0)
        return mag[good], r50[good], good

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.grid(True, color="0.88", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if catalog is not None:
        mag, r50, _ = usable(catalog)
        ax.scatter(mag, r50, s=8, color=DETECTION_COLOR, alpha=0.45, linewidths=0,
                   label=f"SEP detections ({len(mag):,})", zorder=2)
    if stars is not None:
        mag, r50, good = usable(stars)
        clean = (np.asarray(stars["clean_gaia_star"], dtype=bool)[good]
                 if "clean_gaia_star" in stars.colnames else np.ones(len(mag), bool))
        ax.scatter(mag[clean], r50[clean], s=30, color=GAIA_COLOR, edgecolors="white",
                   linewidths=0.6, label=f"Gaia matches, clean ({clean.sum():,})", zorder=3)
        if (~clean).any():
            ax.scatter(mag[~clean], r50[~clean], s=30, facecolors="none",
                       edgecolors=GAIA_COLOR, linewidths=1.1,
                       label=f"Gaia matches, not clean ({(~clean).sum():,})", zorder=3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("AUTO AB magnitude" if use_ab
                  else "AUTO instrumental magnitude, −2.5 log₁₀(flux [DN/s])")
    ax.set_ylabel("Half-light radius r50 [pixel]")
    source = reference.meta.get("INFILE", "")
    default_title = "Size–magnitude" + (f": {source.rsplit('/', 1)[-1]}" if source else "")
    ax.set_title(title or default_title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    return ax


def _split(catalog, stars):
    """(detections, Gaia matches); a Gaia-matched catalog alone counts as matches."""
    if stars is None and catalog is not None and "gaia_match" in catalog.colnames:
        return None, catalog
    return catalog, stars


def _column(table, name, fill=np.nan):
    return np.asarray(np.ma.asarray(table[name], dtype=float).filled(fill))


def _sky_subtracted_image(exposure):
    """``SCI_UNIF`` − ``SKYBKG`` (masked pixels NaN) and the file name."""
    with fits.open(exposure, memmap=False) as hdus:
        image = np.array(hdus["SCI_UNIF"].data, dtype=np.float32)
        mask = hdus["MASK_UNIF"].data != 0
        sky = float(hdus["SCI_UNIF"].header.get("SKYBKG", 0.0))
        name = hdus.filename()
    image -= sky
    image[mask | ~np.isfinite(image)] = np.nan
    return image, str(name)
