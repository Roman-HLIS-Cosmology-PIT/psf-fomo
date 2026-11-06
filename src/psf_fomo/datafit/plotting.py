"""Plots specific to real-data fits. The fit plots are shared with simulations
(`psf_fomo.piffval.plotting`, called through `DataFitResult` methods)."""
import numpy as np
import matplotlib.pyplot as plt

# Detections stay recessive gray; the two selection stages take the first two categorical
# hues (blue, orange; colour-blind separation checked). Marker shape and size differ too, so
# the stages are not told apart by colour alone.
STAGES = (
    ("detection", dict(color="0.75", marker=".", s=6, lw=0)),
    ("candidate", dict(facecolor="none", edgecolor="#eb6834", marker="o", s=28, lw=1.0)),
    ("star", dict(color="#2a78d6", marker="D", s=10, lw=0)),
)
LABELS = {"detection": "detections (stamp on image)", "candidate": "selected by the star finder",
          "star": "PSF stars (after cuts)"}
SIZE_LABELS = {"T": "T = Mxx + Myy [pix²]", "sigma": "σ = det(M)^¼ [pix]"}


def plot_size_mag(table, algorithm=None, size="T", mag="catalog", ax=None):
    """Size vs magnitude of every detection, marked by selection outcome when known.

    Parameters
    ----------
    table : astropy.table.Table
        `StarCatalog.measure` output, or `DataFitResult.selection`. Selection columns
        (``on_image``, ``candidate``, ``star``) are used when present.
    algorithm : str, optional
        Algorithm label of a measurement (``'gauss'``, ``'wmom_0.4'``, ...): size from
        ``<size>_<algorithm>``, flagged measurements left out (counted in the title).
        Default None = the catalog's own size (``size_cat``, e.g. SEP ``r50``).
    size : {'T', 'sigma'}
        Measured size to plot (with ``algorithm``). Default ``'T'`` (pix²).
    mag : {'catalog', 'measured'}
        ``'catalog'``: ``mag`` (or −2.5 log10 ``flux_cat`` without a zero point);
        ``'measured'``: ``mag_<algorithm>``. Default ``'catalog'``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if size not in SIZE_LABELS:
        raise ValueError(f"size must be one of {list(SIZE_LABELS)}")
    if mag not in ("catalog", "measured"):
        raise ValueError("mag must be 'catalog' or 'measured'")
    good = np.ones(len(table), dtype=bool)
    if algorithm is None:
        if mag == "measured":
            raise ValueError("mag='measured' needs an algorithm")
        y, ylabel = _column(table, "size_cat"), "catalog size [pix]"
    else:
        y = _column(table, f"{size}_{algorithm}")
        ylabel = f"{algorithm}: {SIZE_LABELS[size]}"
        good = np.asarray(table[f"flag_{algorithm}"]) == 0
    x, xlabel = _magnitudes(table, algorithm, mag)
    masks = {name: (np.asarray(table[col], dtype=bool) if col in table.colnames
                    else (np.ones(len(table), dtype=bool) if name == "detection" else None))
             for name, col in (("detection", "on_image"), ("candidate", "candidate"),
                               ("star", "star"))}
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
    shown = good & np.isfinite(x) & np.isfinite(y) & (y > 0)
    for stage, style in STAGES:
        if masks[stage] is None:
            continue
        m = masks[stage] & shown
        ax.scatter(x[m], y[m], label=f"{LABELS[stage]}: {int(m.sum())}", **style)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    n_flagged = int((~good & masks["detection"]).sum())
    title = f"{int(masks['detection'].sum())} detections with a stamp"
    if masks["star"] is not None:
        title += f", {int(masks['star'].sum())} PSF stars"
    if algorithm is not None:
        title += f"; {n_flagged} flagged by {algorithm}"
    ax.set_title(title, fontsize=10)
    return ax.figure


def _column(table, name):
    if name not in table.colnames:
        raise ValueError(f"column {name!r} not in the table (have: {table.colnames})")
    return np.asarray(table[name], dtype=float)


def _magnitudes(table, algorithm, mag):
    if mag == "measured":
        return _column(table, f"mag_{algorithm}"), f"{algorithm} magnitude"
    if "mag" in table.colnames:
        return _column(table, "mag"), "catalog magnitude (AB)"
    flux = _column(table, "flux_cat")
    with np.errstate(divide="ignore", invalid="ignore"):
        return -2.5 * np.log10(flux), "−2.5 log10(catalog flux) [DN/s]"
