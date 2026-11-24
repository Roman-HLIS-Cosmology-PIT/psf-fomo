"""Plots specific to real-data fits. The fit plots are shared with simulations
(`psf_fomo.piffval.plotting`, called through `DataFitResult` methods)."""
import numpy as np
import matplotlib.pyplot as plt

# Detections stay recessive gray; the two selection stages take the first two categorical
# hues (blue, orange; color-blind separation checked). Marker shape and size differ too, so
# the stages are not told apart by color alone.
STAGES = (
    ("detection", dict(color="0.75", marker=".", s=6, lw=0)),
    ("candidate", dict(facecolor="none", edgecolor="darkorange", marker="o", s=28, lw=1.0)),
    ("star", dict(color="royalblue", marker="D", s=10, lw=0)),
)
LABELS = {"detection": "detections (stamp on image)", "candidate": "selected by the star finder",
          "star": "PSF stars (after cuts)"}
SIZE_LABELS = {"T": "T = Mxx + Myy [arcsec²]", "sigma": "σ = det(M)^¼ [arcsec]"}


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
        Measured size to plot (with ``algorithm``). Default ``'T'`` (arcsec²).
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


LOCUS_STYLES = {   # same hue order as above; marker shapes differ too
    "selected": dict(color="royalblue", marker="o", s=10, lw=0),
    "saturated": dict(color="darkorange", marker="^", s=18, lw=0),
    "off_locus": dict(color="mediumseagreen", marker="s", s=14, lw=0),
    "faint": dict(color="0.7", marker=".", s=6, lw=0),
    "flagged": dict(color="0.45", marker="x", s=12, lw=0.8),
}


def plot_locus(result):
    """Diagnostics of `select_locus`: three panels.

    1. Size vs magnitude, stars by reason, the saturation limit.
    2. Fractional size residual (to the locus center at each star's position) vs magnitude,
       the ±k × width band, and the binned medians of the saturation scan (deviating bins
       filled) with their uncertainties and the scan's reference level (dotted).
    3. Residual across the SCA for the stars kept in the locus (selected + off-locus): a
       remaining trend means the locus ``order`` is too low.

    Parameters
    ----------
    result : LocusResult

    Returns
    -------
    matplotlib.figure.Figure
    """
    t, s, bins = result.table, result.summary, result.bins
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    reasons = np.asarray(t["reason"])
    limit = s["bright_limit"]
    band = s["k"] * s["width"]

    ax = axes[0]
    for name, style in LOCUS_STYLES.items():
        m = reasons == name
        if m.any():
            ax.scatter(t["mag"][m], t["size"][m], label=f"{name}: {int(m.sum())}", **style)
    ax.set_yscale("log")
    ax.set_xlabel(s["mag"])
    ax.set_ylabel(s["size"])
    ax.set_title("size vs magnitude")

    ax = axes[1]
    for name in ("selected", "saturated", "off_locus"):
        m = reasons == name
        if m.any():
            ax.scatter(t["mag"][m], t["resid"][m], **LOCUS_STYLES[name])
    ax.axhspan(-band, band, color="0.9", zorder=0, label=f"±{s['k']:g} × width")
    used = np.asarray(bins["used"], dtype=bool)
    centers = 0.5 * (np.asarray(bins["mag_lo"]) + np.asarray(bins["mag_hi"]))
    dev = np.asarray(bins["deviates"], dtype=bool)
    for sel, face, label in ((used & ~dev, "white", "bin median"),
                             (used & dev, "k", "bin median, deviates")):
        if sel.any():
            ax.errorbar(centers[sel], np.asarray(bins["median"])[sel],
                        yerr=np.asarray(bins["uncertainty"])[sel], fmt="o", mfc=face,
                        mec="k", ecolor="k", ms=5, capsize=2, label=label)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axhline(s["scan_reference"], color="k", ls=":", lw=1, label="scan reference")
    lim = max(3 * band, np.nanpercentile(np.abs(t["resid"][reasons != "flagged"]), 99))
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(s["mag"])
    ax.set_ylabel("size / locus center − 1")
    ax.set_title(f"locus width {s['width']:.4f}, order {s['order']}")

    for ax in axes[:2]:
        if limit is not None:
            ax.axvline(limit, color="darkorange", ls="--", lw=1,
                       label=f"saturation {limit:.2f} ({s['bright_limit_source']})")
        ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    m = np.isin(reasons, ("selected", "off_locus"))
    vmax = max(band, 1e-6)
    sc = ax.scatter(t["x"][m], t["y"][m], c=t["resid"][m], cmap="RdBu_r", vmin=-vmax,
                    vmax=vmax, s=14)
    fig.colorbar(sc, ax=ax, label="size / locus center − 1")
    ax.set_xlim(0, 4089)
    ax.set_ylim(0, 4089)
    ax.set_aspect("equal")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title("residual across the SCA")
    fig.tight_layout()
    return fig


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
