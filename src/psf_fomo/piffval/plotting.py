"""Plots for Piff fits. Each function takes a `FitResult` and returns the matplotlib figure.

``which`` selects the stars: ``'fit'`` (default), ``'reserve'`` or ``'all'``; stars flagged by
Piff are always left out.
"""
import numpy as np
import matplotlib.pyplot as plt

from .components import component_image, kind_of, leaf_psfs
from .residuals import draw_model_oversampled, fit_sed

QUANTITIES = ("dT_T", "de1", "de2")
LABELS = {"dT_T": "dT/T", "de1": "de1", "de2": "de2"}


def mean_images(psf, stars):
    """Flux-normalised mean data, model and residual (data − model) stamps.

    All stamps must have the same size. Returns a dict of 2-d arrays (``data``, ``model``,
    ``resid``), each the mean of image / fitted flux.
    """
    data, model = [], []
    for star in stars:
        data.append(star.image.array / star.fit.flux)
        model.append(psf.drawStar(star).image.array / star.fit.flux)
    data, model = np.mean(data, axis=0), np.mean(model, axis=0)
    return {"data": data, "model": model, "resid": data - model}


def mean_oversampled(psf, stars, rows, sim, oversample=5):
    """Mean truth, model and model − truth on the oversampled effective-PSF grid.

    Images are unit-flux truth (`Simulation.truth_psf`) and the model on the same grid.
    """
    truth, model = [], []
    for star, row in zip(stars, rows):
        t = sim.truth_psf(int(row["sca"]), int(row["obs"]), int(row["star_id"]),
                          oversample=oversample)
        truth.append(t.array)
        model.append(draw_model_oversampled(psf, star, t).array)
    truth, model = np.mean(truth, axis=0), np.mean(model, axis=0)
    return {"truth": truth, "model": model, "resid": model - truth}


def chi2_by_radius(psf, stars, bin_width=1.0):
    """Mean χ² per pixel in annuli around the star position (native pixels).

    Returns (radius bin centres [pix], mean χ² per pixel, number of pixels per bin).
    """
    r_all, c_all = [], []
    for star in stars:
        model = psf.drawStar(star).image.array
        b = star.image.bounds
        y, x = np.mgrid[b.ymin:b.ymax + 1, b.xmin:b.xmax + 1]
        r_all.append(np.hypot(x - star.image_pos.x, y - star.image_pos.y).ravel())
        c_all.append((star.weight.array * (star.image.array - model) ** 2).ravel())
    r, c = np.concatenate(r_all), np.concatenate(c_all)
    edges = np.arange(0.0, r.max() + bin_width, bin_width)
    idx = np.digitize(r, edges) - 1
    n = np.bincount(idx, minlength=len(edges) - 1)[:len(edges) - 1]
    total = np.bincount(idx, weights=c, minlength=len(edges) - 1)[:len(edges) - 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (edges[1:] + edges[:-1]), total / n, n


def plot_mean_images(result, which="fit", oversampled=False, oversample=5):
    """Mean data / model / residual stamps (native), optionally also oversampled truth/model.

    Parameters
    ----------
    result : FitResult
    which : {'fit', 'reserve', 'all'}
    oversampled : bool
        Add a row with the mean oversampled truth, model and model − truth (needs the
        simulation). Default False.
    oversample : int
        Grid refinement of the oversampled row. Default 5.
    """
    stars, rows = result.selected_stars(which)
    rows_of_images = [("native", mean_images(result.psf, stars), ("data", "model", "resid"))]
    if oversampled:
        rows_of_images.append(("oversampled",
                               mean_oversampled(result.psf, stars, rows, result.sim, oversample),
                               ("truth", "model", "resid")))
    fig, axes = plt.subplots(len(rows_of_images), 3, figsize=(12, 3.8 * len(rows_of_images)),
                             squeeze=False)
    for ax_row, (label, images, keys) in zip(axes, rows_of_images):
        vmax = images[keys[0]].max()
        for ax, key in zip(ax_row, keys):
            if key == "resid":
                lim = np.abs(images[key]).max()
                im = ax.imshow(images[key], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.set_title(f"{label} residual (max |r| / peak = {lim / vmax:.2e})")
            else:
                im = ax.imshow(images[key], origin="lower", cmap="viridis", vmax=vmax)
                ax.set_title(f"{label} mean {key}")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{len(stars)} {which} stars")
    fig.tight_layout()
    return fig


def plot_chi2_radius(result, which="fit", bin_width=1.0, ax=None):
    """Mean χ² per pixel vs distance from the star position (1 = noise level)."""
    stars, _ = result.selected_stars(which)
    r, chi2, n = chi2_by_radius(result.psf, stars, bin_width)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(r, chi2, marker="o")
    ax.axhline(1, color="gray", lw=0.8)
    ax.set_xlabel("radius [native pix]")
    ax.set_ylabel("mean χ² per pixel")
    ax.set_title(f"{len(stars)} {which} stars")
    return ax.figure


def plot_moment_residuals(result, comparison="truth_os", algorithms=None, bins=20):
    """Histograms of dT/T, de1, de2 for fit and reserve stars, per algorithm.

    Parameters
    ----------
    result : FitResult
        Needs ``result.moments``.
    comparison : {'data', 'truth_native', 'truth_os'}
        (``'model_os'`` has no reference; use `plot_field` with ``'T_model'`` etc.)
    algorithms : str or sequence of str, optional
        Algorithm labels (e.g. ``'wmom_0.4'``). Default all measured for this comparison.
    """
    ref_name = _ref_name(comparison)
    m = result.moments
    m = m[(m["comparison"] == comparison) & (m["flag"] == 0)]
    algs = list(dict.fromkeys(m["algorithm"])) if algorithms is None else list(
        np.atleast_1d(algorithms))
    fig, axes = plt.subplots(len(algs), 3, figsize=(13, 3.2 * len(algs)), squeeze=False)
    for ax_row, alg in zip(axes, algs):
        rows = m[m["algorithm"] == alg]
        for ax, q in zip(ax_row, QUANTITIES):
            values = np.asarray(rows[q], dtype=float)
            edges = np.histogram_bin_edges(values[np.isfinite(values)], bins=bins)
            for label, sel in (("fit", ~rows["reserve"]), ("reserve", rows["reserve"])):
                if np.any(sel):
                    v = values[np.asarray(sel)]
                    ax.hist(v, bins=edges, histtype="step",
                            label=f"{label}: {np.nanmean(v):+.2e} ± {np.nanstd(v):.1e}")
            ax.axvline(0, color="gray", lw=0.8)
            ax.set_xlabel(f"{LABELS[q]} (model − {ref_name})")
            ax.set_title(alg)
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_field(result, column, comparison=None, algorithm=None, which="all", ax=None):
    """A per-star quantity across the SCA (scatter at the star positions).

    Parameters
    ----------
    result : FitResult
    column : str
        A column of ``result.table`` (e.g. ``'flux_r'``, ``'chisq_dof'``, ``'dx'``) or, with
        ``comparison`` and ``algorithm``, of ``result.moments`` (e.g. ``'dT_T'``).
    comparison, algorithm : str, optional
        Select moments rows.
    which : {'fit', 'reserve', 'all'}
    """
    if comparison is None:
        table = result.table[result.table["flag"] == 0]
    else:
        m = result.moments
        table = m[(m["comparison"] == comparison) & (m["algorithm"] == algorithm)
                  & (m["flag"] == 0)]
    if which != "all":
        table = table[table["reserve"] == (which == "reserve")]
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.6))
    values = np.asarray(table[column], dtype=float)
    lim = np.nanmax(np.abs(values - np.nanmedian(values)))
    sc = ax.scatter(table["x"], table["y"], c=values, cmap="RdBu_r",
                    vmin=np.nanmedian(values) - lim, vmax=np.nanmedian(values) + lim)
    ax.figure.colorbar(sc, ax=ax, label=column)
    ax.set_xlim(0, 4089)
    ax.set_ylim(0, 4089)
    ax.set_aspect("equal")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    title = column if comparison is None else f"{column}: {comparison}, {algorithm}"
    ax.set_title(title)
    return ax.figure


def plot_components(result, kind="pixelgrid", n_grid=3, oversample=5, size=9):
    """Images of one component (``'pixelgrid'``, ``'kernel'``, ``'optics'``, ``'mtf'``) at
    an n_grid × n_grid set of SCA positions.

    Optics, kernel and MTF are drawn pre-pixel; a PixelGrid includes the pixel.
    """
    leaves = [leaf for leaf in leaf_psfs(result.psf) if kind_of(leaf) == kind]
    if not leaves:
        raise ValueError(f"the model has no {kind!r} component")
    leaf = leaves[0]
    sca = result.config["sca"]
    pos = np.linspace(500, 3588, n_grid)
    fig, axes = plt.subplots(n_grid, n_grid, figsize=(2.6 * n_grid, 2.6 * n_grid),
                             squeeze=False)
    sed_eff = fit_sed(result.psf, getattr(result, "_stars", None) or [])  # chromatic optics
    images = [[component_image(result.psf, leaf, x, y, sca, oversample, size, sed_eff).array
               for x in pos] for y in pos]
    lim = max(np.abs(im).max() for row in images for im in row)
    signed = min(im.min() for row in images for im in row) < 0
    for i, y in enumerate(pos):
        for j, x in enumerate(pos):
            ax = axes[n_grid - 1 - i, j]  # y increases upward
            ax.imshow(images[i][j], origin="lower", cmap="RdBu_r" if signed else "viridis",
                      vmin=-lim if signed else 0, vmax=lim)
            ax.set_title(f"({x:.0f}, {y:.0f})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{kind} component, ×{oversample} grid, {size} native pix")
    fig.tight_layout()
    return fig


def plot_zernikes(result, ax=None):
    """Fitted extra Zernikes (mean ± Piff σ) and, for simulations, the truth; waves."""
    table = result.components()["optics"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(table["noll"], table["fit_mean"], yerr=table["fit_err"], ls="none",
                marker="o", capsize=3, label="fit")
    if "truth_mean" in table.colnames:
        ax.plot(table["noll"], table["truth_mean"], ls="none", marker="x", color="k",
                label="truth")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Noll index")
    ax.set_ylabel("extra aberration [waves]")
    ax.legend()
    return ax.figure


def plot_rho(rho_table):
    """ρ1 … ρ5 (ξ+ ± error) vs separation, one panel per statistic, one line per algorithm.

    Parameters
    ----------
    rho_table : astropy.table.Table
        From `psf_fomo.piffval.rho.rho_stats` / `FitResult.rho`.
    """
    names = ("rho1", "rho2", "rho3", "rho4", "rho5")
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    for ax, name in zip(axes, names):
        rows = rho_table[(rho_table["rho"] == name) & (rho_table["npairs"] > 0)]
        for alg in dict.fromkeys(rows["algorithm"]):
            r = rows[rows["algorithm"] == alg]
            ax.errorbar(r["r"], r["xip"], yerr=r["xip_err"], marker="o", ms=3, capsize=2,
                        label=f"{alg} (n={r['n_used'][0]})")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("θ [arcmin]")
        ax.set_title(f"ρ{name[-1]} ξ+")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def _ref_name(comparison):
    if comparison == "model_os":
        raise ValueError("'model_os' has no reference: plot its *_model columns with "
                         "plot_field, e.g. plot_field(res, 'T_model', 'model_os', 'hsm')")
    return "data" if comparison == "data" else "truth"
