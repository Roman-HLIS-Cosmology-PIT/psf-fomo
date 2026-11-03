"""Plots for moments results. Each function takes a `MomentsResult` and returns the matplotlib
figure it drew on.

Quantities (see `MomentsResult.bias` and `MomentsResult.differences`): ``dT_T`` (fractional
size difference, trace T), ``de1``, ``de2`` (distortion differences).
"""
import numpy as np
import matplotlib.pyplot as plt

QUANTITIES = ("dT_T", "de1", "de2")
LABELS = {"dT_T": "dT/T", "de1": "de1", "de2": "de2"}
MARKERS = {"noisy": "o", "noiseless": "s", "oversampled": "^"}


def plot_bias(result, sim=None, images=None, algorithms=None):
    """Native-pixel minus oversampled moments, mean ± error on the mean, per algorithm.

    Parameters
    ----------
    result : MomentsResult
    sim : str, optional
        Simulation to show. Default the reference simulation.
    images : str or sequence of str, optional
        ``'noisy'`` and/or ``'noiseless'``. Default both (if measured).
    algorithms : str or sequence of str, optional
        Algorithm labels to show. Default all.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sim = result.config["reference"] if sim is None else sim
    summary = result.summary("bias")
    summary = summary[summary["sim"] == sim]
    summary = _keep(summary, "image", images)
    summary = _keep(summary, "algorithm", algorithms)
    algorithms = list(dict.fromkeys(summary["algorithm"]))
    images = list(dict.fromkeys(summary["image"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(algorithms))
    for ax, q in zip(axes, QUANTITIES):
        for k, image in enumerate(images):
            rows = summary[summary["image"] == image]
            idx = [algorithms.index(a) for a in rows["algorithm"]]
            offset = (k - (len(images) - 1) / 2) * 0.15
            ax.errorbar(np.asarray(idx) + offset, rows[q + "_mean"], yerr=rows[q + "_err"],
                        ls="none", marker=MARKERS[image], capsize=3, label=image)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x, algorithms)
        ax.set_ylabel(f"{LABELS[q]} (native − oversampled)")
    axes[0].legend()
    fig.suptitle(f"native vs oversampled, simulation {sim!r}")
    fig.tight_layout()
    return fig


def plot_recovery(result, injected="hsm_os", reference=None, images=None, algorithms=None):
    """Measured PSF errors vs injected ones, one point per simulation × image × algorithm.

    Parameters
    ----------
    result : MomentsResult
    injected : {'hsm_os', 'os'}
        Injected reference on the x axis: HSM on the oversampled images (default) or the same
        algorithm on the oversampled images.
    reference : str, optional
        Reference simulation. Default the configured one.
    images : str or sequence of str, optional
        Image types to show, e.g. ``'noisy'`` or ``['noiseless', 'oversampled']``.
        Default all measured.
    algorithms : str or sequence of str, optional
        Algorithm labels to show, e.g. ``'hsm'`` or ``['hsm', 'wmom_0.4']``. Default all.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if injected not in ("hsm_os", "os"):
        raise ValueError("injected must be 'hsm_os' or 'os'")
    summary = result.summary("differences", reference=reference)
    summary = _keep(summary, "image", images)
    summary = _keep(summary, "algorithm", algorithms)
    algorithms = list(dict.fromkeys(summary["algorithm"]))
    colors = {a: f"C{i}" for i, a in enumerate(algorithms)}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, q in zip(axes, QUANTITIES):
        xs = summary[f"{q}_{injected}_mean"]
        for row in summary:
            ax.errorbar(row[f"{q}_{injected}_mean"], row[q + "_mean"], yerr=row[q + "_err"],
                        marker=MARKERS[row["image"]], color=colors[row["algorithm"]],
                        ls="none", capsize=3)
        finite = np.asarray(xs)[np.isfinite(xs)]
        if len(finite):
            lim = [min(finite.min(), 0), max(finite.max(), 0)]
            ax.plot(lim, lim, color="gray", lw=0.8)
        ax.set_xlabel(f"injected {LABELS[q]} ({injected})")
        ax.set_ylabel(f"measured {LABELS[q]}")
    handles = [plt.Line2D([], [], color=c, marker="o", ls="none", label=a)
               for a, c in colors.items()]
    handles += [plt.Line2D([], [], color="k", marker=m, ls="none", label=i)
                for i, m in MARKERS.items() if i in set(summary["image"])]
    axes[0].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    return fig


def plot_vs(result, column, kind="bias", quantity="dT_T", image="noisy", sim=None, n_bins=8,
            log=False, ax=None):
    """Binned mean ± error on the mean of a quantity vs a per-star column, per algorithm.

    Parameters
    ----------
    result : MomentsResult
    column : str
        Per-star column on the x axis, e.g. ``'phase_x'``, ``'phase_y'``, ``'snr'``, ``'mag'``.
    kind : {'bias', 'differences'}
        Native vs oversampled (`MomentsResult.bias`) or vs the reference simulation
        (`MomentsResult.differences`).
    quantity : {'dT_T', 'de1', 'de2'}
    image : {'noisy', 'noiseless', 'oversampled'}
    sim : str, optional
        Simulation. Default: the reference (bias) or the first other simulation (differences).
    n_bins : int
        Number of bins (equal width, or equal width in log if ``log``). Default 8.
    log : bool
        Logarithmic x axis and bins. Default False.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if kind == "bias":
        table = result.bias()
        sim = result.config["reference"] if sim is None else sim
    elif kind == "differences":
        table = result.differences()
        sim = table["sim"][0] if sim is None else sim
    else:
        raise ValueError("kind must be 'bias' or 'differences'")
    table = table[(table["sim"] == sim) & (table["image"] == image) & (table["flag"] == 0)]
    if len(table) == 0:
        raise ValueError(f"no good rows for sim={sim!r}, image={image!r}")
    x = np.asarray(table[column], dtype=float)
    lo, hi = x.min(), x.max()
    edges = (np.geomspace(lo, hi, n_bins + 1) if log else np.linspace(lo, hi, n_bins + 1))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for algorithm in dict.fromkeys(table["algorithm"]):
        rows = table[table["algorithm"] == algorithm]
        xs, means, errs = _binned(np.asarray(rows[column], dtype=float),
                                  np.asarray(rows[quantity], dtype=float), edges)
        ax.errorbar(xs, means, yerr=errs, marker="o", capsize=3, label=algorithm)
    ax.axhline(0, color="gray", lw=0.8)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(column)
    ax.set_ylabel(f"{LABELS[quantity]} ({kind})")
    ax.set_title(f"{sim}, {image}")
    ax.legend(fontsize=8)
    return ax


def plot_vs_phase(result, kind="bias", quantity="dT_T", image="noisy", sim=None, n_bins=8):
    """`plot_vs` against ``phase_x`` and ``phase_y`` (two panels).

    With fixed-phase simulations, all stars of one simulation fall in one bin; compare
    simulations with different phases instead (one call per simulation).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column in zip(axes, ("phase_x", "phase_y")):
        plot_vs(result, column, kind=kind, quantity=quantity, image=image, sim=sim,
                n_bins=n_bins, ax=ax)
    fig.tight_layout()
    return fig


def plot_vs_snr(result, kind="bias", quantity="dT_T", image="noisy", sim=None, n_bins=8):
    """`plot_vs` against the star S/N, logarithmic bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_vs(result, "snr", kind=kind, quantity=quantity, image=image, sim=sim,
            n_bins=n_bins, log=True, ax=ax)
    fig.tight_layout()
    return fig


def _keep(table, column, values):
    """Rows whose ``column`` is in ``values`` (a string or a sequence); None keeps all."""
    if values is None:
        return table
    values = [values] if isinstance(values, str) else list(values)
    unknown = set(values) - set(table[column])
    if unknown:
        raise ValueError(f"no {column} {sorted(unknown)}; available: "
                         f"{sorted(set(table[column]))}")
    return table[np.isin(table[column], values)]


def _binned(x, y, edges):
    """Bin centres (mean x), mean y and error on the mean per bin; bins with < 2 points skipped."""
    good = np.isfinite(y)
    x, y = x[good], y[good]
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    xs, means, errs = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() < 2:
            continue
        xs.append(x[sel].mean())
        means.append(y[sel].mean())
        errs.append(y[sel].std(ddof=1) / np.sqrt(sel.sum()))
    return xs, means, errs
