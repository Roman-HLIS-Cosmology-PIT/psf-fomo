"""Plots for simulations. Each function takes a `Simulation` and returns the matplotlib
figure or axes it drew on."""
import numpy as np
import matplotlib.pyplot as plt


def _stretch(array):
    """asinh stretch scaled to the background noise, so faint wings and bright cores show."""
    noise = 1.4826 * np.median(np.abs(array - np.median(array)))
    return np.arcsinh(array / max(noise, 1e-12))


def plot_sca(sim, obs=0, sca=None, ax=None):
    """Full-SCA image (asinh stretch) with circles at the star positions.

    Parameters
    ----------
    sim : Simulation
    obs : int
        Observation index. Default 0.
    sca : int, optional
        SCA number; default the first configured SCA.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    sca = sim.config["scas"][0] if sca is None else sca
    image = sim.images[obs, sca]
    rows = sim.catalog(obs, sca)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    extent = (image.xmin - 0.5, image.xmax + 0.5, image.ymin - 0.5, image.ymax + 0.5)
    ax.imshow(_stretch(image.array), origin="lower", cmap="gray", extent=extent)
    ax.scatter(rows["x"], rows["y"], s=80, facecolors="none", edgecolors="r", lw=0.8)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(f"obs {obs}, SCA {sca} [ADU, asinh]")
    return ax


def plot_stamps(sim, obs=0, sca=None, n_max=16):
    """Gallery of star stamps (asinh stretch), labelled with star id and magnitude.

    Parameters
    ----------
    sim : Simulation
    obs : int
        Observation index. Default 0.
    sca : int, optional
        Restrict to one SCA. Default all.
    n_max : int
        Maximum number of stamps shown. Default 16.

    Returns
    -------
    matplotlib.figure.Figure
    """
    stars, rows = sim.stars(sca=sca, obs=obs)
    n = min(len(stars), n_max)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, star, row in zip(axes.ravel(), stars[:n], rows[:n]):
        ax.imshow(_stretch(star.image.array), origin="lower", cmap="viridis")
        ax.set_title(f"id {row['star_id']}, m={row['mag']:.1f}", fontsize=8)
    return fig


def plot_truth_vs_image(sim, star_id, obs=0):
    """Image stamp, noiseless truth re-rendered from config + seed, and their difference.

    Parameters
    ----------
    sim : Simulation
    star_id : int
    obs : int
        Observation index. Default 0.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sca, _ = sim._locate(star_id, obs)
    truth = sim.render_stamp(star_id, obs)
    data = sim.images[obs, sca][truth.bounds]
    resid = data.array - truth.array
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    vmax = truth.array.max()
    for ax, arr, title in [(axes[0], data.array, "image"), (axes[1], truth.array, "truth"),
                           (axes[2], resid, "image − truth")]:
        if arr is resid:
            lim = np.abs(resid).max()
            im = ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{title} [ADU]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"star {star_id}, obs {obs}, SCA {sca}")
    return fig


def plot_phases(sim, obs=None):
    """Sub-pixel phases (phase_x, phase_y) of all stars.

    Parameters
    ----------
    sim : Simulation
    obs : int, optional
        One observation; default all.

    Returns
    -------
    matplotlib.axes.Axes
    """
    tables = sim.catalogs if obs is None else [sim.catalogs[obs]]
    _, ax = plt.subplots(figsize=(4.5, 4.5))
    for table in tables:
        ax.scatter(table["phase_x"], table["phase_y"], s=8)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("phase x [pix]")
    ax.set_ylabel("phase y [pix]")
    return ax


def plot_error_field(sim, column="dilation", obs=0):
    """A per-star truth quantity across each SCA (one panel per SCA).

    Parameters
    ----------
    sim : Simulation
    column : str
        Catalog column, e.g. ``'dilation'``, ``'g1'``, ``'mtf_scale'``, ``'z4'``.
    obs : int
        Observation index. Default 0.

    Returns
    -------
    matplotlib.figure.Figure
    """
    scas = sim.config["scas"]
    table = sim.catalogs[obs]
    vmin, vmax = np.min(table[column]), np.max(table[column])
    fig, axes = plt.subplots(1, len(scas), figsize=(4 * len(scas), 3.8), squeeze=False)
    for ax, sca in zip(axes[0], scas):
        rows = table[table["sca"] == sca]
        sc = ax.scatter(rows["x"], rows["y"], c=rows[column], vmin=vmin, vmax=vmax, s=20)
        ax.set_title(f"SCA {sca}")
        ax.set_aspect("equal")
        ax.set_xlabel("x [pix]")
    axes[0, 0].set_ylabel("y [pix]")
    fig.colorbar(sc, ax=axes[0].tolist(), label=column)
    return fig


def plot_error_vs_obs(sim, column="dilation"):
    """Mean (± star-to-star std) of a per-star truth quantity in each observation and SCA.

    Parameters
    ----------
    sim : Simulation
    column : str
        Catalog column.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _, ax = plt.subplots(figsize=(6, 4))
    for sca in sim.config["scas"]:
        means, stds = [], []
        for table in sim.catalogs:
            values = np.asarray(table[table["sca"] == sca][column])
            means.append(values.mean())
            stds.append(values.std())
        ax.errorbar(np.arange(len(means)), means, yerr=stds, marker="o", capsize=3,
                    label=f"SCA {sca}")
    ax.set_xlabel("observation")
    ax.set_ylabel(column)
    ax.legend()
    return ax
