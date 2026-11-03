"""Moments validation: measure many stars with several algorithms on several image types.

Example
-------
>>> mv = MomentsValidation({"base": sim0, "dil": sim1}, reference="base")
>>> mv.set_algorithms(["hsm", "am", "gauss", "wmom"], wmom_fwhm=0.4)
>>> mv.set_images(["noisy", "noiseless", "oversampled"], oversample=5)
>>> res = mv.run()
>>> res.summary("differences")

Image types (column ``image``):

- ``'noisy'``: the native-pixel stamp cut from the simulated image (`Simulation.stars`).
- ``'noiseless'``: the same stamp re-rendered without noise (`Simulation.render_stamp`), with
  the same weight. It contains no detector effects (IPC, nonlinearity, ...).
- ``'oversampled'``: the unit-flux truth PSF on a grid ``oversample`` times finer
  (`Simulation.truth_psf`); by default the pixel-convolved effective PSF.

Simulations made with the same seed share positions, magnitudes and noise draws, so
differences between them are taken star by star (paired on obs, sca, star_id).
"""
import copy
from pathlib import Path

import numpy as np
from astropy.table import Table, join, vstack

from ..io import provenance, read_yaml, write_yaml
from ..sim import Simulation, load_simulation
from . import plotting
from .measure import (ALGORITHMS, DEFAULT_WMOM_FWHM, algorithm_label, make_star,
                      measure_star, require_ngmix)

IMAGE_TYPES = ("noisy", "noiseless", "oversampled")
STAR_KEYS = ["obs", "sca", "star_id"]
SHAPE_COLUMNS = ("dT_T", "de1", "de2")


class MomentsValidation:
    """Builder for moment measurements on one or more simulations.

    Parameters
    ----------
    simulations : dict
        Name → `Simulation`, or name → path of a saved simulation folder. Only path inputs can
        be written to YAML (`to_yaml`).
    reference : str, optional
        Name of the simulation that differences are taken against. Default: the first one.

    Defaults: algorithms ``hsm``, ``am``, ``gauss``, ``wmom`` (FWHM 0.4″); image types
    ``noisy``, ``noiseless``, ``oversampled`` (×5, pixel included); all SCAs and observations.
    """

    def __init__(self, simulations, reference=None):
        if not simulations:
            raise ValueError("give at least one simulation")
        self.simulations = {}
        paths = {}
        for name, sim in simulations.items():
            if isinstance(sim, Simulation):
                self.simulations[name] = sim
                paths[name] = None
            else:
                self.simulations[name] = load_simulation(sim)
                paths[name] = str(sim)
        names = list(self.simulations)
        reference = names[0] if reference is None else reference
        if reference not in names:
            raise ValueError(f"reference {reference!r} is not one of {names}")
        self.config = {
            "simulations": paths,
            "reference": reference,
            "algorithms": list(ALGORITHMS),
            "wmom_fwhm": [DEFAULT_WMOM_FWHM],
            "images": list(IMAGE_TYPES),
            "oversample": 5,
            "include_pixel": True,
            "scas": None,
            "obs": None,
        }

    def set_algorithms(self, algorithms=ALGORITHMS, wmom_fwhm=DEFAULT_WMOM_FWHM):
        """Choose the measurement algorithms.

        Parameters
        ----------
        algorithms : sequence of str
            Subset of ``'hsm'``, ``'am'``, ``'gauss'``, ``'wmom'``. Default all.
        wmom_fwhm : float or sequence of float
            FWHM of the fixed ``wmom`` weight, arcsec. Several values give one algorithm label
            each (``'wmom_0.3'``, ``'wmom_0.4'``, ...). Default 0.4.
        """
        algorithms = list(algorithms)
        unknown = set(algorithms) - set(ALGORITHMS)
        if unknown or not algorithms:
            raise ValueError(f"algorithms must be a non-empty subset of {ALGORITHMS}")
        fwhm = [float(f) for f in np.atleast_1d(wmom_fwhm)]
        if any(not f > 0 for f in fwhm):
            raise ValueError("wmom_fwhm must be > 0 (arcsec)")
        self.config["algorithms"] = algorithms
        self.config["wmom_fwhm"] = fwhm

    def set_images(self, images=IMAGE_TYPES, oversample=5, include_pixel=True):
        """Choose the image types to measure.

        Parameters
        ----------
        images : sequence of str
            Subset of ``'noisy'``, ``'noiseless'`` (native pixels) and ``'oversampled'``.
            Default all three.
        oversample : int
            Refinement of the oversampled grid (pixel = native / oversample). Default 5.
        include_pixel : bool
            Oversampled image of the pixel-convolved effective PSF (True, default) or of the
            pre-pixel PSF (False).
        """
        images = list(images)
        unknown = set(images) - set(IMAGE_TYPES)
        if unknown or not images:
            raise ValueError(f"images must be a non-empty subset of {IMAGE_TYPES}")
        if int(oversample) < 1:
            raise ValueError("oversample must be >= 1")
        self.config["images"] = images
        self.config["oversample"] = int(oversample)
        self.config["include_pixel"] = bool(include_pixel)

    def select(self, scas=None, obs=None):
        """Restrict the measurement to some SCAs and/or observations (default all)."""
        self.config["scas"] = None if scas is None else [int(s) for s in scas]
        self.config["obs"] = None if obs is None else [int(o) for o in obs]

    def to_yaml(self, path):
        """Write the configuration to YAML. All simulations must have been given as paths."""
        missing = [n for n, p in self.config["simulations"].items() if p is None]
        if missing:
            raise ValueError(f"simulations {missing} were given as objects; save them and "
                             "pass their folders to write a YAML configuration")
        write_yaml(path, self.config)

    @classmethod
    def from_yaml(cls, path):
        """Build from a YAML file written by `to_yaml`."""
        config = read_yaml(path)
        mv = cls(config["simulations"], reference=config["reference"])
        mv.set_algorithms(config["algorithms"], config["wmom_fwhm"])
        mv.set_images(config["images"], config["oversample"], config["include_pixel"])
        mv.select(config["scas"], config["obs"])
        return mv

    def run(self, save_to=None):
        """Measure every selected star of every simulation.

        Parameters
        ----------
        save_to : str or Path, optional
            Folder to save the result to. Default None.

        Returns
        -------
        MomentsResult
        """
        if set(self.config["algorithms"]) - {"hsm"}:
            require_ngmix()  # fail before any rendering, not star by star
        tables = [self._measure_simulation(name, sim)
                  for name, sim in self.simulations.items()]
        result = MomentsResult(vstack(tables), copy.deepcopy(self.config), provenance())
        if save_to is not None:
            result.save(save_to)
        return result

    def _algorithms(self):
        """(label, algorithm, weight_fwhm) for each configured measurement."""
        out = []
        for alg in self.config["algorithms"]:
            fwhms = self.config["wmom_fwhm"] if alg == "wmom" else [None]
            out += [(algorithm_label(alg, f), alg, f) for f in fwhms]
        return out

    def _measure_simulation(self, name, sim):
        cfg = self.config
        scas = sim.config["scas"] if cfg["scas"] is None else cfg["scas"]
        obs_list = range(sim.config["n_obs"]) if cfg["obs"] is None else cfg["obs"]
        rows = []
        for obs in obs_list:
            for sca in scas:
                if sca not in sim.config["scas"]:
                    raise ValueError(f"simulation {name!r} has no SCA {sca}")
                stars, truth_rows = sim.stars(sca=sca, obs=obs)
                for star, row in zip(stars, truth_rows):
                    rows += self._measure_one(name, sim, star, row)
        return Table(rows=rows)

    def _measure_one(self, name, sim, star, row):
        """Rows for one star: every image type × algorithm."""
        cfg = self.config
        obs, sca, star_id = int(row["obs"]), int(row["sca"]), int(row["star_id"])
        # The noiseless stamp is always rendered: it also gives the S/N of the star.
        noiseless = sim.render_stamp(star_id, obs)
        bkg_var = float(row["bkg_var"])
        snr = np.sqrt(np.sum(noiseless.array.astype(float)**2) / bkg_var)
        images = {}
        if "noisy" in cfg["images"]:
            images["noisy"] = star
        if "noiseless" in cfg["images"]:
            images["noiseless"] = make_star(noiseless, star.image_pos, 1.0 / bkg_var)
        if "oversampled" in cfg["images"]:
            fine = sim.truth_psf(sca, obs, star_id, oversample=cfg["oversample"],
                                 include_pixel=cfg["include_pixel"])
            images["oversampled"] = make_star(fine, fine.true_center)
        base = {"sim": name, "obs": obs, "sca": sca, "star_id": star_id,
                "x": float(row["x"]), "y": float(row["y"]),
                "phase_x": float(row["phase_x"]), "phase_y": float(row["phase_y"]),
                "mag": float(row["mag"]), "snr": snr}
        out = []
        for image_type, image_star in images.items():
            for label, alg, fwhm in self._algorithms():
                values = measure_star(image_star, alg, weight_fwhm=fwhm or DEFAULT_WMOM_FWHM)
                out.append({**base, "image": image_type, "algorithm": label, **values})
        return out


class MomentsResult:
    """Measurements from `MomentsValidation.run` (or `load_moments`).

    Attributes
    ----------
    table : astropy.table.Table
        One row per simulation × star × image type × algorithm. Columns: ``sim``, ``obs``,
        ``sca``, ``star_id``, ``x``, ``y`` (pix), ``phase_x``, ``phase_y`` (pix), ``mag``
        (AB), ``snr`` (√(Σ I²/σ²) of the noiseless native stamp), ``image``, ``algorithm``,
        ``flux``, ``du``, ``dv`` (arcsec), ``sigma`` (native pix), ``T`` (native pix²,
        trace), ``e1``, ``e2``, ``g1``, ``g2``, ``flag`` (see `measure.measure_star`).
    config : dict
        The `MomentsValidation` configuration.
    provenance : dict
        Software versions and creation time.
    """

    def __init__(self, table, config, provenance):
        self.table = table
        self.config = config
        self.provenance = provenance

    def bias(self):
        """Native-pixel measurements relative to the oversampled image of the same star.

        For each native row (``noisy``, ``noiseless``) and the same simulation, star and
        algorithm: ``dT_T = (T - T_os) / T_os``, ``de1 = e1 - e1_os``, ``de2 = e2 - e2_os``.
        ``flag`` combines both flags.

        Returns
        -------
        astropy.table.Table
            The native rows with the added columns ``T_os``, ``e1_os``, ``e2_os``, ``dT_T``,
            ``de1``, ``de2``.
        """
        table = self.table
        native = table[table["image"] != "oversampled"]
        os_rows = table[table["image"] == "oversampled"]
        if len(native) == 0 or len(os_rows) == 0:
            raise ValueError("bias() needs native and oversampled measurements")
        out = _attach(native, os_rows, ["sim"] + STAR_KEYS + ["algorithm"],
                      ["T", "e1", "e2", "flag"], "_os")
        out["dT_T"] = (out["T"] - out["T_os"]) / out["T_os"]
        out["de1"] = out["e1"] - out["e1_os"]
        out["de2"] = out["e2"] - out["e2_os"]
        out["flag"] = out["flag"] | out["flag_os"]
        out.remove_column("flag_os")
        return out

    def differences(self, reference=None):
        """Star-by-star differences of each simulation from the reference simulation.

        For each simulation other than the reference, image type and algorithm:
        ``dT_T = (T - T_ref) / T_ref``, ``de1 = e1 - e1_ref``, ``de2 = e2 - e2_ref``
        (likewise ``dg1``, ``dg2``). These are the measured PSF errors. Two "injected"
        references are attached for each star:

        - ``dT_T_os``, ``de1_os``, ``de2_os``: the same algorithm on the oversampled images;
        - ``dT_T_hsm_os``, ``de1_hsm_os``, ``de2_hsm_os``: HSM on the oversampled images
          (the earlier common convention).

        They are NaN when the oversampled (or HSM) measurement was not made.

        Parameters
        ----------
        reference : str, optional
            Reference simulation. Default the configured one.

        Returns
        -------
        astropy.table.Table

        Raises
        ------
        ValueError
            If the paired stars are not at the same positions (simulations not made with the
            same placement and seed).
        """
        reference = self.config["reference"] if reference is None else reference
        table = self.table
        ref = table[table["sim"] == reference]
        others = table[table["sim"] != reference]
        if len(ref) == 0:
            raise ValueError(f"no measurements of reference {reference!r}")
        if len(others) == 0:
            raise ValueError("differences() needs at least two simulations")
        out = _attach(others, ref, STAR_KEYS + ["image", "algorithm"],
                      ["x", "y", "T", "e1", "e2", "g1", "g2", "flag"], "_ref", join_type="inner")
        if len(out) != len(others):
            raise ValueError("some stars have no counterpart in the reference simulation")
        bad = (np.abs(out["x"] - out["x_ref"]) > 1e-6) | (np.abs(out["y"] - out["y_ref"]) > 1e-6)
        if np.any(bad):
            raise ValueError(f"{np.sum(bad)} paired stars are at different positions; "
                             "differences need simulations with the same placement and seed")
        out["dT_T"] = (out["T"] - out["T_ref"]) / out["T_ref"]
        for c in ("e1", "e2", "g1", "g2"):
            out["d" + c] = out[c] - out[c + "_ref"]
        out["flag"] = out["flag"] | out["flag_ref"]
        out.remove_columns(["x_ref", "y_ref", "flag_ref"])

        # Injected errors: the oversampled differences of the same star.
        os_rows = out[out["image"] == "oversampled"]
        keys = ["sim"] + STAR_KEYS
        out = _attach(out, os_rows, keys + ["algorithm"], list(SHAPE_COLUMNS), "_os")
        hsm_os = os_rows[os_rows["algorithm"] == "hsm"]
        out = _attach(out, hsm_os, keys, list(SHAPE_COLUMNS), "_hsm_os")
        return out

    def summary(self, kind="moments", reference=None):
        """Mean, star-to-star std and error on the mean (std/√N) per group.

        Rows with ``flag != 0`` are excluded and counted in ``n_flagged``.

        Parameters
        ----------
        kind : {'moments', 'bias', 'differences'}
            ``'moments'``: raw T, e1, e2, sigma per (sim, image, algorithm).
            ``'bias'``: native vs oversampled (`bias`) per (sim, image, algorithm).
            ``'differences'``: measured vs injected errors (`differences`) per
            (sim, image, algorithm).
        reference : str, optional
            Reference simulation for ``'differences'``.

        Returns
        -------
        astropy.table.Table
            Columns ``<col>_mean``, ``<col>_std``, ``<col>_err`` for each quantity, ``n``
            (stars used) and ``n_flagged``.
        """
        if kind == "moments":
            table, columns = self.table, ["T", "sigma", "e1", "e2"]
        elif kind == "bias":
            table, columns = self.bias(), list(SHAPE_COLUMNS)
        elif kind == "differences":
            table = self.differences(reference)
            columns = list(SHAPE_COLUMNS) + [c + s for s in ("_os", "_hsm_os")
                                             for c in SHAPE_COLUMNS]
        else:
            raise ValueError("kind must be 'moments', 'bias' or 'differences'")
        return summarize(table, ["sim", "image", "algorithm"], columns)

    def save(self, path):
        """Save to a folder: ``moments.fits`` (the table) and ``config.yaml``."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.table.write(path / "moments.fits", overwrite=True)
        write_yaml(path / "config.yaml", {"config": self.config, "provenance": self.provenance})

    def plot_bias(self, **kwargs):
        """Native vs oversampled bias per algorithm (see `plotting.plot_bias`)."""
        return plotting.plot_bias(self, **kwargs)

    def plot_recovery(self, **kwargs):
        """Measured vs injected PSF errors (see `plotting.plot_recovery`)."""
        return plotting.plot_recovery(self, **kwargs)

    def plot_vs_phase(self, **kwargs):
        """Dependence on the sub-pixel phase (see `plotting.plot_vs_phase`)."""
        return plotting.plot_vs_phase(self, **kwargs)

    def plot_vs_snr(self, **kwargs):
        """Dependence on S/N (see `plotting.plot_vs_snr`)."""
        return plotting.plot_vs_snr(self, **kwargs)


def load_moments(path):
    """Load a `MomentsResult` saved with `MomentsResult.save` (or ``run(save_to=...)``)."""
    path = Path(path)
    saved = read_yaml(path / "config.yaml")
    table = Table.read(path / "moments.fits")
    return MomentsResult(table, saved["config"], saved["provenance"])


def summarize(table, group_keys, columns):
    """Mean, std and std/√N of columns per group, over rows with ``flag == 0``.

    Parameters
    ----------
    table : astropy.table.Table
        Must contain ``group_keys``, ``columns`` and ``flag``.
    group_keys : list of str
    columns : list of str

    Returns
    -------
    astropy.table.Table
        One row per group: the keys, ``n``, ``n_flagged`` and ``<col>_mean``, ``<col>_std``,
        ``<col>_err`` (NaN values within good rows are ignored).
    """
    rows = []
    for group in table.group_by(group_keys).groups:
        good = group[group["flag"] == 0]
        row = {k: group[k][0] for k in group_keys}
        row["n"] = len(good)
        row["n_flagged"] = len(group) - len(good)
        for c in columns:
            values = np.asarray(good[c], dtype=float)
            values = values[np.isfinite(values)]
            n = len(values)
            row[c + "_mean"] = values.mean() if n else np.nan
            row[c + "_std"] = values.std(ddof=1) if n > 1 else np.nan
            row[c + "_err"] = values.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append(row)
    return Table(rows=rows)


def _attach(left, right, keys, columns, suffix, join_type="left"):
    """Join ``right[columns]`` (renamed with ``suffix``) onto ``left`` by ``keys``.

    Missing matches (left join) become NaN for float columns and 0 for integer columns.
    """
    sub = right[keys + columns].copy()
    for c in columns:
        sub.rename_column(c, c + suffix)
    if len(sub) == 0:
        out = left.copy()
        for c in columns:
            out[c + suffix] = np.full(len(out), np.nan)
        return out
    out = join(left, sub, keys=keys, join_type=join_type)
    for c in columns:
        col = out[c + suffix]
        if hasattr(col, "mask"):
            fill = 0 if col.dtype.kind in "iu" else np.nan
            out[c + suffix] = col.filled(fill)
    return out
