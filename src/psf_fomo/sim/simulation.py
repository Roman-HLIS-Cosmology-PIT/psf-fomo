"""Simulation builder (`RomanSimulation`), result (`Simulation`) and `load_simulation`.

Example
-------
>>> sim = RomanSimulation(scas=[1], n_stars=9)
>>> sim.set_placement("grid")
>>> sim.add_dilation(1.01, sigma_obs=0.002)
>>> result = sim.run(save_to="out/run01")
>>> stars, rows = result.stars(sca=1)

Every random draw comes from a numpy generator seeded by (seed, obs, sca, purpose), so each
star, error value and noise realization can be regenerated exactly from the configuration.
"""
import copy
import shutil
from pathlib import Path

import numpy as np
import galsim
import galsim.roman
import piff
from astropy.table import Table, vstack

from ..io import provenance, read_yaml, write_yaml
from . import noise as noise_mod
from . import plotting, storage
from .psf import (DEFAULT_N_WAVES, SPATIAL_MODES, apply_errors, clip_to_psd, gaussian_kernel,
                  jitter_kernel, linear_field, make_optical_model, position_profile, optical_profile, random_walk_offsets, timeseries_offsets)
from .stars import draw_magnitudes, draw_positions, mag_to_flux, sub_pixel_phase
from .wcs import DEFAULT_DEC, DEFAULT_RA, make_wcs, pointing_coord

# Purposes of the independent random streams (part of the seed key; do not reorder).
POSITIONS, MAGNITUDES, FIELD, SCATTER, NOISE, OBS_ERRORS, JITTER, KERNEL = range(8)
KERNEL_KEYS = ("mxx", "mxy", "myy")

# Placeholder jitter amplitude for testing: GalSim's galsim.roman.jitter_rms, which GalSim
# documents as a worst-case per-axis RMS for the design. Not a measured Roman value.
DEFAULT_JITTER_RMS_MAS = galsim.roman.jitter_rms * 1000.0


class RomanSimulation:
    """Builder for simulated Roman WFI star images with known truth.

    Create with the common settings, adjust with the ``set_*`` / ``add_*`` methods, then call
    `run`. The whole configuration is a plain dict (``self.config``) that round-trips through
    YAML (`to_yaml`, `from_yaml`).

    Parameters
    ----------
    scas : sequence of int
        SCA numbers, 1 … 18. Default (1,).
    n_obs : int
        Number of observations (exposures). Default 1. Observations share the setup and differ
        through their random draws (positions, fluxes, noise, per-observation errors).
    n_stars : int
        Stars per SCA. Default 1.
    filter : str
        GalSim Roman filter name (e.g. ``'H158'``, ``'J129'``). Default ``'H158'``.
    seed : int
        Master random seed (≥ 0). Default 0.
    stamp_size : int
        Size in native pixels of the postage stamps cut for Piff. Default 25.
    render_size : int
        Size in native pixels of the patch each star is drawn on (wings beyond it are not
        drawn). Default 64.
    max_zernike : int
        Highest Noll index of the truth extra aberrations (4 … 22). Default 22.

    Defaults of the other settings: random placement with random sub-pixel phases; magnitude
    18 (AB); no extra aberrations; no dilation or shear; MTF on at scale 1; no aberration
    scatter; no jitter; chromatic PSFs with a flat SED (drawn through the bandpass); Gaussian
    noise σ = 1 e⁻; simple pixel-scale WCS; Piff's ``'bilinear'`` spatial interpolation of the
    optical PSF.
    """

    def __init__(self, scas=(1,), n_obs=1, n_stars=1, filter="H158", seed=0, stamp_size=25,
                 render_size=64, max_zernike=22):
        scas = [int(s) for s in scas]
        if not scas or any(s < 1 or s > galsim.roman.n_sca for s in scas):
            raise ValueError(f"scas must be a non-empty subset of 1..{galsim.roman.n_sca}")
        if len(set(scas)) != len(scas):
            raise ValueError("scas must not contain duplicates")
        if n_obs < 1 or n_stars < 1:
            raise ValueError("n_obs and n_stars must be >= 1")
        if filter not in galsim.roman.getBandpasses():
            raise ValueError(f"unknown Roman filter {filter!r}")
        if seed < 0:
            raise ValueError("seed must be >= 0")
        if not 4 <= max_zernike <= 22:
            raise ValueError("max_zernike must be in 4..22")
        if stamp_size > render_size:
            raise ValueError("stamp_size must not exceed render_size")
        self.config = {
            "scas": scas,
            "n_obs": int(n_obs),
            "n_stars": int(n_stars),
            "filter": filter,
            "seed": int(seed),
            "stamp_size": int(stamp_size),
            "render_size": int(render_size),
            "max_zernike": int(max_zernike),
            "placement": {"kind": "random", "phase": None, "edge_margin": None},
            "brightness": {"mag": 18.0, "mag_range": None},
            "zernikes": {"values": {}, "sigma_obs": 0.0, "sigma_field": 0.0},
            "dilation": {"value": 1.0, "sigma_obs": 0.0, "sigma_field": 0.0},
            "shear": {"g1": 0.0, "g2": 0.0, "sigma_obs": 0.0, "sigma_field": 0.0},
            "mtf": {"scale": 1.0, "sigma_obs": 0.0, "sigma_field": 0.0},
            "kernel": None,
            "aberration_scatter": 0.0,
            "jitter": None,
            "chromatic": {"sed": "flat", "wave_type": "nm", "flux_type": "fphotons"},
            "noise": {"kind": "gaussian", "sigma": 1.0},
            "wcs": {"kind": "simple", "ra": DEFAULT_RA, "dec": DEFAULT_DEC, "pa": None,
                    "date": None},
            "psf_model": {"spatial_interpolation": "fast", "nominal_interp": "bilinear",
                          "n_waves": DEFAULT_N_WAVES},
        }

    def set_placement(self, kind="random", phase=None, edge_margin=None):
        """Set how stars are placed on each SCA.

        Parameters
        ----------
        kind : {'random', 'grid'}
            Continuous random positions, or an evenly spaced grid with random sub-pixel phases.
        phase : tuple of float, optional
            Fixed sub-pixel phase (fx, fy), each in [-0.5, 0.5]; (0, 0) = pixel centers.
            Default None (random).
        edge_margin : float, optional
            Native pixels kept clear of the SCA edges. Default None = ``render_size // 2 + 1``
            (drawn patches fully on-chip).
        """
        if kind not in ("random", "grid"):
            raise ValueError("kind must be 'random' or 'grid'")
        if phase is not None:
            phase = [float(p) for p in phase]
            if len(phase) != 2 or any(abs(p) > 0.5 for p in phase):
                raise ValueError("phase must be (fx, fy) with each in [-0.5, 0.5]")
        self.config["placement"] = {
            "kind": kind, "phase": phase,
            "edge_margin": None if edge_margin is None else float(edge_margin)}

    def set_brightness(self, mag=None, mag_range=None):
        """Set star magnitudes (AB): a constant ``mag`` or uniform random in ``mag_range``.

        Parameters
        ----------
        mag : float, optional
            Constant magnitude for all stars.
        mag_range : tuple of float, optional
            Limits of a uniform magnitude distribution.
        """
        if (mag is None) == (mag_range is None):
            raise ValueError("give exactly one of mag or mag_range")
        self.config["brightness"] = {
            "mag": None if mag is None else float(mag),
            "mag_range": None if mag_range is None else [float(m) for m in mag_range]}

    def set_zernikes(self, values=None, sigma_obs=0.0, sigma_field=0.0):
        """Set the truth extra Zernike aberrations, in waves.

        Parameters
        ----------
        values : dict, optional
            Noll index (4 … max_zernike) → aberration in waves. Default None (all zero).
        sigma_obs : float or dict
            Standard deviation (waves) of a per-observation offset; a float applies to all
            Zernikes Z4 … Z_max, a dict gives it per Noll index. Default 0.
        sigma_field : float or dict
            Scale (waves) of a random linear field a + b x̂ + c ŷ per observation and SCA
            (x̂ = x / 4088); float or per-index dict as for ``sigma_obs``. Default 0.
        """
        values = {} if values is None else values
        if not isinstance(values, dict):
            raise ValueError("values must be a dict {Noll index: waves}")
        for spec in (values, sigma_obs, sigma_field):
            if isinstance(spec, dict):
                bad = [k for k in spec if not 4 <= int(k) <= self.config["max_zernike"]]
                if bad:
                    raise ValueError(f"Noll indices {bad} outside 4..max_zernike")

        def clean(spec):
            if isinstance(spec, dict):
                return {int(k): float(v) for k, v in spec.items()}
            return float(spec)

        self.config["zernikes"] = {"values": clean(values), "sigma_obs": clean(sigma_obs),
                                   "sigma_field": clean(sigma_field)}

    def add_dilation(self, value, sigma_obs=0.0, sigma_field=0.0):
        """Set the size factor of the optical PSF (1 = no change).

        Parameters
        ----------
        value : float
            Dilation factor. The MTF kernel width is multiplied by it as well
            (docs/conventions.md).
        sigma_obs : float
            Standard deviation of a per-observation additive offset. Default 0.
        sigma_field : float
            Scale of a random linear field across each SCA. Default 0.
        """
        self.config["dilation"] = {"value": float(value), "sigma_obs": float(sigma_obs),
                                   "sigma_field": float(sigma_field)}

    def add_shear(self, g1=0.0, g2=0.0, sigma_obs=0.0, sigma_field=0.0):
        """Set a reduced shear of the optical PSF (world u, v axes).

        Parameters
        ----------
        g1, g2 : float
            Reduced shear components.
        sigma_obs : float
            Standard deviation of per-observation offsets (each of g1, g2). Default 0.
        sigma_field : float
            Scale of random linear fields across each SCA (each of g1, g2). Default 0.
        """
        self.config["shear"] = {"g1": float(g1), "g2": float(g2),
                                "sigma_obs": float(sigma_obs), "sigma_field": float(sigma_field)}

    def add_kernel(self, mxx=0.0, mxy=0.0, myy=0.0, sigma_obs=0.0, sigma_field=0.0):
        """Convolve the PSF with an elliptical Gaussian kernel of second moments M.

        The kernel adds M to the PSF's second moments (before the pixel). It is fixed on the
        sky: not dilated with the optics, applied after shear. ``add_kernel(None)`` turns
        it off.

        Parameters
        ----------
        mxx, mxy, myy : float or None
            Elements of M in native pix² along the world (u, v) axes (× 0.11″² gives arcsec²).
            M must be positive semi-definite (a kernel can only broaden). ``mxx=None``: no
            kernel.
        sigma_obs : float
            Standard deviation (pix²) of a per-observation offset of each element. Default 0.
        sigma_field : float
            Scale (pix²) of a random linear field a + b x̂ + c ŷ of each element per
            observation and SCA. Default 0.

        Notes
        -----
        With variations, a star's M can leave the PSD cone; it is then clipped to the nearest
        PSD matrix (negative eigenvalue set to 0). The catalog columns ``kernel_mxx`` …
        record the M actually used.

        Example
        -------
        >>> sim.add_kernel(0.022, 0.0, 0.022)   # like a 10 % wider MTF in second moments
        """
        if mxx is None:
            self.config["kernel"] = None
            return
        mxx, mxy, myy = float(mxx), float(mxy), float(myy)
        if mxx < 0 or myy < 0 or mxx * myy - mxy * mxy < 0:
            raise ValueError("kernel moments must form a positive semi-definite matrix")
        if sigma_obs < 0 or sigma_field < 0:
            raise ValueError("sigma_obs and sigma_field must be >= 0")
        self.config["kernel"] = {"mxx": mxx, "mxy": mxy, "myy": myy,
                                 "sigma_obs": float(sigma_obs),
                                 "sigma_field": float(sigma_field)}

    def set_mtf(self, scale=1.0, sigma_obs=0.0, sigma_field=0.0):
        """Set the charge-diffusion (MTF) kernel; ``set_mtf(None)`` turns it off.

        Parameters
        ----------
        scale : float or None
            Width factor of the three-Gaussian kernel (1 = nominal). None: no MTF.
        sigma_obs : float
            Standard deviation of a per-observation offset of ``scale``. Default 0.
        sigma_field : float
            Scale of a random linear field of ``scale`` across each SCA. Default 0.
        """
        self.config["mtf"] = None if scale is None else {
            "scale": float(scale), "sigma_obs": float(sigma_obs),
            "sigma_field": float(sigma_field)}

    def add_aberration_scatter(self, sigma):
        """Per-star Gaussian scatter added to every truth Zernike, in waves.

        Note: each star then needs its own optical PSF evaluation (slow for many stars).
        """
        self.config["aberration_scatter"] = float(sigma)

    def add_jitter(self, model="random_walk", rms_mas=DEFAULT_JITTER_RMS_MAS, n_steps=100,
                   t=None, dx_mas=None, dy_mas=None):
        """Add pointing jitter: the PSF is convolved with the distribution of pointing offsets.

        Parameters
        ----------
        model : {'random_walk', 'timeseries'} or None
            ``'random_walk'``: a new random walk per observation, scaled to RMS ``rms_mas``
            per axis. ``'timeseries'``: a user time series (``t``, ``dx_mas``, ``dy_mas``),
            identical in every observation. None turns jitter off.
        rms_mas : float
            RMS per axis in milliarcsec (random walk). The default is a placeholder for
            testing (``galsim.roman.jitter_rms``, GalSim's worst-case design value), not a
            measured Roman value.
        n_steps : int
            Number of time samples of the random walk (≥ 2). Default 100.
        t : array_like
            Times in seconds (timeseries).
        dx_mas, dy_mas : array_like
            Offsets along world u, v in milliarcsec (timeseries).
        """
        if model is None:
            self.config["jitter"] = None
        elif model == "random_walk":
            if n_steps < 2:
                raise ValueError("n_steps must be >= 2")
            self.config["jitter"] = {"model": model, "rms_mas": float(rms_mas),
                                     "n_steps": int(n_steps)}
        elif model == "timeseries":
            t, dx, dy = (np.asarray(a, dtype=float) for a in (t, dx_mas, dy_mas))
            if not (t.ndim == 1 and t.shape == dx.shape == dy.shape and len(t) >= 2):
                raise ValueError("t, dx_mas, dy_mas must be 1-d arrays of equal length >= 2")
            if np.any(np.diff(t) <= 0):
                raise ValueError("t must be strictly increasing")
            self.config["jitter"] = {"model": model, "t": t.tolist(), "dx_mas": dx.tolist(),
                                     "dy_mas": dy.tolist()}
        else:
            raise ValueError("model must be 'random_walk', 'timeseries' or None")

    def set_chromatic(self, sed="flat", wave_type="nm", flux_type="fphotons"):
        """The stars' SED (one shared by all stars), or monochromatic PSFs.

        Chromatic (the default, with a flat SED): the PSF is wavelength dependent and each
        star is drawn through the bandpass, weighted by the SED. Monochromatic
        (``set_chromatic(None)``): the PSF at the bandpass's effective wavelength.

        Parameters
        ----------
        sed : 'flat', str, galsim.SED or None
            ``'flat'`` (default): constant photons per unit wavelength. A file name: read by
            ``galsim.SED`` with ``wave_type`` and ``flux_type``. A ``galsim.SED`` (spectral):
            stored in the configuration as a table sampled every 0.5 nm over the filter's
            wavelength range (photons per nm; finer spectral features are smoothed). None:
            monochromatic.
        wave_type, flux_type : str
            Units of an SED file, as for ``galsim.SED``. Ignored otherwise.

        Example
        -------
        >>> sim.set_chromatic(galsim.SED("vega.txt", wave_type="nm", flux_type="flambda"))
        """
        bandpass = galsim.roman.getBandpasses()[self.config["filter"]]
        self.config["chromatic"] = storage.sed_setting(sed, bandpass, wave_type, flux_type)

    def set_noise(self, kind="gaussian", sigma=1.0, effects=None):
        """Set the noise model. Images are stored in ADU (divided by ``galsim.roman.gain``).

        Parameters
        ----------
        kind : {'gaussian', 'roman-L1', 'roman-L2'}
            Gaussian noise of standard deviation ``sigma``, or Roman noise and detector
            effects: ``'roman-L1'`` applies every effect uncorrected; ``'roman-L2'`` leaves
            out what L1 → L2 processing corrects (nonlinearity, IPC) and the quantization
            (see `psf_fomo.sim.noise`).
        sigma : float
            Gaussian noise in e⁻ per pixel (> 0). Default 1.
        effects : sequence of str, optional
            Roman effects to apply (subset of ``noise.ROMAN_EFFECTS``). Default: the kind's
            own set, ``noise.ROMAN_L1_EFFECTS`` (all) or ``noise.ROMAN_L2_EFFECTS`` (sky,
            poisson, reciprocity, dark, read).

        Examples
        --------
        >>> sim.set_noise("roman-L2")                              # L2-like image
        >>> sim.set_noise("roman-L2", effects=["sky", "poisson", "read"])
        """
        if kind == "gaussian":
            if sigma <= 0:
                raise ValueError("sigma must be > 0")
            self.config["noise"] = {"kind": kind, "sigma": float(sigma)}
        elif kind == "roman":
            raise ValueError("kind 'roman' is now 'roman-L1' (every effect) or 'roman-L2' "
                             "(the effects left in L2 images)")
        elif kind in noise_mod.ROMAN_KINDS:
            effects = list(noise_mod.ROMAN_KINDS[kind] if effects is None else effects)
            unknown = set(effects) - set(noise_mod.ROMAN_EFFECTS)
            if unknown:
                raise ValueError(f"unknown effects {sorted(unknown)}; "
                                 f"allowed: {noise_mod.ROMAN_EFFECTS}")
            if not {"read", "poisson", "dark"} & set(effects):
                raise ValueError("Roman noise needs at least one of 'read', 'poisson', 'dark'")
            self.config["noise"] = {"kind": kind, "effects": effects}
        else:
            raise ValueError("kind must be 'gaussian', 'roman-L1' or 'roman-L2'")

    def set_wcs(self, kind="simple", ra=DEFAULT_RA, dec=DEFAULT_DEC, pa=None, date=None):
        """Set the WCS.

        Parameters
        ----------
        kind : {'simple', 'roman'}
            Plain 0.11″ pixel scale, or the realistic Roman WCS from ``galsim.roman.getWCS``.
        ra, dec : float
            Pointing in degrees. Also used for the sky level with the simple WCS.
        pa : float, optional
            Position angle in degrees; None lets GalSim choose.
        date : str, optional
            Observation date/time, ISO format (e.g. ``'2027-03-20'``); None = GalSim default.
        """
        if kind not in ("simple", "roman"):
            raise ValueError("kind must be 'simple' or 'roman'")
        self.config["wcs"] = {"kind": kind, "ra": float(ra), "dec": float(dec),
                              "pa": None if pa is None else float(pa), "date": date}

    def set_psf_model(self, spatial_interpolation="fast", nominal_interp="bilinear",
                      n_waves="default"):
        """Set how the optical PSF is evaluated across an SCA and across the bandpass.

        Parameters
        ----------
        spatial_interpolation : {'fast', 'piff', 'exact'}
            ``'piff'``: Piff's ``RomanOpticalModel`` exactly — Roman PSFs at the SCA corners
            blended with position weights; chromatic PSFs drawn through the bandpass as Piff
            draws them (slow: every star sums each corner over the filter). ``'fast'``
            (default): the same corner blending, but each corner PSF is summed over the filter
            once per SCA and observation (an effective PSF), so each star is cheap.
            ``'exact'``: ``galsim.roman.getPSF`` at each star's own position (no corner
            blending), summed over the filter per star.
        nominal_interp : {'bilinear', 'five_point'}
            Piff option for the corner blending (``'piff'``, ``'fast'``). Default
            ``'bilinear'`` (four SCA corners).
        n_waves : int, None or 'default'
            Chromatic PSFs, ``'fast'`` and ``'exact'``: number of wavelengths (≥ 2), evenly
            spaced from the filter's blue to red limit, at which the optical PSF is computed
            for the sum over the filter (the PSF is taken as linear in wavelength between
            them); more wavelengths are more accurate and slower to set up. None: one every
            5 nm. ``'default'``: 10, for every filter.
            Must be ``'default'`` or None for ``'piff'``, which is always exact in wavelength.
        """
        if spatial_interpolation not in SPATIAL_MODES:
            raise ValueError(f"spatial_interpolation must be one of {SPATIAL_MODES}")
        if nominal_interp not in ("bilinear", "five_point"):
            raise ValueError("nominal_interp must be 'bilinear' or 'five_point'")
        if spatial_interpolation == "piff":
            if n_waves not in ("default", None):
                raise ValueError("n_waves does not apply to spatial_interpolation='piff' "
                                 "(Piff's model is exact in wavelength)")
            n_waves = None
        elif n_waves == "default":
            n_waves = DEFAULT_N_WAVES
        if n_waves is not None and int(n_waves) < 2:
            raise ValueError("n_waves must be at least 2 (or None)")
        self.config["psf_model"] = {"spatial_interpolation": spatial_interpolation,
                                    "nominal_interp": nominal_interp,
                                    "n_waves": None if n_waves is None else int(n_waves)}

    def to_yaml(self, path):
        """Write the configuration to a YAML file."""
        write_yaml(path, self.config)

    @classmethod
    def from_config(cls, config):
        """Build a simulation from a configuration dict (as in ``self.config``)."""
        sim = cls(scas=config["scas"], n_obs=config["n_obs"], n_stars=config["n_stars"],
                  filter=config["filter"], seed=config["seed"],
                  stamp_size=config["stamp_size"], render_size=config["render_size"],
                  max_zernike=config["max_zernike"])
        sim.config = copy.deepcopy(config)
        sim.config["noise"] = noise_mod.upgrade_noise_config(sim.config["noise"])
        return sim

    @classmethod
    def from_yaml(cls, path):
        """Build a simulation from a YAML file written by `to_yaml`."""
        return cls.from_config(read_yaml(path))

    def run(self, save_to=None, save_noiseless=False):
        """Render all observations and SCAs.

        Parameters
        ----------
        save_to : str or Path, optional
            Folder to save the simulation to. Default None (not saved).
        save_noiseless : bool
            Also keep (and save) the noiseless images. Default False.

        Returns
        -------
        Simulation
        """
        config = copy.deepcopy(self.config)
        truth = _Truth(config)
        images, noiseless, catalogs = {}, {}, []
        for obs in range(config["n_obs"]):
            tables = []
            for sca in config["scas"]:
                table = truth.sca_truth(obs, sca)[0].copy()
                image = galsim.ImageF(galsim.roman.n_pix, galsim.roman.n_pix,
                                      wcs=truth.wcs[sca])
                for i in range(len(table)):
                    truth.draw_star(image, obs, sca, i)
                if save_noiseless:
                    noiseless[obs, sca] = image / galsim.roman.gain
                table["bkg_var"] = truth.add_noise(image, obs, sca, table)
                table["bkg_var"].unit = "adu2"
                images[obs, sca] = image
                tables.append(table)
            catalogs.append(vstack(tables))
        result = Simulation(config, provenance(), images, catalogs,
                            noiseless if save_noiseless else None)
        if save_to is not None:
            result.save(save_to)
        return result


class Simulation:
    """Result of `RomanSimulation.run` (or `load_simulation`).

    Attributes
    ----------
    config : dict
        The full simulation configuration.
    provenance : dict
        Software versions and creation time.
    images : dict
        (obs, sca) → full-SCA galsim.ImageF in ADU, background subtracted.
    catalogs : list of astropy.table.Table
        One star table per observation. Columns: ``star_id``, ``obs``, ``sca``, ``x``, ``y``
        (native pixels, GalSim convention), ``phase_x``, ``phase_y``, ``u``, ``v`` (arcsec,
        simple WCS) or ``ra``, ``dec`` (deg, Roman WCS), ``mag`` (AB), ``flux`` (e⁻),
        ``dilation``, ``g1``, ``g2``, ``mtf_scale`` (if MTF on), ``kernel_mxx``,
        ``kernel_mxy``, ``kernel_myy`` (native pix², if a Gaussian kernel is set), ``z4`` …
        (waves, value at the star), ``sed_id`` (if chromatic), ``bkg_var`` (ADU², expected
        background variance).
    noiseless : dict or None
        (obs, sca) → noiseless image in ADU, if requested.
    """

    def __init__(self, config, provenance, images, catalogs, noiseless=None):
        self.config = config
        self.provenance = provenance
        self.images = images
        self.catalogs = catalogs
        self.noiseless = noiseless
        self._truth = None

    @property
    def truth(self):
        """Deterministic truth model used to re-render stars (built on first use)."""
        if self._truth is None:
            self._truth = _Truth(self.config)
        return self._truth

    def catalog(self, obs=0, sca=None):
        """Star table of one observation, optionally restricted to one SCA."""
        table = self.catalogs[obs]
        return table if sca is None else table[table["sca"] == sca]

    def stars(self, sca=None, obs=0):
        """Piff-ready stars cut from the images, with their truth rows.

        Each star has a ``stamp_size`` stamp, a constant weight 1 / ``bkg_var``, and the
        properties ``sca``, ``chipnum`` (= sca), ``star_id``, ``obs`` and, when star shot noise
        is simulated, ``gain`` (so Piff adds the star's Poisson variance from its model).

        Parameters
        ----------
        sca : int, optional
            SCA number; default None = all SCAs of the observation.
        obs : int
            Observation index. Default 0.

        Returns
        -------
        stars : list of piff.Star
        rows : astropy.table.Table
            The matching catalog rows, in the same order.
        """
        rows = self.catalog(obs, sca)
        if np.any(rows["bkg_var"] <= 0):
            raise ValueError("zero background variance: cannot build Piff weights")
        gain = noise_mod.star_gain(self.config["noise"])
        truth = self.truth
        sed_eff = None
        if truth.sed is not None:
            sed_eff = truth.sed.withFlux(1.0, truth.bandpass) * truth.bandpass
        stars = []
        for row in rows:
            full = self.images[obs, int(row["sca"])]
            pos = galsim.PositionD(float(row["x"]), float(row["y"]))
            bounds = _stamp_bounds(pos, self.config["stamp_size"])
            image = full[bounds].copy()
            weight = galsim.ImageF(bounds, wcs=full.wcs, init_value=1.0 / float(row["bkg_var"]))
            props = {"sca": int(row["sca"]), "chipnum": int(row["sca"]),
                     "star_id": int(row["star_id"]), "obs": int(obs)}
            if gain is not None:
                props["gain"] = gain
            data = piff.StarData(image, pos, weight=weight, pointing=truth.pointing,
                                 properties=props, bandpass=truth.bandpass, sed_eff=sed_eff)
            stars.append(piff.Star(data, None))
        return stars, rows

    def render_stamp(self, star_id, obs=0, size=None):
        """Re-render one star's noiseless stamp from the configuration and seed.

        Drawn exactly as in `RomanSimulation.run` (same patch, same draw call), so it equals
        the noiseless image of an isolated star.

        Parameters
        ----------
        star_id : int
            Star id (catalog ``star_id``).
        obs : int
            Observation index. Default 0.
        size : int, optional
            Stamp size in native pixels. Default ``stamp_size``.

        Returns
        -------
        galsim.Image
            Noiseless stamp in ADU, in the image coordinates of the SCA.
        """
        sca, i = self._locate(star_id, obs)
        patch = self.truth.render_patch(obs, sca, i) / galsim.roman.gain
        row = self.truth.sca_truth(obs, sca)[0][i]
        pos = galsim.PositionD(float(row["x"]), float(row["y"]))
        return patch[_stamp_bounds(pos, size or self.config["stamp_size"])]

    def truth_psf(self, sca, obs, star_id, oversample=5, include_pixel=True, size=None):
        """The truth PSF of one star on a finer grid, unit flux, centered on the image.

        Parameters
        ----------
        sca : int
            SCA number of the star.
        obs : int
            Observation index.
        star_id : int
            Star id.
        oversample : int
            Grid refinement: the output pixel is 1/oversample of the native pixel. Default 5.
        include_pixel : bool
            If True (default), the effective PSF: convolved with the native pixel response,
            then sampled at the finer grid points (the "oversampled image" convention).
            If False, the pre-pixel PSF sampled at the finer grid points.
        size : int, optional
            Size in native pixels; the image has size × oversample pixels. Default
            ``stamp_size``.

        Returns
        -------
        galsim.Image
            Image of the unit-flux PSF, with a WCS that is the local WCS at the star scaled by
            1/oversample. The image sums to the flux fraction inside the box (≈ 0.93 for
            25 native pixels in H158, because of the PSF wings), not to 1.
        """
        found_sca, i = self._locate(star_id, obs)
        if found_sca != sca:
            raise ValueError(f"star {star_id} is on SCA {found_sca}, not {sca}")
        truth = self.truth
        row = truth.sca_truth(obs, sca)[0][i]
        local = truth.wcs[sca].local(galsim.PositionD(float(row["x"]), float(row["y"])))
        prof = truth.profile(obs, sca, i)
        if include_pixel:
            prof = galsim.Convolve(prof, local.toWorld(galsim.Pixel(1.0)))
        n = (size or self.config["stamp_size"]) * oversample
        # jacobian() gives a JacobianWCS for any local WCS (a PixelScale has no getMatrix).
        fine = galsim.JacobianWCS(*(local.jacobian().getMatrix().ravel() / oversample))
        image = galsim.ImageD(n, n, wcs=fine)
        truth.draw(prof, 1.0, image, center=None, method="no_pixel")
        return image

    def save(self, path):
        """Save to a folder (see `psf_fomo.sim.storage` for the layout)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(self.config)
        chromatic = config["chromatic"]
        if chromatic is not None and chromatic["sed"] in ("flat", "table"):
            # Written for Piff's file input (sed_file_name); the config keeps its own copy.
            wave, flux = storage.sample_sed(self.truth.sed, self.truth.bandpass)
            storage.write_sed(path / storage.SED_FILE, wave, flux)
        elif chromatic is not None:
            sed_file = Path(chromatic["sed"])
            if sed_file.resolve() != (path / sed_file.name).resolve():
                shutil.copy(sed_file, path / sed_file.name)
            chromatic["sed"] = sed_file.name  # relative to the saved folder
        write_yaml(path / "config.yaml", {"config": config, "provenance": self.provenance})
        for obs, table in enumerate(self.catalogs):
            headers = {sca: {"GAIN": galsim.roman.gain, "BUNIT": "ADU",
                             "BKGVAR": float(np.median(table[table["sca"] == sca]["bkg_var"]))}
                       for sca in self.config["scas"]}
            storage.write_images(path / storage.image_file_name(obs),
                                 {sca: self.images[obs, sca] for sca in self.config["scas"]},
                                 headers)
            if self.noiseless is not None:
                storage.write_images(path / storage.image_file_name(obs, noiseless=True),
                                     {sca: self.noiseless[obs, sca]
                                      for sca in self.config["scas"]},
                                     headers)
            storage.write_catalog(path / storage.catalog_file_name(obs), table,
                                  per_sca_hdus=self.config["wcs"]["kind"] == "simple")

    def _locate(self, star_id, obs):
        """(sca, index within that SCA) of a star id."""
        n = self.config["n_stars"]
        scas = self.config["scas"]
        if not 0 <= star_id < n * len(scas) or not 0 <= obs < self.config["n_obs"]:
            raise ValueError(f"no star {star_id} in observation {obs}")
        return scas[star_id // n], star_id % n

    def plot_sca(self, obs=0, sca=None, ax=None):
        """Full-SCA image with star markers (see `plotting.plot_sca`)."""
        return plotting.plot_sca(self, obs=obs, sca=sca, ax=ax)

    def plot_stamps(self, obs=0, sca=None, n_max=16):
        """Gallery of star stamps (see `plotting.plot_stamps`)."""
        return plotting.plot_stamps(self, obs=obs, sca=sca, n_max=n_max)

    def plot_truth_vs_image(self, star_id, obs=0):
        """Image stamp, noiseless truth and residual (see `plotting.plot_truth_vs_image`)."""
        return plotting.plot_truth_vs_image(self, star_id, obs=obs)

    def plot_phases(self, obs=None):
        """Sub-pixel phase coverage (see `plotting.plot_phases`)."""
        return plotting.plot_phases(self, obs=obs)

    def plot_error_field(self, column="dilation", obs=0):
        """A truth error across each SCA (see `plotting.plot_error_field`)."""
        return plotting.plot_error_field(self, column=column, obs=obs)

    def plot_error_vs_obs(self, column="dilation"):
        """Per-observation mean of a truth error (see `plotting.plot_error_vs_obs`)."""
        return plotting.plot_error_vs_obs(self, column=column)


def load_simulation(path):
    """Load a simulation saved with `Simulation.save` (or ``run(save_to=...)``).

    Parameters
    ----------
    path : str or Path
        Folder of the saved simulation.

    Returns
    -------
    Simulation
    """
    path = Path(path)
    saved = read_yaml(path / "config.yaml")
    config = saved["config"]
    config["noise"] = noise_mod.upgrade_noise_config(config["noise"])
    chromatic = config["chromatic"]
    if chromatic is not None and chromatic["sed"] not in ("flat", "table"):
        chromatic["sed"] = str(path / chromatic["sed"])
    scas = config["scas"]
    images, noiseless, catalogs = {}, {}, []
    for obs in range(config["n_obs"]):
        for sca, image in storage.read_images(path / storage.image_file_name(obs), scas).items():
            images[obs, sca] = image
        noiseless_file = path / storage.image_file_name(obs, noiseless=True)
        if noiseless_file.exists():
            for sca, image in storage.read_images(noiseless_file, scas).items():
                noiseless[obs, sca] = image
        catalogs.append(storage.read_catalog(path / storage.catalog_file_name(obs)))
    return Simulation(config, saved["provenance"], images, catalogs, noiseless or None)


class _Truth:
    """Everything needed to render any star of a configured simulation, deterministically."""

    def __init__(self, config):
        self.config = config
        self.bandpass = galsim.roman.getBandpasses(AB_zeropoint=True)[config["filter"]]
        self.wcs, self.pointing = make_wcs(config["wcs"], config["scas"])
        self.nz = config["max_zernike"] - 3
        zern = config["zernikes"]
        self.z_value = self._zernike_vector(zern["values"], default=0.0)
        self.z_sigma_obs = self._zernike_vector(zern["sigma_obs"])
        self.z_sigma_field = self._zernike_vector(zern["sigma_field"])
        # A Zernike field needs Piff's 'linear' aberration mode; otherwise one vector per SCA.
        self.linear = bool(np.any(self.z_sigma_field > 0))
        self.sed = _make_sed(config["chromatic"], self.bandpass)
        psf_model = config["psf_model"]
        # Configurations saved before these settings existed used Piff's model.
        self.spatial = psf_model.get("spatial_interpolation", "piff")
        self.n_waves = None if self.spatial == "piff" else psf_model.get("n_waves")
        # Effective-PSF grids twice the render patch: light beyond them never reaches a patch.
        self.extent = 2 * config["render_size"] * galsim.roman.pixel_scale
        self.model = make_optical_model(config["filter"], self.bandpass, config["max_zernike"],
                                        self.linear, psf_model["nominal_interp"],
                                        chromatic=self.sed is not None,
                                        sed=self.sed if self.spatial == "fast" else None,
                                        n_waves=self.n_waves, extent=self.extent)
        self._obs = {}
        self._sca = {}
        self._background = {}

    def rng(self, obs, sca, purpose):
        """Independent generator for (obs, sca, purpose); sca = 0 for per-observation draws."""
        key = np.random.SeedSequence(self.config["seed"], spawn_key=(obs, sca, purpose))
        return np.random.default_rng(key)

    def _zernike_vector(self, spec, default=None):
        if isinstance(spec, dict):
            vec = np.zeros(self.nz)
            for noll, value in spec.items():
                vec[int(noll) - 4] = value
            return vec
        return np.full(self.nz, float(spec if default is None else default))

    def obs_truth(self, obs):
        """Per-observation error offsets and jitter kernel (shared by all SCAs)."""
        if obs not in self._obs:
            cfg = self.config
            rng = self.rng(obs, 0, OBS_ERRORS)
            mtf = cfg["mtf"]
            # Draw every offset even when its sigma is 0, so the streams do not shift
            # when one setting changes.
            out = {
                "dilation": rng.normal(0.0, cfg["dilation"]["sigma_obs"]),
                "g1": rng.normal(0.0, cfg["shear"]["sigma_obs"]),
                "g2": rng.normal(0.0, cfg["shear"]["sigma_obs"]),
                "mtf_scale": rng.normal(0.0, mtf["sigma_obs"] if mtf else 0.0),
                "zernikes": rng.normal(0.0, self.z_sigma_obs),
                "jitter": self._jitter(obs),
            }
            self._obs[obs] = out
        return self._obs[obs]

    def _jitter(self, obs):
        jit = self.config["jitter"]
        if jit is None:
            return None
        if jit["model"] == "random_walk":
            dx, dy, w = random_walk_offsets(jit["rms_mas"], jit["n_steps"],
                                            self.rng(obs, 0, JITTER))
        else:
            dx, dy, w = timeseries_offsets(jit["t"], jit["dx_mas"], jit["dy_mas"])
        return jitter_kernel(dx, dy, w)

    def sca_truth(self, obs, sca):
        """Truth star table of one (obs, sca) and the Piff Zernike parameters of each star."""
        if (obs, sca) in self._sca:
            return self._sca[obs, sca]
        cfg = self.config
        n = cfg["n_stars"]
        place = cfg["placement"]
        margin = (cfg["render_size"] // 2 + 1 if place["edge_margin"] is None
                  else place["edge_margin"])
        x, y = draw_positions(n, place["kind"], self.rng(obs, sca, POSITIONS), margin,
                              place["phase"])
        mag = draw_magnitudes(n, self.rng(obs, sca, MAGNITUDES), **cfg["brightness"])
        off = self.obs_truth(obs)
        field_rng = self.rng(obs, sca, FIELD)

        def field(base, offset, sigma_field):
            coeffs = field_rng.normal(0.0, sigma_field, 3)
            return base + offset + linear_field(coeffs, x, y)

        mtf = cfg["mtf"]
        dilation = field(cfg["dilation"]["value"], off["dilation"],
                         cfg["dilation"]["sigma_field"])
        g1 = field(cfg["shear"]["g1"], off["g1"], cfg["shear"]["sigma_field"])
        g2 = field(cfg["shear"]["g2"], off["g2"], cfg["shear"]["sigma_field"])
        mtf_scale = field(mtf["scale"] if mtf else 1.0, off["mtf_scale"],
                          mtf["sigma_field"] if mtf else 0.0)
        zc = field_rng.normal(0.0, self.z_sigma_field, size=(3, self.nz))
        a = self.z_value + off["zernikes"] + zc[0]
        scatter = self.rng(obs, sca, SCATTER).normal(0.0, cfg["aberration_scatter"],
                                                     size=(n, self.nz))
        z_star = linear_field((a, zc[1], zc[2]), x, y) + scatter
        if self.linear:
            params = np.hstack([a + scatter, np.tile(zc[1], (n, 1)), np.tile(zc[2], (n, 1))])
        else:
            params = a + scatter

        table = Table()
        first_id = cfg["scas"].index(sca) * n
        table["star_id"] = np.arange(first_id, first_id + n)
        table["obs"] = np.full(n, obs)
        table["sca"] = np.full(n, sca)
        table["x"] = x
        table["y"] = y
        table["phase_x"] = sub_pixel_phase(x)
        table["phase_y"] = sub_pixel_phase(y)
        world = [self.wcs[sca].toWorld(galsim.PositionD(xi, yi)) for xi, yi in zip(x, y)]
        if self.pointing is None:
            table["u"] = [w.x for w in world]
            table["v"] = [w.y for w in world]
        else:
            table["ra"] = [w.ra.deg for w in world]
            table["dec"] = [w.dec.deg for w in world]
        table["mag"] = mag
        table["flux"] = mag_to_flux(mag, self.bandpass)
        table["dilation"] = dilation
        table["g1"] = g1
        table["g2"] = g2
        if mtf:
            table["mtf_scale"] = mtf_scale
        for k in range(self.nz):
            table[f"z{k + 4}"] = z_star[:, k]
            table[f"z{k + 4}"].description = "extra aberration at the star, waves"
        kernel = self._kernel_moments(obs, sca, x, y)
        if kernel is not None:
            for key, values in zip(KERNEL_KEYS, kernel):
                table[f"kernel_{key}"] = values
                table[f"kernel_{key}"].description = "Gaussian-kernel moment, native pix2"
        if self.sed is not None:
            table["sed_id"] = np.zeros(n, dtype=int)
        for name, unit in [("x", "pix"), ("y", "pix"), ("phase_x", "pix"), ("phase_y", "pix"),
                           ("u", "arcsec"), ("v", "arcsec"), ("ra", "deg"), ("dec", "deg"),
                           ("mag", "mag")]:
            if name in table.colnames:
                table[name].unit = unit
        # FITS has no electron unit, so the unit goes in the description instead.
        table["flux"].description = "detected flux, e-"
        self._sca[obs, sca] = (table, params)
        return self._sca[obs, sca]

    def _kernel_moments(self, obs, sca, x, y):
        """Per-star Gaussian-kernel moments (mxx, mxy, myy) arrays, PSD-clipped, or None.

        The kernel has its own random streams, so adding it leaves all other draws unchanged.
        """
        kern = self.config.get("kernel")  # absent in configs saved before it existed
        if kern is None:
            return None
        offsets = self.rng(obs, 0, KERNEL).normal(0.0, kern["sigma_obs"], 3)
        coeffs = self.rng(obs, sca, KERNEL).normal(0.0, kern["sigma_field"], (3, 3))
        values = [kern[key] + off + linear_field(c, x, y)
                  for key, off, c in zip(KERNEL_KEYS, offsets, coeffs)]
        clipped = np.array([clip_to_psd(*m) for m in zip(*values)])
        return clipped[:, 0], clipped[:, 1], clipped[:, 2]

    def profile(self, obs, sca, i):
        """Unit-flux truth PSF (no pixel) of star i of (obs, sca), world coordinates."""
        table, params = self.sca_truth(obs, sca)
        row = table[i]
        x, y = float(row["x"]), float(row["y"])
        if self.spatial == "exact":
            # The catalog's aberrations at the star (incl. any field and per-star scatter).
            zernikes = [float(row[f"z{k + 4}"]) for k in range(self.nz)]
            optical = position_profile(sca, self.config["filter"], self.bandpass, x, y,
                                       self.wcs[sca], zernikes, sed=self.sed,
                                       n_waves=self.n_waves, extent=self.extent)
        else:
            optical = optical_profile(self.model, params[i], x, y, sca, self.wcs[sca],
                                      self.pointing)
        dilation = float(row["dilation"])
        mtf_width = dilation * float(row["mtf_scale"]) if self.config["mtf"] else None
        kernel = None
        if "kernel_mxx" in table.colnames:
            moments = [float(row[f"kernel_{key}"]) for key in KERNEL_KEYS]
            if any(moments):
                kernel = gaussian_kernel(*moments)
        return apply_errors(optical, self.wcs[sca].local(galsim.PositionD(x, y)),
                            dilation=dilation, g1=float(row["g1"]), g2=float(row["g2"]),
                            kernel=kernel, jitter=self.obs_truth(obs)["jitter"],
                            mtf_width=mtf_width)

    def draw(self, prof, flux, image, center, method):
        """Draw a unit-flux truth profile with the given flux (e⁻) onto an image.

        A chromatic profile (``'piff'`` mode) is drawn as Piff's RomanOptics draws its model:
        × flux × the effective SED (SED × bandpass, unit flux), through the flat bandpass.
        Effective (``'fast'``, ``'exact'``) and monochromatic profiles are achromatic; their
        flux is scaled, not reset, so light outside an effective-PSF grid stays dropped.
        """
        if isinstance(prof, galsim.ChromaticObject):
            sed_eff = self.sed * self.bandpass
            (prof * flux * sed_eff).drawImage(self.model.flat_bandpass, image=image,
                                              method=method, center=center)
        else:
            prof.withScaledFlux(flux).drawImage(image=image, method=method, center=center)

    def render_patch(self, obs, sca, i):
        """Noiseless ``render_size`` patch of star i in e⁻, in SCA image coordinates."""
        row = self.sca_truth(obs, sca)[0][i]
        pos = galsim.PositionD(float(row["x"]), float(row["y"]))
        patch = galsim.ImageD(_stamp_bounds(pos, self.config["render_size"]),
                              wcs=self.wcs[sca].local(pos))
        self.draw(self.profile(obs, sca, i), float(row["flux"]), patch, pos, "auto")
        return patch

    def draw_star(self, image, obs, sca, i):
        """Add star i of (obs, sca) to a full-SCA image (e⁻)."""
        patch = self.render_patch(obs, sca, i)
        image[patch.bounds] += patch

    def background(self, sca):
        """Expected sky + thermal background of an SCA in e⁻ (Roman noise only)."""
        if sca not in self._background:
            template = galsim.ImageF(galsim.roman.n_pix, galsim.roman.n_pix, wcs=self.wcs[sca])
            if self.pointing is None:
                world_pos = pointing_coord(self.config["wcs"])
            else:
                world_pos = self.wcs[sca].toWorld(template.true_center)
            self._background[sca] = noise_mod.background_image(
                template, self.config["filter"], self.bandpass, world_pos)
        return self._background[sca]

    def add_noise(self, image, obs, sca, table):
        """Add the configured noise to a full-SCA image (e⁻ → ADU, in place).

        Returns the expected background variance (ADU²) at each star of ``table``.
        """
        noise = self.config["noise"]
        rng = self.rng(obs, sca, NOISE)
        if noise["kind"] == "gaussian":
            noise_mod.add_gaussian_noise(image, noise["sigma"], rng)
            return noise_mod.background_variance(noise) + np.zeros(len(table))
        bkg = self.background(sca)
        noise_mod.add_roman_noise(image, noise["effects"], bkg, rng)
        ix = np.round(np.asarray(table["x"])).astype(int) - bkg.xmin
        iy = np.round(np.asarray(table["y"])).astype(int) - bkg.ymin
        return noise_mod.background_variance(noise, bkg.array[iy, ix])


def _make_sed(chromatic, bandpass):
    """The SED shared by all stars, normalized to unit flux in the bandpass (or None)."""
    if chromatic is None:
        return None
    return storage.sed_from_setting(chromatic).withFlux(1.0, bandpass)


def _stamp_bounds(pos, size):
    """Bounds of a size × size stamp around the pixel containing pos (GalSim convention)."""
    ix, iy = int(round(pos.x)), int(round(pos.y))
    lo = size // 2
    return galsim.BoundsI(ix - lo, ix - lo + size - 1, iy - lo, iy - lo + size - 1)
