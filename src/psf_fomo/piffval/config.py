"""Piff configuration builder (`FitConfig`).

One builder for every Piff model this package fits; Piff runs and reference-solver runs both
take their model from `FitConfig.piff_config`. The generated structure is::

    Sum[ Convolve[ RomanOptics, kernel, MTF ], Simple(PixelGrid) ]

where each optional piece is left out when not requested (a Convolve or Sum with a single
component is not built). Order inside the Convolve: optics first (fitted first), then the
kernel (Piff gives it ``init='delta'`` by default), then the fixed MTF.

Units follow Piff: arcsec for sizes and positions (PixelGrid scale, GaussianMixture sigmas),
except ``GaussianMoments`` parameters, which are native pix². Piff's ``GaussianBlur`` (kernel
kind ``'blur'``, Piff's ``gaussian_blur`` branch only) uses ``[ixx, iyy, ixy]`` in arcsec²;
`blur_params` / `blur_moments` convert.

Example
-------
>>> cfg = FitConfig()
>>> cfg.set_optics("free", max_zernike=11, prior_sigma=0.05)
>>> cfg.add_kernel("signed", order=0)
>>> cfg.add_pixelgrid(size=5, scale=0.11, order=1)
>>> piff_dict = cfg.piff_config("H158")
"""
import copy

import numpy as np
import galsim.roman

from ..io import read_yaml, write_yaml
from ..sim.psf import MTF_BASE_SIGMA, MTF_REL_SIGMAS, MTF_WEIGHTS

OPTICS_MODES = ("free", "pinned")
KERNEL_KINDS = ("signed", "positive", "gaussian", "blur")
KERNEL_INTERPS = ("Polynomial", "BasisPolynomial")
PIXEL_SCALE = galsim.roman.pixel_scale
SHAPE_METHODS = ("hsm", "gauss", "am", "wmom")

# "Pinned" optics: a prior this narrow holds the extra aberrations at 0 (the nominal GalSim
# Roman PSF). Only Z4 is kept, so each Piff iteration costs a single finite-difference step.
PINNED_PRIOR_SIGMA = 1e-10
PINNED_MAX_ZERNIKE = 4


class FitConfig:
    """Builder for the Piff PSF model and fit settings.

    Create it, adjust with the ``set_*`` / ``add_*`` methods, then `piff_config` gives the
    Piff ``psf`` dict. The settings are a plain dict (``self.config``) that round-trips
    through YAML (`to_yaml`, `from_yaml`).

    Defaults: free RomanOptics with Piff's defaults (Z4–Z22, prior σ 0.05 waves, constant per
    SCA, bilinear), fixed MTF on, no kernel, no PixelGrid, no outlier rejection, Piff's
    iteration defaults (max_iter 30, min_iter 2, chisq_thresh 0.1), Piff's shape method
    ``'hsm'``.
    """

    def __init__(self):
        self.config = {
            "optics": {"mode": "free", "max_zernike": 22, "prior_sigma": 0.05,
                       "aberration_interp": "constant", "nominal_interp": "bilinear"},
            "mtf": True,
            "kernel": None,
            "pixelgrid": None,
            "outliers": None,
            "iterations": {"max_iter": 30, "min_iter": 2, "chisq_thresh": 0.1},
            "shape_method": {"method": "hsm", "weight_fwhm": None},
            "piff_psf": None,
        }

    def set_optics(self, mode="free", max_zernike=22, prior_sigma=0.05,
                   aberration_interp="constant", nominal_interp="bilinear"):
        """Configure the Piff ``RomanOptics`` component (always present).

        Parameters
        ----------
        mode : {'free', 'pinned'}
            ``'free'``: extra Zernikes Z4 … ``max_zernike`` fitted per SCA with a Gaussian
            prior toward 0. ``'pinned'``: held at 0 (nominal GalSim Roman optics) by a prior
            of σ = 1e-10 on Z4 only; ``max_zernike`` and ``prior_sigma`` are ignored.
        max_zernike : int
            Highest fitted Noll index, 4 … 22. Default 22 (Piff's default). Each Piff
            iteration costs (max_zernike − 2) × 4 ``galsim.roman.getPSF`` calls.
        prior_sigma : float, sequence of float, or None
            Prior width in waves: one value for all Zernikes, or one per Zernike
            (Z4 … Z_max). None = no prior. Default 0.05 (Piff's default).
        aberration_interp : {'constant', 'linear', 'global'}
            Piff option: one vector per SCA, a linear field a + b x̂ + c ŷ per SCA, or one
            vector shared by all SCAs. Default ``'constant'``.
        nominal_interp : {'bilinear', 'five_point'}
            Piff option for the spatial interpolation of the sample-point PSFs. Use the same as
            the simulation (default ``'bilinear'``).
        """
        if mode not in OPTICS_MODES:
            raise ValueError(f"mode must be one of {OPTICS_MODES}")
        if not 4 <= max_zernike <= 22:
            raise ValueError("max_zernike must be in 4..22")
        if aberration_interp not in ("constant", "linear", "global"):
            raise ValueError("aberration_interp must be 'constant', 'linear' or 'global'")
        if nominal_interp not in ("bilinear", "five_point"):
            raise ValueError("nominal_interp must be 'bilinear' or 'five_point'")
        if prior_sigma is not None:
            values = np.atleast_1d(np.asarray(prior_sigma, dtype=float))
            if values.ndim != 1 or len(values) not in (1, max_zernike - 3):
                raise ValueError(f"prior_sigma must be a scalar or have {max_zernike - 3} "
                                 "values (Z4 … Z_max)")
            if np.any(values <= 0):
                raise ValueError("prior_sigma values must be > 0")
            prior_sigma = float(values[0]) if np.ndim(prior_sigma) == 0 else values.tolist()
        self.config["optics"] = {"mode": mode, "max_zernike": int(max_zernike),
                                 "prior_sigma": prior_sigma,
                                 "aberration_interp": aberration_interp,
                                 "nominal_interp": nominal_interp}

    def set_mtf(self, on=True):
        """Include the fixed charge-diffusion kernel (Piff ``GaussianMixture``, ``Fixed``).

        Same three Gaussians as the simulation truth at ``mtf_scale = 1`` (CLAUDE.md §5.3),
        sigmas converted to arcsec with ``galsim.roman.pixel_scale``. It cannot be fitted.
        """
        self.config["mtf"] = bool(on)

    def add_kernel(self, kind="signed", order=0, init=None, max_nfev=30, interp="Polynomial"):
        """Add a fitted convolution kernel; ``add_kernel(None)`` removes it.

        Parameters
        ----------
        kind : {'signed', 'positive', 'gaussian', 'blur'} or None
            ``'signed'`` / ``'positive'``: our ``GaussianMoments`` (M in pix², flux and centre
            fixed in its per-star fit). ``'gaussian'``: Piff's ``Gaussian`` (scale, g1, g2;
            its per-star fit also moves flux and centre), ``init='delta'``. ``'blur'``: Piff's
            ``GaussianBlur`` (only on Piff's ``gaussian_blur`` branch; ``[ixx, iyy, ixy]`` in
            arcsec², PSD-projected, flux and centre fixed in its fit, starts at zero).
        order : int
            Order of the interpolator in (u, v). Needs at least (order+1)(order+2)/2 stars.
            Default 0 (one kernel for the SCA).
        init : sequence of 3 float, optional
            Starting (Mxx, Mxy, Myy) in native pix² for ``'signed'``, ``'positive'`` and
            ``'blur'`` (converted to GaussianBlur's ``[ixx, iyy, ixy]`` arcsec²). Default: the
            model's default ((0.01, 0, 0.01) positive, 0 signed and blur).
        max_nfev : int
            ``GaussianMoments`` evaluations per star and iteration. Default 30.
        interp : {'Polynomial', 'BasisPolynomial'}
            Interpolator. ``'BasisPolynomial'`` (joint linearised solve with Piff's line
            search) only for ``'blur'``: the other kernels have no quadratic ``chisq``.
            Default ``'Polynomial'``.
        """
        if kind is None:
            self.config["kernel"] = None
            return
        if kind not in KERNEL_KINDS:
            raise ValueError(f"kind must be one of {KERNEL_KINDS} or None")
        if order < 0:
            raise ValueError("order must be >= 0")
        if interp not in KERNEL_INTERPS:
            raise ValueError(f"interp must be one of {KERNEL_INTERPS}")
        if interp == "BasisPolynomial" and kind != "blur":
            raise ValueError("BasisPolynomial is only available for kind='blur'")
        self.config["kernel"] = {"kind": kind, "order": int(order),
                                 "init": None if init is None else [float(v) for v in init],
                                 "max_nfev": int(max_nfev), "interp": interp}

    def add_pixelgrid(self, size=5, scale=0.11, order=1, interp="Lanczos(7)", centered=True,
                      init=None):
        """Add a ``PixelGrid`` component in a Sum; ``add_pixelgrid(None)`` removes it.

        Parameters
        ----------
        size : int or None
            Grid points per side. Default 5. None removes the grid.
        scale : float
            Grid spacing in arcsec. Default 0.11 (one native pixel).
        order : int
            ``BasisPolynomial`` order in (u, v). Default 1.
        interp : str
            GalSim interpolant. Default ``'Lanczos(7)'`` (Piff's default).
        centered : bool
            Piff's centroid constraint on the grid. Default True.
        init : {None, 'zero', 'gauss', 'delta'}
            Piff init; None lets the Sum choose (``'zero'``, which forces ``fit_flux=True``).
        """
        if size is None:
            self.config["pixelgrid"] = None
            return
        if size < 1 or scale <= 0 or order < 0:
            raise ValueError("need size >= 1, scale > 0 and order >= 0")
        if init not in (None, "zero", "gauss", "delta"):
            raise ValueError("init must be None, 'zero', 'gauss' or 'delta'")
        self.config["pixelgrid"] = {"size": int(size), "scale": float(scale),
                                    "order": int(order), "interp": str(interp),
                                    "centered": bool(centered), "init": init}

    def set_outliers(self, chisq_nsigma=None, max_remove=None, max_centroid=None):
        """Outlier rejection after each iteration (Piff ``Chisq`` and ``Centroid``).

        Parameters
        ----------
        chisq_nsigma : float, optional
            Reject stars whose χ² exceeds the ``nsigma``-equivalent tail for their dof.
        max_remove : int or float, optional
            Chisq only: at most this many (or this fraction, if < 1) per iteration.
        max_centroid : float, optional
            Reject stars whose fitted centre is more than this many arcsec from the catalog
            position.

        All None (default): no outlier rejection.
        """
        if chisq_nsigma is None and max_centroid is None:
            self.config["outliers"] = None
            return
        self.config["outliers"] = {
            "chisq_nsigma": None if chisq_nsigma is None else float(chisq_nsigma),
            "max_remove": max_remove,
            "max_centroid": None if max_centroid is None else float(max_centroid)}

    def set_iterations(self, max_iter=30, min_iter=2, chisq_thresh=0.1):
        """Piff's fit-loop controls (stop when Δχ² < chisq_thresh × dof after min_iter)."""
        if max_iter < 1 or min_iter < 1 or chisq_thresh < 0:
            raise ValueError("need max_iter >= 1, min_iter >= 1, chisq_thresh >= 0")
        self.config["iterations"] = {"max_iter": int(max_iter), "min_iter": int(min_iter),
                                     "chisq_thresh": float(chisq_thresh)}

    def set_piff_shape_method(self, method="hsm", weight_fwhm=None):
        """Piff's global shape method during the fit.

        Piff uses it to initialise PixelGrid (``gauss``/``zero`` init) and GSObject models,
        and in its own selection and stats. It is set only while the fit runs and restored
        afterwards; our moment diagnostics are configured separately and do not depend on it.

        Parameters
        ----------
        method : {'hsm', 'gauss', 'am', 'wmom'}
            Default ``'hsm'`` (Piff's default). Note: HSM fails on most native H158 stamps
            (Phase 3); Piff does not check that flag when initialising.
        weight_fwhm : float, optional
            Weight FWHM in arcsec, required for ``'wmom'``.
        """
        if method not in SHAPE_METHODS:
            raise ValueError(f"method must be one of {SHAPE_METHODS}")
        if method == "wmom" and weight_fwhm is None:
            raise ValueError("wmom needs weight_fwhm (arcsec)")
        self.config["shape_method"] = {
            "method": method, "weight_fwhm": None if weight_fwhm is None else float(weight_fwhm)}

    @classmethod
    def from_piff_config(cls, psf_config):
        """Use a hand-written Piff ``psf`` dict instead of the builder options.

        The iteration, outlier and option settings of this object are then ignored for the
        model (the dict is used as is); the shape-method setting still applies.
        """
        cfg = cls()
        cfg.config["piff_psf"] = copy.deepcopy(psf_config)
        return cfg

    def piff_config(self, filter_name):
        """The Piff ``psf`` configuration dict.

        Parameters
        ----------
        filter_name : str
            Roman filter (GalSim or Roman F name), normally the simulation's filter.

        Returns
        -------
        dict
            Ready for ``piff.PSF.process``.
        """
        if self.config["piff_psf"] is not None:
            return copy.deepcopy(self.config["piff_psf"])
        c = self.config
        convolved = [self._optics(filter_name)]
        if c["kernel"] is not None:
            convolved.append(self._kernel())
        if c["mtf"]:
            convolved.append(mtf_component())
        psf = convolved[0] if len(convolved) == 1 else {"type": "Convolve",
                                                        "components": convolved}
        if c["pixelgrid"] is not None:
            psf = {"type": "Sum", "components": [psf, self._pixelgrid()]}
        psf.update(c["iterations"])
        if c["outliers"] is not None:
            psf["outliers"] = self._outliers()
        return psf

    def _optics(self, filter_name):
        o = self.config["optics"]
        pinned = o["mode"] == "pinned"
        return {"type": "RomanOptics", "filter_name": filter_name,
                "chromatic": False,  # Piff's default is True; chromatic fits are deferred
                "max_zernike": PINNED_MAX_ZERNIKE if pinned else o["max_zernike"],
                "aberration_prior_sigma": PINNED_PRIOR_SIGMA if pinned else o["prior_sigma"],
                "aberration_interp": o["aberration_interp"],
                "nominal_interp": o["nominal_interp"]}

    def _kernel(self):
        k = self.config["kernel"]
        if k["kind"] == "gaussian":
            model = {"type": "Gaussian", "init": "delta"}
        elif k["kind"] == "blur":
            model = {"type": "GaussianBlur"}
            if k["init"] is not None:
                model["init"] = blur_params(*k["init"])
        else:
            # Imported here so that building configs does not need the models registered
            # until a Piff object is made from them.
            from .. import models  # noqa: F401  (registers GaussianMoments)
            model = {"type": "GaussianMoments", "signed": k["kind"] == "signed",
                     "max_nfev": k["max_nfev"]}
            if k["init"] is not None:
                model["init"] = list(k["init"])
        # Settings saved before 'interp' existed mean Polynomial.
        return {"type": "Simple", "model": model,
                "interp": {"type": k.get("interp", "Polynomial"), "order": k["order"]}}

    def _pixelgrid(self):
        p = self.config["pixelgrid"]
        model = {"type": "PixelGrid", "size": p["size"], "scale": p["scale"],
                 "interp": p["interp"], "centered": p["centered"]}
        if p["init"] is not None:
            model["init"] = p["init"]
        return {"type": "Simple", "model": model,
                "interp": {"type": "BasisPolynomial", "order": p["order"]}}

    def _outliers(self):
        o = self.config["outliers"]
        out = []
        if o["chisq_nsigma"] is not None:
            chisq = {"type": "Chisq", "nsigma": o["chisq_nsigma"]}
            if o["max_remove"] is not None:
                chisq["max_remove"] = o["max_remove"]
            out.append(chisq)
        if o["max_centroid"] is not None:
            out.append({"type": "Centroid", "max_offset": o["max_centroid"]})
        return out

    def to_yaml(self, path):
        """Write the settings to a YAML file."""
        write_yaml(path, self.config)

    @classmethod
    def from_config(cls, config):
        """Build from a settings dict (as in ``self.config``)."""
        cfg = cls()
        cfg.config.update(copy.deepcopy(config))
        return cfg

    @classmethod
    def from_yaml(cls, path):
        """Build from a YAML file written by `to_yaml`."""
        return cls.from_config(read_yaml(path))


def blur_params(mxx, mxy, myy):
    """GaussianBlur parameters ``[ixx, iyy, ixy]`` (arcsec²) of M = (Mxx, Mxy, Myy) in pix².

    Both are world (u, v) axes; note the different order.
    """
    s = PIXEL_SCALE**2
    return [float(mxx) * s, float(myy) * s, float(mxy) * s]


def blur_moments(params):
    """Inverse of `blur_params`: (Mxx, Mxy, Myy) in pix² of ``[ixx, iyy, ixy]`` in arcsec²."""
    ixx, iyy, ixy = (float(v) / PIXEL_SCALE**2 for v in params)
    return ixx, ixy, iyy


def mtf_component():
    """Piff config of the fixed MTF: ``Simple(GaussianMixture, Fixed)``, sigmas in arcsec."""
    sigmas = [r * MTF_BASE_SIGMA * galsim.roman.pixel_scale for r in MTF_REL_SIGMAS]
    return {"type": "Simple",
            "model": {"type": "GaussianMixture", "weights": list(MTF_WEIGHTS),
                      "sigmas": sigmas},
            "interp": {"type": "Fixed"}}
