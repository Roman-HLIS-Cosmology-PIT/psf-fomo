"""``GaussianMoments``: a signed convolution kernel parameterized by its second moments.

Experimental: a model component added by this package (not part of Piff), still being tested.

Parameters (Mxx, Mxy, Myy) in native pix² along the world (u, v) axes (× 0.11″² gives
arcsec²). Convolution adds moments, M_total = M_psf + M, so to first order the image responds
linearly to them. Use it as a component of a Piff ``Convolve`` with a ``Polynomial``
interpolator (it is a nonlinear model, so ``BasisPolynomial`` does not apply).

K(M) = (1 − Σw) G(M0) + Σ_j w_j G(M0 + h E_j), w_j = (M − M0)_j / h, with tiny Gaussians G
(M0 = m0·I, σ0 = √m0 ≈ 0.05 pix), E_j the unit xx, xy, yy matrices and h = m0/2. Moments of a
weighted sum of Gaussians are the weighted sum of their moments, so K has flux 1 and second
moments exactly M for any M, including negative eigenvalues (sharpening along an axis). Linear
in M, no boundary; start at 0. Convolving a PSF with K gives PSF + ½ M : ∇∇PSF, accurate to
first order in M relative to the PSF's own moments (≈ 0.3 pix² pre-pixel for H158).

For a kernel that can only broaden (a PSD Gaussian), use Piff's ``GaussianBlur`` (kernel kind
``'blur'`` in `psf_fomo.piff_simfit.PiffFitConfig.add_kernel`); it replaced our earlier
``signed=False`` mode.

The per-star fit varies only M; flux and center stay fixed (Piff's reflux updates them). This
differs from Piff's ``Gaussian`` model, whose per-star fit also moves flux and center.

The model registers with Piff under the type name ``GaussianMoments`` when this module is
imported (``import psf_fomo.models``).
"""
import numpy as np
import scipy.optimize
import galsim
import galsim.roman
import piff
from piff.config import LoggerWrapper

from ..sim.psf import gaussian_kernel

PIXEL_SCALE = galsim.roman.pixel_scale  # arcsec per native pixel, also with the Roman WCS


class GaussianMoments(piff.Model):
    """Signed convolution kernel with parameters (Mxx, Mxy, Myy) in native pix².

    Parameters
    ----------
    signed : bool
        Must be True (default). Kept so that configs written with ``signed: True`` still
        load; the former ``signed=False`` mode is Piff's ``GaussianBlur`` now.
    init : sequence of 3 float, optional
        Starting (Mxx, Mxy, Myy) in pix². Default (0, 0, 0). Piff's composite
        ``default_init`` (e.g. ``'delta'`` inside a Convolve) is ignored.
    m0 : float
        Variance in pix² of the base Gaussians. Default 0.0025.
    diff_step : float
        Finite-difference step of the per-star fit (scipy ``diff_step``; ≈ absolute step in
        pix² for |M| < 1). Default 1e-3.
    max_nfev : int
        Maximum model evaluations per star and Piff iteration. Default 30. Stars that reach it
        are counted per Piff iteration in ``max_nfev_history`` (last one: ``n_max_nfev``).
    logger : logging.Logger, optional
        Passed by Piff; unused.

    Example
    -------
    In a Piff config::

        {'type': 'Simple', 'model': {'type': 'GaussianMoments'},
         'interp': {'type': 'Polynomial', 'order': 0}}
    """

    # _type_name is set after the class body (below) so that re-importing this module
    # (notebook autoreload) replaces the registration instead of raising in Piff's
    # Model.__init_subclass__.
    _type_name = None
    _method = "auto"  # the pixel is applied once when the full PSF is drawn
    _centered = True

    _E = (np.array([[1.0, 0.0], [0.0, 0.0]]),
          np.array([[0.0, 1.0], [1.0, 0.0]]),
          np.array([[0.0, 0.0], [0.0, 1.0]]))

    def __init__(self, signed=True, init=None, m0=0.0025, diff_step=1e-3, max_nfev=30,
                 logger=None):
        if not signed:
            raise ValueError("GaussianMoments(signed=False) was removed; use Piff's "
                             "GaussianBlur (kernel kind 'blur') for a broadening-only kernel")
        if init is None:
            init = (0.0, 0.0, 0.0)
        self.init = tuple(float(v) for v in init)
        if len(self.init) != 3:
            raise ValueError("init must be (Mxx, Mxy, Myy)")
        self.m0 = float(m0)
        if self.m0 <= 0:
            raise ValueError("m0 must be > 0")
        self.diff_step = float(diff_step)
        self.max_nfev = int(max_nfev)
        self.kwargs = {"signed": True, "init": np.array(self.init), "m0": self.m0,
                       "diff_step": self.diff_step, "max_nfev": self.max_nfev}
        self.max_nfev_history = []  # per Piff iteration: stars that stopped at max_nfev
        self._base = None
        self.set_num(None)

    @classmethod
    def _fix_kwargs(cls, kwargs):
        # FITS round trip returns the init list as an array.
        kwargs["init"] = [float(v) for v in np.atleast_1d(kwargs["init"])]

    def _base_parts(self):
        """The fixed Gaussians G(M0) and G(M0 + h E_j), built once."""
        if self._base is None:
            m0 = self.m0 * np.eye(2)
            h = 0.5 * self.m0  # keeps M0 + h E_xy positive definite
            g0 = gaussian_kernel(m0[0, 0], m0[0, 1], m0[1, 1])
            gj = [gaussian_kernel(m[0, 0], m[0, 1], m[1, 1]) for m in (m0 + h * e for e in self._E)]
            self._base = (g0, gj, h, np.array([self.m0, 0.0, self.m0]))
        return self._base

    def getProfile(self, params):
        """Unit-flux kernel profile (arcsec) for params (Mxx, Mxy, Myy) in pix²."""
        params = np.asarray(params, dtype=float)
        g0, gj, h, m0vec = self._base_parts()
        w = (params - m0vec) / h
        terms = [(1.0 - w.sum()) * g0] + [wj * g for wj, g in zip(w, gj) if wj != 0.0]
        return galsim.Sum(terms) if len(terms) > 1 else terms[0]

    def initialize(self, star, logger=None, default_init=None):
        """Start every star at ``init`` (``default_init`` from composites is ignored)."""
        fit = star.fit.newParams(np.array(self.init), params_var=np.zeros(3), num=self._num)
        return piff.Star(star.data, fit)

    @property
    def n_max_nfev(self):
        """Number of stars that stopped at ``max_nfev`` in the last Piff iteration."""
        return self.max_nfev_history[-1] if self.max_nfev_history else 0

    def initialize_iteration(self):
        """Called by Piff at the start of each iteration: start a new ``max_nfev`` count."""
        self.max_nfev_history.append(0)

    def fit(self, star, logger=None, convert_func=None, draw_method=None):
        """Least-squares fit of (Mxx, Mxy, Myy) for one star, flux and center fixed.

        ``convert_func`` folds in the other Convolve components (supplied by Piff).
        """
        logger = LoggerWrapper(logger)
        p0 = np.array(star.fit.get_params(self._num), dtype=float)
        data = star.image.array.ravel()
        sw = np.sqrt(star.weight.array.ravel())
        method = draw_method if draw_method is not None else self._method

        def resid(p):
            prof = self.getProfile(p).shift(star.fit.center) * star.fit.flux
            if convert_func is not None:
                prof = convert_func(prof)
            image = star.image.copy()
            prof.drawImage(image=image, method=method, center=star.image_pos)
            return sw * (data - image.array.ravel())

        res = scipy.optimize.least_squares(resid, p0, method="trf",
                                           diff_step=self.diff_step, max_nfev=self.max_nfev)
        if res.status == 0:  # stopped at max_nfev: not converged for this star
            if not self.max_nfev_history:  # fit called outside a Piff iteration
                self.max_nfev_history.append(0)
            self.max_nfev_history[-1] += 1
        logger.debug("GaussianMoments fit: %s -> %s (nfev=%d, status=%d)",
                     p0, res.x, res.nfev, res.status)
        params_var = np.diag(np.linalg.pinv(res.jac.T @ res.jac))
        chisq = float(np.sum(res.fun ** 2))
        # As Piff's own models: no parameters subtracted (they are interpolated); reflux
        # sets the final dof.
        dof = int(np.count_nonzero(sw))
        fit = star.fit.newParams(res.x, params_var=params_var, num=self._num,
                                 chisq=chisq, dof=dof)
        return piff.Star(star.data, fit)


GaussianMoments._type_name = "GaussianMoments"
piff.Model.valid_model_types["GaussianMoments"] = GaussianMoments
