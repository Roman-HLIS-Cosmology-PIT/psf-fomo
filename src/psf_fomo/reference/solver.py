"""Reference solver: the joint χ² optimum of the model Piff fits.

Piff fits a composite PSF block by block (each component in turn, then flux and center,
iterated). The reference solver takes the *same* Piff PSF object and minimizes χ² over all
model parameters at once with ``scipy.optimize.least_squares``; every model image is drawn by
Piff's own ``psf.drawStar``. Comparing Piff's solution with this optimum separates optimizer
limitations (Piff above the optimum of its own model) from model limitations (the optimum
itself is biased or has excess χ²).

Parameterization (one block per leaf component):

- ``RomanOptics``: one vector per SCA (one SCA per fit here); its Gaussian prior toward 0 is
  included as extra residuals p / σ.
- ``Simple`` with ``Polynomial`` / ``BasisPolynomial``: per-star parameters are polynomials in
  the interpolator's own keys (normally u, v) with Piff's own set of terms, evaluated on
  centered and scaled coordinates (same span, better conditioned). ``Mean``: a constant.
- Frozen (held at the given values): ``Fixed`` components, ``RomanOptics`` pinned by a prior
  σ < 1e-6, and any component kind listed in ``freeze``.

Simplifications:

- Star centers are held fixed and fluxes are profiled in closed form at every evaluation
  (variable projection). Centers: ``'catalog'`` (default; offset 0 = the truth for a
  simulation) or ``'fitted'`` (Piff's fitted centers; for real data, which has no truth).
  Piff also fits centers, so the reference χ² is an upper bound on the true optimum of Piff's
  problem.
- The star table, components and moments of a solution come from `star_table`,
  `components` and `moment_table`: simulation versions here; real-data fits use the subclass
  `psf_fomo.datafit.reference.DataReferenceSolver`.
- A centered PixelGrid is optimized over the full grid (Piff restricts it to zero centroid and
  fits the star center instead); constraints inside a model's ``fit`` are not imposed.
- Only fit stars (not reserve, not flagged) enter χ², so only they are drawn during the
  optimization; reserve and flagged stars are drawn at the reported solutions
  (``report='all'``, default) or not at all (``report='fit'``).

Degrees of freedom: dof = weighted pixels of fit stars − one flux per fit star − free model
coefficients (docs/conventions.md). Piff's own dof is pixels − 3 per star.

Speed: derivatives are per star (each star depends only on its own parameters; polynomial
coefficients enter through the chain rule). Components that enter the image linearly (e.g. a
PixelGrid in a Sum, detected by a second-difference test) have their derivative images computed
once and reused. Reported solutions are always re-drawn exactly.
"""
import time

import numpy as np
import scipy.optimize
import galsim.roman
import piff
from piff.config import LoggerWrapper

from ..piffval import diagnostics
from ..piffval.config import blur_params
from ..piffval.components import components as component_tables, kind_of, leaf_psfs
from ..piffval.residuals import moment_residuals

# Finite-difference steps per component kind (parameter units: waves, pix², grid values).
STEPS = {"optics": 1e-3, "kernel": 1e-3, "pixelgrid": 1e-4, "other": 1e-4}
# Piff's GaussianBlur works in arcsec²: the same 1e-3 pix² step.
BLUR_STEP = 1e-3 * galsim.roman.pixel_scale**2
PINNED_PRIOR = 1e-6  # RomanOptics with a prior narrower than this is frozen (pinned mode)


class Leaf:
    """One parameterized component: per-star parameters P = K @ C, C of shape (n_terms, plen)."""

    def __init__(self, leaf, stars, frozen):
        self.psf = leaf
        self.num = leaf._num
        self.kind = kind_of(leaf)
        self.name = leaf.model._type_name
        self.P_given = np.array([np.asarray(s.fit.get_params(self.num), dtype=float).ravel()
                                 for s in stars])
        self.plen = self.P_given.shape[1]
        self.K = basis_matrix(leaf, stars)
        self.prior_sigma = None
        if self.kind == "optics" and leaf.model.prior_sigma is not None:
            self.prior_sigma = np.asarray(leaf.model.prior_sigma, dtype=float)
        pinned = self.prior_sigma is not None and np.max(self.prior_sigma) < PINNED_PRIOR
        self.frozen = (frozen or pinned or type(leaf.interp).__name__ == "Fixed")
        self.step = BLUR_STEP if self.name == "GaussianBlur" else STEPS.get(self.kind,
                                                                            STEPS["other"])
        self.C_given = np.linalg.lstsq(self.K, self.P_given, rcond=None)[0]
        scale = max(np.abs(self.P_given).max(), 1e-30)
        # How well the basis reproduces the given (Piff) parameters; ~0 when the basis matches
        # Piff's interpolator.
        self.proj_err = float(np.abs(self.K @ self.C_given - self.P_given).max() / scale)
        self.linear = None   # set by the linearity test
        self.dP = None       # cached derivative images (n_fit, plen, npix) if linear

    @property
    def nterms(self):
        return self.K.shape[1]

    @property
    def ncoef(self):
        return self.nterms * self.plen

    def per_star(self, C, idx=None):
        """Per-star parameters of stars ``idx`` (default all) for coefficients C."""
        K = self.K if idx is None else self.K[idx]
        return K @ C.reshape(self.nterms, self.plen)


def basis_matrix(leaf, stars):
    """(n_stars, n_terms) basis of a leaf's interpolator, on centered and scaled coordinates.

    Uses Piff's own choice of terms, so the span equals Piff's interpolation space.
    """
    interp = leaf.interp
    name = type(interp).__name__
    n = len(stars)
    if name in ("Mean", "Fixed", "RomanSCAInterp"):
        return np.ones((n, 1))
    if name == "Polynomial":
        if interp.orders is not None or interp.poly_type != "poly":
            raise ValueError("reference solver supports Polynomial with one order and "
                             "poly_type='poly' only")
        keys = ("u", "v")
        exps = interp._generate_indices(interp.order)
    elif name == "BasisPolynomial":
        keys = tuple(interp._keys)
        grids = np.meshgrid(*[np.arange(o + 1) for o in interp._orders], indexing="ij")
        exps = list(zip(*[g[interp._mask] for g in grids]))
    else:
        raise ValueError(f"reference solver: unsupported interpolator {name!r} "
                         f"for component {leaf.model._type_name!r}")
    vals = np.array([[float(s.data[k]) for k in keys] for s in stars])
    center = vals.mean(axis=0)
    span = np.abs(vals - center).max(axis=0)
    vals = (vals - center) / np.where(span > 0, span, 1.0)
    return np.column_stack([np.prod(vals ** np.asarray(e), axis=1) for e in exps])


class ReferenceSolver:
    """Joint χ² optimum of a Piff PSF model on a set of stars.

    Parameters
    ----------
    psf : piff.PSF
        The model, fitted or only initialized (e.g. `PiffValidation.initialize`). Its
        structure and drawing are used; its interpolators are not changed.
    stars : list of piff.Star
        Stars with data images and fit parameters (``psf.stars`` after a fit, or the
        initialized stars). Their ``is_reserve`` / ``is_flagged`` marks are respected.
    rows : astropy.table.Table
        Simulation catalog rows of the stars (same order).
    sim : Simulation, optional
        Needed for moment residuals in `evaluate`.
    freeze : sequence of str, optional
        Component kinds to hold at their given values (``'optics'``, ``'kernel'``,
        ``'pixelgrid'``). Fixed components and pinned optics are always frozen.
    centers : {'catalog', 'fitted'}
        Star centers held during the optimization: the catalog position (default; the truth
        in a simulation) or the stars' given (Piff-fitted) centers.

    Example
    -------
    >>> ref = ReferenceSolver(res.psf, *res.fitted_stars(), sim=res.sim)
    >>> out = ref.solve(starts=("given", "zero"))
    >>> comparison_table({"piff": out["given"], **out["optimum"]})
    """

    def __init__(self, psf, stars, rows, sim=None, freeze=(), centers="catalog"):
        if len(stars) != len(rows):
            raise ValueError("stars and rows have different lengths")
        if centers not in ("catalog", "fitted"):
            raise ValueError("centers must be 'catalog' or 'fitted'")
        self.psf = psf
        self.stars = list(stars)
        self.rows = rows
        self.sim = sim
        self.centers = centers
        self.leaves = [Leaf(leaf, self.stars, kind_of(leaf) in freeze)
                       for leaf in leaf_psfs(psf)]
        self.free = [lf for lf in self.leaves if not lf.frozen]
        # Unit flux, the held center, the given parameter structure.
        self.base_fits = [s.fit.withNew(flux=1.0, center=(0.0, 0.0) if centers == "catalog"
                                        else tuple(s.fit.center)) for s in self.stars]
        self.D = np.array([s.image.array.ravel() for s in self.stars], dtype=float)
        self.W = np.array([s.weight.array.ravel() for s in self.stars], dtype=float)
        self.use = np.array([not s.is_reserve and not s.is_flagged for s in self.stars])
        if not self.use.any():
            raise ValueError("no fit stars (all reserve or flagged)")
        # Reserve / flagged stars do not enter χ², so the optimization draws only these.
        self.fit_idx = np.flatnonzero(self.use)
        self.sw = np.sqrt(self.W[self.fit_idx])
        self.n, self.npix = self.D.shape
        self.dof = (int(np.count_nonzero(self.W[self.fit_idx])) - len(self.fit_idx)
                    - self.nfree)
        self.n_draws = 0
        self._cache = None
        self._lin_ref = None

    @property
    def nfree(self):
        return sum(lf.ncoef for lf in self.free)

    def x_start(self, which="given"):
        """Start vector: ``'given'`` (the stars' parameters) or ``'zero'`` (kernel and grid
        components at zero; optics and Piff Gaussians at their given values). Piff's
        GaussianBlur (PSD-projected) starts at 0.01 pix² per axis."""
        parts = []
        for lf in self.free:
            if which == "given" or lf.kind == "optics" or lf.name == "Gaussian":
                parts.append(lf.C_given.ravel())  # Gaussian (scale, g) cannot start at 0
            elif which == "zero":
                C = np.zeros((lf.nterms, lf.plen))
                # M = 0 is the apex of the PSD cone, where a PSD-projected kernel is not
                # differentiable (forward steps in Mxy leave the cone): start at a small round
                # kernel, 0.01 pix² per axis (constant term only; the first basis column is
                # the constant).
                if lf.name == "GaussianBlur":
                    C[0] = blur_params(0.01, 0.0, 0.01)
                parts.append(C.ravel())
            else:
                raise ValueError("which must be 'given' or 'zero'")
        return np.concatenate(parts) if parts else np.zeros(0)

    def unpack(self, x):
        Cs, j = [], 0
        for lf in self.leaves:
            if lf.frozen:
                Cs.append(lf.C_given)
            else:
                Cs.append(x[j:j + lf.ncoef].reshape(lf.nterms, lf.plen))
                j += lf.ncoef
        return Cs

    def per_star_params(self, x, idx=None):
        """List over leaves of (len(idx), plen) parameter arrays of stars ``idx`` (default
        all)."""
        return [lf.per_star(C, idx) for lf, C in zip(self.leaves, self.unpack(x))]

    def _fit_for(self, i, Pi):
        fit = self.base_fits[i]
        for lf, p in zip(self.leaves, Pi):
            p = np.asarray(p, dtype=float)
            if lf.kind == "pixelgrid" and p.sum() == 0.0:
                # GalSim's InterpolatedImage refuses a grid summing to exactly zero (e.g. an
                # all-zero start); a 1e-12 nudge of the center is negligible for a unit-flux
                # model.
                p = p.copy()
                p[p.size // 2] += 1e-12
            fit = fit.newParams(p, num=lf.num)
        return fit

    def draw_star(self, i, Pi):
        self.n_draws += 1
        star = piff.Star(self.stars[i].data, self._fit_for(i, Pi))
        return self.psf.drawStar(star).image.array.ravel()

    def images(self, x):
        """(n_fit, npix) unit-flux model images of the fit stars at parameters x."""
        if self._cache is not None and np.array_equal(x, self._cache[0]):
            return self._cache[1]
        Ps = self.per_star_params(x, self.fit_idx)
        if self._lin_ref is not None:
            # Every free component is linear: m = m_ref + Σ_leaves dP · (P − P_ref).
            P_ref, m_ref = self._lin_ref
            m = m_ref.copy()
            for li, lf in enumerate(self.leaves):
                if not lf.frozen:
                    m += np.einsum("nk,nkp->np", Ps[li] - P_ref[li], lf.dP)
        else:
            m = np.array([self.draw_star(i, [P[j] for P in Ps])
                          for j, i in enumerate(self.fit_idx)])
        self._cache = (x.copy(), m)
        return m

    def profiled(self, m, idx=None):
        """Closed-form fluxes of stars ``idx`` (default the fit stars) for their model images
        m, and χ² of the fit stars among them."""
        idx = self.fit_idx if idx is None else idx
        W, D = self.W[idx], self.D[idx]
        F = np.sum(W * m * D, axis=1) / np.sum(W * m * m, axis=1)
        use = self.use[idx]
        chi2 = float(np.sum(W[use] * (D[use] - F[use, None] * m[use]) ** 2))
        return F, chi2

    def residual(self, x):
        m = self.images(x)
        F, _ = self.profiled(m)
        r = [(self.sw * (self.D[self.fit_idx] - F[:, None] * m)).ravel()]
        for lf, C in zip(self.leaves, self.unpack(x)):
            if not lf.frozen and lf.prior_sigma is not None:
                r.append((C / lf.prior_sigma[None, :]).ravel())
        return np.concatenate(r)

    def _detect_linearity(self, Ps):
        """Second-difference test on the first fit star, two parameters per free leaf.

        ``Ps``: per-star parameters of the fit stars (`per_star_params` with ``fit_idx``).
        """
        i = int(self.fit_idx[0])
        Pi = [P[0].copy() for P in Ps]
        base = self.draw_star(i, Pi)
        for li, lf in enumerate(self.leaves):
            if lf.frozen:
                continue
            ok = True
            for k in sorted({0, lf.plen // 2}):
                h = 10 * lf.step
                P1 = [p.copy() for p in Pi]
                P1[li][k] += h
                P2 = [p.copy() for p in Pi]
                P2[li][k] += 2 * h
                m1, m2 = self.draw_star(i, P1), self.draw_star(i, P2)
                d1, d2 = m1 - base, m2 - 2 * m1 + base
                if np.abs(d2).max() > 1e-4 * max(np.abs(d1).max(), 1e-30):
                    ok = False
            lf.linear = ok

    def _leaf_derivs(self, li, Ps, m):
        """d(image_j)/d(param_jk) for every fit star j and parameter k of leaf li
        (one-sided); ``Ps`` and ``m`` are those of the fit stars.

        Parameters outer, stars inner: RomanOptics shares one vector across stars, and its
        sample-point PSFs are cached for the current vector, so each perturbed vector costs
        one set of getPSF calls for all stars.
        """
        lf = self.leaves[li]
        dP = np.empty((len(self.fit_idx), lf.plen, self.npix))
        for k in range(lf.plen):
            for j, i in enumerate(self.fit_idx):
                Pp = [P[j].copy() for P in Ps]
                Pp[li][k] += lf.step
                dP[j, k] = (self.draw_star(i, Pp) - m[j]) / lf.step
        return dP

    def jacobian(self, x):
        m = self.images(x)
        F, _ = self.profiled(m)
        Ps = self.per_star_params(x, self.fit_idx)
        if any(lf.linear is None for lf in self.free):
            self._detect_linearity(Ps)
        a = self.sw * m  # model directions per star
        aa = np.maximum(np.sum(a * a, axis=1), 1e-300)
        blocks = []
        for li, lf in enumerate(self.leaves):
            if lf.frozen:
                continue
            if lf.linear and lf.dP is not None:
                dP = lf.dP
            else:
                dP = self._leaf_derivs(li, Ps, m)
                if lf.linear:
                    lf.dP = dP
            # Column (j, k): d m_i = K[i, j] dP[i, k]. With fluxes profiled the residual
            # Jacobian is −F_i times that direction minus its component along the star's own
            # model image (Kaufman's variable-projection form).
            K = lf.K[self.fit_idx]
            V = self.sw[:, None, None, :] * K[:, :, None, None] * dP[:, None, :, :]
            coef = np.einsum("np,njkp->njk", a, V) / aa[:, None, None]
            Vp = V - coef[..., None] * a[:, None, None, :]
            J = -(F[:, None, None, None] * Vp)  # (n_fit, nterms, plen, npix)
            blocks.append(J.transpose(0, 3, 1, 2).reshape(-1, lf.ncoef))
        rows = [np.hstack(blocks)]
        col = 0
        for lf in self.free:
            if lf.prior_sigma is not None:
                R = np.zeros((lf.ncoef, self.nfree))
                R[:, col:col + lf.ncoef] = np.diag(np.tile(1.0 / lf.prior_sigma, lf.nterms))
                rows.append(R)
            col += lf.ncoef
        if self._lin_ref is None and all(lf.linear for lf in self.free):
            self._lin_ref = ([P.copy() for P in Ps], m.copy())
        return np.vstack(rows)

    def evaluate(self, x, label="", moments=None, info=None, report="all"):
        """Diagnostics of the solution x, drawn exactly (no linearization).

        Parameters
        ----------
        x : numpy.ndarray
            Free coefficients.
        label : str
            Name stored in the summary.
        moments : dict, optional
            Keyword arguments of `psf_fomo.piffval.residuals.moment_residuals` (see
            `moment_table`). Default None = no moment residuals.
        info : dict, optional
            Extra entries for the summary (e.g. optimizer status).
        report : {'all', 'fit'}
            Stars drawn and reported: all (default; reserve stars give out-of-sample
            diagnostics) or only the fit stars (faster with many reserve stars). χ² is the
            fit stars' either way.

        Returns
        -------
        ReferenceResult
        """
        idx = self._report_idx(report)
        Ps = self.per_star_params(x, idx)
        stars = [piff.Star(self.stars[i].data, self._fit_for(i, [P[j] for P in Ps]))
                 for j, i in enumerate(idx)]
        self.n_draws += len(stars)
        m = np.array([self.psf.drawStar(s).image.array.ravel() for s in stars])
        F, chi2 = self.profiled(m, idx)
        final = []
        for j, (i, s) in enumerate(zip(idx, stars)):
            c2 = float(np.sum(self.W[i] * (self.D[i] - F[j] * m[j]) ** 2))
            npix = int(np.count_nonzero(self.W[i]))
            final.append(piff.Star(s.data, s.fit.withNew(flux=float(F[j]), chisq=c2,
                                                         dof=npix - 1)))
        rows = self.rows[idx]
        table = self.star_table(final, rows)
        summary = {"label": label, "chisq": chi2, "dof": self.dof,
                   "chisq_dof": chi2 / self.dof if self.dof > 0 else float("nan"),
                   "n_free": self.nfree, "n_fit_stars": len(self.fit_idx),
                   "n_reported": len(idx), "centers": self.centers,
                   "leaves": [leaf_summary(lf) for lf in self.leaves]}
        summary.update(info or {})
        moment_table = None if moments is None else self.moment_table(final, table, moments)
        return ReferenceResult(final, table, summary, moment_table,
                               self.components(final, rows), x=np.array(x, dtype=float))

    def reweighted_stars(self, x, reserve=True):
        """All stars at the solution x after Piff's per-iteration update of weights and centers.

        For each star (reserve ones only if ``reserve``; flagged ones never): the model at x
        is drawn with the flux profiled on the current weights, its Poisson variance is added
        to the background weights (``Star.addPoisson``, a no-op without a ``gain`` property),
        then Piff's reflux fits flux and center with the new weights. The other stars only get
        the parameters at x. A new solver on these stars continues from x (start
        ``'given'``) with the updated weights and centers.

        Returns
        -------
        list of piff.Star
        """
        Ps = self.per_star_params(x)
        update = self.use | (reserve & np.array([not s.is_flagged for s in self.stars]))
        out = []
        for i, s in enumerate(self.stars):
            star = piff.Star(s.data, self._fit_for(i, [P[i] for P in Ps]))
            if update[i]:
                m = self.psf.drawStar(star).image.array.ravel()
                flux = np.sum(self.W[i] * m * self.D[i]) / np.sum(self.W[i] * m * m)
                star = star.withFlux(float(flux))
                star = star.addPoisson(self.psf.drawStar(star))
                # reflux_stars flags a star whose reflux fails (as in Piff's fit).
                star = self.psf.reflux_stars([star], LoggerWrapper(None))[0][0]
                self.n_draws += 6  # 2 here, up to 4 in the reflux
            out.append(star)
        return out

    def _report_idx(self, report):
        if report == "all":
            return np.arange(self.n)
        if report == "fit":
            return self.fit_idx
        raise ValueError("report must be 'all' or 'fit'")

    def star_table(self, stars, rows):
        """Star table of a solution (simulation: `piffval.diagnostics.star_table`); ``rows``
        are the catalog rows of ``stars``."""
        return diagnostics.star_table(self.psf, stars, rows)

    def components(self, stars, rows):
        """Component summaries of a solution, compared with the simulation truth."""
        return component_tables(self.psf, stars, rows)

    def moment_table(self, stars, table, settings):
        """Moment residuals of a solution (`moment_residuals` with the simulation)."""
        if self.sim is None:
            raise ValueError("moment residuals need the simulation (sim=...)")
        return moment_residuals(self.psf, stars, table, self.sim, **settings)

    def solve(self, starts=("given",), max_nfev=50, moments=None, report="all", given=True):
        """Joint optimum from one or more starting points.

        Only the fit stars are drawn during the optimization (only they enter χ²); the
        reported solutions are re-drawn exactly for the stars chosen by ``report``.

        Parameters
        ----------
        starts : sequence of {'given', 'zero'}
            Starting points (see `x_start`). The same χ² from both is evidence of the
            global optimum. Default ('given',).
        max_nfev : int
            Maximum χ² evaluations per start (Jacobian evaluations not counted). Default 50.
        moments : dict, optional
            Moment-residual settings for the reported solutions (see `evaluate`).
        report : {'all', 'fit'}
            Stars in the reported solutions: all (default) or only the fit stars (no reserve
            diagnostics; much faster when most stars are reserve). See `evaluate`.
        given : bool
            Also report the given parameters (``'given'``). Default True; False saves one
            exact evaluation (and its moments).

        Returns
        -------
        dict
            ``'given'`` (if ``given``): `ReferenceResult` of the given parameters with the
            held centers and profiled fluxes (for a fitted PSF: Piff's solution on the
            reference's terms); ``'optimum'``: {start: ReferenceResult}. Each optimum's
            summary has ``nfev``, ``njev``, ``status``, ``converged`` (False when scipy
            stopped at ``max_nfev``: status 0), ``seconds``.
        """
        self._report_idx(report)  # validate before any work
        out = {}
        if given:
            out["given"] = self.evaluate(self.x_start("given"), "given", moments,
                                         report=report)
        out["optimum"] = {}
        if self.nfree == 0:
            return out
        for start in starts:
            # scipy's 'trf' sizes its first trust region from |x0|: round-off-level values
            # (e.g. a zero start projected on the basis) would make it tiny and stall.
            x0 = self.x_start(start)
            x0 = np.where(np.abs(x0) < 1e-9, 0.0, x0)
            self._cache = None
            t0 = time.time()
            sol = scipy.optimize.least_squares(self.residual, x0, jac=self.jacobian,
                                               method="trf", x_scale="jac", max_nfev=max_nfev)
            info = {"start": start, "nfev": int(sol.nfev), "njev": int(sol.njev),
                    "status": int(sol.status), "converged": bool(sol.status > 0),
                    "message": str(sol.message), "seconds": time.time() - t0}
            out["optimum"][start] = self.evaluate(sol.x, f"optimum ({start})", moments, info,
                                                  report)
        return out


def leaf_summary(lf):
    """Short description of one leaf for summaries."""
    return {"kind": lf.kind, "type": lf.name, "frozen": bool(lf.frozen),
            "n_terms": int(lf.nterms), "n_params": int(lf.plen),
            "linear": None if lf.linear is None else bool(lf.linear),
            "proj_err": lf.proj_err}


class ReferenceResult:
    """A reference solution: stars at the solution, their table, summary, moments, components.

    Attributes
    ----------
    stars : list of piff.Star
        Stars with the solution's parameters, profiled flux, the held center.
    table : astropy.table.Table
        As `FitResult.table` (``du = dv = 0`` with catalog centers; ``dof`` = pixels − 1
        per star), or the real-data star table for `DataReferenceSolver`.
    summary : dict
        χ², reference dof, ``chisq_dof``, ``n_free``, per-leaf info, optimizer status.
    moments : astropy.table.Table or None
        As `FitResult.moments`.
    components : dict
        As `FitResult.components()`. The optics ``fit_err`` column still holds Piff's
        parameter σ of the given solution (the reference does not estimate errors).
    x : numpy.ndarray or None
        The solver's free coefficients of this solution.
    """

    def __init__(self, stars, table, summary, moments, components, x=None):
        self.stars = stars
        self.table = table
        self.summary = summary
        self.moments = moments
        self.components = components
        self.x = x


def reference_from_fit(result, freeze=(), starts=("given",), max_nfev=50, moments="same",
                       report="all", given=True):
    """Reference optimum of the model of a `FitResult` (starting from Piff's solution).

    Parameters
    ----------
    result : FitResult
        A Piff fit (needs its stars: in memory, or reloaded with the simulation).
    freeze, starts, max_nfev, report, given
        See `ReferenceSolver` and `ReferenceSolver.solve`.
    moments : 'same', dict or None
        Moment settings: ``'same'`` (default) uses the fit's diagnostics settings when it
        computed moments; None skips them.

    Returns
    -------
    dict
        As `ReferenceSolver.solve`.
    """
    stars, rows = result.fitted_stars()
    if moments == "same":
        diag = result.config.get("diagnostics", {})
        moments = ({k: v for k, v in diag.items() if k != "moments"}
                   if diag.get("moments") and result.sim is not None else None)
    solver = ReferenceSolver(result.psf, stars, rows, sim=result.sim, freeze=freeze)
    return solver.solve(starts=starts, max_nfev=max_nfev, moments=moments, report=report,
                        given=given)


def reference_from_config(validation, freeze=(), starts=("zero",), max_nfev=50, moments=None,
                          report="all", given=True):
    """Reference optimum of a `PiffValidation`'s model without running Piff's fit.

    Piff only initializes the model (`PiffValidation.initialize`); the solver starts from that
    initialization (``'given'``) and/or from zero kernels and grids (``'zero'``, default).
    ``report``, ``given``: see `ReferenceSolver.solve`.

    Returns
    -------
    dict
        As `ReferenceSolver.solve`.
    """
    psf, stars, rows = validation.initialize()
    solver = ReferenceSolver(psf, stars, rows, sim=validation.sim, freeze=freeze)
    return solver.solve(starts=starts, max_nfev=max_nfev, moments=moments, report=report,
                        given=given)


def comparison_table(results, comparison="truth_os", algorithm="hsm"):
    """One row per solution: χ²/dof, flux ratios, and moment residuals of the fit stars.

    Parameters
    ----------
    results : dict
        Label → `ReferenceResult`, `FitResult` or `DataFitResult` (a fit's row uses Piff's
        own χ² and dof; ReferenceResults use the reference dof, so compare Piff with the
        reference via the ``'given'`` solution of `ReferenceSolver.solve`).
    comparison, algorithm : str
        Which moment residuals to summarize (if present). For real data use
        ``comparison='data'`` (e.g. with ``algorithm='gauss'``).

    Returns
    -------
    astropy.table.Table
        Columns ``label``, ``chisq``, ``chisq_dof``, ``flux_r``, ``flux_tot_r`` (NaN without
        truth), ``flux_cat_r`` (only when a result has it), ``dT_T``, ``dT_T_std``, ``de1``,
        ``de1_std``, ``de2``, ``de2_std``, ``n_moments`` (unflagged fit stars), ``converged``.
    """
    from astropy.table import Table

    flux_cols = ["flux_r", "flux_tot_r"]
    if any("flux_cat_r" in res.table.colnames for res in results.values()):
        flux_cols.append("flux_cat_r")
    out = []
    for label, res in results.items():
        summary = getattr(res, "summary", None)
        summary = summary if isinstance(summary, dict) else res.fit_summary
        t = res.table
        fit = t[(~t["reserve"]) & (t["flag"] == 0)]
        row = {"label": label, "chisq": summary.get("chisq", np.nan),
               "chisq_dof": summary["chisq_dof"]}
        for name in flux_cols:
            row[name] = (float(np.nanmean(fit[name])) if name in fit.colnames and len(fit)
                         else np.nan)
        row["converged"] = summary.get("converged", None)
        m = res.moments
        if m is not None:
            m = m[(m["comparison"] == comparison) & (m["algorithm"] == algorithm)
                  & (~m["reserve"]) & (m["flag"] == 0)]
        for q in ("dT_T", "de1", "de2"):
            values = np.asarray(m[q], dtype=float) if m is not None else np.array([])
            row[q] = float(values.mean()) if values.size else np.nan
            row[q + "_std"] = float(values.std(ddof=1)) if values.size > 1 else np.nan
        row["n_moments"] = 0 if m is None else len(m)
        row["converged"] = "" if row["converged"] is None else str(row["converged"])
        out.append(row)
    return Table(rows=out)
