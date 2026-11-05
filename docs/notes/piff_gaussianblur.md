# Piff PR "GaussianBlur" — study notes (2026-11-04)

**Where the code is.** There is no separate `Piff_gaussianblur/` folder. The PR branch is
checked out in our usual clone **`Piff/`**: branch `gaussian_blur`, HEAD `da203ae`, three
commits on top of our pinned `6ada133` (`e42021c` "Add GaussianBlur class", `70b9805` "Make
GaussianBlur work correctly with BasisPolynomial", `da203ae` "Add Fixed to the PixelGrid io
test"). If Piff is installed in editable mode from `Piff/`, **every run now uses the PR branch**,
including our normal baseline; `psf_fomo.io.piff_commit()` will record `da203ae`.

Files changed (`git diff 6ada133 HEAD`): new `piff/gaussian_blur.py` (288 lines),
`piff/basis_interp.py` (+line search), `piff/simplepsf.py` (+`chisq_func` callback),
`piff/model.py` (+`nonlinear_chisq`, +`constrain_params`), `piff/__init__.py` (exports
`GaussianBlur`, `FourierGaussian`), docs, tests (`tests/test_gaussian_blur.py` new; small edits
to `test_pixel.py`, `test_simple.py`, `test_stats.py`).

## 1. `GaussianBlur` model (`piff/gaussian_blur.py`)

| question | answer (from source) |
|---|---|
| parameters, order | `[ixx, iyy, ixy]` — **note the order** (ours: `(Mxx, Mxy, Myy)`) |
| units / axes | **arcsec²**, world (u, v): the profile is a GalSim object in world coordinates, `kval = exp(−½(ixx kx² + iyy ky² + 2 ixy kx ky))`. Native pix² = arcsec² / 0.11². |
| profile | `FourierGaussian`: a GSObject defined only in k-space (`_is_analytic_x = False`), unit flux, centred. Covariance [[ixx, ixy], [ixy, iyy]]: convolution adds exactly these moments. |
| init | `init=None` (default) → `[0, 0, 0]` = delta function. `init` may be a scalar trace (split equally) or a 3-vector; must be PSD (else `ValueError`). Composite `default_init` is ignored (zero is already 'delta'). |
| flux / centre | **not fitted** by the model: `fit`/`chisq` change only the 3 params; flux and centre come from `star.fit` (Piff's reflux updates them). Same as our `GaussianMoments`; unlike Piff's `Gaussian`. `_centered = True`, `_model_can_be_offset = False`. |
| PSD constraint | **projection**, not bounds or reparameterisation: `constrain_params` sets negative eigenvalues to 0 (nearest PSD matrix in Frobenius norm) and rounds `ixy` to the physical side. `getProfile` always projects; the *raw* (possibly non-PSD) params stay on the star / in the interpolator. So it can only broaden — the same model family as our positive `GaussianMoments` (which uses the same eigenvalue clip). |
| zero blur | Exactly supported: `kval ≡ 1`; `_maxk = 1e100`, `_stepk = 1e100` for zero eigenvalues; rank-1 blur also exact (no floor; ours adds a 1e-6 pix floor). Derivatives at 0 are analytic and finite (one-sided in effect, because negative steps are projected). |
| `_method` | `'auto'`: pixel applied once when the full Convolve is drawn (same as ours, RomanOptics, GaussianMixture). |
| derivatives | analytic in k-space: `∂kval/∂ixx = −½kx² kval`, `∂/∂iyy = −½ky²`, `∂/∂ixy = −kx ky`; drawn through `convert_func` (i.e. convolved with the other Convolve components). |
| `chisq(star)` (BasisPolynomial path) | linearises at the **projected** params, `b` shifted by `A (projected − raw)` so the step is relative to the raw params. `dof = N(w≠0)`. |
| `fit(star)` (Polynomial / Mean path) | **one** Gauss–Newton step per Piff iteration: proposal = raw + lstsq(A, b), projected to PSD, then a bounded `minimize_scalar` line search on the segment raw → proposal, plus explicit checks of both ends. Stays put if no fraction improves χ² (test `test_no_improving_step`). No per-star `max_nfev`, no `params_var`. |

`model.nonlinear_chisq = True` on GaussianBlur (default False for every other model).
`Model.constrain_params` default = identity (called only by GaussianBlur).

## 2. Interpolators and the line search

- **Supported**: any interpolator. `Polynomial`/`Mean`/`Fixed` use `fit()` per star then fit
  the (projected, PSD) per-star params. `BasisPolynomial` uses `chisq()` (A, b per star) and a
  joint solve (`SimplePSF.quadratic_chisq` is True when the model has `chisq` and the interp
  has `degenerate_points`).
- **Line search** (`BasisInterp.solve(..., chisq_func=None)` → `_line_search`): only when
  `quadratic_chisq and model.nonlinear_chisq`. `SimplePSF` builds `chisq_func` = Σ over fit
  stars of the rendered residual² at trial coefficients (re-interpolated, drawn with each star's
  `convert_func`). Bounded Brent on fraction ∈ [0, 1] + both ends.
- **Existing linear models (PixelGrid)**: `nonlinear_chisq` is False → `solve` is called
  exactly as before; `chisq_func=None` skips every new line. RomanOptics, GaussianMixture,
  our GaussianMoments and Piff's Gaussian do not touch the changed code. **No change to our
  baseline is expected from the source**. The notebook `piff_gaussianblur.ipynb` runs the
  baseline models (optics only, PG, free ± PG) on the branch as part of its own tests.
- Side effect: a BasisPolynomial blur makes the Convolve (and Sum) `degenerate_points = True`,
  so outlier rejection is skipped at iteration 0 (as with a PixelGrid already).
- Cost of the BasisPolynomial path per Piff iteration: 4 draws per star (residual + 3
  derivatives) + (≈ 10–25 line-search evaluations) × N stars full draws.

## 3. Expected behaviour (to test)

- **Polynomial order 0** = average of per-star *projected* params → the same
  clip-before-average boundary bias as our positive kernel's per-star fits (§9.6) is expected
  at zero true blur (and a one-GN-step-per-iteration fit may need more Piff iterations).
- **BasisPolynomial order 0** averages the linearised problems *before* projecting → should
  sit at zero cleanly (no clipping bias); raw params may go non-PSD where the data want
  sharpening (projected to the wall when drawn).
- **Reference optimum**: the model family is identical to positive `GaussianMoments` (Gaussian
  of the PSD projection of M), so blur and positive optima should agree to solver precision;
  Polynomial vs BasisPolynomial order 0 give the same reference parameter space. Differences
  between kinds should appear only in Piff's fits (optimiser), not in the optimum.
- Cannot sharpen: shear g1 = 0.02 and the flipped combined case (dil 0.98, MTF 0.98) should
  hit the wall as our positive kernel does.

## 4. What our code must change to use it

- `FitConfig.add_kernel`: new kind `'blur'` → `{'type': 'GaussianBlur', 'init': ...}`; `init`
  given in our pix² (Mxx, Mxy, Myy) converted to `[ixx, iyy, ixy]` arcsec²; new option
  `interp='Polynomial' | 'BasisPolynomial'` (BasisPolynomial only for `'blur'`; our
  GaussianMoments has no `chisq`, §10 gotcha).
- `piffval/components.py`: `kind_of` maps `'GaussianBlur'` → `'kernel'`; `kernel_moments`
  converts `constrain_params(params)` → (ixx, ixy, iyy)/0.11², `clipped` when the projection
  changes the raw params. `component_image` / `plot_components('kernel')`: a bare
  `FourierGaussian` drawn without a pixel needs an FFT with maxk ≈ 1e100 at (near-)zero blur →
  draw the equivalent real-space `gaussian_kernel(M)` for display instead.
- `reference/solver.py`: GaussianBlur is nonlinear (the second-difference test will say so;
  nothing to force). Finite-difference step must be per model, not per kind: `STEPS['kernel']
  = 1e-3` is pix² for GaussianMoments but would be 1e-3 arcsec² ≈ 0.08 pix² for GaussianBlur →
  use 1e-3 × 0.11² arcsec². `x_start('zero')`: same apex issue as positive GaussianMoments
  (forward differences at M = 0 leave the cone) → start at the equivalent of (0.01, 0, 0.01)
  pix². `basis_matrix` already handles both Polynomial and BasisPolynomial for any model.
- `piffval/diagnostics.py`: `kernel_max_nfev` stays None for blur (no per-star `max_nfev`);
  nothing else is kernel-specific. Convergence flag unchanged (Piff's warning).
- `docs/piffval.md`, `CLAUDE.md` §4/§11: pin note (PR branch vs `6ada133`).

## 5. Differences from our kernels

| | GaussianBlur (PR) | positive GaussianMoments | signed GaussianMoments |
|---|---|---|---|
| params | [ixx, iyy, ixy] arcsec² | (Mxx, Mxy, Myy) pix² | (Mxx, Mxy, Myy) pix² |
| profile | exact Gaussian, k-space only | real-space Gaussian, 1e-6 pix floor | sum of 4 tiny Gaussians (σ₀ ≈ 0.05 pix), some negative weights |
| moments | exactly projected M | exactly clipped M | exactly M |
| sign | PSD only (projection) — broadens only | PSD only (bounds on Mxx, Myy + clip) | any sign (sharpening allowed) |
| accuracy | exact for any PSD M | exact for any PSD M | first order in M (differs at ≈ 0.05 pix²) |
| linear in M | no | no | yes (reference caches it) |
| start | 0 allowed (exact delta, analytic derivatives) | must be inside the cone, default (0.01, 0, 0.01) | 0 |
| per-star fit | 1 GN step + line search per Piff iteration, analytic derivatives | scipy `trf` to convergence (≤ `max_nfev`), finite differences | same as positive, no bounds |
| interpolators | Polynomial / Mean / **BasisPolynomial** (joint, line search) | Polynomial only | Polynomial only |
| flux / centre in its fit | fixed | fixed | fixed |
