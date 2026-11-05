# Phase 4 findings: reference optimum and kernels (2026-11-04)

Results from `notebooks/phase4_piff.ipynb` (section "earlier reference tests"), run by the
user. Setup of every case: SCA 5, simple WCS, 30 stars, 10⁵ e⁻ per star, Gaussian noise
σ = 5 e⁻, stamp 25, pinned optics, PG 5×5 with `BasisPolynomial` order 2, kernels order 0,
Piff `zero` start for the reference unless noted. Moments: oversampled HSM, model − truth,
fit stars; "std" is star-to-star std (error on mean = std/√30).

## How to run a reference optimum without a Piff fit

`psf_fomo.reference.solver.reference_from_config(PiffValidation, starts=("zero",),
moments=dict(native=..., wmom_fwhm=[...], oversampled=["hsm"]))`. Piff only initialises
(`PiffValidation.initialize`: `initialize_flux_center` + `initialize_params`); `out["given"]` is
that initial state ("initial state (no Piff fit)" of prev_code), `out["optimum"][start]` the
joint optimum. 1–2 min per case vs ~20 min for Piff + reference on hard cases (Piff runs to
`max_iter` = 20 without converging there).

## Agreement with prev_code

The reference optimum reproduces the prev_code notebook (`check_piff_PRs.ipynb`):

| case | χ²/dof old → new | flux_r old → new | kernel M old → new [pix²] |
|---|---|---|---|
| signed + PG, MTF ×1.05 | 0.996 → 0.997 | 1.0003 → 1.0015 | (0.0074, 0.0001, 0.0073) → (0.0076, 0.0001, 0.0072) |
| signed + PG, dilation 1.05 | 3.140 → 3.183 | 1.0935 → 1.0967 | (0.0526, −0.0015, 0.0550) → (0.0535, −0.0014, 0.0537) |
| signed + PG, g1 0.02 | 1.66 → 1.72 | — | (0.0176, 0.0002, −0.0212) → (0.0179, 0, −0.0205) |
| signed + PG, dil 1.02 + MTF 1.02 + g (0.02, −0.02) | 2.24 → 2.34 (*) | 1.031 → 1.033 | (0.0424, −0.0141, 0.0082) → (0.0436, −0.0142, 0.0080) |

(*) prev_code had 100 stars and free optics; new run 30 stars, pinned optics.
Remaining differences: different star samples, T = Mxx + Myy instead of HSM σ², upstream
`GaussianMixture` MTF instead of the old patch.

## Overlapping stars (simulation trap)

`sim/stars.py:draw_positions` has an edge margin but **no minimum separation**. With 100
random stars and stamp 25 on one SCA the expected number of pairs closer than a stamp is
≈ 0.75 (30 stars: ≈ 0.07). One bright neighbour (10⁵ e⁻, σ 5 e⁻) in a stamp gave χ²/dof ≈ 900
for the *initial state* and every model (vs 27 with 30 stars), while mean moments were
unaffected (the oversampled truth has no neighbour). Check per-star χ² and nearest-neighbour
distances before trusting χ² of many-star runs. A minimum-separation option is not
implemented (Phase 2 change, needs approval).

## Results by error type

Closed loop, no error: PG and positive kernel reach χ²/dof ≈ 0.997, moment residuals < 1e-4.
The positive-kernel optimum is exactly M = 0: the joint optimum has no boundary bias (§9.6's
T_k ≈ 0.002 comes from Piff's per-star clipped fits). Caveat: the `zero` start is the cone
apex, so "found" and "stuck" are indistinguishable there.

MTF width (expected per-axis ΔM ≈ 0.106 (s² − 1): 0.0109 at ×1.05, 0.022 at ×1.1):

| model | χ²/dof | M per axis | dT/T (std) |
|---|---|---|---|
| signed, ×1.05 | 1.051 | 0.0084 | −5.8e-4 (1.6e-5) |
| signed + PG, ×1.05 | 0.997 | 0.0074 | −4e-5 (2.3e-4) |
| positive, ×1.1 | 1.207 | 0.0172 | −1.06e-3 (3.3e-5) |
| positive + PG, ×1.1 | 1.000 | 0.015 | −2e-5 (2.3e-4) |

A single Gaussian cannot match the three-Gaussian MTF; the χ² fit matches the core, so M is
below the unweighted second-moment increase and a kernel alone leaves a small, very
significant size bias. PG removes it (χ² → 1) but raises per-star moment scatter ~10×
(150 coefficients follow noise) and takes part of the width (grid flux ≠ 0), so kernel M
alone no longer measures the MTF error.

Shear g1 = 0.02 (initial de1 = −0.0217):

| model | χ²/dof | de1 | dT/T | M |
|---|---|---|---|---|
| PG | 3.74 | −3.8e-3 | +3e-4 | — |
| positive | 5.41 | −1.19e-2 | +1.27e-2 | (0.0079, 0, 0.0000) |
| positive + PG | 2.67 | −1.6e-3 | +2.2e-3 | (0.0198, 0, 0.0000) |
| signed | 3.22 | −3e-4 | +4e-4 | (0.0088, 0, −0.0086) |
| signed + PG | 1.72 | +1.3e-4 | +4.6e-4 | (0.0179, 0, −0.0205) |

The positive kernel sits on the PSD wall (Myy = 0) and trades shape error for a size error.
Signed alone finds the pure-shape ΔMxx ≈ −ΔMyy. With a PG the kernel is ~2× larger and the
grid compensates (kernel/grid split not unique). χ² > 1 remains: a geometric shear also
stretches the rings.

Dilation 1.05: signed alone χ² 15.6, dT/T +6e-3 (convolution cannot rescale diffraction
rings). + PG: χ² ≈ 3.2–3.4, moments right (≈ 1e-3), but flux_r 1.09 vs grid sum −0.07
(flux_tot_r 1.013). Positive + PG (χ² 3.37, M 0.065) differs from signed + PG (3.18, M 0.054)
although M is well inside the cone: at M ≈ 0.05 pix² the signed kernel is no longer a Gaussian
(first-order only), so the two are different profile families for large corrections.

Combined (dil 1.02, MTF 1.02, g 0.02/−0.02, 30 stars):

| model | χ²/dof | dT/T | de1 | M |
|---|---|---|---|---|
| PG | 9.77 | −4.6e-3 | −3.7e-3 | — |
| signed | 7.22 | +2.2e-3 | +3.3e-4 | (0.0220, −0.0086, 0.0040) |
| signed + PG | 2.34 | +1.6e-4 | +7.0e-4 | (0.0436, −0.0142, 0.0080) |
| positive + PG | 2.37 | +1.0e-4 | +6.4e-4 | (0.0501, −0.0166, 0.0083) |

Positive + PG ≈ signed + PG here because the needed M is inside the PSD cone: isotropic
broadening (dilation + MTF, T/2 ≈ 0.026) exceeds the shear part (≈ 0.023); eigenvalues
≈ 0.049 and 0.003. The clip never activates. This is fragile: larger shear relative to the
broadening, or broadening of the other sign (model wider than truth, e.g. dilation 0.98 /
MTF ×0.98), puts the optimum outside the cone. Proposed check (not yet run): the same
combined case with dilation 0.98, MTF ×0.98.

## Why the signed kernel is a sum of four Gaussians

A kernel must be a function with flux 1, zero centroid and second moments M. A Gaussian
with a negative-eigenvalue M does not exist (non-normalisable; a deconvolution in k-space),
and no non-negative function has negative variance, so sharpening needs negative lobes.
Moments are linear in the weights of a sum of fixed Gaussians: 1 flux + 3 moment conditions
→ 4 Gaussians (base G(m0·I), σ0 = 0.05 pix, and G(m0·I + h E_j) for xx, xy, yy). The
differences are finite-difference derivatives in M, so PSF ⊛ K ≈ PSF + ½ M : ∇∇PSF: 3
unrestricted parameters, linear in M (the reference solver caches it as linear), but only
first-order accurate (visible at M ≈ 0.05 pix²).

## Piff interplay seen in these runs

- Piff's block-wise fit stops at `max_iter` = 20 unconverged on large errors (signed + PG,
  dilation 1.05: Piff χ²/dof 10.3, flux_r 0.95, grid +0.028 vs optimum 3.18, 1.097, −0.076):
  optimiser limit (§9.2), not model limit.
