"""Reference optimum of a real-data fit (without truth).

The joint χ² optimum of the *same* Piff model on the same stars, as for simulations
(`psf_fomo.reference.ReferenceSolver`), to tell optimizer limits (Piff above the optimum of
its own model) from model limits (the optimum itself leaves χ² above the noise). Real data has
no true positions, so the star centers are held at Piff's fitted centers by default; the
reference χ² is then an upper bound on the optimum of Piff's problem (which also fits
centers). ``centers='catalog'`` holds them at the detection-catalog positions instead (a
measurement, not the truth).

The solution's star table, components and moments use the real-data versions (no truth):
`psf_fomo.piff_datafit.diagnostics.star_table`, components without truth columns, moments
``'data'`` / ``'model_os'``.

Example
-------
>>> out = res.reference(starts=("given", "zero"))      # res: PiffDataFitResult
>>> from psf_fomo.reference import comparison_table
>>> comparison_table({"Piff": res, "Piff, profiled": out["given"],
...                   "optimum": out["optimum"]["given"]}, comparison="data", algorithm="gauss")
"""
from ..piff_simfit.components import components as component_tables
from ..piff_simfit.residuals import drawable, moment_residuals
from ..reference.solver import ReferenceSolver
from . import diagnostics


class DataReferenceSolver(ReferenceSolver):
    """`ReferenceSolver` for a real-data fit: Piff's fitted centers, data-only diagnostics.

    Parameters
    ----------
    psf : piff.PSF
        The fitted model.
    stars : list of piff.Star
        Fitted stars with their data (`PiffDataFitResult.fitted_stars`).
    catalog : astropy.table.Table
        The fit's Piff catalog or selection table (matched to the stars by ``id``).
    freeze : sequence of str, optional
        Component kinds held at their given values (see `ReferenceSolver`).
    centers : {'fitted', 'catalog'}
        Held centers: Piff's fitted ones (default) or the catalog positions.

    Stars Piff left with a non-finite flux or center (excluded from its fit) are left out here
    too; ``n_excluded`` counts them.
    """

    def __init__(self, psf, stars, catalog, freeze=(), centers="fitted"):
        usable = [s for s in stars if drawable(s)]
        self.n_excluded = len(stars) - len(usable)
        rows = diagnostics.catalog_rows(usable, catalog)
        super().__init__(psf, usable, rows, sim=None, freeze=freeze, centers=centers)

    def star_table(self, stars, rows):
        return diagnostics.star_table(self.psf, stars, rows)

    def components(self, stars, rows):
        return component_tables(self.psf, stars, rows, truth=False)

    def moment_table(self, stars, table, settings):
        return moment_residuals(self.psf, stars, table, sim=None, **settings)


def reference_from_data_fit(result, freeze=(), starts=("given",), max_nfev=50,
                            moments="same", centers="fitted", report="all", given=True,
                            refresh=None):
    """Reference optimum of the model of a `PiffDataFitResult`, starting from Piff's solution.

    The solver holds the stars' weights and centers fixed. With model Poisson weights they
    depend on the model, which Piff updates after every iteration (Poisson variance of the
    current model, then a reflux of flux and center). ``refresh`` rounds do the same at the
    optimum and optimize again from it (iteratively reweighted least squares, the fixed point
    Piff converges to). A Piff fit's stars already carry its last update, so the default is
    no refresh for a fit and 2 for `PiffDataFit.initialize` results (whose weights and centers
    come from the initial model).

    Parameters
    ----------
    result : PiffDataFitResult
    freeze : sequence of str
        Component kinds held fixed (``'optics'``, ``'kernel'``, ``'pixelgrid'``).
    starts : sequence of {'given', 'zero'}
        Starting points (see `ReferenceSolver.x_start`). Default ('given',).
    max_nfev : int
        Maximum χ² evaluations per start. Default 50.
    moments : 'same', dict or None
        ``'same'`` (default): the fit's diagnostics settings (if it computed moments); a dict
        of `moment_residuals` settings; None: no moments.
    centers : {'fitted', 'catalog'}
        Held centers. Default Piff's fitted ones.
    report : {'all', 'fit'}
        Stars drawn at the reported solutions: all (default) or only the fit stars (much
        faster with a large reserve fraction; no reserve diagnostics). The optimization
        itself always draws only the fit stars.
    given : bool
        Also report Piff's solution (``'given'``; for an initialized result, the initial
        model). Default True.
    refresh : int or None
        Rounds of weight and center update at the optimum, each followed by a new
        optimization from it (`ReferenceSolver.reweighted_stars`). Default None: 2 for a
        `PiffDataFit.initialize` result, 0 for a Piff fit.

    Returns
    -------
    dict
        ``'given'`` (if ``given``): Piff's parameters with the held centers and profiled
        fluxes, on the reference's χ² and dof (dof = weighted pixels − fit stars − free
        coefficients); ``'optimum'``: {start: `ReferenceResult`}, with ``converged`` False
        when scipy stopped at ``max_nfev`` (in the last round). With refresh rounds each
        optimum's summary has ``rounds``: per round ``chisq_dof`` (on that round's weights),
        ``nfev``, ``seconds``, ``converged`` and ``max_dx`` (largest change of a free
        coefficient from the previous round; small = converged).
    """
    stars, _ = result.fitted_stars()
    if moments == "same":
        diag = result.config.get("diagnostics", {})
        moments = ({k: v for k, v in diag.items() if k != "moments"}
                   if diag.get("moments") else None)
    if refresh is None:
        refresh = 2 if result.fit_summary.get("initialized_only") else 0
    if refresh < 0:
        raise ValueError("refresh must be ≥ 0")
    solver = DataReferenceSolver(result.psf, stars, result.selection, freeze, centers)
    if refresh == 0:
        return solver.solve(starts=starts, max_nfev=max_nfev, moments=moments, report=report,
                            given=given)

    solver._report_idx(report)  # validate before any work
    out = {}
    if given:
        out["given"] = solver.evaluate(solver.x_start("given"), "given", moments,
                                       report=report)
    out["optimum"] = {}
    if solver.nfree == 0:
        return out
    for start in starts:
        # Intermediate rounds report only the fit stars and no moments (cheap); the last
        # round is reported as asked.
        sol = solver.solve(starts=(start,), max_nfev=max_nfev, report="fit",
                           given=False)["optimum"][start]
        rounds = [round_summary(sol, None)]
        current = solver
        for k in range(refresh):
            last = k == refresh - 1
            new_stars = current.reweighted_stars(sol.x, reserve=last and report == "all")
            current = DataReferenceSolver(result.psf, new_stars, result.selection, freeze,
                                          centers)
            previous = sol.x
            sol = current.solve(starts=("given",), max_nfev=max_nfev,
                                moments=moments if last else None,
                                report=report if last else "fit",
                                given=False)["optimum"]["given"]
            rounds.append(round_summary(sol, previous))
        sol.summary.update(label=f"optimum ({start})", start=start, refresh=refresh,
                           rounds=rounds)
        out["optimum"][start] = sol
    return out


def round_summary(sol, previous_x):
    """χ²/dof, optimizer status and coefficient change of one refresh round."""
    s = sol.summary
    max_dx = (None if previous_x is None or len(sol.x) == 0
              else float(abs(sol.x - previous_x).max()))
    return {"chisq_dof": s["chisq_dof"], "nfev": s["nfev"], "seconds": s["seconds"],
            "converged": s["converged"], "max_dx": max_dx}
