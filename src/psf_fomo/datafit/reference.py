"""Reference optimum of a real-data fit (CLAUDE.md §8, without truth).

The joint χ² optimum of the *same* Piff model on the same stars, as for simulations
(`psf_fomo.reference.ReferenceSolver`), to tell optimiser limits (Piff above the optimum of
its own model) from model limits (the optimum itself leaves χ² above the noise). Real data has
no true positions, so the star centres are held at Piff's fitted centres by default; the
reference χ² is then an upper bound on the optimum of Piff's problem (which also fits
centres). ``centers='catalog'`` holds them at the detection-catalog positions instead (a
measurement, not the truth).

The solution's star table, components and moments use the real-data versions (no truth):
`psf_fomo.datafit.diagnostics.star_table`, components without truth columns, moments
``'data'`` / ``'model_os'``.

Example
-------
>>> out = res.reference(starts=("given", "zero"))      # res: DataFitResult
>>> from psf_fomo.reference import comparison_table
>>> comparison_table({"Piff": res, "Piff, profiled": out["given"],
...                   "optimum": out["optimum"]["given"]}, comparison="data", algorithm="gauss")
"""
from ..piffval.components import components as component_tables
from ..piffval.residuals import moment_residuals
from ..reference.solver import ReferenceSolver
from . import diagnostics


class DataReferenceSolver(ReferenceSolver):
    """`ReferenceSolver` for a real-data fit: Piff's fitted centres, data-only diagnostics.

    Parameters
    ----------
    psf : piff.PSF
        The fitted model.
    stars : list of piff.Star
        Fitted stars with their data (`DataFitResult.fitted_stars`).
    catalog : astropy.table.Table
        The fit's Piff catalog or selection table (matched to the stars by ``id``).
    freeze : sequence of str, optional
        Component kinds held at their given values (see `ReferenceSolver`).
    centers : {'fitted', 'catalog'}
        Held centres: Piff's fitted ones (default) or the catalog positions.
    """

    def __init__(self, psf, stars, catalog, freeze=(), centers="fitted"):
        rows = diagnostics.catalog_rows(stars, catalog)
        super().__init__(psf, stars, rows, sim=None, freeze=freeze, centers=centers)

    def star_table(self, stars):
        return diagnostics.star_table(self.psf, stars, self.rows)

    def components(self, stars):
        return component_tables(self.psf, stars, self.rows, truth=False)

    def moment_table(self, stars, table, settings):
        return moment_residuals(self.psf, stars, table, sim=None, **settings)


def reference_from_data_fit(result, freeze=(), starts=("given",), max_nfev=50,
                            moments="same", centers="fitted"):
    """Reference optimum of the model of a `DataFitResult`, starting from Piff's solution.

    Parameters
    ----------
    result : DataFitResult
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
        Held centres. Default Piff's fitted ones.

    Returns
    -------
    dict
        ``'given'``: Piff's parameters with the held centres and profiled fluxes, on the
        reference's χ² and dof (dof = weighted pixels − fit stars − free coefficients);
        ``'optimum'``: {start: `ReferenceResult`}, with ``converged`` False when scipy
        stopped at ``max_nfev``.
    """
    stars, _ = result.fitted_stars()
    if moments == "same":
        diag = result.config.get("diagnostics", {})
        moments = ({k: v for k, v in diag.items() if k != "moments"}
                   if diag.get("moments") else None)
    solver = DataReferenceSolver(result.psf, stars, result.selection, freeze, centers)
    return solver.solve(starts=starts, max_nfev=max_nfev, moments=moments)
