"""Reference solver: joint χ² optimum of the model Piff fits (CLAUDE.md §8).

- `ReferenceSolver`: the solver on a Piff PSF and its stars.
- `reference_from_fit`: optimum of a `FitResult`'s model, compared with Piff's solution.
- `reference_from_config`: optimum of a `PiffValidation`'s model without a Piff fit.
- `comparison_table`: χ²/dof, flux ratios and moment residuals side by side.
"""
from .solver import (ReferenceResult, ReferenceSolver, comparison_table, reference_from_config,
                     reference_from_fit)

__all__ = ["ReferenceSolver", "ReferenceResult", "reference_from_fit",
           "reference_from_config", "comparison_table"]
