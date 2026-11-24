import psf_fomo


def test_package_imports():
    import psf_fomo.io
    import psf_fomo.sim
    from psf_fomo.sim import RomanSimulation, Simulation, load_simulation  # noqa: F401
    import psf_fomo.moments
    from psf_fomo.moments import MomentsValidation, MomentsResult, load_moments  # noqa: F401
    import psf_fomo.piff_simfit
    from psf_fomo.piff_simfit import PiffFitConfig, PiffSimFit, PiffSimFitResult  # noqa: F401
    import psf_fomo.piff_datafit
    from psf_fomo.piff_datafit import PiffDataFit, PiffDataFitResult  # noqa: F401
    import psf_fomo.reference

    assert psf_fomo.__version__
