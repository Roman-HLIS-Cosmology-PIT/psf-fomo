import psf_fomo


def test_package_imports():
    import psf_fomo.io
    import psf_fomo.sim
    from psf_fomo.sim import RomanSimulation, Simulation, load_simulation  # noqa: F401
    import psf_fomo.moments
    import psf_fomo.piffval
    import psf_fomo.reference

    assert psf_fomo.__version__
