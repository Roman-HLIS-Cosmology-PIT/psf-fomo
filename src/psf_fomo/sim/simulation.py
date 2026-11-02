"""Simulation builder and result object.

Planned contents:

- ``RomanSimulation``: builder. The constructor takes the common settings (SCAs, number of
  observations, stars per SCA, filter, seed); dedicated methods add placement, brightness,
  PSF errors, noise and WCS; ``run()`` renders and returns a ``Simulation``. The full
  configuration round-trips through YAML (``to_yaml`` / ``from_yaml``).
- ``Simulation``: images, per-observation catalogs, truth and config; access to Piff-ready
  stars and truth PSFs; ``save()``.
- ``load_simulation(path)``: returns the same ``Simulation`` object as ``run()``.
"""
