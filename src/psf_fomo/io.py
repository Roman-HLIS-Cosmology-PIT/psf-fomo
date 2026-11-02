"""Shared FITS/YAML helpers and provenance.

Planned contents:

- writing/reading YAML configs;
- writing/reading FITS tables with units in the column metadata;
- a provenance record saved with every output: psf-fomo, GalSim, Piff (and ngmix/TreeCorr when
  used) versions, plus the Piff git commit. Piff's version string does not distinguish ``main``
  from the release, so the commit is read from the ``.git/HEAD`` file of the Piff checkout
  when available.
"""
