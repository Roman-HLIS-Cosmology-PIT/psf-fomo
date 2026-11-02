"""Saving and loading simulations.

Planned layout of a saved simulation:

- one FITS image file per observation, one image HDU per SCA;
- one catalog file per observation: HDU 1 is the full star table (with an ``sca`` column).
  With the simple WCS, per-SCA subset tables follow in further HDUs, because Piff's file input
  pairs one catalog with each image and does not filter by chip;
- the configuration and provenance as YAML;
- SEDs stored once each and referenced by id from the catalog.
"""
