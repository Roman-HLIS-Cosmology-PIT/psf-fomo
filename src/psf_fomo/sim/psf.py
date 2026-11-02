"""Truth PSF construction.

Planned contents:

- the optical PSF from Piff's ``RomanOpticalModel.getProfile`` with the truth Zernike
  aberrations (the same spatial approximation Piff fits with);
- the PSF error chain applied to it: dilation → shear → jitter → MTF (three-Gaussian
  charge-diffusion kernel) → shift, each error with optional per-observation and per-field
  variation;
- caching of sample-point PSFs per (SCA, observation, aberrations, wavelength).
"""
