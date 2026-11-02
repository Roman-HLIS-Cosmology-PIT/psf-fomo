"""``GaussianMoments`` convolution kernel parameterised by its second moments (Phase 4).

Planned contents: parameters (Mxx, Mxy, Myy) in native pix²; ``signed=False`` is a PSD-clipped
Gaussian that only broadens, ``signed=True`` is a linear combination of tiny Gaussians whose
second moments equal M of either sign.
"""
