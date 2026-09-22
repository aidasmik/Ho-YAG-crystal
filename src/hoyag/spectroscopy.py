"""Room-temperature Ho:YAG pump spectroscopy utilities for Stage 2P.

The 295 K absorption model below is a local surrogate constructed from the
published peak wavelength, peak cross section and FWHM values near 1.9 um.
It is intended for picosecond pump overlap around 1907.7 nm, not as a
replacement for the complete measured 1700-2200 nm spectrum.
"""

from __future__ import annotations

import numpy as np

C0 = 299_792_458.0

HOYAG_ABSORPTION_PEAKS_295K = (
    (1907.3, 1.259e-24, 4.5),
    (1928.3, 5.77e-25, 2.5),
    (1932.5, 6.88e-25, 3.0),
    (1973.5, 1.99e-25, 2.4),
)


def pump_absorption_cross_section_295K(wavelength_m):
    """Approximate Ho:YAG 295 K pump absorption cross section."""
    wavelength_nm = np.asarray(wavelength_m, dtype=float) * 1e9
    sigma = np.zeros_like(wavelength_nm, dtype=float)
    for center_nm, peak_m2, fwhm_nm in HOYAG_ABSORPTION_PEAKS_295K:
        sigma += peak_m2 * np.exp(
            -4.0 * np.log(2.0) * ((wavelength_nm - center_nm) / fwhm_nm) ** 2
        )
    if sigma.ndim == 0:
        return float(sigma)
    return sigma


def temporal_spectral_weights(field, time):
    """Return envelope-frequency offsets and energy-proportional FFT weights."""
    arr = np.asarray(field, dtype=np.complex128)
    if arr.shape[0] != time.nt:
        raise ValueError("first field axis must match time.nt")
    spectrum = np.fft.fft(arr, axis=0)
    power = np.abs(spectrum) ** 2
    if arr.ndim > 1:
        power = np.sum(power, axis=tuple(range(1, arr.ndim)))
    return time.frequency_hz, power


def effective_pump_absorption_cross_section_295K(
    field,
    time,
    center_wavelength_m: float,
) -> float:
    """Spectrum-weighted effective sigma_a for a coherent pulse envelope.

    This is an energy-spectrum-weighted cross section, so it is the correct
    first-order scalar surrogate for total pump-energy attenuation. The Stage 2P
    rate equations reuse it with the carrier photon energy; for the current
    1--10 ps bandwidths near 1907.7 nm the photon-energy correction is below
    about 2e-4 relative. It does not reproduce frequency-dependent spectral
    reshaping inside a saturated crystal.
    """
    if center_wavelength_m <= 0:
        raise ValueError("center_wavelength_m must be positive")

    frequency_offset_hz, weight = temporal_spectral_weights(field, time)
    nu0 = C0 / center_wavelength_m
    nu = nu0 + frequency_offset_hz
    valid = nu > 0

    sigma = np.zeros_like(nu, dtype=float)
    sigma[valid] = pump_absorption_cross_section_295K(C0 / nu[valid])

    denom = float(np.sum(weight))
    if denom <= 0:
        raise ValueError("field has zero spectral energy")
    return float(np.sum(sigma * weight) / denom)
