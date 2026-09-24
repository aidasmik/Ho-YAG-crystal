"""Fluorescence photon spectrum inferred from Yb:LuAG emission cross sections.

The input cross sections are figure-guided reconstructions. Einstein A/B
reciprocity gives spontaneous photons per wavelength proportional to
sigma_em(lambda)/lambda**4 in a nondispersive-index approximation. This is a
normalized spectral *shape*, not an independently measured photon spectrum.
"""

from dataclasses import dataclass

import numpy as np

from .model import C, H, YbLuAGMaterial, _spectra


@dataclass(frozen=True)
class FluorescenceSpectrum:
    wavelength_nm: np.ndarray
    photon_probability_per_nm: np.ndarray
    energy_equivalent_wavelength_nm: float
    mean_photon_energy_J: float
    scope: str = ("derived from McCumber-consistent reconstructed cross sections; "
                  "optically thin intrinsic shape; escape/reabsorption not modeled")


def fluorescence_spectrum(material: YbLuAGMaterial) -> FluorescenceSpectrum:
    """Return normalized spontaneous-photon spectrum on the archive wavelength grid.

    The shape is normalized only over 880–1150 nm. The fraction of experimental
    fluorescence outside this interval is unknown.
    The energy-equivalent wavelength is hc/<E>, not the arithmetic mean lambda.
    """
    wavelength, _, _, _ = _spectra()
    emission = np.array([material.cross_sections_m2(float(w))[1] for w in wavelength])
    photon_weight = emission / wavelength**4
    norm = float(np.trapezoid(photon_weight, wavelength))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("fluorescence spectrum has no positive integral")
    probability = photon_weight / norm
    mean_energy = float(H * C / 1e-9 *
                        np.trapezoid(probability / wavelength, wavelength))
    return FluorescenceSpectrum(wavelength.copy(), probability,
                                H * C / mean_energy / 1e-9,
                                mean_energy)
