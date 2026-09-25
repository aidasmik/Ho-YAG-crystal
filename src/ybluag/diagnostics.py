"""Bounds and missing-input checks; these are not device predictions."""

from __future__ import annotations

import math
import numpy as np

from .model import C, H, YbLuAGMaterial, spectral_cross_sections_m2


def gain_feasibility(material: YbLuAGMaterial, *, thickness_m: float,
                     pump_passes: int, signal_traversals: int,
                     regenerative_round_trips: int | None = None,
                     input_energy_J: float | None = None,
                     requested_output_energy_J: float | None = None,
                     injection_efficiency: float = 1.0,
                     extraction_efficiency: float = 1.0,
                     held_roundtrip_retention: float = 1.0,
                     disk_hr_reflectivity: float = 1.0,
                     ideal_relay_power_retention: float = 1.0) -> dict:
    """Unsaturated gain ceilings under homogeneous fixed-temperature assumptions.

    The pump asymptote omits spontaneous decay and signal load, so even this
    tighter ceiling is optimistic. Pump passes affect achieved inversion, not
    the pump-wavelength thermodynamic asymptote.
    """
    if thickness_m <= 0 or signal_traversals < 1 or pump_passes < 1:
        raise ValueError("positive thickness and pass counts required")
    if not np.isfinite(ideal_relay_power_retention) or not 0 < ideal_relay_power_retention <= 1:
        raise ValueError("ideal relay retention must be in (0, 1]")
    sa, se = material.cross_sections_m2(material.signal_wavelength_nm)
    pa, pe = material.cross_sections_m2(material.pump_wavelength_nm)
    n = material.number_density_m3
    beta_transparent = sa/(sa+se)
    beta_pump_limit = pa/(pa+pe)
    saturation_fluence = H*C/(material.signal_wavelength_nm*1e-9)/(sa+se)
    traversals = (2*regenerative_round_trips if regenerative_round_trips is not None
                  else signal_traversals)
    gain_full = math.exp(n*se*thickness_m*traversals)
    gain_pump = math.exp(n*((sa+se)*beta_pump_limit-sa)*thickness_m*traversals)
    configured_losses = (injection_efficiency*extraction_efficiency*
                         held_roundtrip_retention**(regenerative_round_trips or 0)*
                         disk_hr_reflectivity**(regenerative_round_trips or 0))
    if regenerative_round_trips is None:
        configured_losses *= ideal_relay_power_retention**(traversals-1)
    requested_gain = (requested_output_energy_J/input_energy_J
                      if requested_output_energy_J is not None and input_energy_J else None)
    return {
        "pump_passes": pump_passes,
        "material_traversals": traversals,
        "regenerative_round_trips": regenerative_round_trips,
        "signal_transparency_inversion": beta_transparent,
        "pump_asymptotic_inversion": beta_pump_limit,
        "saturation_fluence_J_m2": saturation_fluence,
        "full_inversion_unsaturated_gain_ceiling": gain_full,
        "pump_asymptotic_unsaturated_gain_ceiling": gain_pump,
        "configured_optical_retention": configured_losses,
        "pump_ceiling_after_configured_losses": gain_pump*configured_losses,
        "requested_energy_gain": requested_gain,
        "requested_energy_exceeds_pump_ceiling": (requested_gain > gain_pump*configured_losses
                                                  if requested_gain is not None else None),
        "scope": "Upper bounds: homogeneous 20 °C cross sections, no saturation, "
                 "no spatial hole burning. Pump asymptote neglects finite lifetime "
                 "and signal depletion. Optical retention is separate from gain-medium "
                 "extraction and omits unspecified aperture and switch transients.",
    }


def hardware_validity(*, pump_nm: float, coating_band_nm: tuple[float, float],
                      switching_time_s: float | None = None,
                      cavity_roundtrip_time_s: float | None = None) -> dict:
    """Report unknowns instead of assigning zero device risk."""
    band_status = ("outside_specified_target_band" if not coating_band_nm[0] <= pump_nm <= coating_band_nm[1]
                   else "inside_target_band_not_measured")
    switching = ("unknown" if switching_time_s is None or cavity_roundtrip_time_s is None
                 else "slower_than_round_trip" if switching_time_s >= cavity_roundtrip_time_s
                 else "faster_than_round_trip")
    return {
        "pump_coating_status": band_status,
        "pump_coating_measured_reflectivity": None,
        "switching_status": switching,
        "ase_parasitic_risk": "unknown: geometry-dependent spontaneous-emission transport missing",
        "nonlinear_B_integral": None,
        "nonlinear_B_integral_reason": "bulk nonlinear index and intracavity temporal peak field unavailable",
        "coating_absorption_W": None,
        "coating_absorption_reason": "measured coating absorptance unavailable",
        "damage_margin": None,
        "damage_margin_reason": "coating and crystal damage data for the actual pulse unavailable",
    }


def spectral_gain_screen(material: YbLuAGMaterial, *, source_fwhm_fs: float,
                         stretched_fwhm_ps: float, shared_inversion: float | None = None,
                         material_traversals: int, thickness_m: float,
                         excited_fraction_by_slice=None, density_m3_by_slice=None,
                         temperature_K_by_slice=None, incident_fluence_J_m2=None,
                         architecture="ideal_multipass") -> dict:
    """Wavelength-resolved *small-signal* screen using one shared inversion.

    The transform-limited source intensity FWHM sets a Gaussian power spectrum.
    Stretching preserves this assumed spectral magnitude. Spectral phase,
    time-resolved gain depletion, gain narrowing and compression are unsolved.
    """
    if source_fwhm_fs <= 0 or stretched_fwhm_ps*1000 < source_fwhm_fs:
        raise ValueError("invalid source or stretched duration")
    center = material.signal_wavelength_nm
    bandwidth = (center*1e-9)**2/C * (0.441/(source_fwhm_fs*1e-15))*1e9
    wavelengths = np.linspace(center-2*bandwidth, center+2*bandwidth, 41)
    if wavelengths[0] < material.spectral_range_nm[0] or wavelengths[-1] > material.spectral_range_nm[1]:
        return {"status": "not_calculated", "reason": "source band extends beyond reconstructed spectra"}
    if architecture != "ideal_multipass":
        return {"status": "not_calculated", "reason":
                "regenerative wavelength-dependent cavity propagation is not implemented"}
    weights = np.exp(-4*math.log(2)*((wavelengths-center)/bandwidth)**2)
    weights /= weights.sum()
    spatial = excited_fraction_by_slice is not None
    if spatial:
        beta = np.asarray(excited_fraction_by_slice, dtype=float)
        density = np.asarray(density_m3_by_slice, dtype=float)
        incident = np.asarray(incident_fluence_J_m2, dtype=float)
        if (beta.ndim != 3 or density.shape != beta.shape or
                incident.shape != beta.shape[1:] or
                not np.all(np.isfinite(beta)) or np.any((beta < 0) | (beta > 1)) or
                not np.all(np.isfinite(density)) or np.any(density < 0) or
                not np.all(np.isfinite(incident)) or np.any(incident < 0) or
                not np.sum(incident) > 0):
            raise ValueError("invalid spatial spectral population, density or incident field")
        temperature = (np.full(beta.shape, material.temperature_K)
                       if temperature_K_by_slice is None else
                       np.broadcast_to(np.asarray(temperature_K_by_slice, float), beta.shape))
        dz = thickness_m / beta.shape[0]
        small_signal = []
        for wavelength in wavelengths:
            sa, se = material.local_cross_sections_m2(float(wavelength), temperature)
            log_gain = np.sum(density * (se*beta-sa*(1-beta)), axis=0) * dz * material_traversals
            if np.max(log_gain) > 700:
                return {"status": "not_calculated", "reason": "small-signal exponential exceeds numeric range"}
            small_signal.append(float(np.sum(incident * np.exp(log_gain)) / np.sum(incident)))
        small_signal = np.asarray(small_signal)
    else:
        if shared_inversion is None or not 0 <= shared_inversion <= 1:
            raise ValueError("supply spatial population or valid homogeneous inversion")
        sa, se = np.asarray([material.cross_sections_m2(float(w)) for w in wavelengths]).T
        g = material.number_density_m3*(se*shared_inversion-sa*(1-shared_inversion))
        small_signal = np.exp(g*thickness_m*material_traversals)
    return {
        "status": "spatial_small_signal_screen_only" if spatial else "small_signal_screen_only",
        "wavelength_nm": wavelengths.tolist(),
        "input_spectral_weights": weights.tolist(),
        "unsaturated_gain": small_signal.tolist(),
        "shared_inversion": None if spatial else shared_inversion,
        "spatial_weighting": ("incident-fluence-weighted exp(integrated local gain)" if spatial
                              else "homogeneous analytic reference"),
        "source_fwhm_nm": bandwidth,
        "stretch_preserves_spectral_magnitude": True,
        "spectral_phase": None,
        "compressed_duration_s": None,
        "reason": ("One spatial pre-pulse inversion shared by all wavelength bins; ideal relays, no spectral saturation or cavity wavelength dependence."
                   if spatial else
                   "Homogeneous inversion reference; no spectral-bin-specific inversion. This is not a saturated pulse prediction."),
    }
