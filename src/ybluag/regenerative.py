"""Short-pulse Yb:LuAG regenerative amplifier with an explicit cavity map.

The Frantz–Nodvik map is exact for fluence in a two-manifold, homogeneous
short-pulse slice with fixed cross sections. It does not resolve chirp,
spectral gain narrowing, nonlinear phase, or the pulse's temporal reshaping.
"""

from dataclasses import dataclass
import math

import numpy as np

from hoyag.propagation import Grid2D, angular_spectrum_transfer, optical_power
from hoyag.resonator import ThinDiskResonator, fluence_transfer

from .model import C, H, YbLuAGMaterial
from .multipass_pump import steady_multipass_pump, transport_multipass_pump

# LuAG Sellmeier at 1.03 µm plus the measured 7 at.% film index offset.
# These are assumptions for a 12 at.% bulk disk until its index is measured.
LUAG_PHASE_INDEX_ASSUMED = 1.8302
LUAG_GROUP_INDEX_ASSUMED = 1.8488


@dataclass(frozen=True)
class RegenerativeCavity:
    round_trips: int = 10
    air_gap_m: float = 0.25
    mirror_radius_m: float = 0.5
    disk_hr_reflectivity: float = 0.9995
    held_roundtrip_retention: float = 0.98
    injection_efficiency: float = 0.9
    extraction_efficiency: float = 0.9
    disk_diameter_m: float = 0.010

    def validate(self, material: YbLuAGMaterial, thickness_m: float,
                 grid: Grid2D, repetition_rate_Hz: float) -> ThinDiskResonator:
        if isinstance(self.round_trips, bool) or not isinstance(self.round_trips, int) or not 1 <= self.round_trips <= 60:
            raise ValueError("regenerative round trips must be 1–60")
        for name in ("air_gap_m", "mirror_radius_m", "disk_hr_reflectivity",
                     "held_roundtrip_retention", "injection_efficiency",
                     "extraction_efficiency", "disk_diameter_m"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid regenerative {name}")
        for name in ("disk_hr_reflectivity", "held_roundtrip_retention",
                     "injection_efficiency", "extraction_efficiency"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must not exceed one")
        cavity = ThinDiskResonator(
            disk_diameter_m=self.disk_diameter_m, disk_thickness_m=thickness_m,
            air_gap_m=self.air_gap_m, output_mirror_radius_m=self.mirror_radius_m,
            output_transmission=1e-12, disk_hr_reflectivity=self.disk_hr_reflectivity,
            other_roundtrip_loss=1-self.held_roundtrip_retention,
            wavelength_m=material.signal_wavelength_nm*1e-9,
            host_index=LUAG_PHASE_INDEX_ASSUMED,
            host_group_index=LUAG_GROUP_INDEX_ASSUMED)
        if not cavity.stable:
            raise ValueError("cold regenerative cavity is geometrically unstable")
        if grid.nx*grid.dx < self.disk_diameter_m or grid.ny*grid.dy < self.disk_diameter_m:
            raise ValueError("FFT window must contain the disk")
        if self.round_trips*cavity.roundtrip_time_s >= 1/repetition_rate_Hz:
            raise ValueError("regenerative storage time exceeds seed period")
        return cavity


def amplify_regenerative(material: YbLuAGMaterial, grid: Grid2D, seed_field,
                         seed_energy_J: float, pump_W_m2, density_scale,
                         thickness_m: float, pump_passes: int,
                         repetition_rate_Hz: float, cavity_settings: RegenerativeCavity,
                         fluorescence_photon_energy_J: float,
                         fluorescence_escape_yield: float):
    """Converge the periodic inversion and return a cycle energy ledger.

    The held Pockels/polarizer state is represented by a measured-or-assumed
    power retention. Switching is instantaneous at an integer round trip.
    Pump rates are recomputed during recovery as the inversion changes.
    """
    seed = np.asarray(seed_field, complex)
    scale = np.asarray(density_scale, float)
    pump = np.asarray(pump_W_m2, float)
    if seed.shape != grid.shape or pump.shape != grid.shape or scale.ndim != 3 or scale.shape[1:] != grid.shape:
        raise ValueError("regenerative field, pump, and dopant grids disagree")
    if not np.isfinite(seed_energy_J) or seed_energy_J <= 0 or not np.isfinite(repetition_rate_Hz) or repetition_rate_Hz <= 0:
        raise ValueError("invalid seed energy or repetition rate")
    cavity = cavity_settings.validate(material, thickness_m, grid, repetition_rate_Hz)
    dz = thickness_m/scale.shape[0]
    pixel_area = grid.dx*grid.dy
    photon_J = H*C/(material.signal_wavelength_nm*1e-9)
    sigma_a, sigma_e = material.cross_sections_m2(material.signal_wavelength_nm)
    saturation_J_m2 = photon_J/(sigma_a+sigma_e)
    density = material.number_density_m3*scale
    x, y = grid.mesh
    aperture = (x*x+y*y) <= (cavity_settings.disk_diameter_m/2)**2
    mirror_phase = np.exp(-1j*2*np.pi/(material.signal_wavelength_nm*1e-9)*
                          (x*x+y*y)/cavity_settings.mirror_radius_m)
    transfer = angular_spectrum_transfer(grid, material.signal_wavelength_nm*1e-9,
                                         cavity_settings.air_gap_m)

    def travel(field):
        return np.fft.ifft2(np.fft.fft2(field)*transfer)

    def recover(beta, duration_s, chunks):
        absorbed_integral = np.zeros_like(beta)
        excited_integral = np.zeros_like(beta)
        for _ in range(chunks):
            dt = duration_s/chunks
            midpoint, absorbed, _ = transport_multipass_pump(
                material, pump, scale, thickness_m, pump_passes, beta)
            up, down = material.rates_s1(midpoint, 0)
            rate = up+down+1/material.lifetime_s
            equilibrium = up/rate
            factor = -np.expm1(-rate*dt)
            excited_integral += equilibrium*dt + (beta-equilibrium)*factor/rate
            absorbed_integral += absorbed*dt
            beta = equilibrium + (beta-equilibrium)*np.exp(-rate*dt)
        return beta, absorbed_integral, excited_integral

    def disk_pass(field, beta, order, signal_ledger):
        for iz in order:
            incoming = np.abs(field)**2
            g = density[iz]*((sigma_a+sigma_e)*beta[iz]-sigma_a)*dz
            outgoing = fluence_transfer(incoming, g, saturation_J_m2)
            change = outgoing-incoming
            signal_ledger[iz] += change
            population_change = np.divide(change, density[iz]*dz*photon_J,
                                          out=np.zeros_like(change),
                                          where=density[iz] > 0)
            beta[iz] -= population_change
            if np.min(beta[iz]) < -1e-8 or np.max(beta[iz]) > 1+1e-8:
                raise RuntimeError("regenerative fluence map violates Yb population bounds")
            beta[iz] = np.clip(beta[iz], 0, 1)
            field *= np.sqrt(np.divide(outgoing, incoming,
                                       out=np.ones_like(outgoing), where=incoming > 0))
        return field

    pump_steady = steady_multipass_pump(material, pump, scale, thickness_m, pump_passes)
    beta_before = pump_steady.excited_fraction_by_slice.copy()
    roundtrip_s = cavity.roundtrip_time_s
    period_s = 1/repetition_rate_Hz
    orders = (range(scale.shape[0]), range(scale.shape[0]-1, -1, -1))
    for cycle in range(1, 101):
        beta = beta_before.copy()
        field = seed*np.sqrt(seed_energy_J*cavity_settings.injection_efficiency)
        signal_ledger = np.zeros_like(beta)
        absorbed_integral = np.zeros_like(beta)
        excited_integral = np.zeros_like(beta)
        history = []
        for iround in range(cavity_settings.round_trips):
            before = optical_power(field, grid)
            field *= aperture
            disk_pass(field, beta, orders[0], signal_ledger)
            field *= math.sqrt(cavity_settings.disk_hr_reflectivity)
            disk_pass(field, beta, orders[1], signal_ledger)
            after_disk = optical_power(field, grid)
            field = travel(field)
            field *= mirror_phase*math.sqrt(cavity_settings.held_roundtrip_retention)
            field = travel(field)*aperture
            after_return = optical_power(field, grid)
            history.append((before, after_disk, after_return))
            beta, a, b = recover(beta, roundtrip_s, 1)
            absorbed_integral += a
            excited_integral += b
        stored = optical_power(field, grid)
        output_field = field*math.sqrt(cavity_settings.extraction_efficiency)
        output_energy = optical_power(output_field, grid)
        beta, a, b = recover(beta, period_s-cavity_settings.round_trips*roundtrip_s, 8)
        absorbed_integral += a
        excited_integral += b
        residual = float(np.max(np.abs(beta-beta_before)))
        beta_before = beta
        if residual <= 1e-6:
            break
    else:
        raise RuntimeError("regenerative Yb population did not converge within 100 seed periods")
    mean_beta = excited_integral/period_s
    pump_absorbed = absorbed_integral/period_s
    signal_gain = signal_ledger*repetition_rate_Hz
    fluorescence = (density*mean_beta/material.lifetime_s*
                    fluorescence_escape_yield*fluorescence_photon_energy_J*dz)
    heat_slices = (pump_absorbed-signal_gain-fluorescence)/dz
    return {
        "output_field": output_field,
        "heat_W_m3_by_slice": heat_slices,
        "pump_absorbed_W_m2_by_slice": pump_absorbed,
        "signal_gain_W_m2_by_slice": signal_gain,
        "escaping_fluorescence_W_m2_by_slice": fluorescence,
        "roundtrip_energy_J": np.asarray(history),
        "roundtrip_time_s": roundtrip_s,
        "storage_time_s": cavity_settings.round_trips*roundtrip_s,
        "cold_cavity_waist_m": cavity.waist_m,
        "stored_energy_J": stored,
        "output_energy_J": output_energy,
        "injection_efficiency": cavity_settings.injection_efficiency,
        "extraction_efficiency": cavity_settings.extraction_efficiency,
        "held_roundtrip_retention": cavity_settings.held_roundtrip_retention,
        "cycles": cycle,
        "residual": residual,
        "mean_excited_fraction_before_pulse": float(np.mean(beta_before)),
        "pump_steady_iterations": pump_steady.iterations,
        "saturation_fluence_J_m2": saturation_J_m2,
    }
