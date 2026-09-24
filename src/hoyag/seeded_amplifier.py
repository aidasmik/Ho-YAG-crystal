"""Externally seeded encounter/relay path over one shared Ho:YAG crystal state.

This is a generic API, not a claimed complete hardware topology. Each material
traversal updates the same population array. Seed and pump inputs stay separate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .inhomogeneity import HoDensityField
from .population_state import validate_populations
from .populations import HoYAGFourLevelParams
from .signal import propagate_structured_signal_saturated
from .temporal import TimeGrid, propagate_spatiotemporal, spatiotemporal_energy


def wrap_2pi(phase):
    value=np.asarray(phase,float)
    if not np.all(np.isfinite(value)):
        raise ValueError('SLM phase must be finite')
    return np.mod(value,2*np.pi)


def apply_phase_modulator(field, phi_pattern, phi_correction=0., *,
                          phi_applied=None, amplitude_transmission=1.):
    """Ideal phase-only by default; return separate requested/applied masks."""
    arr=np.asarray(field,complex)
    if arr.ndim!=3:
        raise ValueError('seed field must have (time,y,x) shape')
    pattern=np.broadcast_to(np.asarray(phi_pattern,float),arr.shape[-2:])
    correction=np.broadcast_to(np.asarray(phi_correction,float),arr.shape[-2:])
    requested=wrap_2pi(pattern+correction)
    applied=requested if phi_applied is None else wrap_2pi(
        np.broadcast_to(np.asarray(phi_applied,float),arr.shape[-2:]))
    amplitude=np.broadcast_to(np.asarray(amplitude_transmission,float),arr.shape[-2:])
    if np.any(~np.isfinite(amplitude)) or np.any((amplitude<0)|(amplitude>1)):
        raise ValueError('amplitude transmission must be finite in [0,1]')
    return {'field_after':amplitude[None]*arr*np.exp(1j*applied[None]),
            'phi_pattern':pattern.copy(), 'phi_correction':correction.copy(),
            'phi_requested':requested, 'phi_applied':applied.copy(),
            'amplitude_transmission':amplitude.copy(),
            'idealized_response':phi_applied is None}


@dataclass
class SharedCrystalState:
    density: HoDensityField
    populations_by_slice: np.ndarray
    temperature_K: np.ndarray | None = None
    stress_Pa: np.ndarray | None = None
    displacement_m: np.ndarray | None = None

    def __post_init__(self):
        self.populations_by_slice=validate_populations(
            self.populations_by_slice,self.density.values_m3).copy()


@dataclass(frozen=True)
class DiskEncounter:
    traversals: int = 1
    first_direction: int = 1
    relay_distance_m: float = 0.
    relay_power_transmission: float = 1.
    reflection_power_transmission: float = 1.

    def __post_init__(self):
        if self.traversals not in (1,2) or self.first_direction not in (-1,1):
            raise ValueError('encounter requires one or two directed traversals')
        if self.relay_distance_m<0 or not all(0<=x<=1 for x in
                (self.relay_power_transmission,self.reflection_power_transmission)):
            raise ValueError('invalid relay or reflection setting')


def amplify_seeded_pulse(field, grid, time: TimeGrid, seed_energy_J: float,
                         crystal: SharedCrystalState, encounters: list[DiskEncounter],
                         *, params=None):
    """Run a declared external seed through ordered visits to one crystal.

    Relay lengths and losses must be supplied by the caller. Population recovery,
    pump steps, temperature feedback and hardware timing are separate updates.
    """
    if not encounters:
        raise ValueError('at least one disk encounter is required')
    crystal.density.validate_grid(grid)
    if not np.isfinite(seed_energy_J) or seed_energy_J<=0:
        raise ValueError('seed energy must be positive')
    params=params or HoYAGFourLevelParams()
    signal=np.asarray(field,complex)
    if signal.shape!=(time.nt,*grid.shape):
        raise ValueError('seed field shape mismatch')
    energy=seed_energy_J
    records=[]
    for encounter_id,encounter in enumerate(encounters,1):
        for traversal in range(encounter.traversals):
            direction=encounter.first_direction*(-1 if traversal%2 else 1)
            if direction==1:
                density=crystal.density
                populations=crystal.populations_by_slice
            else:
                density=HoDensityField(crystal.density.values_m3[::-1],crystal.density.length_m)
                populations=crystal.populations_by_slice[:,::-1]
            result=propagate_structured_signal_saturated(
                signal,grid,time,energy,populations,density,params)
            crystal.populations_by_slice=(result.final_populations_by_slice if direction==1
                                           else result.final_populations_by_slice[:,::-1]).copy()
            signal=result.field_out
            energy=result.output_energy_J
            records.append({'encounter':encounter_id,'traversal':traversal+1,
                            'direction':direction,'input_energy_J':result.input_energy_J,
                            'output_energy_J':energy,
                            'signal_energy_change_by_slice_J':result.signal_energy_change_by_slice_J.tolist(),
                            'shared_crystal_state':True})
            if traversal+1<encounter.traversals:
                signal*=np.sqrt(encounter.reflection_power_transmission)
                energy*=encounter.reflection_power_transmission
        if encounter.relay_distance_m:
            signal=propagate_spatiotemporal(signal,grid,time,params.laser_wavelength_m,
                                            encounter.relay_distance_m)
            energy=spatiotemporal_energy(signal,grid,time)
        signal*=np.sqrt(encounter.relay_power_transmission)
        energy*=encounter.relay_power_transmission
    return {'application_mode':'seeded_multipass_amplifier',
            'seeded_amplifier_configuration_complete':False,
            'field_out':signal,'output_energy_J':energy,
            'disk_encounters':len(encounters),
            'material_traversals':sum(e.traversals for e in encounters),
            'traversal_records':records,
            'crystal':crystal}
