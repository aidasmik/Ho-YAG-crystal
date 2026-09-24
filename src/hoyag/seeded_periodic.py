"""Bounded, short-pulse Ho:YAG amplifier with a shared crystal state.

Pump and signal use photon-conserving Frantz--Nodvik kicks. The four-manifold
state relaxes between 10 kHz events. Heat follows the local optical/population
energy ledger. This is a short-pulse approximation, not a resolved temporal
envelope or a measured amplifier assembly.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import numpy as np

from .heat import HeatSpectroscopy, fluorescence_power_density, ion_energy_density
from .inhomogeneity import HoDensityField, relax_inhomogeneous_populations_dark
from .population_state import validate_populations
from .populations import C0, H, I7, I8, HoYAGFourLevelParams, four_level_rhs
from .propagation import Grid2D, angular_spectrum_propagate, optical_power
from .resonator import fluence_transfer
from .pump_source import resolve_pump_source


@dataclass(frozen=True)
class PeriodicAmplifierSettings:
    seed_energy_J: float = 10e-9  # YbSLAM proposal scenario, not a Ho measurement
    seed_fwhm_s: float = 10e-12  # explicit assumption; proposal says only picosecond
    repetition_rate_Hz: float = 10_000.
    pump_energy_J: float = 1e-3  # archived Ho reference scenario
    pump_fwhm_s: float = 10e-12
    pump_waist_m: float = .5e-3
    signal_traversals: int = 10
    pump_traversals: int = 2
    pump_reflectivity: float = .995
    signal_relay_transmission: float = 1.
    cavity_ejection_efficiency: float = 1.
    relay_distance_m: float = 0.  # ideal unit-magnification relay when zero
    max_cycles: int = 400
    population_tolerance: float = 2e-5
    cpu_workers: int = 1

    def __post_init__(self):
        positive = ('seed_energy_J','seed_fwhm_s','repetition_rate_Hz',
                    'pump_energy_J','pump_fwhm_s','pump_waist_m','population_tolerance')
        if any(not np.isfinite(getattr(self,k)) or getattr(self,k)<=0 for k in positive):
            raise ValueError('pulse, pump, repetition, and tolerance values must be positive and finite')
        if self.signal_traversals<1 or self.pump_traversals<1 or self.max_cycles<1:
            raise ValueError('traversal and cycle counts must be positive')
        if isinstance(self.cpu_workers,bool) or not isinstance(self.cpu_workers,int) or not 1<=self.cpu_workers<=16:
            raise ValueError('cpu_workers must be an integer from 1 to 16')
        if (not 0<self.pump_reflectivity<=1 or
            not 0<self.signal_relay_transmission<=1 or
            not 0<self.cavity_ejection_efficiency<=1):
            raise ValueError('reflectivity, relay transmission and cavity ejection must be in (0,1]')
        if not np.isfinite(self.relay_distance_m) or self.relay_distance_m<0:
            raise ValueError('relay distance must be finite and nonnegative')
        if self.seed_fwhm_s+(self.signal_traversals-1)*self.relay_distance_m/C0 >= 1/self.repetition_rate_Hz:
            raise ValueError('seed and inter-pass flight times must be shorter than the repetition period')


def _ground_state(density: HoDensityField):
    state=np.zeros((4,*density.values_m3.shape))
    state[I8]=density.values_m3
    return state


def _kick(fluence, state, density, dz, sigma_em, sigma_abs, photon_energy):
    fsat=photon_energy/(sigma_abs+sigma_em)
    gain=dz*(sigma_em*state[I7]-sigma_abs*state[I8])
    after=fluence_transfer(fluence,gain,fsat)
    after=np.where(density>0,after,fluence)
    delta=(after-fluence)/(photon_energy*dz)
    state[I7]-=delta
    state[I8]+=delta
    validate_populations(state,density,error_type=FloatingPointError)
    return after


def _relax_tile(task):
    state,density,duration,params=task
    return relax_inhomogeneous_populations_dark(state,density,duration,params)


def _dark_recovery(state,density,duration,params,workers,executor):
    if workers==1:
        return relax_inhomogeneous_populations_dark(state,density,duration,params)
    ny=density.shape[1]
    edges=np.linspace(0,ny,workers+1,dtype=int)
    tasks=((np.ascontiguousarray(state[:,:,lo:hi,:]),
            np.ascontiguousarray(density[:,lo:hi,:]),duration,params)
           for lo,hi in zip(edges[:-1],edges[1:]))
    result=np.empty_like(state)
    for (lo,hi),tile in zip(zip(edges[:-1],edges[1:]),executor.map(_relax_tile,tasks)):
        result[:,:,lo:hi,:]=tile
    return result


def _interpass_recovery(state,density,duration,params,workers,executor):
    """Evolve the same four manifolds during the declared optical relay flight."""
    if duration <= 0:
        return state
    if duration > 1e-8:
        return _dark_recovery(state,density,duration,params,workers,executor)
    # A midpoint step is accurate for the sub-10-ns relay legs supported here.
    k1=four_level_rhs(state,params)
    midpoint=validate_populations(state+.5*duration*k1,density,error_type=FloatingPointError)
    return validate_populations(state+duration*four_level_rhs(midpoint,params),density,
                                error_type=FloatingPointError)


def _one_cycle(seed, pump_profile, grid, density, state, cfg, params, hot_phase,
               pump_absorption_m2, spectroscopy, temperature_K, workers, executor):
    """One pump/seed pair; phase is a one-traversal screen from prior thermal solve."""
    dz=density.dz_m; pixel=grid.dx*grid.dy
    ep=H*C0/params.pump_wavelength_m
    es=H*C0/params.laser_wavelength_m
    pump_f=cfg.pump_energy_J*pump_profile
    pump_net=np.zeros_like(density.values_m3)
    for visit in range(cfg.pump_traversals):
        indices=range(density.nz) if visit%2==0 else range(density.nz-1,-1,-1)
        for iz in indices:
            before=pump_f
            pump_f=_kick(before,state[:,iz],density.values_m3[iz],dz,
                         params.sigma_em_pump_m2,pump_absorption_m2,ep)
            pump_net[iz]+=(before-pump_f)/dz
        if visit+1<cfg.pump_traversals:
            pump_f*=cfg.pump_reflectivity
    before_signal=state.copy()
    field=seed*np.sqrt(cfg.seed_energy_J/optical_power(seed,grid))
    input_f=np.abs(field)**2
    signal_net=np.zeros_like(density.values_m3)
    pass_records=[]
    pass_diagnostics=[]
    interpass_fluorescence=np.zeros_like(density.values_m3)
    passive_transport_change=0.
    relay_loss=0.
    total_density=float(np.sum(density.values_m3)*dz*pixel)
    if total_density<=0:
        raise ValueError('active Ho density is required')
    for visit in range(cfg.signal_traversals):
        incoming=np.abs(field)**2
        before_pass=float(np.sum(incoming)*pixel)
        before_upper=float(np.sum(state[I7])*dz*pixel)
        beam_density=float(np.sum(incoming*np.sum(density.values_m3,axis=0))*dz*pixel)
        if beam_density<=0:
            raise ValueError('seed field does not overlap active Ho medium')
        beam_upper_before=float(np.sum(incoming*np.sum(state[I7],axis=0))*dz*pixel)
        before_stored=float(np.sum(ion_energy_density(state,spectroscopy))*dz*pixel)
        local_log_gain=np.sum((params.sigma_em_laser_m2*state[I7]-
                               params.sigma_abs_laser_m2*state[I8])*dz,axis=0)
        weighted_log_gain=float(np.sum(incoming*local_log_gain)*pixel/before_pass)
        active=density.values_m3.sum(axis=0)>0
        beam_temperature=float(np.sum(incoming*temperature_K*active)/np.sum(incoming*active))
        indices=range(density.nz) if visit%2==0 else range(density.nz-1,-1,-1)
        for iz in indices:
            before=np.abs(field)**2
            after=_kick(before,state[:,iz],density.values_m3[iz],dz,
                        params.sigma_em_laser_m2,params.sigma_abs_laser_m2,es)
            ratio=np.divide(after,before,out=np.ones_like(after),where=before>0)
            field*=np.sqrt(ratio)
            signal_net[iz]+=(after-before)/dz
        if hot_phase is not None:
            field*=np.exp(1j*hot_phase)
        after_pass=float(np.sum(np.abs(field)**2)*pixel)
        extracted=after_pass-before_pass
        volume_extracted=float(np.sum(signal_net)*dz*pixel)-sum(
            item['net_stimulated_transfer_J'] for item in pass_diagnostics)
        if abs(extracted-volume_extracted)>1e-8*max(before_pass,abs(extracted),1e-20):
            raise FloatingPointError('pass optical gain does not match local stimulated transfer')
        after_upper=float(np.sum(state[I7])*dz*pixel)
        beam_upper_after=float(np.sum(incoming*np.sum(state[I7],axis=0))*dz*pixel)
        after_stored=float(np.sum(ion_energy_density(state,spectroscopy))*dz*pixel)
        if abs((before_upper-after_upper)*es-extracted)>1e-8*max(before_pass,abs(extracted),1e-20):
            raise FloatingPointError('pass inversion depletion does not match optical transfer')
        pass_records.append((before_pass,after_pass))
        pass_diagnostics.append({
            'pass':visit+1,'direction':'front_to_rear' if visit%2==0 else 'rear_to_front',
            'seed_energy_J':before_pass,'disk_exit_energy_J':after_pass,
            'small_signal_log_gain':weighted_log_gain,
            'saturated_energy_gain':after_pass/before_pass,
            'net_stimulated_transfer_J':extracted,
            'inversion_before_fraction':before_upper/total_density,
            'inversion_after_fraction':after_upper/total_density,
            'beam_weighted_inversion_before_fraction':beam_upper_before/beam_density,
            'beam_weighted_inversion_after_fraction':beam_upper_after/beam_density,
            'stored_ion_energy_before_J':before_stored,
            'stored_ion_energy_after_J':after_stored,
            'stored_laser_energy_before_J':before_upper*es,
            'stored_laser_energy_after_J':after_upper*es,
            'beam_weighted_temperature_K':beam_temperature})
        if visit+1<cfg.signal_traversals:
            if cfg.relay_distance_m:
                field=angular_spectrum_propagate(field,grid,params.laser_wavelength_m,
                                                  cfg.relay_distance_m)
            after_propagation=float(np.sum(np.abs(field)**2)*pixel)
            passive_transport_change+=after_pass-after_propagation
            field*=np.sqrt(cfg.signal_relay_transmission)
            after_relay=float(np.sum(np.abs(field)**2)*pixel)
            relay_loss+=after_propagation-after_relay
            delay=cfg.relay_distance_m/C0
            if delay:
                pre=state.copy()
                state[:]=_interpass_recovery(state,density.values_m3,delay,params,workers,executor)
                interpass_fluorescence+=.5*(
                    fluorescence_power_density(pre,params,spectroscopy)+
                    fluorescence_power_density(state,params,spectroscopy))*delay
    return (field,input_f,pump_net,signal_net,pass_records,pass_diagnostics,
            before_signal,interpass_fluorescence,passive_transport_change,relay_loss)


def solve_periodic_seeded_amplifier(seed, grid: Grid2D, density: HoDensityField,
                                    cfg: PeriodicAmplifierSettings,
                                    *, params=None, initial_populations=None,
                                    hot_phase_rad=None, spectroscopy=None,
                                    temperature_K=None):
    """Iterate pump, depleted seed, and dark recovery to a periodic pulse state.

    A caller must reject ``converged=False`` before claiming steady heat/optics.
    Heat is cycle averaged from the local optical ledger. Fluorescence during
    the dark interval is approximated by trapezoidal quadrature.
    """
    params=params or HoYAGFourLevelParams()
    spectroscopy=spectroscopy or HeatSpectroscopy()
    density.validate_grid(grid)
    seed=np.asarray(seed,complex)
    if seed.shape!=grid.shape or not np.all(np.isfinite(seed)) or optical_power(seed,grid)<=0:
        raise ValueError('seed must be a finite nonzero complex spatial field')
    if hot_phase_rad is not None:
        hot_phase_rad=np.broadcast_to(np.asarray(hot_phase_rad,float),grid.shape)
        if not np.all(np.isfinite(hot_phase_rad)):
            raise ValueError('hot phase must be finite')
    temperature=np.full(grid.shape,295.) if temperature_K is None else np.broadcast_to(
        np.asarray(temperature_K,float),grid.shape)
    if not np.all(np.isfinite(temperature)) or np.any(temperature<=0):
        raise ValueError('temperature_K must be finite and above absolute zero')
    state=_ground_state(density) if initial_populations is None else validate_populations(
        initial_populations,density.values_m3)
    x,y=grid.mesh
    pump=np.exp(-2*(x*x+y*y)/cfg.pump_waist_m**2)
    pump/=float(np.sum(pump)*grid.dx*grid.dy)
    pump_source=resolve_pump_source(params.pump_wavelength_m,cfg.pump_fwhm_s)
    pump_absorption=pump_source.effective_absorption_m2()
    period=1/cfg.repetition_rate_Hz
    converged=False
    workers=min(cfg.cpu_workers,grid.ny)
    context=(ProcessPoolExecutor(max_workers=workers) if workers>1 else nullcontext())
    with context as executor:
        for cycle in range(1,cfg.max_cycles+1):
            before=state.copy()
            (out,input_f,pump_net,signal_net,records,pass_diagnostics,before_signal,
             interpass_fluorescence,passive_transport_change,relay_loss)=_one_cycle(
                seed,pump,grid,density,state,cfg,params,hot_phase_rad,pump_absorption,
                spectroscopy,temperature,workers,executor)
            after_signal=state.copy()
            state=_dark_recovery(state,density.values_m3,
                                 period-cfg.seed_fwhm_s-(cfg.signal_traversals-1)*cfg.relay_distance_m/C0,
                                 params,workers,executor)
            active=density.values_m3>0
            residual=float(np.max(np.abs(state-before)[:,active]/density.values_m3[active]))
            if residual<=cfg.population_tolerance:
                converged=True
                break
    fluorescence=interpass_fluorescence+.5*(
        fluorescence_power_density(after_signal,params,spectroscopy)+
        fluorescence_power_density(state,params,spectroscopy))*(
        period-cfg.seed_fwhm_s-(cfg.signal_traversals-1)*cfg.relay_distance_m/C0)
    stored=ion_energy_density(state,spectroscopy)-ion_energy_density(before,spectroscopy)
    heat_J_m3=pump_net-signal_net-stored-fluorescence
    disk_exit=float(np.sum(np.abs(out)**2)*grid.dx*grid.dy)
    signal_extracted=float(np.sum(signal_net)*density.dz_m*grid.dx*grid.dy)
    initial_laser_stored=pass_diagnostics[0]['stored_laser_energy_before_J']
    ejected=disk_exit*cfg.cavity_ejection_efficiency
    ejection_loss=disk_exit-ejected
    optical_residual=(cfg.seed_energy_J+signal_extracted-passive_transport_change-
                      relay_loss-ejection_loss-ejected)
    scale=max(cfg.seed_energy_J,abs(signal_extracted),disk_exit,1e-20)
    if abs(optical_residual)>1e-8*scale:
        raise FloatingPointError('seed/material/relay/ejection optical energy does not close')
    if abs(signal_extracted-sum(r['net_stimulated_transfer_J'] for r in pass_diagnostics))>1e-8*scale:
        raise FloatingPointError('summed pass extraction does not match local optical ledger')
    return {'converged':converged,'cycles':cycle,'population_residual':residual,
            'cpu_workers':workers,
            'field_out':out*np.sqrt(cfg.cavity_ejection_efficiency),
            'input_fluence_J_m2':input_f,
            'output_fluence_J_m2':np.abs(out)**2*cfg.cavity_ejection_efficiency,
            'input_energy_J':cfg.seed_energy_J,
            'disk_exit_energy_J':disk_exit,'output_energy_J':ejected,
            'pump_absorbed_J':float(np.sum(pump_net)*density.dz_m*grid.dx*grid.dy),
            'signal_extracted_J':signal_extracted,
            'gain_medium_extraction_efficiency':signal_extracted/initial_laser_stored if initial_laser_stored>0 else None,
            'initial_stored_laser_energy_J':initial_laser_stored,
            'cavity_ejection_efficiency':cfg.cavity_ejection_efficiency,
            'cavity_ejection_loss_J':ejection_loss,
            'relay_loss_J':relay_loss,
            'passive_transport_change_J':passive_transport_change,
            'optical_energy_balance_residual_J':optical_residual,
            'heat_W_m3':heat_J_m3*cfg.repetition_rate_Hz,
            'populations_before_signal':before_signal,
            'pass_records_J':records,'pass_diagnostics':pass_diagnostics,
            'populations_before_pump':state,
            'population_change_energy_J':float(np.sum(stored)*density.dz_m*grid.dx*grid.dy),
            'fluorescence_energy_J':float(np.sum(fluorescence)*density.dz_m*grid.dx*grid.dy),
            'heat_energy_J':float(np.sum(heat_J_m3)*density.dz_m*grid.dx*grid.dy),
            'heat_ledger_closure_J':float(np.sum(pump_net-signal_net-stored-fluorescence-heat_J_m3)*density.dz_m*grid.dx*grid.dy),
            'pump_source':pump_source.summary(),
            'assumptions':['short-pulse Frantz-Nodvik gain; temporal envelope and GVD omitted',
                           'pump and seed events are sequential; dark fluorescence uses trapezoidal quadrature',
                           'fixed Ho cross sections at 295 K reference; no validated temperature series',
                           'temperature is a fixed macrostate during each picosecond seed train']}
