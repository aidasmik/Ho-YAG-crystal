"""Stage 7W: polarization-pair-resolved adiabatic cavity/assembly closure.

Reuses the audited pump, population, photon, heat, plate, bond and Jones physics.
The change is candidate selection, branch identity and convergence bookkeeping.
Distinct eigenfields are never mixed to conceal a polarization switch.
"""
from __future__ import annotations
from dataclasses import asdict
import time
import numpy as np
from .populations import HoYAGFourLevelParams
from .pump_source import resolve_pump_source
from .numerical_quality import cavity_sampling_diagnostic
from .thermal_resonator import area_averaged_lg0, sample_cycle_heat
from .vector_cavity import (VectorRoundTrip, PlaneExchange, normalize_vector,
                           aligned_distance, passive_mode_losses)
from .coupled_resonator import (HotCavitySettings, CoupledHotCavityResult,
    FieldCoupledLaser, PlateAssembly, time_averaged_populations, initial_vector_modes,
    _weighted_relative)
from .polarization_tracking import (TrackingSettings, normalized_modes,
    solve_tracked_eigenfields, invariant_subspace_distance, mixture_distance)


def convergence_metrics(used, predicted, power, gain_used, gain_predicted, *,
                        previous_fields=None, previous_powers=None,
                        previous_gain=None):
    """All residuals are undamped and preserve actual polarization and power."""
    power=np.asarray(power,float)
    total=max(float(power.sum()),1e-20)
    out={'field_residual':max(aligned_distance(a,b) for a,b in zip(used,predicted)),
         'subspace_residual':invariant_subspace_distance(used,predicted),
         'mixture_stationarity':mixture_distance(used,power,predicted,power),
         'gain_stationarity':float(np.max(abs(np.asarray(gain_predicted)-gain_used))),
         'modal_power_relative_change':None,'mixture_relative_change':None,
         'modal_gain_change':None}
    if previous_powers is not None:
        before=np.asarray(previous_powers,float)
        out['modal_power_relative_change']=float(np.sum(abs(power-before))/max(total,float(before.sum()),1e-20))
        out['mixture_relative_change']=mixture_distance(used,power,previous_fields,before)
        out['modal_gain_change']=float(np.max(abs(np.asarray(gain_used)-previous_gain)))
    return out


def accepted_closure(row, settings):
    """Retain every original field/thermal/power gate; add pair/mixture gates."""
    limits={'field_residual':settings.field_tolerance,
            'subspace_residual':settings.field_tolerance,
            'mixture_stationarity':settings.field_tolerance,
            'mixture_relative_change':settings.field_tolerance,
            'heat_residual':settings.heat_relative_tolerance,
            'temperature_change_K':settings.temperature_tolerance_K,
            'displacement_change_m':settings.displacement_tolerance_m,
            'output_relative_change':settings.power_relative_tolerance,
            'modal_power_relative_change':settings.power_relative_tolerance,
            'loss_residual':settings.loss_tolerance,
            'gain_stationarity':settings.loss_tolerance,
            'modal_gain_change':settings.loss_tolerance,
            'eigen_residual':settings.eigen_tolerance}
    gates={name:(row.get(name) is not None and np.isfinite(row[name]) and row[name]<=limit)
           for name,limit in limits.items()}
    gates['candidate_bank_completed']=row.get('eigen_converged') is True
    gates['no_split_polarization_family']=not row.get('split_boundary_pairs',[])
    gates['no_new_branch_replacement']=not row.get('photon_seed_reset_branches',[])
    gates['periodic_optical_state']=row.get('optical_periodic_converged') is True
    return gates


def run_polarization_hot_cavity(grid,mesh,cavity,pump_energy_J,assembly_configuration,*,
        repetition_rate_Hz=1e4,pump_duration_s=10e-12,pump_waist_m=.5e-3,
        params=None,density_m3=None,initial_fields=None,mode_count=2,settings=None,
        tracking=None,spectroscopy=None,progress=None,pump_source=None,
        pump_absorption_m2=None):
    """Two-or-more vector eigenbranches with separate incoherent photon states.

    Two is a conservative minimum for THIS polarization-unfiltered cavity, not
    a universal physical theorem. Legacy single-mode controls remain available
    through run_coupled_hot_cavity and are never silently promoted to two modes.
    New eigenfields replace old guesses directly; only the heat-source iteration
    is relaxed. No polarizer, averaging of modal powers, or coherent beat model is
    introduced. Stored photon seeds preserve matched labels; a genuinely replaced
    branch resets all photon seeds and must settle again before acceptance.
    """
    s=settings or HotCavitySettings(eigen_candidates=6)
    tracking=tracking or TrackingSettings()
    if not isinstance(mode_count,int) or isinstance(mode_count,bool) or mode_count<2:
        raise ValueError('Stage 7W requires at least two explicit vector branches; use legacy for one-mode controls')
    if s.eigen_candidates<mode_count+2:
        raise ValueError('at least two additional candidate eigenfields are required')
    if not np.isfinite(pump_energy_J) or pump_energy_J<=0:
        raise ValueError('pump energy must be finite and positive')
    if not np.allclose(np.diff(mesh.z_edges_m),cavity.disk_thickness_m/mesh.nz,rtol=1e-12,atol=0):
        raise ValueError('uniform depth cells required by modal population solver')
    if not np.isclose(mesh.r_edges_m[-1],cavity.disk_diameter_m/2,rtol=0,atol=1e-12):
        raise ValueError('material and cavity radii disagree')
    p=params or HoYAGFourLevelParams()
    source=resolve_pump_source(p.pump_wavelength_m,pump_duration_s,source=pump_source,
                              absorption_override_m2=pump_absorption_m2)
    density=mesh.field(p.N_total_m3 if density_m3 is None else density_m3,'Ho density')
    if np.any(density<=0):raise ValueError('active modal cells require positive Ho density')
    fields=(initial_vector_modes(grid,cavity,tuple(i//2 for i in range(mode_count)))
            if initial_fields is None else np.asarray(initial_fields,complex).copy())
    if fields.shape!=(mode_count,2,*grid.shape):raise ValueError('initial vector-field shape mismatch')
    fields=normalized_modes(fields)
    exchange=PlaneExchange(grid,mesh,order=s.projection_order)
    assembly=PlateAssembly(mesh,grid,assembly_configuration)
    pump=area_averaged_lg0(mesh,pump_waist_m)
    temperature,displacement,screens=assembly.solve(np.zeros(mesh.shape))
    passive=VectorRoundTrip(grid,cavity,screens)
    losses,_=passive_mode_losses(passive,fields)
    history=[];previous_heat=previous_temperature=previous_u=None
    previous_power=previous_fields=previous_powers=previous_gain=None
    initial_fractions=initial_log_photons=None
    optical=heat=mean_fractions=eigen=predicted=assembly_heat=None
    used_fields=fields.copy();used_losses=losses.copy()
    converged=False;status='outer iteration limit reached';streak=0
    for iteration in range(s.max_outer_iterations):
        begin=time.perf_counter();used_fields=fields.copy();used_losses=losses.copy()
        profiles=[];errors=[]
        for field in used_fields:
            profile,error=exchange.visit_averaged_mode(field,passive)
            if error>s.max_projection_error:raise ValueError('optical/material quadrature needs refinement')
            profiles.append(profile);errors.append(error)
        model=FieldCoupledLaser(cavity,exchange.area,density.reshape(mesh.nz,-1),
                  np.asarray(profiles),pump,params=p,pump_source=source,
                  mode_labels=tuple(f'tracked vector branch {i}' for i in range(mode_count)))
        model.set_roundtrip_losses(used_losses)
        tick=time.perf_counter()
        optical=model.run(pump_energy_J,repetition_rate_Hz,
            max_cycles=s.optical_max_cycles,min_cycles=s.optical_min_cycles,
            rtol=s.optical_rtol,periodic_tolerance=s.optical_population_tolerance,
            energy_tolerance=s.optical_energy_tolerance,max_period_cycles=s.max_period_cycles,
            initial_fractions=initial_fractions,initial_log_photons=initial_log_photons,
            pump_fwhm_s=pump_duration_s)
        optical_seconds=time.perf_counter()-tick
        if not optical.periodic_converged:
            status='optical pump cycle did not converge; no steady polarized hot-cavity result'
            break
        tick=time.perf_counter()
        heat=sample_cycle_heat(model,optical,pump_energy_J,repetition_rate_Hz,
                              spectroscopy=spectroscopy,rtol=s.optical_rtol/2)
        mean_fractions=time_averaged_populations(model,optical,pump_energy_J,
                              repetition_rate_Hz,rtol=s.optical_rtol/2)
        replay_seconds=time.perf_counter()-tick
        raw_heat=heat.heat_W_m3.reshape(mesh.shape)
        source_residual=None if previous_heat is None else _weighted_relative(raw_heat,previous_heat,mesh.volumes_m3)
        assembly_heat=(raw_heat.copy() if previous_heat is None else
                 (1-s.heat_relaxation)*previous_heat+s.heat_relaxation*raw_heat)
        tick=time.perf_counter()
        temperature,displacement,screens=assembly.solve(assembly_heat)
        assembly_seconds=time.perf_counter()-tick
        g=density.reshape(mesh.nz,-1)*(p.sigma_em_laser_m2*mean_fractions[2]-p.sigma_abs_laser_m2*mean_fractions[3])
        gain_screen=exchange.surface_on_grid(g.sum(axis=0)*model.dz)
        op=VectorRoundTrip(grid,cavity,screens,gain_screen)
        tick=time.perf_counter()
        eigen=solve_tracked_eigenfields(op,used_fields,candidates=s.eigen_candidates,
                         tolerance=s.eigen_tolerance,maxiter=s.eigen_maxiter,tracking=tracking)
        eigen_seconds=time.perf_counter()-tick
        predicted=eigen.fields
        power=float(heat.budget['output_W'])
        powers=np.asarray(heat.budget['output_W_by_mode'],float)
        row={'iteration':iteration+1,'output_W':power,'output_W_by_mode':powers.tolist(),
             'heat_W':heat.budget['heat_W'],'assembly_heat_W':float(np.sum(assembly_heat*mesh.volumes_m3)),
             'heat_residual':source_residual,
             'temperature_change_K':None if previous_temperature is None else float(max(
                  np.max(abs(temperature.disk_temperature_K-previous_temperature[0])),
                  np.max(abs(temperature.plate_temperature_K-previous_temperature[1])))),
             'displacement_change_m':None if previous_u is None else float(max(
                  np.max(abs(displacement.disk_u_m-previous_u[0])),
                  np.max(abs(displacement.plate_u_m-previous_u[1])))),
             'output_relative_change':None if previous_power is None else abs(power-previous_power)/max(abs(power),abs(previous_power),pump_energy_J*repetition_rate_Hz*1e-9),
             'eigen_converged':bool(eigen.converged),
             'eigen_residual':float(np.max(eigen.residuals)) if np.all(np.isfinite(eigen.residuals)) else None,
             'eigen_operator_calls':eigen.operator_calls,
             'optical_periodic_converged':bool(optical.periodic_converged),
             'optical_period_cycles':optical.period_cycles,'optical_cycles':optical.cycles_simulated,
             'projection_error':float(max(errors)),
             'disk_peak_temperature_K':float(temperature.disk_temperature_K.max()),
             'plate_peak_temperature_K':float(temperature.plate_temperature_K.max()),
             'thermal_balance_error_W':temperature.balance_error_W,
             'mechanical_residual':displacement.free_residual_relative,
             'local_energy_error':heat.budget['local_ledger_L1_error_over_incident'],
             'optical_seconds':optical_seconds,'replay_seconds':replay_seconds,
             'assembly_seconds':assembly_seconds,'eigen_seconds':eigen_seconds,
             'tracking':eigen.diagnostics,
             'split_boundary_pairs':eigen.diagnostics.get('split_boundary_pairs',[]),
             'photon_seed_reset_branches':eigen.diagnostics.get('photon_seed_reset_branches',[])}
        if not eigen.converged:
            status=eigen.status
            row.update(field_residual=None,subspace_residual=None,loss_residual=None,
                       all_convergence_gates_passed=False,iteration_seconds=time.perf_counter()-begin)
            history.append(row)
            if progress:progress(row)
            break
        passive_new=VectorRoundTrip(grid,cavity,screens)
        new_losses,_=passive_mode_losses(passive_new,predicted)
        new_profiles=[]
        for f in predicted:
            prof,err=exchange.visit_averaged_mode(f,passive_new)
            if err>s.max_projection_error:raise ValueError('predicted eigenfield projection needs refinement')
            new_profiles.append(prof)
        gain_used=model.log_roundtrip_gain(mean_fractions)
        gain_pred=2*model.dz*(np.asarray(new_profiles)@(g.sum(axis=0)*model.area))
        row.update(convergence_metrics(used_fields,predicted,powers,gain_used,gain_pred,
                 previous_fields=previous_fields,previous_powers=previous_powers,previous_gain=previous_gain))
        row['loss_residual']=float(np.max(abs(new_losses-used_losses)))
        growth=np.array([np.log(max(np.sum(abs(op.propagate(f)[0])**2),1e-300)) for f in used_fields])
        row['field_rate_log_gain_mismatch']=float(np.max(abs(growth-(gain_used-used_losses))))
        row['modal_log_roundtrip_gain']=gain_used.tolist()
        row['passive_log_roundtrip_loss']=used_losses.tolist()
        gates=accepted_closure(row,s)
        row['convergence_gates']=gates
        row['all_convergence_gates_passed']=bool(all(gates.values()))
        row['iteration_seconds']=time.perf_counter()-begin
        history.append(row)
        if progress:progress(row)
        streak=streak+1 if all(gates.values()) else 0
        if iteration+1>=s.minimum_outer_iterations and streak>=s.consecutive_converged:
            converged=True;status='converged polarization-resolved adiabatic closure';break
        if iteration+1==s.max_outer_iterations:break
        previous_heat=assembly_heat.copy()
        previous_temperature=(temperature.disk_temperature_K.copy(),temperature.plate_temperature_K.copy())
        previous_u=(displacement.disk_u_m.copy(),displacement.plate_u_m.copy())
        previous_power=power;previous_fields=used_fields.copy();previous_powers=powers.copy()
        previous_gain=gain_used.copy()
        initial_fractions=heat.final_fractions.copy()
        initial_log_photons=(None if row['photon_seed_reset_branches'] else heat.final_log_photons.copy())
        if initial_log_photons is None:
            previous_fields=previous_powers=previous_gain=None
        # Keep each separately resolved eigenfield, rather than a non-eigenfield
        # linear blend of two split branches. Per-mode phase is already aligned.
        fields=predicted.copy();passive=passive_new;losses=new_losses.copy()
    metadata={'model':'Stage 7W polarization-resolved adiabatic eigenbranch closure',
         'mode_count':mode_count,'pump_energy_J':pump_energy_J,'repetition_rate_Hz':repetition_rate_Hz,
         'pump_duration_s':pump_duration_s,'pump_waist_m':pump_waist_m,
         'pump_source':source.summary(),'cavity':cavity.summary(),
         'settings':asdict(s),'tracking_settings':asdict(tracking),
         'field_update':'full independently resolved eigenfields; field_relaxation setting is not used',
         'optical_sampling':cavity_sampling_diagnostic(grid,cavity),
         'mesh_convergence_verified':False,'validated_for_dataset':False,'dataset_ready':False,
         'physical_solver_base':'88cd102ea4c0d34eeb49a22639721ff4957f062b',
         'limits':['incoherent modal dynamics, no phase-locked or coherent beating prediction',
                   'mode shapes frozen within each fast optical pump cycle',
                   'finite candidate bank is not a global stability proof',
                   'fixed spectroscopy and prescribed bonded interface',
                   'positive active Ho density and thin collapsed-disk optical screens']}
    return CoupledHotCavityResult(converged,status,history,used_fields,predicted,optical,heat,
        mean_fractions,assembly_heat,temperature,displacement,screens,eigen,used_losses,metadata)
