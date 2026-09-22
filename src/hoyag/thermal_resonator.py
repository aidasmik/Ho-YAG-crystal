"""Stage 5: energy-resolved oscillator cycles and reduced thermal feedback.

Uses the existing Stage 4R oscillator without altering its population equations.
The cycle source is integrated over actual adaptive time intervals, including
period-multiplied operation; output-coupler losses are not disk bulk heating.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss
from .heat import ManifoldEnergies, heat_rates, stored_energy_density, pulse_heat_from_states
from .populations import I7
from .resonator import radial_laser
from .thermal_optics import (radial_source_to_volume, thermal_opd, fit_thermal_lens,
                             gaussian_hot_cavity)


def measure_heat_cycle(model, initial_fractions, initial_log_photons,
                       pump_energy_J: float, repetition_rate_Hz: float, *,
                       energies: ManifoldEnergies | None = None,
                       rtol: float = 1e-7, quadrature_order: int = 6) -> dict:
    """One pump event + one dark/laser interval, including all energy storage.

    Local fields are J/m^3 integrated over this cycle. Global quantities are J.
    The pump return path uses Stage 4R's sequential short-pulse fluence model.
    Mean branch photon energies are provisional unless supplied explicitly.
    """
    if not np.isfinite(pump_energy_J) or pump_energy_J < 0:
        raise ValueError("pump energy must be finite and nonnegative")
    if not np.isfinite(repetition_rate_Hz) or repetition_rate_Hz <= 0:
        raise ValueError("repetition rate must be positive")
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("invalid integration tolerance")
    if not isinstance(quadrature_order,int) or not 2 <= quadrature_order <= 16:
        raise ValueError("quadrature order must be an integer from 2 to 16")
    en = energies or ManifoldEnergies()
    before = model.full_fractions(np.asarray(initial_fractions,float).copy())
    model.check_fractions(before[:3])
    log0 = np.asarray(initial_log_photons,float)
    if log0.shape != (model.nm,) or np.any(~np.isfinite(log0)):
        raise ValueError("invalid initial photon state")
    period = 1/repetition_rate_Hz
    post, absorbed, escaped_pump, pump_loss = model.pump_kick(before[:3],pump_energy_J)
    n_before = before*model.density[None]
    n_post = model.full_fractions(post)*model.density[None]
    pump_density = model.ep*(n_post[I7]-n_before[I7])
    prompt_heat = pulse_heat_from_states(n_before,n_post,pump_density,energies=en)
    y0 = np.r_[post.ravel(),log0,np.zeros(model.nm)]
    atol = np.r_[np.full(3*model.nc,1e-11),np.full(model.nm,rtol*.1),
                 np.full(model.nm,max(pump_energy_J,1e-12)*1e-11)]
    sol = solve_ivp(model.rhs,(0,period),y0,method="BDF",jac=model.jacobian,
                    rtol=rtol,atol=atol,dense_output=True,max_step=period/20)
    if not sol.success:
        raise RuntimeError(sol.message)
    # Gaussian quadrature on EVERY adaptive interval resolves narrow laser bursts.
    nodes,weights = leggauss(quadrature_order)
    half=np.diff(sol.t)/2;mid=(sol.t[1:]+sol.t[:-1])/2
    ts=(mid[:,None]+half[:,None]*nodes).ravel()
    ws=(half[:,None]*weights).ravel()
    ys=sol.sol(ts)
    three=ys[:3*model.nc].reshape(3,model.nz,model.ns,-1)
    f=np.concatenate((three,(1-three.sum(axis=0))[None]),axis=0)
    n=f*model.density[None,:,:,None]
    photons=np.exp(ys[3*model.nc:3*model.nc+model.nm])
    intensity=2*model.es/model.trt*(model.modes.T@photons)
    rates=heat_rates(n,model.params,signal_intensity_W_m2=intensity[None],
                     energies=en,cavity_spontaneous_fraction=model.nm*model.beta_sp)
    integrated={key:np.sum(value*ws,axis=-1) for key,value in rates.items()}
    final=sol.y[:3*model.nc,-1].reshape(3,model.nz,model.ns)
    final_log=sol.y[3*model.nc:3*model.nc+model.nm,-1].copy()
    model.check_fractions(final)
    n_final=model.full_fractions(final)*model.density[None]
    ionic_delta=stored_energy_density(n_final-n_before,en)
    heat=prompt_heat+integrated['heat']
    def total(a):return float(np.sum(a*model.volume))
    signal=total(integrated['signal_extracted'])
    fluorescence=total(integrated['fluorescence_escape'])
    cavity_sp=total(integrated['spontaneous_to_cavity'])
    delta_u=total(ionic_delta)
    qtotal=total(heat)
    output_J=float(sol.y[-model.nm:,-1].sum())
    optical_loss_J=output_J*model.kloss/model.kout
    photon_delta=model.es*float(np.exp(final_log).sum()-np.exp(log0).sum())
    local_residual=pump_density-integrated['signal_extracted']-integrated['fluorescence_escape']-integrated['spontaneous_to_cavity']-ionic_delta-heat
    return {
        'heat_J_m3':heat,'prompt_heat_J_m3':prompt_heat,
        'pump_absorbed_J_m3':pump_density,'signal_extracted_J_m3':integrated['signal_extracted'],
        'fluorescence_J_m3':integrated['fluorescence_escape'],
        'ionic_energy_change_J_m3':ionic_delta,
        'mean_populations_m3':np.sum(n*ws,axis=-1)/period,
        'final_fractions':final,'final_log_photons':final_log,
        'budget_J':{
            'incident_pump':pump_energy_J,'absorbed_pump':absorbed,
            'escaped_pump':escaped_pump,'pump_optics_loss':pump_loss,
            'heat':qtotal,'prompt_heat':total(prompt_heat),
            'fluorescence_escape':fluorescence,'spontaneous_to_cavity':cavity_sp,
            'stimulated_signal':signal,'output_coupler':output_J,
            'all_cavity_optical_losses':optical_loss_J,
            'ionic_storage_change':delta_u,'photon_storage_change':photon_delta,
            'local_first_law_residual':total(local_residual),
            'whole_system_residual':absorbed-qtotal-fluorescence-optical_loss_J-delta_u-photon_delta,
            'pump_map_residual':absorbed-total(pump_density)},
        'maximum_local_balance_residual_J_m3':float(np.max(np.abs(local_residual))),
        'ode_steps':len(sol.t)-1,'period_s':period,
    }


def cycle_averaged_heat(model, result, *, energies=None, rtol=1e-7,
                        quadrature_order=6, allow_transient=False) -> dict:
    """Average over the detected optical period, not an arbitrary one-pulse state."""
    if not result.periodic_converged and not allow_transient:
        raise ValueError("oscillator is not periodic; explicitly request transient accounting")
    count=result.period_cycles or min(4,result.cycles_simulated)
    frac=result.fractions_before_next_pump.copy();log=result.log_photon_number.copy()
    pump=result.metadata['pump_energy_J'];rep=result.metadata['repetition_rate_Hz']
    cases=[]
    for _ in range(count):
        one=measure_heat_cycle(model,frac,log,pump,rep,energies=energies,
                               rtol=rtol,quadrature_order=quadrature_order)
        cases.append(one);frac=one['final_fractions'];log=one['final_log_photons']
    duration=count/rep
    budget={key:sum(c['budget_J'][key] for c in cases)/duration for key in cases[0]['budget_J']}
    q=sum(c['heat_J_m3'] for c in cases)/duration
    return {'source_W_m3':q,'budget_W':budget,'cycles_averaged':count,
            'periodic':bool(result.periodic_converged),
            'mean_populations_m3':sum(c['mean_populations_m3'] for c in cases)/count,
            'final_fractions':frac,'final_log_photons':log,
            'closure_relative':abs(budget['whole_system_residual'])/max(abs(budget['absorbed_pump']),1e-30)}


def thermal_feedback(cavity, solver, pump_energy_J: float, *,
                     repetition_rate_Hz=1e4, pump_waist_m=.5e-3,
                     radial_points=36, z_slices=4, max_outer=12, mixing=.5,
                     tolerance=.005, dn_dT_K1=9.1e-6, energies=None,
                     oscillator_options=None, progress=None) -> dict:
    """Under-relaxed, axisymmetric near-Gaussian thermo-optic feedback.

    Each outer step recomputes the periodic oscillator, signed cycle heat, 3-D
    conduction, and the quadratic hot-cavity mode. The remaining nonquadratic
    OPD is measured but not fed to this reduced modal model. Use the full OPD
    with thermal_cavity_roundtrip for coherent field studies. No sigma(T),
    thermoelastic deformation, or simultaneous counterpropagating ps pump.
    """
    if not isinstance(max_outer,int) or max_outer<1 or not 0<mixing<=1 or not 0<tolerance<1:
        raise ValueError("invalid outer iteration controls")
    if not np.isclose(cavity.disk_thickness_m,solver.thickness_m,rtol=1e-12,atol=0) or not np.isclose(cavity.disk_diameter_m,solver.diameter_m,rtol=1e-12,atol=0):
        raise ValueError("thermal and optical disk geometries must agree")
    model,r=radial_laser(cavity,radial_points=radial_points,z_slices=z_slices,
                        pump_waist_m=pump_waist_m)
    options={'max_cycles':450,'min_cycles':30,'periodic_tolerance':1e-5,
             'energy_tolerance':.002,'rtol':2e-6,'max_period_cycles':8}
    options.update(oscillator_options or {})
    if 'initial_fractions' in options or 'initial_log_photons' in options:
        raise ValueError("outer loop manages its own same-phase warm starts")
    frac=log=None;power=0.;history=[];previous_temperature=None;previous_output=None
    converged=False;reason='outer iteration limit reached'
    for outer in range(max_outer):
        mode=gaussian_hot_cavity(cavity,power)
        if not mode['stable']:
            raise RuntimeError("thermal cavity is unstable; reduced Gaussian model cannot continue")
        w=mode['waist_m']
        profile=2/(np.pi*w*w)*np.exp(-2*r*r/(w*w))
        if not np.isclose(profile@model.area,1,rtol=2e-5,atol=0):
            raise ValueError("hot mode is underresolved/clipped; increase radial quadrature")
        model.modes=profile[None]
        oscillator=model.run(pump_energy_J,repetition_rate_Hz,initial_fractions=frac,
                             initial_log_photons=log,**options)
        if not oscillator.periodic_converged:
            raise RuntimeError("inner oscillator failed periodic convergence; increase cycles or inspect dynamics")
        heat=cycle_averaged_heat(model,oscillator,energies=energies)
        if heat['closure_relative']>5e-4:
            raise RuntimeError("optical/thermal energy accounting failed; refine ODE/quadrature")
        source,mapping=radial_source_to_volume(r,heat['source_W_m3'],model.area,model.dz,solver)
        temperature=solver.steady(source)
        opd=thermal_opd(temperature.temperature_K,solver,
                         reference_K=solver.boundary.sink_temperature_K,dn_dT_K1=dn_dT_K1)
        x,y=solver.grid.mesh
        fit=fit_thermal_lens(opd,solver.grid,fit_radius_m=2*w,weights=np.exp(-2*(x*x+y*y)/(w*w)))
        target=fit.mean_power_m1
        target_mode=gaussian_hot_cavity(cavity,target)
        output=heat['budget_W']['output_coupler']
        dp=abs(target-power)/max(abs(target),abs(power),1e-3)
        dt=np.inf if previous_temperature is None else float(np.max(abs(temperature.temperature_K-previous_temperature)))/max(float(np.max(temperature.temperature_K)-solver.boundary.sink_temperature_K),.01)
        do=np.inf if previous_output is None else abs(output-previous_output)/max(abs(output),abs(previous_output),1e-8)
        row={'iteration':outer,'waist_m':w,'power_used_m1':power,'power_target_m1':target,
             'max_temperature_K':float(np.max(temperature.temperature_K)),
             'heat_W':heat['budget_W']['heat'],'output_W':output,
             'nonquadratic_opd_rms_m':fit.residual_rms_m,'lens_residual':dp,
             'temperature_residual':dt,'output_residual':do,
             'inner_cycles':oscillator.cycles_simulated,'optical_period_cycles':oscillator.period_cycles,
             'target_stable':bool(target_mode['stable'])}
        history.append(row)
        if progress is not None:progress(row)
        if not target_mode['stable']:
            reason='thermal lens reached cavity stability boundary';break
        if outer>0 and max(dp,dt,do)<tolerance:
            converged=True;reason='thermal and optical outer residuals met';break
        previous_temperature=temperature.temperature_K.copy();previous_output=output
        frac=heat['final_fractions'];log=heat['final_log_photons']
        power=(1-mixing)*power+mixing*target
    return {'converged':converged,'reason':reason,'history':history,'heat':heat,
            'source_W_m3':source,'temperature':temperature,'single_pass_opd_m':opd,
            'lens_fit':fit,'mapping':mapping,'oscillator':oscillator,'model':model,
            'radius_m':r,'energies_provenance':(energies or ManifoldEnergies()).provenance}
