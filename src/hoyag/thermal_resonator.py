"""Stage 5: cycle-resolved heat -> circular-disk diffusion -> hot resonator.

The existing Stage 4R photon/population model is reused, not copied into the
thermal code. The heat source is integrated over its actual optical cycle.
Optional feedback updates a prescribed Gaussian mode via a fitted parabolic
thermal lens. Full nonparabolic phase screens are supported separately by
thermal_optics; this is NOT a fully self-consistent 3-D Fox-Li oscillator.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc
from numpy.polynomial.legendre import leggauss
from .populations import HoYAGFourLevelParams, I7
from .resonator import ThinDiskResonator, ModalThinDiskLaser, OscillatorResult
from .heat import HeatSpectroscopy, heat_budget, ion_energy_density, pump_kick_budget
from .thermal import DiskThermalMesh, DiskHeatSolver, DiskCooling, ThermalSolution
from .thermal_optics import thermal_opd, fit_radial_thermal_lens, hot_gaussian_mode, LensFit


def area_averaged_lg0(mesh: DiskThermalMesh, waist_m: float, charge=0):
    """Exact annular probability / annular area, not pointwise sampling."""
    if not np.isfinite(waist_m) or waist_m<=0: raise ValueError('waist must be positive')
    if not isinstance(charge,(int,np.integer)): raise ValueError('charge must be an integer')
    cumulative=gammainc(abs(charge)+1,2*(mesh.r_edges_m/waist_m)**2)
    probability=np.diff(cumulative)
    intensity=probability/(np.pi*np.diff(mesh.r_edges_m**2))
    return np.broadcast_to(intensity[:,None],(mesh.nr,mesh.nphi)).ravel().copy()


def modal_laser_on_thermal_mesh(mesh: DiskThermalMesh, cavity: ThinDiskResonator, *,
                                waist_m=None, charges=(0,), pump_waist_m=.5e-3,
                                params=None, density_m3=None, pump_absorption_m2=1.223454786e-24,
                                spontaneous_fraction_per_mode=1e-8):
    """Use identical optical and thermal control volumes: no resampling of Q."""
    if not np.isclose(mesh.r_edges_m[-1],cavity.disk_diameter_m/2,rtol=1e-12,atol=0) or not np.isclose(mesh.z_edges_m[-1],cavity.disk_thickness_m,rtol=1e-12,atol=0):
        raise ValueError('cavity and thermal disk dimensions must agree')
    if not np.allclose(np.diff(mesh.z_edges_m),cavity.disk_thickness_m/mesh.nz,rtol=1e-12,atol=0):
        raise ValueError('Stage 4R modal model requires equal depth slices')
    if not charges or len(set(charges))!=len(charges): raise ValueError('charges must be nonempty and unique')
    p=params or HoYAGFourLevelParams();w=cavity.waist_m if waist_m is None else waist_m
    modes=np.array([area_averaged_lg0(mesh,w,l) for l in charges])
    pump=area_averaged_lg0(mesh,pump_waist_m)
    density=mesh.field(p.N_total_m3 if density_m3 is None else density_m3,'Ho density')
    return ModalThinDiskLaser(cavity,mesh.face_areas_m2.ravel(),density.reshape(mesh.nz,-1),
        modes,pump,params=p,pump_absorption_m2=pump_absorption_m2,
        spontaneous_fraction_per_mode=spontaneous_fraction_per_mode,
        mode_labels=tuple(f'LG(0,{l})' for l in charges))


@dataclass
class CycleHeatResult:
    heat_W_m3: np.ndarray
    pump_absorbed_W_m3: np.ndarray
    stimulated_signal_W_m3: np.ndarray
    fluorescence_W_m3: np.ndarray
    storage_change_W_m3: np.ndarray
    final_fractions: np.ndarray
    final_log_photons: np.ndarray
    budget: dict


def _cycle_quadrature(sol,model,spectroscopy,order):
    """Gaussian quadrature on EVERY adaptive BDF interval, using dense output."""
    nodes,weights=leggauss(order)
    half=.5*np.diff(sol.t);middle=.5*(sol.t[:-1]+sol.t[1:])
    times=(middle[:,None]+half[:,None]*nodes).ravel()
    wt=(half[:,None]*weights).ravel()
    totals=np.zeros((3,model.nz,model.ns))
    for start in range(0,len(times),512):
        stop=min(start+512,len(times)); y=sol.sol(times[start:stop]);count=stop-start
        three=y[:3*model.nc].reshape(3,model.nz,model.ns,count)
        frac=np.concatenate((three,(1-three.sum(axis=0))[None]),axis=0)
        photons=np.exp(y[3*model.nc:3*model.nc+model.nm])
        intensity=2*model.es/model.trt*(model.modes.T@photons)
        b=heat_budget(frac*model.density[None,:,:,None],model.params,spectroscopy,
                      signal_intensity_W_m2=intensity[None])
        for i,a in enumerate((b.heat_W_m3,b.signal_net_W_m3,b.fluorescence_W_m3)):
            totals[i]+=np.sum(a*wt[None,None,start:stop],axis=-1)
    return totals


def sample_cycle_heat(model: ModalThinDiskLaser, optical_state: OscillatorResult,
                      pump_energy_J, repetition_rate_Hz=1e4, *, spectroscopy=None,
                      cycles=None, rtol=5e-7, quadrature_order=6, allow_transient=False):
    """Replay a full detected period with a first-law ledger at every voxel.

    dU is computed from actual endpoints, not assumed zero at startup or at an
    imperfect periodic solution. Radiative power is total spontaneous emission
    from ions, including the tiny cavity-seeded fraction. Mirror losses remain
    optical losses unless separate measured coating heat flux is supplied.
    """
    if not np.isfinite(pump_energy_J) or pump_energy_J<=0 or not np.isfinite(repetition_rate_Hz) or repetition_rate_Hz<=0:
        raise ValueError('pump energy and repetition rate must be finite and positive')
    if not np.isfinite(rtol) or rtol<=0 or not isinstance(quadrature_order,int) or quadrature_order<2:
        raise ValueError('invalid quadrature or ODE tolerance')
    if not optical_state.periodic_converged and not allow_transient:
        raise ValueError('optical state is not periodic; explicitly allow finite-window transient heating')
    ncycles=optical_state.period_cycles if cycles is None else cycles
    if not isinstance(ncycles,(int,np.integer)) or ncycles<1: raise ValueError('positive cycle count required')
    spec=spectroscopy or HeatSpectroscopy();period=1/repetition_rate_Hz
    state=np.asarray(optical_state.fractions_before_next_pump,float).copy()
    logph=np.asarray(optical_state.log_photon_number,float).copy()
    model.check_fractions(state)
    initial=model.full_fractions(state)*model.density[None]
    total=np.zeros((4,model.nz,model.ns)) # pump, heat, signal, radiation: J/m^3
    quadrature_error=np.zeros_like(total[0]);out_J=np.zeros(model.nm);escaped=0.;loss=0.
    for _ in range(ncycles):
        before=model.full_fractions(state)*model.density[None]
        state,absorbed,escape,mirror_loss=model.pump_kick(state,pump_energy_J)
        after=model.full_fractions(state)*model.density[None]
        kick=pump_kick_budget(before,after,model.ep,spec)
        if not np.isclose(np.sum(kick['pump_J_m3']*model.volume),absorbed,rtol=1e-9,atol=pump_energy_J*1e-12):
            raise FloatingPointError('pump kick and local absorption energy disagree')
        total[0]+=kick['pump_J_m3'];total[1]+=kick['heat_J_m3']
        escaped+=escape;loss+=mirror_loss
        y0=np.r_[state.ravel(),logph,np.zeros(model.nm)]
        atol=np.r_[np.full(3*model.nc,1e-11),np.full(model.nm,rtol*.1),np.full(model.nm,pump_energy_J*1e-11)]
        sol=solve_ivp(model.rhs,(0,period),y0,method='BDF',jac=model.jacobian,
                      rtol=rtol,atol=atol,max_step=period/20,dense_output=True)
        if not sol.success: raise RuntimeError(sol.message)
        integral=_cycle_quadrature(sol,model,spec,quadrature_order)
        comparison=_cycle_quadrature(sol,model,spec,max(2,quadrature_order//2))
        total[1:]+=integral;quadrature_error+=np.max(abs(integral-comparison),axis=0)
        state=sol.y[:3*model.nc,-1].reshape(3,model.nz,model.ns)
        logph=sol.y[3*model.nc:3*model.nc+model.nm,-1]
        model.check_fractions(state);out_J+=sol.y[-model.nm:,-1]
    elapsed=ncycles*period
    final=model.full_fractions(state)*model.density[None]
    stored=ion_energy_density(final,spec)-ion_energy_density(initial,spec)
    closure=total[0]-total[1]-total[2]-total[3]-stored
    sums=np.sum(total*model.volume[None],axis=(1,2))/elapsed
    error=np.sum(abs(closure)*model.volume)/(ncycles*pump_energy_J)
    budget={
        'pump_incident_W':pump_energy_J*repetition_rate_Hz,'pump_absorbed_W':float(sums[0]),
        'heat_W':float(sums[1]),'stimulated_signal_W':float(sums[2]),'fluorescence_W':float(sums[3]),
        'ion_storage_change_W':float(np.sum(stored*model.volume)/elapsed),
        'output_W':float(out_J.sum()/elapsed),'output_W_by_mode':(out_J/elapsed).tolist(),
        'pump_escaped_W':float(escaped/elapsed),'pump_mirror_loss_W':float(loss/elapsed),
        'local_ledger_L1_error_over_incident':float(error),
        'quadrature_L1_difference_over_incident':float(np.sum(quadrature_error*model.volume)/(ncycles*pump_energy_J)),
        'cycles_averaged':int(ncycles),'interval_s':elapsed,
        'population_cycle_drift':float(np.max(abs(final-initial)/model.density[None])),
        'heat_spectroscopy_provenance':spec.provenance,
        'fluorescence_transport':'all photons leave ion subsystem; no radiation reabsorption',
        'coating_heat_included':False,
    }
    if error>1e-4: raise FloatingPointError(f'cycle energy ledger not converged: {error:g} of incident power')
    return CycleHeatResult(total[1]/elapsed,total[0]/elapsed,total[2]/elapsed,total[3]/elapsed,
                           stored/elapsed,state,logph,budget)


@dataclass
class ThermalResonatorResult:
    optical: OscillatorResult
    heat: CycleHeatResult
    thermal: ThermalSolution
    single_pass_opd_m: np.ndarray
    lens: LensFit
    iterations: list[dict]
    feedback_converged: bool
    mode_radius_used_m: float
    mode_radius_predicted_m: float | None
    status: str


def run_thermal_resonator(mesh: DiskThermalMesh, cavity: ThinDiskResonator, pump_energy_J,
                          repetition_rate_Hz=1e4, *, cooling=None, params=None,
                          spectroscopy=None, conductivity_W_mK=14., dn_dT_K1=9.1e-6,
                          mass_density_kg_m3=4560., heat_capacity_J_kgK=680.,
                          reference_temperature_K=293.15, pump_waist_m=.5e-3, pump_duration_fwhm_s=10e-12,
                          max_feedback_iterations=1, mode_tolerance=1e-3,
                          heat_tolerance=3e-3, relaxation=.6, optical_max_cycles=420,
                          initial_fractions=None, initial_log_photons=None, progress=None):
    """Optical-period / thermal-steady-state iteration for the selected Gaussian.

    max_feedback_iterations=1 is one-way heating of the cold-mode solution.
    With >1, the fitted thermal lens updates Gaussian radius and therefore
    pump/mode overlap, spatial saturation, heat and output power. All optical
    solves must converge; thermal feedback has its own independent status.

    This axisymmetric parabolic feedback is approximate. No temperature-dependent
    cross sections, aberration-induced diffraction loss, or stress are invented.
    The separate 3-D thermal solver and full FFT phase operator accept asymmetry.
    """
    if mesh.nphi!=1: raise ValueError('Gaussian radius feedback currently requires axisymmetry')
    if not isinstance(max_feedback_iterations,int) or max_feedback_iterations<1 or not 0<relaxation<=1 or mode_tolerance<=0 or heat_tolerance<=0:
        raise ValueError('invalid feedback settings')
    w=cavity.waist_m;state=initial_fractions;logph=initial_log_photons
    solver=DiskHeatSolver(mesh,conductivity_W_mK,cooling,
                          density_kg_m3=mass_density_kg_m3,heat_capacity_J_kgK=heat_capacity_J_kgK)
    history=[];previous_heat=None;converged=False;predicted=None;status='one-way cold-mode heating'
    for iteration in range(max_feedback_iterations):
        model=modal_laser_on_thermal_mesh(mesh,cavity,waist_m=w,params=params,pump_waist_m=pump_waist_m)
        optical=model.run(pump_energy_J,repetition_rate_Hz,max_cycles=optical_max_cycles,
            min_cycles=20,rtol=5e-7,max_period_cycles=8,periodic_tolerance=1e-7,energy_tolerance=1e-4,
            pump_fwhm_s=pump_duration_fwhm_s,initial_fractions=state,initial_log_photons=logph)
        if not optical.periodic_converged:
            raise RuntimeError('optical oscillator did not reach a periodic state; no steady thermal claim is made')
        heat=sample_cycle_heat(model,optical,pump_energy_J,repetition_rate_Hz,spectroscopy=spectroscopy)
        thermal=solver.steady(heat.heat_W_m3.reshape(mesh.shape))
        opd=thermal_opd(mesh,thermal.temperature_K,reference_temperature_K=reference_temperature_K,dn_dT_K1=dn_dT_K1)
        weight=model.modes[0]*model.area
        lens=fit_radial_thermal_lens(mesh.r_m,opd[:,0],weight,fit_radius_m=1.5*w)
        try:
            hot=hot_gaussian_mode(cavity,lens.single_pass_power_m1)
            predicted=hot.disk_radius_m;residual=abs(predicted-w)/w
        except ValueError:
            predicted=None;residual=np.inf;status='parabolic hot cavity is unstable'
        heat_change=np.inf if previous_heat is None else abs(heat.budget['heat_W']-previous_heat)/max(abs(heat.budget['heat_W']),1e-15)
        row={'iteration':iteration+1,'mode_radius_used_m':w,'mode_radius_predicted_m':predicted,
             'mode_relative_residual':float(residual) if np.isfinite(residual) else None,'heat_relative_change':None if previous_heat is None else float(heat_change),
             'heat_W':heat.budget['heat_W'],'output_W':heat.budget['output_W'],
             'peak_cell_temperature_K':float(thermal.temperature_K.max()),
             'single_pass_lens_power_m1':lens.single_pass_power_m1,
             'opd_fit_rms_m':lens.weighted_rms_residual_m,'optical_cycles':optical.cycles_simulated,
             'optical_period_pump_cycles':optical.period_cycles}
        history.append(row)
        if progress is not None: progress(row)
        if predicted is None: break
        if max_feedback_iterations>1 and residual<mode_tolerance and heat_change<heat_tolerance:
            converged=True;status='converged axisymmetric parabolic-mode thermal feedback';break
        if iteration+1==max_feedback_iterations:
            if max_feedback_iterations>1: status='thermal feedback iteration limit reached'
            break
        state=heat.final_fractions;logph=heat.final_log_photons
        previous_heat=heat.budget['heat_W'];w=(1-relaxation)*w+relaxation*predicted
    return ThermalResonatorResult(optical,heat,thermal,opd,lens,history,converged,w,predicted,status)
