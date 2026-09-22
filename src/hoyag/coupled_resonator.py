"""Stage 7: cycle-resolved, spatial-eigenfield / thermo-mechanical closure.

This is an adiabatic modal closure, not optical-carrier FDTD or roundtrip-by-
roundtrip coherent Maxwell-Bloch dynamics. Vector eigenfields change each slow
iteration; incoherent modal photon populations and local Ho states are solved
through the fast pump cycle. Temperature, interface and plate mechanics are
then recomputed. No prescribed Gaussian waist or parabolic lens is used.
"""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
import math
import numpy as np
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss
from .pump_source import resolve_pump_source
from .numerical_quality import cavity_sampling_diagnostic
from .populations import HoYAGFourLevelParams
from .resonator import ModalThinDiskLaser, lg0_field
from .thermal_resonator import sample_cycle_heat, area_averaged_lg0
from .thermal import ThermalBoundary
from .cooling_plate import (DiskPlateHeatSolver, cooling_plate_mesh,
                            ThermalMaterial, sample_temperature)
from .thermomechanics import (DiskPlateMesh, ElasticMaterial, BondedInterface,
                             solve_disk_plate)
from .stress_optics import (build_hot_disk_screens, CubicElastoOptic,
                           crystal_axes_111)
from .vector_cavity import (VectorRoundTrip, PlaneExchange, normalize_vector,
                           solve_vector_eigenfields, passive_mode_losses,
                           aligned_distance, subspace_distance)


class FieldCoupledLaser(ModalThinDiskLaser):
    """Existing four-manifold/photon equations with eigenfield-dependent loss.

    Additional diffractive/aperture loss is outside the crystal heat ledger.
    Mirror logarithmic output coupling retains the existing mean-field convention.
    Gain is NOT counted again as negative cavity loss.
    """
    def set_roundtrip_losses(self, logarithmic_losses):
        loss = np.asarray(logarithmic_losses, float)
        if loss.shape != (self.nm,) or not np.all(np.isfinite(loss)):
            raise ValueError('one finite passive logarithmic loss per mode required')
        additional = loss-self.cavity.logarithmic_loss
        if np.min(additional) < -1e-9:
            raise ValueError('passive field loss cannot be smaller than specified mirror loss')
        self.additional_decay = np.maximum(additional, 0.)/self.trt

    def rhs(self, time, y):
        out = super().rhs(time, y)
        out[3*self.nc:3*self.nc+self.nm] -= getattr(self, 'additional_decay', 0.)
        return out
    # The extra term is constant in d(log photons)/dt, so the inherited analytic
    # Jacobian remains exact. Changing a temperature does not change spectroscopy.


def time_averaged_populations(model, optical, pump_energy_J, repetition_rate_Hz,
                               *, rtol=1e-6, order=4):
    """Average local populations over a FULL detected pump-period multiple.

    Independent replay from the same optical state used by sample_cycle_heat.
    This is not the post-pulse population incorrectly treated as a cycle average.
    """
    if not optical.periodic_converged or not optical.period_cycles:
        raise ValueError('periodic optical state required for averaged gain')
    state = optical.fractions_before_next_pump.copy()
    logph = optical.log_photon_number.copy()
    period = 1./repetition_rate_Hz
    nodes, weights = leggauss(order)
    avg = np.zeros((4, model.nz, model.ns))
    for _ in range(optical.period_cycles):
        state, _, _, _ = model.pump_kick(state, pump_energy_J)
        y0 = np.r_[state.ravel(),logph,np.zeros(model.nm)]
        atol = np.r_[np.full(3*model.nc,1e-11),np.full(model.nm,rtol*.1),
                      np.full(model.nm,pump_energy_J*1e-11)]
        sol = solve_ivp(model.rhs,(0,period),y0,method='BDF',jac=model.jacobian,
                        rtol=rtol,atol=atol,max_step=period/20,dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        half = .5*np.diff(sol.t); midpoint = .5*(sol.t[:-1]+sol.t[1:])
        time = (midpoint[:,None]+half[:,None]*nodes).ravel()
        weight = (half[:,None]*weights).ravel()
        for start in range(0,len(time),256):
            stop=min(start+256,len(time)); count=stop-start
            y=sol.sol(time[start:stop])[:3*model.nc].reshape(3,model.nz,model.ns,count)
            f=np.concatenate((y,(1-y.sum(axis=0))[None]),axis=0)
            avg += np.sum(f*weight[None,None,None,start:stop],axis=-1)
        state=sol.y[:3*model.nc,-1].reshape(3,model.nz,model.ns)
        logph=sol.y[3*model.nc:3*model.nc+model.nm,-1]
    avg /= optical.period_cycles*period
    if np.min(avg)<-1e-8 or not np.allclose(avg.sum(axis=0),1.,atol=1e-8):
        raise FloatingPointError('invalid cycle-averaged populations')
    return avg


class PlateAssembly:
    """Bridge all Stage 6 thermal, mechanical and Jones components, unchanged."""
    def __init__(self, mesh, grid, configuration):
        self.mesh, self.grid, self.configuration = mesh, grid, deepcopy(configuration)
        cfg=self.configuration;g=cfg['geometry'];th=cfg['thermal'];mech=cfg['mechanical']
        if not np.isclose(mesh.r_edges_m[-1],g['disk_radius_m'],rtol=0,atol=1e-12) or not np.isclose(mesh.z_edges_m[-1],g['disk_thickness_m'],rtol=0,atol=1e-12):
            raise ValueError('crystal and assembly geometry disagree')
        plate=cooling_plate_mesh(mesh,radius_m=g['plate_radius_m'],
                    thickness_m=g['plate_thickness_m'],nz=cfg['numerics']['plate_thermal_nz'])
        self.heat_solver=DiskPlateHeatSolver(mesh,plate,
                    contact_conductance_W_m2K=th['interface_conductance_W_m2K'],
                    coolant=ThermalBoundary(th['coolant_temperature_K'],th['coolant_conductance_W_m2K']),
                    disk_material=ThermalMaterial(**th['disk']),plate_material=ThermalMaterial(**th['plate']))
        self.fem=DiskPlateMesh.make(radius_m=g['disk_radius_m'],disk_thickness_m=g['disk_thickness_m'],
                    plate_radius_m=g['plate_radius_m'],plate_thickness_m=g['plate_thickness_m'],
                    **cfg['numerics']['mechanical'])
        self.disk_material=ElasticMaterial(**mech['disk'])
        self.plate_material=ElasticMaterial(**mech['plate'])
        x,y=grid.mesh; self.xy=np.stack((x,y),axis=-1)

    def solve(self, heat_W_m3):
        cfg=self.configuration;mech=cfg['mechanical'];op=cfg['optics']
        temperature=self.heat_solver.steady(heat_W_m3)
        td=sample_temperature(self.mesh,temperature.disk_temperature_K,self.fem.disk.centers_m)
        tp=sample_temperature(self.heat_solver.plate,temperature.plate_temperature_K,
                     self.fem.plate.centers_m,z_offset_m=self.fem.disk_thickness_m)
        displacement=solve_disk_plate(self.fem,td,tp,disk_material=self.disk_material,
                     plate_material=self.plate_material,interface=BondedInterface(**mech['bond']),
                     support=mech['plate_support'],front_pressure_Pa=mech['front_pressure_Pa'])
        screens=build_hot_disk_screens(self.fem,displacement,self.mesh,
                     temperature.disk_temperature_K,self.xy,
                     index=op['index'],wavelength_m=op['wavelength_m'],dn_dT_K1=op['dn_dT_K1'],
                     reference_temperature_K=op['reference_temperature_K'],material=self.disk_material,
                     coefficients=CubicElastoOptic(**op['cubic_elasto_optic']),
                     crystal_axes=crystal_axes_111(op['crystal_azimuth_rad']))
        return temperature,displacement,screens


@dataclass(frozen=True)
class HotCavitySettings:
    max_outer_iterations: int = 16
    minimum_outer_iterations: int = 3
    consecutive_converged: int = 2
    heat_relaxation: float = .6
    field_relaxation: float = .7
    field_tolerance: float = 2e-3
    heat_relative_tolerance: float = 5e-3
    temperature_tolerance_K: float = .03
    displacement_tolerance_m: float = 2e-10
    power_relative_tolerance: float = 5e-3
    loss_tolerance: float = 1e-4
    eigen_tolerance: float = 5e-7
    eigen_maxiter: int = 600
    eigen_candidates: int = 4
    optical_max_cycles: int = 420
    optical_min_cycles: int = 16
    optical_rtol: float = 2e-6
    optical_population_tolerance: float = 1e-6
    optical_energy_tolerance: float = 3e-4
    max_period_cycles: int = 8
    projection_order: int = 4
    max_projection_error: float = .05

    def __post_init__(self):
        for key,value in vars(self).items():
            if not np.isfinite(value) or value<=0:
                raise ValueError(f'{key} must be finite and positive')
        for key in ('max_outer_iterations','minimum_outer_iterations','consecutive_converged',
                    'eigen_maxiter','eigen_candidates','optical_max_cycles','optical_min_cycles',
                    'max_period_cycles','projection_order'):
            if not isinstance(getattr(self,key),int) or isinstance(getattr(self,key),bool):
                raise ValueError(f'{key} must be integer')
        if self.heat_relaxation>1 or self.field_relaxation>1:
            raise ValueError('relaxation must be <=1')
        if self.optical_min_cycles>self.optical_max_cycles:
            raise ValueError('optical_min_cycles exceeds limit')


@dataclass
class CoupledHotCavityResult:
    converged: bool
    status: str
    history: list
    fields_used: np.ndarray
    fields_predicted: np.ndarray | None
    optical_state: object | None
    cycle_heat: object | None
    mean_fractions: np.ndarray | None
    assembly_heat_W_m3: np.ndarray | None
    temperature: object | None
    displacement: object | None
    screens: object | None
    eigenfields: object | None
    roundtrip_losses_used: np.ndarray
    metadata: dict


def _weighted_relative(a,b,volume):
    numerator=np.sum(abs(a-b)**2*volume)
    denominator=max(np.sum(abs(a)**2*volume),np.sum(abs(b)**2*volume),1e-40)
    return float(np.sqrt(numerator/denominator))


def initial_vector_modes(grid,cavity,charges=(0,)):
    """LG fields are initial guesses only, not constraints on final eigenfields."""
    x,y=grid.mesh;fields=[]
    for i,charge in enumerate(charges):
        f=np.zeros((2,*grid.shape),complex)
        f[i%2]=lg0_field(x,y,cavity.waist_m,charge)
        fields.append(normalize_vector(f))
    return np.asarray(fields)


def run_coupled_hot_cavity(grid,mesh,cavity,pump_energy_J,assembly_configuration,*,
                           repetition_rate_Hz=1e4,pump_duration_s=10e-12,
                           pump_waist_m=.5e-3,params=None,density_m3=None,
                           initial_fields=None,mode_count=1,settings=None,
                           spectroscopy=None,progress=None,pump_source=None,pump_absorption_m2=None):
    """Close all slow feedback channels with undamped residual checks.

    Retains asymmetric 3-D heat, arbitrary complex vector modes, a finite cooler,
    interface forces and both crystal surfaces. Modal intensity is averaged over
    the two disk visits and treated as depth-independent inside this 1 mm disk.
    Population and heat retain all depth cells. Modes are incoherent and frozen
    within a pump cycle; competition beyond mode_count is not resolved.
    """
    settings=settings or HotCavitySettings();p=params or HoYAGFourLevelParams()
    source=resolve_pump_source(p.pump_wavelength_m,pump_duration_s,
                              source=pump_source,absorption_override_m2=pump_absorption_m2)
    if mode_count<1 or mode_count>settings.eigen_candidates:
        raise ValueError('invalid mode count')
    if not np.allclose(np.diff(mesh.z_edges_m),cavity.disk_thickness_m/mesh.nz,rtol=1e-12,atol=0):
        raise ValueError('modal rate solver requires uniform depth cells')
    if not np.isclose(mesh.r_edges_m[-1],cavity.disk_diameter_m/2,rtol=0,atol=1e-12):
        raise ValueError('cavity and thermal mesh radii disagree')
    if not np.isfinite(pump_energy_J) or pump_energy_J<=0:
        raise ValueError('pump energy must be positive')
    density=mesh.field(p.N_total_m3 if density_m3 is None else density_m3,'Ho density')
    if np.any(density<=0):
        raise ValueError('modal cells need positive density; undoped modal voxels not yet supported')
    fields=initial_vector_modes(grid,cavity,tuple(i//2 for i in range(mode_count))) if initial_fields is None else np.asarray(initial_fields,complex).copy()
    if fields.shape != (mode_count,2,*grid.shape):
        raise ValueError('initial field shape mismatch')
    fields=np.asarray([normalize_vector(f) for f in fields])
    if mode_count>1 and np.linalg.matrix_rank(fields.reshape(mode_count,-1))<mode_count:
        raise ValueError('initial fields must be linearly independent')
    exchange=PlaneExchange(grid,mesh,order=settings.projection_order)
    assembly=PlateAssembly(mesh,grid,assembly_configuration)
    pump=area_averaged_lg0(mesh,pump_waist_m)
    history=[];previous_heat=None;previous_temperature=None;previous_u=None
    previous_power=None;initial_fractions=None;initial_log_photons=None
    # This initial assembly also includes any specified mounting preload or
    # mismatch between bath and stress-free temperature, rather than ignoring it.
    temperature,displacement,screens=assembly.solve(np.zeros(mesh.shape))
    passive=VectorRoundTrip(grid,cavity,screens)
    losses,_=passive_mode_losses(passive,fields)
    optical=heat=mean_fractions=eigen=None;predicted=None;assembly_heat=None
    status='outer iteration limit reached';converged=False;good=0
    for iteration in range(settings.max_outer_iterations):
        used_fields=fields.copy();used_losses=losses.copy()
        profiles=[];errors=[]
        for field in used_fields:
            a,e=exchange.visit_averaged_mode(field,passive)
            if e>settings.max_projection_error:
                raise ValueError('field/material projection needs refinement')
            profiles.append(a);errors.append(e)
        model=FieldCoupledLaser(cavity,exchange.area,density.reshape(mesh.nz,-1),
                      np.asarray(profiles),pump,params=p,pump_source=source,
                      mode_labels=tuple(f'vector eigenbranch {i}' for i in range(mode_count)))
        model.set_roundtrip_losses(used_losses)
        optical=model.run(pump_energy_J,repetition_rate_Hz,
                      max_cycles=settings.optical_max_cycles,min_cycles=settings.optical_min_cycles,
                      rtol=settings.optical_rtol,periodic_tolerance=settings.optical_population_tolerance,
                      energy_tolerance=settings.optical_energy_tolerance,max_period_cycles=settings.max_period_cycles,
                      initial_fractions=initial_fractions,initial_log_photons=initial_log_photons,
                      pump_fwhm_s=pump_duration_s)
        if not optical.periodic_converged:
            status='optical pump cycle did not converge; no steady hot-cavity result'
            break
        heat=sample_cycle_heat(model,optical,pump_energy_J,repetition_rate_Hz,
                      spectroscopy=spectroscopy,rtol=settings.optical_rtol/2)
        mean_fractions=time_averaged_populations(model,optical,pump_energy_J,
                      repetition_rate_Hz,rtol=settings.optical_rtol/2)
        raw_heat=heat.heat_W_m3.reshape(mesh.shape)
        # Only the iteration is relaxed. Residuals compare the unrelaxed new
        # source against the previously used source, preventing false convergence.
        source_residual=np.inf if previous_heat is None else _weighted_relative(raw_heat,previous_heat,mesh.volumes_m3)
        assembly_heat=raw_heat.copy() if previous_heat is None else ((1-settings.heat_relaxation)*previous_heat+settings.heat_relaxation*raw_heat)
        temperature,displacement,screens=assembly.solve(assembly_heat)
        gain=density.reshape(mesh.nz,-1)*(p.sigma_em_laser_m2*mean_fractions[2]-p.sigma_abs_laser_m2*mean_fractions[3])
        gain_screen=exchange.surface_on_grid(gain.sum(axis=0)*model.dz)
        operator=VectorRoundTrip(grid,cavity,screens,gain_screen)
        eigen=solve_vector_eigenfields(operator,used_fields,candidates=settings.eigen_candidates,
                      tolerance=settings.eigen_tolerance,maxiter=settings.eigen_maxiter)
        predicted=eigen.fields
        passive_new=VectorRoundTrip(grid,cavity,screens)
        new_losses,_=passive_mode_losses(passive_new,predicted)
        field_residual=max(aligned_distance(a,b) for a,b in zip(used_fields,predicted))
        subspace_residual=subspace_distance(used_fields,predicted)
        temp_residual=np.inf if previous_temperature is None else max(
                np.max(abs(temperature.disk_temperature_K-previous_temperature[0])),
                np.max(abs(temperature.plate_temperature_K-previous_temperature[1])))
        displacement_residual=np.inf if previous_u is None else max(
                np.max(abs(displacement.disk_u_m-previous_u[0])),np.max(abs(displacement.plate_u_m-previous_u[1])))
        power=heat.budget['output_W']
        power_residual=np.inf if previous_power is None else abs(power-previous_power)/max(abs(power),abs(previous_power),pump_energy_J*repetition_rate_Hz*1e-9)
        loss_residual=float(np.max(abs(new_losses-used_losses)))
        # Diagnose the thin-screen/mean-field gain correspondence; not used as a
        # hidden correction to either rates or mirror loss.
        pair_gain=model.log_roundtrip_gain(mean_fractions)
        rayleigh_growth=[]
        for field in used_fields:
            rayleigh_growth.append(math.log(max(np.sum(abs(operator.propagate(field)[0])**2),1e-300)))
        correspondence=float(np.max(abs(np.asarray(rayleigh_growth)-(pair_gain-used_losses))))
        row={'iteration':iteration+1,'output_W':power,'heat_W':heat.budget['heat_W'],
             'assembly_heat_W':float(np.sum(assembly_heat*mesh.volumes_m3)),
             'field_residual':field_residual,'subspace_residual':subspace_residual,
             'heat_residual':None if previous_heat is None else source_residual,
             'temperature_change_K':None if previous_temperature is None else float(temp_residual),
             'displacement_change_m':None if previous_u is None else float(displacement_residual),
             'output_relative_change':None if previous_power is None else float(power_residual),
             'loss_residual':loss_residual,'eigen_residual':float(np.max(eigen.residuals)),
             'eigen_converged':eigen.converged,'eigen_operator_calls':eigen.operator_calls,
             'field_rate_log_gain_mismatch':correspondence,
             'projection_error':float(max(errors)),'optical_cycles':optical.cycles_simulated,
             'optical_period_cycles':optical.period_cycles,
             'disk_peak_temperature_K':float(temperature.disk_temperature_K.max()),
             'plate_peak_temperature_K':float(temperature.plate_temperature_K.max()),
             'thermal_balance_error_W':temperature.balance_error_W,
             'mechanical_residual':displacement.free_residual_relative,
             'local_energy_error':heat.budget['local_ledger_L1_error_over_incident']}
        history.append(row)
        if progress:progress(row)
        if not eigen.converged:
            status=eigen.status;break
        okay=(field_residual<settings.field_tolerance and source_residual<settings.heat_relative_tolerance
              and temp_residual<settings.temperature_tolerance_K
              and displacement_residual<settings.displacement_tolerance_m
              and power_residual<settings.power_relative_tolerance
              and loss_residual<settings.loss_tolerance)
        good=good+1 if okay else 0
        if iteration+1>=settings.minimum_outer_iterations and good>=settings.consecutive_converged:
            converged=True;status='converged adiabatic vector-eigenfield/assembly closure';break
        if iteration+1==settings.max_outer_iterations:break
        previous_heat=assembly_heat.copy()
        previous_temperature=(temperature.disk_temperature_K.copy(),temperature.plate_temperature_K.copy())
        previous_u=(displacement.disk_u_m.copy(),displacement.plate_u_m.copy())
        previous_power=power
        initial_fractions=heat.final_fractions;initial_log_photons=heat.final_log_photons
        # Phase-aligned field damping does not constrain the modal spatial form.
        fields=np.asarray([normalize_vector((1-settings.field_relaxation)*a+settings.field_relaxation*b)
                             for a,b in zip(used_fields,predicted)])
        passive=passive_new;losses,_=passive_mode_losses(passive,fields)
    metadata={'model':'adiabatic cycle-averaged vector-eigenfield/modal-photon closure',
          'base_revision':'a4db1b616d75d208dc5d9963d727164f2b680438',
          'mode_count':mode_count,'pump_energy_J':pump_energy_J,'repetition_rate_Hz':repetition_rate_Hz,
          'pump_source':source.summary(),'optical_sampling':cavity_sampling_diagnostic(grid,cavity),
          'mesh_convergence_verified':False,'validated_for_dataset':False,
          'pump_duration_s':pump_duration_s,'pump_waist_m':pump_waist_m,'cavity':cavity.summary(),
          'settings':vars(settings),'limits':['incoherent modal competition, not coherent mode beating',
          'spatial mode frozen during each fast pump cycle','gain screen uses cycle-averaged populations',
          'thin-disk depth-independent modal intensity and collapsed phase screens',
          'fixed spectroscopy; prescribed bonded-interface conductance and stiffness',
          'positive Ho density in active modal cells','finite number of candidate eigenmodes, no global stability proof']}
    return CoupledHotCavityResult(converged,status,history,used_fields,predicted,optical,heat,
          mean_fractions,assembly_heat,temperature,displacement,screens,eigen,used_losses,metadata)
