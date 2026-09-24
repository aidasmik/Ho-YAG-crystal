"""Structured-light probes against archived or recalculated Ho:YAG backgrounds.

``weak_probe`` reuses archived Stage 7W population fractions. The
``full_seeded_modal`` retains its historical API name but calculates a fixed-mode
oscillator background: periodic four-manifold populations, heat, bonded
thermoelastic assembly, and a photoelastic Jones screen. Each displayed input
then makes a separate, undepleted one-pass probe of that common background.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import j0
from scipy.interpolate import RegularGridInterpolator

from .inhomogeneity import HoDensityField
from .population_state import validate_populations
from .propagation import (Grid2D, angular_spectrum_propagate, gaussian_beam,
                          hermite_gaussian, laguerre_gaussian, optical_power)
from .signal import propagate_structured_signal_small_signal
from .resonator import ModalThinDiskLaser, ThinDiskResonator
from .thermal_resonator import area_averaged_lg0, sample_cycle_heat
from .coupled_resonator import PlateAssembly
from .stress_optics import apply_jones, geometric_transmission_opd
from .thermal import DiskThermalMesh
from .populations import HoYAGFourLevelParams
from .pump_source import resolve_pump_source
from .seeded_amplifier import apply_phase_modulator
from .snapshots import ScientificSnapshot
from .thermal_optics import polar_to_cartesian
from .seeded_periodic import PeriodicAmplifierSettings, solve_periodic_seeded_amplifier
from .refractive_response import HoIndexResponse


@dataclass(frozen=True)
class GallerySettings:
    mean_ho_density_m3: float = 1.52e26
    density_seed: int = 17
    cluster_count: int = 24
    cluster_contrast: float = 0.27
    cluster_radius_min_m: float = 0.20e-3
    cluster_radius_max_m: float = 1.25e-3
    waist_m: float = 0.408e-3
    input_power_W: float = 1.0
    post_disk_distance_m: float = 0.25
    phase_mask_name: str = 'none'
    phase_strength_rad: float = np.pi
    solver_mode: str = 'weak_probe'
    selected_beam: str = 'Gaussian TEM00'
    seed_energy_J: float = 10e-9
    seed_fwhm_s: float = 10e-12
    signal_traversals: int = 10
    relay_distance_m: float = 0.
    cavity_ejection_efficiency: float = 1.
    cpu_workers: int = 4
    dn_dHo_m3: float | None = None
    dn_dExcited_m3: float | None = None
    index_provenance: str = ''

    def __post_init__(self):
        values=(self.mean_ho_density_m3,self.cluster_contrast,
                self.cluster_radius_min_m,self.cluster_radius_max_m,self.waist_m,
                self.input_power_W,self.post_disk_distance_m,self.phase_strength_rad)
        if not all(np.isfinite(value) for value in values):
            raise ValueError('gallery settings must be finite')
        if min(self.mean_ho_density_m3,self.cluster_radius_min_m,self.waist_m,self.input_power_W) <= 0:
            raise ValueError('density, widths, and input power must be positive')
        if (self.post_disk_distance_m < 0 or self.cluster_contrast < 0 or
            self.cluster_radius_max_m < self.cluster_radius_min_m):
            raise ValueError('invalid cluster contrast, radii, or post-disk distance')
        if (isinstance(self.density_seed,bool) or not isinstance(self.density_seed,int) or
            isinstance(self.cluster_count,bool) or not isinstance(self.cluster_count,int) or
            self.cluster_count < 2):
            raise ValueError('density seed must be an integer and cluster count >=2')
        if self.phase_mask_name not in PHASE_MASKS:
            raise ValueError(f'phase mask must be one of {PHASE_MASKS}')
        if self.solver_mode not in SOLVER_MODES:
            raise ValueError(f'solver mode must be one of {SOLVER_MODES}')
        if self.selected_beam not in BEAM_NAMES:
            raise ValueError('unknown selected beam')
        if (self.seed_energy_J<=0 or self.seed_fwhm_s<=0 or self.signal_traversals<1 or
            self.relay_distance_m<0 or not 0<self.cavity_ejection_efficiency<=1):
            raise ValueError('invalid seeded-amplifier settings')
        if isinstance(self.cpu_workers,bool) or not isinstance(self.cpu_workers,int) or not 1<=self.cpu_workers<=16:
            raise ValueError('cpu_workers must be an integer from 1 to 16')
        HoIndexResponse(self.dn_dHo_m3,self.dn_dExcited_m3,self.index_provenance)


PHASE_MASKS=('none','vortex+1','vortex-1','vortex+2',
             'defocus','astigmatic','axicon')
SOLVER_MODES=('weak_probe','full_seeded_modal','periodic_seeded_amplifier')
BEAM_NAMES=('Gaussian TEM00','Helical LG(0,+1)','Double helix LG(0,+2)',
            'Hermite-Gaussian HG(1,1)','Needle Bessel-Gaussian','Flattop super-Gaussian')


def nonuniform_density(grid: Grid2D, z_edges_m, disk_radius_m: float,
                       settings: GallerySettings = GallerySettings()) -> HoDensityField:
    """Seeded 3-D Ho-rich and Ho-poor Gaussian clusters at mixed length scales."""
    z_edges = np.asarray(z_edges_m, float)
    if z_edges.ndim != 1 or len(z_edges) < 2 or not np.all(np.diff(z_edges) > 0):
        raise ValueError('z_edges_m must increase')
    if disk_radius_m <= 0 or settings.mean_ho_density_m3 <= 0:
        raise ValueError('invalid disk/density settings')
    x, y = grid.mesh
    mask = np.hypot(x, y) <= disk_radius_m
    if not np.any(mask):
        raise ValueError('grid misses the disk')
    length = float(z_edges[-1] - z_edges[0])
    z = .5 * (z_edges[:-1] + z_edges[1:]) - z_edges[0]
    rng=np.random.default_rng(settings.density_seed)
    mixture=np.zeros((len(z),*grid.shape),float)
    signs=np.r_[1.,-1.,rng.choice((-1.,1.),settings.cluster_count-2)]
    rng.shuffle(signs)
    for index in range(settings.cluster_count):
        # Several seeded defects cross the pumped central region; others span
        # the wider disk. Both location sets are random and reproducible.
        central=index<max(2,settings.cluster_count//4)
        reach=min(1.5e-3,.6*disk_radius_m) if central else .9*disk_radius_m
        radius=reach*np.sqrt(rng.random())
        theta=rng.uniform(0,2*np.pi)
        cx,cy=radius*np.cos(theta),radius*np.sin(theta)
        cz=rng.uniform(0,length)
        width=np.exp(rng.uniform(np.log(settings.cluster_radius_min_m),
                                  np.log(settings.cluster_radius_max_m)))
        depth_width=rng.uniform(.16,.75)*length
        strength=signs[index]*rng.uniform(.6,1.4)
        transverse=np.exp(-((x-cx)**2+(y-cy)**2)/(2*width**2))
        axial=np.exp(-(z-cz)**2/(2*depth_width**2))
        mixture+=strength*axial[:,None,None]*transverse[None]
    active=mixture[:,mask]
    deviation=float(active.std())
    if deviation<=0:
        raise ValueError('random cluster field has no variation')
    standardized=(mixture-float(active.mean()))/deviation
    # A bounded log-density perturbation keeps every active voxel positive.
    multiplier=np.exp(np.clip(settings.cluster_contrast*standardized,-.55,.55))
    # Preserve the requested arithmetic mean over active disk voxels.
    active_mean = float(multiplier[:,mask].mean())
    values = np.where(mask[None],settings.mean_ho_density_m3*multiplier/active_mean,0.)
    return HoDensityField(values,length)


def phase_pattern(grid: Grid2D, settings: GallerySettings = GallerySettings()) -> np.ndarray:
    """Requested ideal SLM phase; the vortex names specify integer winding."""
    x,y=grid.mesh
    name=settings.phase_mask_name
    if name=='none':return np.zeros(grid.shape)
    if name.startswith('vortex'):
        charge=int(name.removeprefix('vortex'))
        return charge*np.arctan2(y,x)
    xn=x/settings.waist_m;yn=y/settings.waist_m
    if name=='defocus':return settings.phase_strength_rad*(xn*xn+yn*yn)
    if name=='astigmatic':return settings.phase_strength_rad*(xn*xn-yn*yn)
    if name=='axicon':return settings.phase_strength_rad*np.hypot(xn,yn)
    raise ValueError(f'unsupported phase mask: {name}')


def frozen_populations_on_grid(snapshot: ScientificSnapshot, grid: Grid2D,
                               density: HoDensityField) -> np.ndarray:
    """Interpolate archived cylindrical fractions; rescale to local Ho density."""
    a = snapshot.arrays
    required = ('mean_fractions','raw_heat_W_m3','r_edges_m','z_edges_m')
    if any(key not in a for key in required):
        raise ValueError('saved reference lacks population/mesh arrays')
    heat = a['raw_heat_W_m3']
    mesh = DiskThermalMesh(a['r_edges_m'],a['z_edges_m'],heat.shape[-1])
    if density.nz != mesh.nz or not np.isclose(density.length_m,mesh.z_edges_m[-1]-mesh.z_edges_m[0]):
        raise ValueError('density and frozen population depth meshes disagree')
    fractions = np.asarray(a['mean_fractions'],float).reshape(4,*mesh.shape)
    mapped = np.empty((4,density.nz,*grid.shape),float)
    for level in range(4):
        for iz in range(density.nz):
            mapped[level,iz] = polar_to_cartesian(mesh,fractions[level,iz],grid,outside=0.)
    mapped = np.clip(mapped,0,None)
    total = mapped.sum(axis=0)
    active = density.values_m3 > 0
    if np.any(total[active] <= 0):
        raise ValueError('population interpolation lost active cells')
    np.divide(mapped,total[None],out=mapped,where=total[None]>0)
    populations = mapped*density.values_m3[None]
    return validate_populations(populations,density.values_m3)


def input_modes(grid: Grid2D, settings: GallerySettings = GallerySettings()) -> dict[str,np.ndarray]:
    """Return Gaussian, helical, needle, and flattop structured inputs."""
    w=settings.waist_m
    x,y=grid.mesh
    radius=np.hypot(x,y)
    # A finite-aperture Bessel-Gaussian gives a narrow central needle while
    # retaining a finite power integral on the computational grid.
    needle_zero_radius=0.32*w
    needle_kr=2.4048255577/needle_zero_radius
    needle=j0(needle_kr*radius)*np.exp(-(radius/(2.4*w))**2)
    # The eighth-order super-Gaussian has a broad, nearly uniform central
    # plateau and a smooth numerical edge.
    flattop=np.exp(-(radius/w)**8)
    modes = {'Gaussian TEM00':gaussian_beam(grid,w),
             'Helical LG(0,+1)':laguerre_gaussian(grid,0,1,w),
             'Double helix LG(0,+2)':laguerre_gaussian(grid,0,2,w),
             'Hermite-Gaussian HG(1,1)':hermite_gaussian(grid,1,1,w),
             'Needle Bessel-Gaussian':needle.astype(np.complex128),
             'Flattop super-Gaussian':flattop.astype(np.complex128)}
    return {name:field*np.sqrt(settings.input_power_W/optical_power(field,grid))
            for name,field in modes.items()}


def _cartesian_to_polar(values, grid: Grid2D, mesh: DiskThermalMesh) -> np.ndarray:
    """Interpolate a Cartesian z-stack onto the thermal solver's (z,r,phi) mesh."""
    stack=np.asarray(values,float)
    if stack.shape!=(mesh.nz,*grid.shape):
        raise ValueError('Cartesian stack and thermal depth mesh disagree')
    z=.5*(mesh.z_edges_m[:-1]+mesh.z_edges_m[1:])
    zz,rr,pp=np.meshgrid(mesh.z_m,mesh.r_m,mesh.phi_rad,indexing='ij')
    points=np.column_stack((zz.ravel(),(rr*np.sin(pp)).ravel(),(rr*np.cos(pp)).ravel()))
    interp=RegularGridInterpolator((z,grid.y,grid.x),stack,bounds_error=False,fill_value=0.)
    return interp(points).reshape(mesh.shape)


def _plane_to_polar(values, grid: Grid2D, mesh: DiskThermalMesh) -> np.ndarray:
    rr,pp=np.meshgrid(mesh.r_m,mesh.phi_rad,indexing='ij')
    points=np.column_stack(((rr*np.sin(pp)).ravel(),(rr*np.cos(pp)).ravel()))
    interp=RegularGridInterpolator((grid.y,grid.x),np.asarray(values,float),
                                   bounds_error=False,fill_value=0.)
    return interp(points).reshape(mesh.nr,mesh.nphi)


def _hot_solver_context(snapshot: ScientificSnapshot):
    """Load the audited Stage 7 assembly and cavity configuration beside a snapshot."""
    case_dir=Path(str(snapshot.metadata['run_id']))
    summary=json.loads((case_dir/'summary.json').read_text())
    case=summary['case']; physics=case['physics']; numerics=case['numerics']
    geometry=physics['assembly']['geometry']
    mesh=DiskThermalMesh(np.asarray(snapshot.arrays['r_edges_m'],float),
                         np.asarray(snapshot.arrays['z_edges_m'],float),
                         int(np.asarray(snapshot.arrays['raw_heat_W_m3']).shape[-1]))
    cavity=ThinDiskResonator(**{key:physics['cavity'][key] for key in (
        'disk_diameter_m','disk_thickness_m','air_gap_m','output_mirror_radius_m',
        'output_transmission','disk_hr_reflectivity','other_roundtrip_loss',
        'wavelength_m','host_index','host_group_index','pump_hr_reflectivity')})
    assembly=json.loads(json.dumps(physics['assembly']))
    assembly['numerics']={'plate_thermal_nz':numerics['plate_thermal_nz'],
                          'mechanical':json.loads(json.dumps(numerics['mechanical']))}
    return case,mesh,cavity,assembly,geometry


def _run_full_seeded_modal(snapshot: ScientificSnapshot, grid: Grid2D,
                           density: HoDensityField, seeds_before_slm: dict[str,np.ndarray],
                           seeds: dict[str,np.ndarray], settings: GallerySettings) -> tuple[dict,dict]:
    """Build one fixed-mode oscillator background, then probe each input weakly."""
    case,thermal_mesh,cavity,assembly_cfg,_geometry=_hot_solver_context(snapshot)
    # Reuse the audited polar control volumes for the stiff four-manifold ODE.
    # The optical display grid has far too many sites for a bounded BDF cycle.
    areas=thermal_mesh.face_areas_m2.ravel()
    mode_values=np.asarray([_plane_to_polar(np.abs(field)**2,grid,thermal_mesh).ravel()
                            for field in seeds.values()])
    mode_values/=mode_values@areas[:,None]
    pump=area_averaged_lg0(thermal_mesh,case['physics']['pump']['waist_m'])
    pump/=float(pump@areas)
    density_active=_cartesian_to_polar(density.values_m3,grid,thermal_mesh).reshape(density.nz,-1)
    if np.any(density_active<=0):
        raise ValueError('generated Ho density did not cover every polar control volume')
    params=HoYAGFourLevelParams()
    source=resolve_pump_source(params.pump_wavelength_m,
                               case['physics']['pump']['duration_s'])
    model=ModalThinDiskLaser(cavity,areas,density_active,mode_values,pump,params=params,
                             mode_labels=tuple(seeds),pump_source=source,
                             spontaneous_fraction_per_mode=1e-8)
    pump_energy=case['physics']['pump']['energy_J']; repetition=case['physics']['pump']['repetition_rate_Hz']
    optical=model.run(pump_energy,repetition_rate_Hz=repetition,max_cycles=320,min_cycles=16,
                      max_period_cycles=4,pump_fwhm_s=case['physics']['pump']['duration_s'])
    if not optical.periodic_converged:
        raise RuntimeError('fixed-mode oscillator background did not reach a periodic pump state')
    heat,mean_fractions=sample_cycle_heat(model,optical,pump_energy,repetition,
                                          return_mean_fractions=True)
    thermal_assembly=PlateAssembly(thermal_mesh,grid,assembly_cfg)
    temperature,displacement,screens=thermal_assembly.solve(heat.heat_W_m3.reshape(thermal_mesh.shape))
    populations=np.zeros((4,density.nz,*grid.shape),float)
    polar_fractions=mean_fractions.reshape(4,*thermal_mesh.shape)
    for level in range(4):
        for iz in range(density.nz):
            populations[level,iz]=polar_to_cartesian(thermal_mesh,polar_fractions[level,iz],grid,outside=0.)
    populations=np.clip(populations,0,None)
    totals=populations.sum(axis=0)
    np.divide(populations,totals[None],out=populations,where=totals[None]>0)
    populations*=density.values_m3[None]
    validate_populations(populations,density.values_m3)
    outcomes={}
    for name,field in seeds.items():
        material=propagate_structured_signal_small_signal(field,grid,populations,density,params=params)
        vector=np.stack((material.field_out,np.zeros_like(material.field_out)),axis=-1)
        vector=apply_jones(vector,screens.inward_jones)
        transmission_opd=geometric_transmission_opd(
            screens.front_uz_m,screens.rear_uz_m,index=cavity.host_index)
        vector*=np.exp(2j*np.pi*transmission_opd/
                       float(snapshot.metadata['wavelength_m']))[...,None]
        output_vector=np.asarray([angular_spectrum_propagate(vector[...,pol],grid,
            float(snapshot.metadata['wavelength_m']),settings.post_disk_distance_m)
            for pol in range(2)])
        output_intensity=np.sum(np.abs(output_vector)**2,axis=0)
        output_power=float(np.sum(output_intensity)*grid.dx*grid.dy)
        input_power=float(np.sum(np.abs(field)**2)*grid.dx*grid.dy)
        outcomes[name]={'seed_before_slm':seeds_before_slm[name],'input_field':field,
                        'output_field':output_vector[0],'output_vector':output_vector,
                        'input_intensity':np.abs(field)**2,'output_intensity':output_intensity,
                        'input_power_W':input_power,'output_power_W':output_power,
                        'disk_exit_power_W':float(np.sum(np.sum(np.abs(vector)**2,axis=-1))*grid.dx*grid.dy),
                        'disk_power_gain':output_power/input_power,
                        'cross_polarized_fraction':float(np.sum(np.abs(output_vector[1])**2)*grid.dx*grid.dy/
                                                          max(output_power,1e-30))}
    diagnostics={'solver_mode':'full_seeded_modal','status':'completed_periodic_modal_state',
                 'periodic_cycles':optical.period_cycles,'pump_cycles':optical.cycles_simulated,
                 'oscillator_background_output_W':float(heat.budget['output_W']),
                 'pump_heat_W':float(heat.budget['heat_W']),
                 'pump_source':source.summary(),
                 'cycle_energy_ledger_relative_error':float(heat.budget['local_ledger_L1_error_over_incident']),
                 'thermal_peak_disk_K':float(temperature.disk_temperature_K.max()),
                 'thermal_peak_plate_K':float(temperature.plate_temperature_K.max()),
                 'thermal_balance_error_W':float(temperature.balance_error_W),
                 'mechanical_residual':float(displacement.free_residual_relative),
                 'photoelastic_screen':'audited Stage 6 Jones screen',
                 'population_model':'cycle-averaged saturated fixed-mode oscillator; displayed probes do not deplete it',
                 'thermal_optical_feedback':False,
                 'probe_population_depletion':False,
                 'mesh_convergence_verified':False,
                 'cavity_eigenfield_update':False}
    return outcomes,diagnostics


def _run_periodic_seeded(snapshot, grid, density, seeds_before_slm, seeds, settings):
    """One external seed, with pump, depletion, dark recovery, and hot assembly."""
    case,mesh,cavity,assembly_cfg,_=_hot_solver_context(snapshot)
    pump=case['physics']['pump']
    cfg=PeriodicAmplifierSettings(seed_energy_J=settings.seed_energy_J,
        seed_fwhm_s=settings.seed_fwhm_s,repetition_rate_Hz=pump['repetition_rate_Hz'],
        pump_energy_J=pump['energy_J'],pump_fwhm_s=pump['duration_s'],pump_waist_m=pump['waist_m'],
        signal_traversals=settings.signal_traversals,
        pump_reflectivity=cavity.pump_hr_reflectivity,
        relay_distance_m=settings.relay_distance_m,
        cavity_ejection_efficiency=settings.cavity_ejection_efficiency,
        cpu_workers=settings.cpu_workers)
    name=settings.selected_beam
    assembly=PlateAssembly(mesh,grid,assembly_cfg)
    index_response=HoIndexResponse(settings.dn_dHo_m3,settings.dn_dExcited_m3,
                                   settings.index_provenance)
    phase=None;temperature_xy=None;previous=None;converged=False;error=None;power_error=None
    for outer in range(4):
        pulse=solve_periodic_seeded_amplifier(seeds[name],grid,density,cfg,
              hot_phase_rad=phase,temperature_K=temperature_xy)
        if not pulse['converged']:
            raise RuntimeError('seeded pump/seed population cycle did not converge within bounded cycles')
        heat=_cartesian_to_polar(pulse['heat_W_m3'],grid,mesh)
        optical_heat=float(np.sum(pulse['heat_W_m3'])*density.dz_m*grid.dx*grid.dy)
        polar_heat=float(np.sum(heat*mesh.volumes_m3))
        if not np.isfinite(polar_heat) or abs(polar_heat)<1e-30:
            raise RuntimeError('thermal heat projection is degenerate')
        heat*=optical_heat/polar_heat
        temperature,displacement,screens=assembly.solve(heat)
        temperature_xy=polar_to_cartesian(mesh,np.mean(temperature.disk_temperature_K,axis=0),
                                          grid,outside=293.15)
        opd=(screens.thermal_single_pass_opd_m+
             screens.photoelastic_mean_single_pass_opd_m+
             geometric_transmission_opd(screens.front_uz_m,screens.rear_uz_m,
                                        index=cavity.host_index))
        opd+=index_response.single_pass_opd_m(density,pulse['populations_before_signal'])
        new_phase=2*np.pi*opd/cavity.wavelength_m
        # Piston has no effect on gain or intensity and should not impede closure.
        new_phase-=new_phase[grid.ny//2,grid.nx//2]
        if phase is not None:
            amplitude=np.sqrt(pulse['input_fluence_J_m2'])
            error=float(np.sqrt(np.sum(amplitude**2*(new_phase-phase)**2)/np.sum(amplitude**2)))
            power_error=abs(pulse['output_energy_J']-previous)/max(pulse['output_energy_J'],1e-30)
            if error<2e-3 and power_error<5e-3:
                converged=True
                phase=new_phase
                break
        phase=new_phase;previous=pulse['output_energy_J']
    # The last thermal screen is applied consistently to one final population cycle.
    if not converged:
        raise RuntimeError('seeded thermal-optical closure did not converge within four outer steps')
    pulse=solve_periodic_seeded_amplifier(seeds[name],grid,density,cfg,
                                          hot_phase_rad=phase,temperature_K=temperature_xy)
    output=angular_spectrum_propagate(pulse['field_out'],grid,cavity.wavelength_m,
                                       settings.post_disk_distance_m)
    input_irr=pulse['input_fluence_J_m2']*cfg.repetition_rate_Hz
    output_irr=np.abs(output)**2*cfg.repetition_rate_Hz
    output_energy=float(np.sum(np.abs(output)**2)*grid.dx*grid.dy)
    input_scale=np.sqrt(settings.seed_energy_J/optical_power(seeds[name],grid)*cfg.repetition_rate_Hz)
    outcomes={name:{'seed_before_slm':seeds_before_slm[name]*input_scale,
                    'input_field':seeds[name]*input_scale,
                    'output_field':output*np.sqrt(cfg.repetition_rate_Hz),
                    'input_intensity':input_irr,'output_intensity':output_irr,
                    'input_power_W':settings.seed_energy_J*cfg.repetition_rate_Hz,
                    'output_power_W':output_energy*cfg.repetition_rate_Hz,
                    'disk_exit_power_W':pulse['disk_exit_energy_J']*cfg.repetition_rate_Hz,
                    'disk_power_gain':pulse['disk_exit_energy_J']/settings.seed_energy_J,
                    'input_pulse_energy_J':settings.seed_energy_J,
                    'output_pulse_energy_J':output_energy,
                    'gain_medium_extraction_efficiency':pulse['gain_medium_extraction_efficiency'],
                    'cavity_ejection_efficiency':cfg.cavity_ejection_efficiency}}
    diag={'solver_mode':'periodic_seeded_amplifier','status':'periodic_and_thermal_fixed_point',
          'population_cycles':pulse['cycles'],'population_residual':pulse['population_residual'],
          'cpu_workers':pulse['cpu_workers'],
          'thermal_outer_iterations':outer+1,'thermal_peak_disk_K':float(temperature.disk_temperature_K.max()),
          'phase_closure_rms_rad':error,'power_closure_relative':power_error,
          'pump_absorbed_W':pulse['pump_absorbed_J']*cfg.repetition_rate_Hz,
          'pump_source':pulse['pump_source'],
          'signal_extracted_W':pulse['signal_extracted_J']*cfg.repetition_rate_Hz,
          'signal_extracted_J':pulse['signal_extracted_J'],
          'gain_medium_extraction_efficiency':pulse['gain_medium_extraction_efficiency'],
          'initial_stored_laser_energy_J':pulse['initial_stored_laser_energy_J'],
          'disk_exit_energy_J':pulse['disk_exit_energy_J'],
          'cavity_ejection_efficiency':cfg.cavity_ejection_efficiency,
          'cavity_ejection_loss_J':pulse['cavity_ejection_loss_J'],
          'relay_loss_J':pulse['relay_loss_J'],
          'passive_transport_change_J':pulse['passive_transport_change_J'],
          'optical_energy_balance_residual_J':pulse['optical_energy_balance_residual_J'],
          'heat_W':pulse['heat_energy_J']*cfg.repetition_rate_Hz,
          'heat_ledger_closure_J':pulse['heat_ledger_closure_J'],
          'gaussian_peak_seed_power_W':settings.seed_energy_J/(settings.seed_fwhm_s*np.sqrt(np.pi/(4*np.log(2)))),
          'thermal_balance_error_W':float(temperature.balance_error_W),
          'pass_records_J':pulse['pass_records_J'],
          'pass_diagnostics':pulse['pass_diagnostics'],
          'cross_section_temperature_status':'fixed 295 K Ho reference; no validated Ho emission/reabsorption temperature series',
          'pass_temperature_status':'beam-weighted converged steady thermal map; no transient temperature change between picosecond traversals',
          'refractive_index_model':{'host':'YAG n, dn/dT, and photoelastic reference',
              'dn_dHo_m3':settings.dn_dHo_m3,'dn_dExcited_m3':settings.dn_dExcited_m3,
              'provenance':settings.index_provenance,
              'excited_population_time':'pre-signal frozen state; in-pulse electronic lens not resolved'},
          'phase_feedback':'scalar mean thermal/photoelastic/transmission OPD applied at each material visit',
          'hot_phase_affects_gain':settings.relay_distance_m>0,
          'hardware_status':'illustrative bonded copper heatsink, ideal relay when distance=0; cavity ejection is a separate post-disk loss, coating heat not modeled',
          'mesh_convergence_verified':False,'experimental_calibration':False,
          'pulse_model':'short-pulse fluence kick; FWHM used for peak-power interpretation, not temporal propagation'}
    raw={'populations_before_pump_m3':pulse['populations_before_pump'],
         'populations_before_signal_m3':pulse['populations_before_signal'],
         'heat_W_m3_cartesian':pulse['heat_W_m3'],
         'heat_W_m3_polar':heat,
         'temperature_disk_K':temperature.disk_temperature_K,
         'temperature_optical_grid_K':temperature_xy,
         'temperature_plate_K':temperature.plate_temperature_K,
         'hot_phase_single_pass_rad':phase,
         'front_displacement_m':screens.front_uz_m,
         'rear_displacement_m':screens.rear_uz_m,
         'thermal_r_edges_m':mesh.r_edges_m,
         'thermal_z_edges_m':mesh.z_edges_m,
         'thermal_phi_rad':mesh.phi_rad}
    return outcomes,diag,raw


def simulate_gallery(snapshot: ScientificSnapshot,
                     settings: GallerySettings = GallerySettings()) -> dict:
    """Propagate each input through one disk and the same free-space leg."""
    a=snapshot.arrays
    n=96 if settings.solver_mode=='periodic_seeded_amplifier' else len(a['x_m'])
    width=float(a['x_m'][1]-a['x_m'][0])*len(a['x_m'])
    grid=Grid2D(n,n,width/n,width/n)
    radius=float(a['r_edges_m'][-1])
    density=nonuniform_density(grid,a['z_edges_m'],radius,settings)
    wavelength=float(snapshot.metadata['wavelength_m'])
    pattern=phase_pattern(grid,settings)
    seeds_before_slm=input_modes(grid,settings)
    modulated_seeds={}
    phase_meta=None
    for name,seed in seeds_before_slm.items():
        slm=apply_phase_modulator(seed[None],pattern)
        field=slm['field_after'][0]
        modulated_seeds[name]=field
        if phase_meta is None: phase_meta=slm
    if settings.solver_mode=='full_seeded_modal':
        outcomes,solver_diagnostics=_run_full_seeded_modal(snapshot,grid,density,
                                                           seeds_before_slm,modulated_seeds,settings)
        raw_fields={}
    elif settings.solver_mode=='periodic_seeded_amplifier':
        outcomes,solver_diagnostics,raw_fields=_run_periodic_seeded(snapshot,grid,density,
                                                           seeds_before_slm,modulated_seeds,settings)
    else:
        raw_fields={}
        outcomes={}
        populations=frozen_populations_on_grid(snapshot,grid,density)
        solver_diagnostics={'solver_mode':'weak_probe','status':'completed_frozen_population_probe',
                            'population_model':'archived cycle-averaged fractions',
                            'cavity_eigenfield_update':False}
    for name,seed in (() if settings.solver_mode!='weak_probe' else seeds_before_slm.items()):
        field=modulated_seeds[name]
        result=propagate_structured_signal_small_signal(field,grid,populations,density)
        output=angular_spectrum_propagate(result.field_out,grid,wavelength,
                                           settings.post_disk_distance_m)
        outcomes[name]={'seed_before_slm':seed,'input_field':field,'output_field':output,
                        'input_power_W':optical_power(field,grid),
                        'output_power_W':optical_power(output,grid),
                        'disk_exit_power_W':result.output_power,
                        'disk_power_gain':result.power_gain}
    return {'grid':grid,'density':density,'outcomes':outcomes,
            'raw_fields':raw_fields,
            'phase_pattern_rad':phase_meta['phi_pattern'],
            'phase_correction_rad':phase_meta['phi_correction'],
            'phase_requested_rad':phase_meta['phi_requested'],
            'phase_applied_rad':phase_meta['phi_applied'],
            'settings':settings,'wavelength_m':wavelength,
            'reference_state_id':snapshot.metadata['state_id'],
            'model':('periodic short-pulse externally seeded amplifier with thermal phase feedback'
                     if settings.solver_mode=='periodic_seeded_amplifier' else
                     'fixed-mode oscillator thermal background with separate weak seeded probes'
                     if settings.solver_mode=='full_seeded_modal' else
                     'weak one-traversal seeded probe; frozen archived population fractions'),
            'solver_diagnostics':solver_diagnostics}
