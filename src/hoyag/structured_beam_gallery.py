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


PHASE_MASKS=('none','vortex+1','vortex-1','vortex+2',
             'defocus','astigmatic','axicon')
SOLVER_MODES=('weak_probe','full_seeded_modal')


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


def simulate_gallery(snapshot: ScientificSnapshot,
                     settings: GallerySettings = GallerySettings()) -> dict:
    """Propagate each input through one disk and the same free-space leg."""
    a=snapshot.arrays
    grid=Grid2D(len(a['x_m']),len(a['y_m']),
                float(a['x_m'][1]-a['x_m'][0]),float(a['y_m'][1]-a['y_m'][0]))
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
    else:
        outcomes={}
        populations=frozen_populations_on_grid(snapshot,grid,density)
        solver_diagnostics={'solver_mode':'weak_probe','status':'completed_frozen_population_probe',
                            'population_model':'archived cycle-averaged fractions',
                            'cavity_eigenfield_update':False}
    for name,seed in (() if settings.solver_mode=='full_seeded_modal' else seeds_before_slm.items()):
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
            'phase_pattern_rad':phase_meta['phi_pattern'],
            'phase_correction_rad':phase_meta['phi_correction'],
            'phase_requested_rad':phase_meta['phi_requested'],
            'phase_applied_rad':phase_meta['phi_applied'],
            'settings':settings,'wavelength_m':wavelength,
            'reference_state_id':snapshot.metadata['state_id'],
            'model':('fixed-mode oscillator thermal background with separate weak seeded probes'
                     if settings.solver_mode=='full_seeded_modal' else
                     'weak one-traversal seeded probe; frozen archived population fractions'),
            'solver_diagnostics':solver_diagnostics}
