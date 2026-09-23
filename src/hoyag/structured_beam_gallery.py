"""Weak structured-light probes through a declared nonuniform Ho:YAG disk.

The saved Stage 7W population *fractions* are frozen. A new Ho concentration
map scales those fractions locally; pump, heat, and mechanics are not re-solved.
This is an externally seeded one-traversal probe, not an oscillator prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inhomogeneity import HoDensityField
from .population_state import validate_populations
from .propagation import (Grid2D, angular_spectrum_propagate, gaussian_beam,
                          hermite_gaussian, laguerre_gaussian, optical_power)
from .signal import propagate_structured_signal_small_signal
from .seeded_amplifier import apply_phase_modulator
from .snapshots import ScientificSnapshot
from .thermal import DiskThermalMesh
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


PHASE_MASKS=('none','vortex+1','vortex-1','vortex+2',
             'defocus','astigmatic','axicon')


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
    """One Gaussian control, two helical LG charges, and a four-lobe HG mode."""
    w=settings.waist_m
    modes = {'Gaussian TEM00':gaussian_beam(grid,w),
             'Helical LG(0,+1)':laguerre_gaussian(grid,0,1,w),
             'Double helix LG(0,+2)':laguerre_gaussian(grid,0,2,w),
             'Hermite-Gaussian HG(1,1)':hermite_gaussian(grid,1,1,w)}
    return {name:field*np.sqrt(settings.input_power_W/optical_power(field,grid))
            for name,field in modes.items()}


def simulate_gallery(snapshot: ScientificSnapshot,
                     settings: GallerySettings = GallerySettings()) -> dict:
    """Propagate each weak input through one disk and the same free-space leg."""
    a=snapshot.arrays
    grid=Grid2D(len(a['x_m']),len(a['y_m']),
                float(a['x_m'][1]-a['x_m'][0]),float(a['y_m'][1]-a['y_m'][0]))
    radius=float(a['r_edges_m'][-1])
    density=nonuniform_density(grid,a['z_edges_m'],radius,settings)
    populations=frozen_populations_on_grid(snapshot,grid,density)
    wavelength=float(snapshot.metadata['wavelength_m'])
    pattern=phase_pattern(grid,settings)
    outcomes={}
    for name,seed in input_modes(grid,settings).items():
        slm=apply_phase_modulator(seed[None],pattern)
        field=slm['field_after'][0]
        result=propagate_structured_signal_small_signal(field,grid,populations,density)
        output=angular_spectrum_propagate(result.field_out,grid,wavelength,
                                           settings.post_disk_distance_m)
        outcomes[name]={'seed_before_slm':seed,'input_field':field,'output_field':output,
                        'input_power_W':optical_power(field,grid),
                        'output_power_W':optical_power(output,grid),
                        'disk_exit_power_W':result.output_power,
                        'disk_power_gain':result.power_gain}
    return {'grid':grid,'density':density,'outcomes':outcomes,
            'phase_pattern_rad':slm['phi_pattern'],
            'phase_correction_rad':slm['phi_correction'],
            'phase_requested_rad':slm['phi_requested'],
            'phase_applied_rad':slm['phi_applied'],
            'settings':settings,'wavelength_m':wavelength,
            'reference_state_id':snapshot.metadata['state_id'],
            'model':'weak one-traversal seeded probe; frozen archived population fractions'}
