"""Stage 5C/D: thermo-refractive OPD and double-pass cavity phase.

Only dn/dT is included. End-face bulging, stress/photoelasticity, coating phase,
and temperature-dependent Ho spectroscopy are not implied by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .thermal import DiskThermalMesh
from .resonator import ThinDiskResonator, cavity_roundtrip


def thermal_opd(mesh: DiskThermalMesh, temperature_K, *, reference_temperature_K=293.15,
                dn_dT_K1=9.1e-6, passes=1):
    """OPD [m] on (r,phi), integrating every depth cell once per traversal."""
    if not np.isfinite(reference_temperature_K) or reference_temperature_K<=0:
        raise ValueError('reference temperature must be finite and positive')
    if not isinstance(passes,(int,np.integer)) or isinstance(passes,bool) or passes<1:
        raise ValueError('passes must be a positive integer')
    t=mesh.field(temperature_K,'temperature')
    if np.min(t)<=0: raise ValueError('temperature must be above absolute zero')
    d=mesh.field(dn_dT_K1,'dn/dT')
    return passes*np.sum(d*(t-reference_temperature_K)*np.diff(mesh.z_edges_m)[:,None,None],axis=0)


def polar_to_cartesian(mesh: DiskThermalMesh, values_r_phi, grid, *, outside=0.):
    """Interpolate a scalar disk map to the optical grid; periodic in azimuth.

    Interpolation is for phase/visualization, not conservative heat remapping.
    At r=0 and the outer half-cell the nearest radial cell is used.
    """
    a=np.asarray(values_r_phi,float)
    if a.shape!=(mesh.nr,mesh.nphi) or not np.all(np.isfinite(a)):
        raise ValueError('map must be finite with shape (nr,nphi)')
    x,y=grid.mesh;r=np.hypot(x,y);inside=r<=mesh.r_edges_m[-1]
    rr=np.clip(r,mesh.r_m[0],mesh.r_m[-1])
    if mesh.nphi==1:
        v=np.interp(rr,mesh.r_m,a[:,0])
    else:
        phi=np.mod(np.arctan2(y,x),2*np.pi)
        angle=np.r_[mesh.phi_rad[-1]-2*np.pi,mesh.phi_rad,mesh.phi_rad[0]+2*np.pi]
        extended=np.concatenate((a[:,-1:],a,a[:,:1]),axis=1)
        v=RegularGridInterpolator((mesh.r_m,angle),extended)(np.column_stack((rr.ravel(),phi.ravel()))).reshape(r.shape)
    return np.where(inside,v,outside)


def thermal_cavity_roundtrip(field,grid,cavity: ThinDiskResonator, single_pass_opd_m,
                              single_pass_log_gain=0.0):
    """Full OPD screen on BOTH disk visits; same reference plane as Stage 4R.

    Pass SINGLE-traversal OPD, not double-pass OPD. A constant OPD changes the
    round-trip phase by 2*k0*OPD but cannot change normalized modal overlap.
    """
    opd=np.broadcast_to(np.asarray(single_pass_opd_m,float),grid.shape)
    if not np.all(np.isfinite(opd)): raise ValueError('OPD must be finite')
    screen=np.exp(2j*np.pi*opd/cavity.wavelength_m)
    back,output=cavity_roundtrip(np.asarray(field)*screen,grid,cavity,single_pass_log_gain)
    return back*screen,output


@dataclass(frozen=True)
class LensFit:
    single_pass_power_m1: float
    single_pass_focal_length_m: float
    piston_m: float
    weighted_rms_residual_m: float
    fit_radius_m: float


def fit_radial_thermal_lens(radius_m,opd_m,weights=None,*,fit_radius_m):
    """Fit OPD = piston - r^2/(2*f); focusing lens has positive 1/f.

    The fit radius and weights matter for an aberrated thermal OPD. They are
    part of the reported result, not hidden global material constants.
    """
    r=np.asarray(radius_m,float);o=np.asarray(opd_m,float)
    w=np.ones_like(r) if weights is None else np.asarray(weights,float)
    if r.ndim!=1 or r.shape!=o.shape or r.shape!=w.shape or not np.all(np.isfinite([r,o,w])) or np.any(w<0):
        raise ValueError('radius, OPD and nonnegative weights must be matching finite 1D arrays')
    if not np.isfinite(fit_radius_m) or fit_radius_m<=0 or np.any(r<0): raise ValueError('invalid radii')
    mask=(r<=fit_radius_m)&(w>0)
    if np.count_nonzero(mask)<3: raise ValueError('at least three weighted radius samples required')
    a=np.column_stack((np.ones(mask.sum()),(r[mask]/fit_radius_m)**2))
    wt=w[mask]/w[mask].sum(); coef=np.linalg.lstsq(a*np.sqrt(wt)[:,None],o[mask]*np.sqrt(wt),rcond=None)[0]
    power=-2*coef[1]/fit_radius_m**2
    rms=np.sqrt(np.sum(wt*(a@coef-o[mask])**2))
    return LensFit(float(power),float(np.inf if abs(power)<1e-14 else 1/power),float(coef[0]),float(rms),float(fit_radius_m))


@dataclass(frozen=True)
class HotGaussianMode:
    disk_radius_m: float
    output_radius_m: float
    q_at_disk_m: complex
    trace_half: float
    single_pass_lens_power_m1: float


def hot_gaussian_mode(cavity: ThinDiskResonator, single_pass_lens_power_m1=0.):
    """Parabolic thermal-screen approximation to the plane-HR cavity eigenmode.

    M = F P C P F, with one single-pass thermal lens F on each disk traversal.
    q convention matches Stage 4R's exp(-i*k_transverse^2*z/2*k0) FFT operator:
    Im(q)<0 and the field's quadratic phase is +k0*r^2*Re(1/q)/2.
    A nonparabolic phase is still applied in full by thermal_cavity_roundtrip.
    """
    power=float(single_pass_lens_power_m1)
    if not np.isfinite(power): raise ValueError('lens power must be finite')
    L=cavity.reduced_length_m
    f=np.array([[1.,0.],[-power,1.]])
    p=np.array([[1.,L],[0.,1.]])
    c=np.array([[1.,0.],[-2/cavity.output_mirror_radius_m,1.]])
    matrix=f@p@c@p@f;A,B,C,D=matrix.ravel();mu=(A+D)/2
    if abs(mu)>=1-1e-12: raise ValueError('thermal lens makes cavity unstable or marginal')
    roots=np.roots([C,D-A,-B]); q=next((complex(v) for v in roots if v.imag<0),None)
    if q is None: raise ValueError('no confined Gaussian cavity mode')
    qoc=q/(1-power*q)+L
    w=np.sqrt(cavity.wavelength_m/(np.pi*np.imag(1/q)))
    woc=np.sqrt(cavity.wavelength_m/(np.pi*np.imag(1/qoc)))
    return HotGaussianMode(float(w),float(woc),q,float(mu),power)


def normalized_overlap(reference,field):
    a=np.asarray(reference,complex);b=np.asarray(field,complex)
    if a.shape!=b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('fields must have matching finite arrays')
    norm=np.vdot(a,a).real*np.vdot(b,b).real
    if norm<=0: raise ValueError('zero field in overlap')
    return float(np.clip(abs(np.vdot(a,b))**2/norm,0,1))
