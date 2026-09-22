"""Stage 5C/D: thermo-optic OPD, phase screens and reduced hot-cavity modes.

No surface displacement or stress-optic term is included. Every phase supplied
below is SINGLE traversal unless an explicit pass count says otherwise.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .propagation import Grid2D
from .resonator import ThinDiskResonator, cavity_roundtrip
from .thermal import DiskHeatSolver


def thermal_opd(temperature_K, solver: DiskHeatSolver, *, reference_K=293.15,
                dn_dT_K1=9.1e-6, passes: int = 1) -> np.ndarray:
    """Path integral of delta n: OPD=passes*sum_z(dn/dT*(T-Tref)*dz)."""
    if not np.isfinite(reference_K) or reference_K <= 0 or not np.isfinite(dn_dT_K1):
        raise ValueError("invalid thermo-optic parameters")
    if not isinstance(passes,(int,np.integer)) or passes < 1:
        raise ValueError("passes must be a positive integer")
    t = np.asarray(temperature_K,float)
    if t.shape != solver.shape or np.any(~np.isfinite(t[solver.mask])):
        raise ValueError("temperature must match the thermal volume")
    return passes*dn_dT_K1*solver.dz*np.sum(np.where(solver.mask,t-reference_K,0.0),axis=0)


def phase_from_opd(opd_m, wavelength_m: float):
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength must be finite and positive")
    opd = np.asarray(opd_m,float)
    if np.any(~np.isfinite(opd)):
        raise ValueError("OPD must be finite")
    return 2*np.pi*opd/wavelength_m


def thermal_cavity_roundtrip(field, grid: Grid2D, cavity: ThinDiskResonator,
                             single_pass_opd_m, single_pass_log_gain=0.0):
    """Two physical disk crossings: phase once before OC and once on return.

    The base cavity operator already counts the two gain traversals and losses.
    Do not pass a double-pass OPD here or the thermal lens will be doubled again.
    """
    opd = np.asarray(single_pass_opd_m,float)
    if opd.shape != grid.shape:
        raise ValueError("OPD shape must match optical grid")
    ph = np.exp(1j*phase_from_opd(opd,cavity.wavelength_m))
    back, output = cavity_roundtrip(np.asarray(field)*ph,grid,cavity,single_pass_log_gain)
    return back*ph, output


@dataclass
class LensFit:
    power_matrix_m1: np.ndarray
    fitted_opd_m: np.ndarray
    residual_rms_m: float
    piston_m: float

    @property
    def mean_power_m1(self):
        return float(np.trace(self.power_matrix_m1)/2)

    @property
    def focal_lengths_m(self):
        eig = np.linalg.eigvalsh(self.power_matrix_m1)
        return np.divide(1.,eig,out=np.full(2,np.inf),where=np.abs(eig)>1e-14)


def fit_thermal_lens(opd_m, grid: Grid2D, *, fit_radius_m: float,
                     weights=None) -> LensFit:
    """Fit piston/tilt and a full quadratic OPD; converging power is -Hessian.

    This is a paraxial local fit, NOT removal of the remaining aberrations.
    Weights can select the part of the disk sampled by the current optical mode.
    """
    if not np.isfinite(fit_radius_m) or fit_radius_m <= 0:
        raise ValueError("fit radius must be finite and positive")
    opd = np.asarray(opd_m,float)
    if opd.shape != grid.shape or np.any(~np.isfinite(opd)):
        raise ValueError("finite OPD on optical grid required")
    x,y = grid.mesh
    w = np.ones(grid.shape) if weights is None else np.asarray(weights,float)
    if w.shape != grid.shape or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite, nonnegative and match grid")
    mask = (x*x+y*y <= fit_radius_m**2) & (w>0)
    if np.count_nonzero(mask) < 6:
        raise ValueError("lens fit is underresolved")
    xx,yy = x/fit_radius_m,y/fit_radius_m
    A = np.stack((np.ones_like(x),xx,yy,xx*xx/2,xx*yy,yy*yy/2),axis=-1)
    sw = np.sqrt(w[mask]/w[mask].max())
    coeff,_,rank,_ = np.linalg.lstsq(A[mask]*sw[:,None],opd[mask]*sw,rcond=None)
    if rank < 6:
        raise ValueError("lens fit is rank deficient")
    fit = A@coeff
    power = -np.array([[coeff[3],coeff[4]],[coeff[4],coeff[5]]])/fit_radius_m**2
    rms = np.sqrt(np.sum(w[mask]*(opd[mask]-fit[mask])**2)/np.sum(w[mask]))
    return LensFit(power,fit,float(rms),float(coeff[0]))


def gaussian_hot_cavity(cavity: ThinDiskResonator, single_pass_power_m1: float) -> dict:
    """ABCD fixed point with one thin-lens screen on EACH disk traversal.

    This axisymmetric quadratic approximation is used for the slow feedback
    loop. The full nonquadratic OPD remains available to the FFT field solver.
    """
    power = float(single_pass_power_m1)
    if not np.isfinite(power):
        raise ValueError("lens power must be finite")
    S = np.array([[1.,0.],[-power,1.]])
    P = np.array([[1.,cavity.reduced_length_m],[0.,1.]])
    M = np.array([[1.,0.],[-2/cavity.output_mirror_radius_m,1.]])
    rt = S@P@M@P@S
    A,B,C,D = rt.ravel()
    half_trace = (A+D)/2
    if not abs(half_trace) < 1:
        return {"stable":False,"half_trace":float(half_trace),"roundtrip_matrix":rt}
    roots = np.roots([C,D-A,-B])
    q = next(complex(v) for v in roots if v.imag>0)
    w = np.sqrt(-cavity.wavelength_m/(np.pi*(1/q).imag))
    # Outward lens first, then propagation to the OC.
    qo = q/(1-power*q)+cavity.reduced_length_m
    wo = np.sqrt(-cavity.wavelength_m/(np.pi*(1/qo).imag))
    return {"stable":True,"half_trace":float(half_trace),"waist_m":float(w),
            "oc_spot_m":float(wo),"q_m":q,"roundtrip_matrix":rt}


def radial_source_to_volume(radius_m, source_W_m3, areas_m2, source_dz_m: float,
                            solver: DiskHeatSolver):
    """Conserve integrated positive/negative power when mapping radial quadrature.

    Linear radial interpolation; exact overlap of source/destination z cells.
    Per-layer power renormalization compensates raster quadrature error. It is
    reported so excessive interpolation error is not silently concealed.
    """
    r,q,area = np.asarray(radius_m,float),np.asarray(source_W_m3,float),np.asarray(areas_m2,float)
    if r.ndim!=1 or area.shape!=r.shape or q.ndim!=2 or q.shape[1]!=len(r):
        raise ValueError("expected radial nodes/areas and (nz,nr) heat source")
    if np.any(np.diff(r)<=0) or np.any(area<=0) or np.any(~np.isfinite(q)) or np.any(~np.isfinite(r)) or np.any(~np.isfinite(area)):
        raise ValueError("invalid radial sampling")
    if not np.isfinite(source_dz_m) or source_dz_m<=0 or not np.isclose(q.shape[0]*source_dz_m,solver.thickness_m,rtol=1e-10,atol=0):
        raise ValueError("source and thermal thickness differ")
    x,y = solver.grid.mesh; rr=np.hypot(x,y); mask=solver.mask[0]
    nz_src=len(q); mapped=np.zeros((nz_src,*solver.grid.shape)); factors=[]
    pixel_area=solver.grid.dx*solver.grid.dy
    for iz in range(nz_src):
        for sign in (1,-1):
            values=np.maximum(sign*q[iz],0.0)
            target=float(values@area)
            if target==0:continue
            raster=np.where(mask,np.interp(rr,r,values),0.0)
            raw=float(raster.sum()*pixel_area)
            if raw<=0:raise ValueError("heat feature missed by thermal grid")
            factor=target/raw;factors.append(factor)
            mapped[iz]+=sign*factor*raster
    src_edges=np.arange(nz_src+1)*source_dz_m
    dest_edges=np.arange(solver.nz+1)*solver.dz
    overlap=np.maximum(0.,np.minimum(dest_edges[1:,None],src_edges[None,1:])-np.maximum(dest_edges[:-1,None],src_edges[None,:-1]))
    full=np.einsum('ij,jyx->iyx',overlap/solver.dz,mapped)
    return full,{"power_before_W":float(np.sum(q*area)*source_dz_m),
                 "power_after_W":float(full.sum()*solver.cell_volume_m3),
                 "min_raster_correction":min(factors,default=1.),
                 "max_raster_correction":max(factors,default=1.)}
