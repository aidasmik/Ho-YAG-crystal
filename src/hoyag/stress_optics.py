"""Stage 6: stress-optic tensors and ordered, reciprocal Jones propagation.

The default uses cubic YAG elasto-optic coefficients p11=-.029, p12=.0091,
p44=-.0615 (Bričkus & Dement'ev 2016, DOI 10.3952/physics.v56i1.3272).
Only ELASTIC strain S:sigma enters this term. Thermal expansion already present
in stress-free dn/dT must not be added a second time. Coefficients are a host
reference, not a verified Ho:YAG measurement at 2.09 um.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .thermomechanics import ElasticMaterial,YAG_ELASTIC,positive,stress_tensor


@dataclass(frozen=True)
class CubicElastoOptic:
    p11:float=-.029
    p12:float=.0091
    p44:float=-.0615
    def __post_init__(self):
        if not np.all(np.isfinite([self.p11,self.p12,self.p44])):raise ValueError('finite photoelastic coefficients required')


def crystal_axes_111(azimuth_rad=0.):
    """Columns are lab x,y,z in cubic crystal coordinates; z // [111]."""
    if not np.isfinite(azimuth_rad):raise ValueError('azimuth must be finite')
    q=np.column_stack((np.array([1.,-1,0])/np.sqrt(2),
                       np.array([1.,1,-2])/np.sqrt(6),
                       np.ones(3)/np.sqrt(3)))
    c,s=np.cos(azimuth_rad),np.sin(azimuth_rad)
    return q@np.array([[c,-s,0],[s,c,0],[0,0,1]])


def _rotation(q):
    q=np.asarray(q,float)
    if q.shape!=(3,3) or not np.all(np.isfinite(q)) or not np.allclose(q.T@q,np.eye(3),rtol=0,atol=1e-12) or np.linalg.det(q)<0:
        raise ValueError('crystal axes must be a proper orthonormal 3x3 rotation')
    return q


def stress_impermeability(stress_Pa,*,material:ElasticMaterial=YAG_ELASTIC,
                          coefficients=CubicElastoOptic(),crystal_axes=None):
    """Delta B (inverse relative dielectric tensor), lab axes, dimensionless.

    Stress order xx,yy,zz,xy,xz,yz (tensor shear). Cubic p44 multiplies
    engineering shear strain: Bxy = p44 * 2*elastic_exy.
    """
    st=stress_tensor(stress_Pa)
    if not np.all(np.isfinite(st)):raise ValueError('stress must be finite')
    q=_rotation(crystal_axes_111() if crystal_axes is None else crystal_axes)
    elastic=((1+material.poisson)*st-material.poisson*np.trace(st,axis1=-2,axis2=-1)[...,None,None]*np.eye(3))/material.young_Pa
    e=np.einsum('ia,...ab,jb->...ij',q,elastic,q,optimize=True)
    p=coefficients
    db=2*p.p44*e
    diagonal=p.p12*np.trace(e,axis1=-2,axis2=-1)[...,None]+(p.p11-p.p12)*np.diagonal(e,axis1=-2,axis2=-1)
    db[...,np.arange(3),np.arange(3)]=diagonal
    return np.einsum('ai,...ab,bj->...ij',q,db,q,optimize=True)


def photoelastic_index_matrix(stress_Pa,*,index=1.799104526293235,**kwargs):
    """Transverse Delta n for propagation along lab z; true symmetric eigensolve."""
    positive(index,'index')
    db=stress_impermeability(stress_Pa,**kwargs)[...,:2,:2]
    b=db+np.eye(2)/index**2
    val,vec=np.linalg.eigh(b)
    if np.min(val)<=0:raise ValueError('nonpositive impermeability: outside linear model')
    dn=1/np.sqrt(val)-index
    return np.einsum('...ia,...a,...ja->...ij',vec,dn,vec)


def geometric_roundtrip_opd(front_uz_m,rear_uz_m,*,index=1.799104526293235):
    """Small-deformation reflected OPD referenced to a FIXED external plane.

    z increases from optical front into plate. Includes change in external air
    path AND internal crystal thickness. Rear HR follows the CRYSTAL rear face,
    not the plate face when the interface is compliant.
    deltaOPD = 2[(1-n)u_front + n*u_rear].
    Rigid translation gives 2u; fixed rear expansion gives 2(n-1)*delta_thickness.
    """
    positive(index,'index')
    f,r=np.broadcast_arrays(np.asarray(front_uz_m,float),np.asarray(rear_uz_m,float))
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(r)):raise ValueError('displacements must be finite')
    return 2*((1-index)*f+index*r)


def ordered_jones(index_change_by_slice,dz_m,wavelength_m):
    """Near-normal paraxial Jones matrices in a fixed lab x/y basis.

    `index_change_by_slice`: (nz,...,2,2), real symmetric. Returns front->HR
    and HR->front traversals. Retarders in different slices need not commute.
    A reciprocal scalar mirror does not complex-conjugate the field.
    """
    positive(wavelength_m,'wavelength')
    dn=np.asarray(index_change_by_slice,float)
    if dn.ndim<3 or dn.shape[-2:]!=(2,2) or not np.all(np.isfinite(dn)) or not np.allclose(dn,dn.swapaxes(-1,-2),rtol=0,atol=1e-14):
        raise ValueError('finite symmetric 2x2 index matrices required')
    dz=np.broadcast_to(np.asarray(dz_m,float),(dn.shape[0],))
    if not np.all(np.isfinite(dz)) or np.any(dz<=0):raise ValueError('slice thickness must be positive')
    inward=np.broadcast_to(np.eye(2,dtype=complex),dn.shape[1:]).copy()
    outward=inward.copy()
    k=2*np.pi/wavelength_m
    for a,d in zip(dn,dz):
        val,vec=np.linalg.eigh(a)
        h=np.einsum('...ia,...a,...ja->...ij',vec,np.exp(1j*k*d*val),vec)
        inward=h@inward
        outward=outward@h
    return inward,outward


def apply_jones(field,jones):
    """field (...,2), Jones (...,2,2); all non-polarization axes broadcast."""
    f=np.asarray(field,complex);j=np.asarray(jones,complex)
    if f.shape[-1]!=2 or j.shape[-2:]!=(2,2) or not np.all(np.isfinite(f)) or not np.all(np.isfinite(j)):
        raise ValueError('invalid field or Jones matrix')
    return np.einsum('...ij,...j->...i',j,f)


@dataclass
class HotDiskScreens:
    inward_jones:np.ndarray
    outward_jones:np.ndarray
    geometry_roundtrip_opd_m:np.ndarray
    thermal_single_pass_opd_m:np.ndarray
    photoelastic_mean_single_pass_opd_m:np.ndarray
    photoelastic_retardance_bound_rad:np.ndarray
    front_uz_m:np.ndarray
    rear_uz_m:np.ndarray
    wavelength_m:float

    @property
    def mean_roundtrip_opd_m(self):
        return self.geometry_roundtrip_opd_m+2*(self.thermal_single_pass_opd_m+self.photoelastic_mean_single_pass_opd_m)


def build_hot_disk_screens(assembly_mesh,displacement,thermal_mesh,temperature_K,xy_query,*,
                           index=1.799104526293235,wavelength_m=2.0903e-6,
                           dn_dT_K1=9.1e-6,reference_temperature_K=293.15,
                           material=YAG_ELASTIC,coefficients=CubicElastoOptic(),crystal_axes=None):
    """Map FEM mechanics and Stage 5/6 temperature onto arbitrary optical XY.

    Stress is volume-weighted to nodes before interpolation. The small polygonal
    FEM rim is extended to the analytical circle by nearest-boundary recovery.
    Outside the disk all optical operators are identity, not extra absorption.
    """
    from .cooling_plate import sample_temperature
    positive(reference_temperature_K,'reference temperature')
    if not np.isfinite(dn_dT_K1):raise ValueError('dn/dT must be finite')
    xy=np.asarray(xy_query,float)
    if xy.shape[-1]!=2 or not np.all(np.isfinite(xy)):raise ValueError('finite xy[...,2] required')
    shape=xy.shape[:-1];flat=xy.reshape(-1,2)
    inside=np.linalg.norm(flat,axis=1)<=assembly_mesh.radius_m
    # Only sample inside; no fictitious crystal/plate optical contribution outside.
    chosen=flat[inside]
    nz=thermal_mesh.nz;dz=np.diff(thermal_mesh.z_edges_m)
    ni=np.broadcast_to(np.eye(2,dtype=complex),(len(flat),2,2)).copy();no=ni.copy()
    geom=np.zeros(len(flat));th=np.zeros(len(flat));pe=np.zeros(len(flat));ret=np.zeros(len(flat))
    uf=np.zeros(len(flat));ur=np.zeros(len(flat))
    if len(chosen):
        z=thermal_mesh.z_m
        points=np.concatenate([np.column_stack((chosen,np.full(len(chosen),zz))) for zz in z])
        temp=sample_temperature(thermal_mesh,temperature_K,points).reshape(nz,-1)
        stress=assembly_mesh.disk.interpolate_layers(displacement.disk_nodal_stress_Pa,chosen,z)
        dn=photoelastic_index_matrix(stress,index=index,material=material,coefficients=coefficients,crystal_axes=crystal_axes)
        delta_t=dn_dT_K1*(temp-reference_temperature_K)
        all_dn=dn+delta_t[...,None,None]*np.eye(2)
        ji,jo=ordered_jones(all_dn,dz,wavelength_m)
        ni[inside]=ji;no[inside]=jo
        u=assembly_mesh.disk.interpolate_layers(displacement.disk_u_m,chosen,[0,assembly_mesh.disk_thickness_m])
        uf[inside]=u[0,:,2];ur[inside]=u[1,:,2]
        geom[inside]=geometric_roundtrip_opd(u[0,:,2],u[1,:,2],index=index)
        th[inside]=np.sum(delta_t*dz[:,None],axis=0)
        pe[inside]=np.sum(np.trace(dn,axis1=-2,axis2=-1)*.5*dz[:,None],axis=0)
        val=np.linalg.eigvalsh(dn)
        ret[inside]=2*np.pi/wavelength_m*np.sum((val[...,1]-val[...,0])*dz[:,None],axis=0)
    return HotDiskScreens(ni.reshape(shape+(2,2)),no.reshape(shape+(2,2)),
                          *[a.reshape(shape) for a in (geom,th,pe,ret,uf,ur)],wavelength_m)


def hot_disk_cavity_roundtrip(field,grid,cavity,screens:HotDiskScreens,*,single_pass_log_gain=0.):
    """Stage 4R round trip for a two-polarization field (2,ny,nx).

    Full thermal+photoelastic Jones operator on EACH disk visit. Total geometric
    reflected OPD is split symmetrically across visits, matching the existing
    collapsed-disk screen convention. Do NOT also call thermal_cavity_roundtrip:
    its dn/dT contribution is already present here.
    """
    from .resonator import cavity_roundtrip
    f=np.asarray(field,complex)
    if f.shape!=(2,*grid.shape):raise ValueError('field must have shape (2,ny,nx)')
    if screens.geometry_roundtrip_opd_m.shape!=grid.shape:raise ValueError('screen and grid shape mismatch')
    if not np.isclose(cavity.wavelength_m,screens.wavelength_m,rtol=1e-12,atol=0):raise ValueError('screen/cavity wavelengths differ')
    half=np.exp(1j*np.pi*screens.geometry_roundtrip_opd_m/screens.wavelength_m)
    outgoing=apply_jones(f.transpose(1,2,0),screens.outward_jones)*half[...,None]
    back=[];out=[]
    for pol in range(2):
        a,b=cavity_roundtrip(outgoing[...,pol],grid,cavity,single_pass_log_gain)
        back.append(a);out.append(b)
    returned=apply_jones(np.stack(back,axis=-1),screens.inward_jones)*half[...,None]
    return returned.transpose(2,0,1),np.stack(out)
