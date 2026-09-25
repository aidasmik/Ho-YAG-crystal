"""Stage 6: finite cooling plate coupled to the Stage 5 disk heat solver.

Heat transfer across the bond is reciprocal and energy-conserving. Its finite
area-specific resistance is distinct from the plate/coolant resistance. A
prescribed contact map is allowed; there is no unvalidated pressure-h law.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import block_diag,coo_matrix,diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import RegularGridInterpolator
from .thermal import DiskThermalMesh,DiskCooling,ThermalBoundary,DiskHeatSolver
from .thermomechanics import positive,integer


@dataclass(frozen=True)
class ThermalMaterial:
    conductivity_W_mK: float | np.ndarray
    density_kg_m3: float
    heat_capacity_J_kgK: float
    def __post_init__(self):
        conductivity=np.asarray(self.conductivity_W_mK,float)
        if np.any(~np.isfinite(conductivity)) or np.any(conductivity<=0):
            raise ValueError('conductivity_W_mK must be finite and positive')
        positive(self.density_kg_m3,'density_kg_m3')
        positive(self.heat_capacity_J_kgK,'heat_capacity_J_kgK')

YAG_THERMAL=ThermalMaterial(14.,4560.,680.)
# Rounded SI conversions of the CDA C10100 room-temperature table.
COPPER_THERMAL=ThermalMaterial(391.16,8940.,385.2)


def cooling_plate_mesh(disk:DiskThermalMesh,*,radius_m=10e-3,thickness_m=3e-3,nz=8,outer_rings=12):
    positive(radius_m,'plate radius');positive(thickness_m,'plate thickness')
    integer(nz,'plate nz');integer(outer_rings,'outer_rings')
    if radius_m<disk.r_edges_m[-1]:raise ValueError('plate must cover the disk')
    r=disk.r_edges_m if radius_m==disk.r_edges_m[-1] else np.r_[disk.r_edges_m,np.linspace(disk.r_edges_m[-1],radius_m,outer_rings+1)[1:]]
    return DiskThermalMesh(r,np.linspace(0,thickness_m,nz+1),disk.nphi)


@dataclass
class AssemblyTemperature:
    disk_temperature_K:np.ndarray
    plate_temperature_K:np.ndarray
    interface_flux_W_m2:np.ndarray
    disk_contact_temperature_K:np.ndarray
    plate_contact_temperature_K:np.ndarray
    input_heat_W:float
    outward_heat_W:dict
    stored_energy_change_J:float
    balance_error_W:float
    relative_balance_error:float


class DiskPlateHeatSolver:
    def __init__(self,disk:DiskThermalMesh,plate:DiskThermalMesh,*,
                 contact_conductance_W_m2K=1e5,
                 coolant=ThermalBoundary(293.15,1e4),
                 disk_material=YAG_THERMAL,plate_material=COPPER_THERMAL,
                 disk_front=None,disk_rim=None,plate_rim=None):
        if plate.nphi!=disk.nphi or plate.nr<disk.nr or not np.array_equal(plate.r_edges_m[:disk.nr+1],disk.r_edges_m):
            raise ValueError('plate must use the identical disk radial prefix and nphi')
        self.disk=disk;self.plate=plate;self.coolant=coolant
        insulated=ThermalBoundary(coolant.bath_temperature_K,0.)
        self.ds=DiskHeatSolver(disk,disk_material.conductivity_W_mK,
            DiskCooling(rear=insulated,front=disk_front or insulated,rim=disk_rim or insulated),
            density_kg_m3=disk_material.density_kg_m3,heat_capacity_J_kgK=disk_material.heat_capacity_J_kgK)
        self.ps=DiskHeatSolver(plate,plate_material.conductivity_W_mK,
            DiskCooling(rear=coolant,front=insulated,rim=plate_rim or insulated),
            density_kg_m3=plate_material.density_kg_m3,heat_capacity_J_kgK=plate_material.heat_capacity_J_kgK)
        self.nd=int(np.prod(disk.shape));self.np=int(np.prod(plate.shape))
        h=np.broadcast_to(np.asarray(contact_conductance_W_m2K,float),(disk.nr,disk.nphi)).copy()
        if np.any(np.isnan(h)) or np.any(h<0):raise ValueError('contact h must be >=0, optionally +inf')
        self.h=h
        self.rd=np.diff(disk.z_edges_m)[-1]/(2*self.ds.k[-1])
        self.rp=np.diff(plate.z_edges_m)[0]/(2*self.ps.k[0,:disk.nr])
        active=h>0
        resistance=np.full(h.shape,np.inf)
        resistance[active]=self.rd[active]+self.rp[active]+1/h[active]
        self.g=disk.face_areas_m2/resistance
        di=np.arange(self.nd).reshape(disk.shape)[-1].ravel()
        pi=self.nd+np.arange(self.np).reshape(plate.shape)[0,:disk.nr].ravel()
        g=self.g.ravel()
        contact=coo_matrix((np.r_[g,g,-g,-g],(np.r_[di,pi,di,pi],np.r_[di,pi,pi,di])),shape=(self.nd+self.np,)*2)
        self.K=block_diag((self.ds.K,self.ps.K),format='csc')+contact.tocsc()
        self.capacity=np.r_[self.ds.capacity_J_K.ravel(),self.ps.capacity_J_K.ravel()]
        # Solve temperature increments to avoid cancellation of bath-scale fluxes.
        ref=coolant.bath_temperature_K;rhs=[]
        for solver in (self.ds,self.ps):
            a=np.zeros(solver.mesh.shape)
            for sl,conductance,bath in solver.boundaries.values():a[sl]+=conductance*(bath-ref)
            rhs.append(a.ravel())
        self.boundary_rhs=np.concatenate(rhs)

    def _check_steady(self):
        if not np.any(self.g>0):raise ValueError('disk and plate are thermally disconnected')
        if not any(np.any(g>0) for solver in (self.ds,self.ps) for _,g,_ in solver.boundaries.values()):
            raise ValueError('no heat-removing boundary')

    def _source(self,disk_q,plate_q):
        dq=self.disk.field(disk_q,'disk heat');pq=self.plate.field(plate_q,'plate heat')
        return np.r_[(dq*self.disk.volumes_m3).ravel(),(pq*self.plate.volumes_m3).ravel()]

    def _result(self,temperature,source,delta_energy=0.,dt=None):
        if not np.all(np.isfinite(temperature)) or np.min(temperature)<=0:raise FloatingPointError('invalid temperature')
        td=temperature[:self.nd].reshape(self.disk.shape)
        tp=temperature[self.nd:].reshape(self.plate.shape)
        flux=self.g*(td[-1]-tp[0,:self.disk.nr])/self.disk.face_areas_m2
        face_d=td[-1]-flux*self.rd
        face_p=tp[0,:self.disk.nr]+flux*self.rp
        flows={f'disk_{k}':v for k,v in self.ds.boundary_heat_flows(td).items()}
        flows.update({f'plate_{k}':v for k,v in self.ps.boundary_heat_flows(tp).items()})
        pin=float(np.sum(source));error=pin-sum(flows.values())-(0 if dt is None else delta_energy/dt)
        return AssemblyTemperature(td,tp,flux,face_d,face_p,pin,flows,float(delta_energy),float(error),float(abs(error)/max(abs(pin),sum(abs(v) for v in flows.values()),1e-15)))

    def steady(self,disk_heat_W_m3,plate_heat_W_m3=0.):
        self._check_steady();src=self._source(disk_heat_W_m3,plate_heat_W_m3)
        t=self.coolant.bath_temperature_K+spsolve(self.K,src+self.boundary_rhs)
        return self._result(t,src)

    def advance(self,disk_temperature_K,plate_temperature_K,disk_heat_W_m3,dt_s,plate_heat_W_m3=0.):
        positive(dt_s,'time step')
        t0=np.r_[self.disk.field(disk_temperature_K).ravel(),self.plate.field(plate_temperature_K).ravel()]
        if np.min(t0)<=0:raise ValueError('initial temperatures must be positive')
        src=self._source(disk_heat_W_m3,plate_heat_W_m3);mass=self.capacity/dt_s
        ref=self.coolant.bath_temperature_K
        t=ref+spsolve(self.K+diags(mass),src+self.boundary_rhs+mass*(t0-ref))
        return self._result(t,src,float(np.sum(self.capacity*(t-t0))),dt_s)


def sample_temperature(mesh:DiskThermalMesh,temperature_K,points_xyz_m,*,z_offset_m=0.):
    """Temperature at tetra centers; periodic phi, nearest half-cell extension."""
    t=mesh.field(temperature_K,'temperature');p=np.asarray(points_xyz_m,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.all(np.isfinite(p)):raise ValueError('points must be finite (n,3)')
    r=np.hypot(p[:,0],p[:,1]);z=p[:,2]-z_offset_m
    if np.any(r>mesh.r_edges_m[-1]+1e-12) or np.any(z < -1e-12) or np.any(z>mesh.z_edges_m[-1]+1e-12):
        raise ValueError('points outside thermal body')
    rr=np.clip(r,mesh.r_m[0],mesh.r_m[-1]);zz=np.clip(z,mesh.z_m[0],mesh.z_m[-1])
    if mesh.nphi==1:
        return RegularGridInterpolator((mesh.z_m,mesh.r_m),t[:,:,0],bounds_error=False,fill_value=None)(np.c_[zz,rr])
    phi=np.mod(np.arctan2(p[:,1],p[:,0]),2*np.pi)
    angles=np.r_[mesh.phi_rad[-1]-2*np.pi,mesh.phi_rad,mesh.phi_rad[0]+2*np.pi]
    ext=np.concatenate((t[:,:,-1:],t,t[:,:,:1]),axis=2)
    return RegularGridInterpolator((mesh.z_m,mesh.r_m,angles),ext,bounds_error=False,fill_value=None)(np.c_[zz,rr,phi])
