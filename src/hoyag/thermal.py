"""Stage 5B: conservative finite-volume heat diffusion in a circular disk.

Mesh order (z,r,phi). nphi=1 is axisymmetric; nphi>=3 resolves asymmetric 3-D
heating. z=0 is the front optical face, z=thickness is the cooled rear HR face.
No air cells, square corners, or rod-mount W/K constants are used.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

@dataclass(frozen=True)
class DiskThermalMesh:
    r_edges_m: np.ndarray
    z_edges_m: np.ndarray
    nphi: int = 1

    def __post_init__(self):
        for name in ('r_edges_m','z_edges_m'):
            a=np.asarray(getattr(self,name),float).copy()
            if a.ndim!=1 or len(a)<2 or not np.all(np.isfinite(a)) or a[0]!=0 or np.any(np.diff(a)<=0):
                raise ValueError(f'{name} must start at zero and increase strictly')
            a.setflags(write=False); object.__setattr__(self,name,a)
        if isinstance(self.nphi,bool) or not isinstance(self.nphi,(int,np.integer)) or (self.nphi!=1 and self.nphi<3):
            raise ValueError('nphi must be 1 (axisymmetric) or an integer >=3')

    @classmethod
    def disk(cls, nr=64, nz=12, nphi=1, radius_m=5e-3, thickness_m=1e-3, radial_exponent=1.5):
        for name,n in (('nr',nr),('nz',nz)):
            if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<1: raise ValueError(f'{name} must be a positive integer')
        if not np.all(np.isfinite([radius_m,thickness_m,radial_exponent])) or min(radius_m,thickness_m,radial_exponent)<=0:
            raise ValueError('mesh lengths/exponent must be finite and positive')
        return cls(radius_m*np.linspace(0,1,nr+1)**radial_exponent,np.linspace(0,thickness_m,nz+1),nphi)

    @property
    def nr(self): return len(self.r_edges_m)-1
    @property
    def nz(self): return len(self.z_edges_m)-1
    @property
    def shape(self): return (self.nz,self.nr,self.nphi)
    @property
    def r_m(self): return .5*(self.r_edges_m[:-1]+self.r_edges_m[1:])
    @property
    def z_m(self): return .5*(self.z_edges_m[:-1]+self.z_edges_m[1:])
    @property
    def phi_rad(self): return (np.arange(self.nphi)+.5)*2*np.pi/self.nphi
    @property
    def face_areas_m2(self):
        ring=np.pi*np.diff(self.r_edges_m**2)/self.nphi
        return np.broadcast_to(ring[:,None],(self.nr,self.nphi)).copy()
    @property
    def volumes_m3(self): return np.diff(self.z_edges_m)[:,None,None]*self.face_areas_m2[None]

    def field(self, value, name='field'):
        try: a=np.broadcast_to(np.asarray(value,float),self.shape).copy()
        except ValueError as e: raise ValueError(f'{name} must broadcast to {self.shape}') from e
        if not np.all(np.isfinite(a)): raise ValueError(f'{name} must be finite')
        return a

@dataclass(frozen=True)
class ThermalBoundary:
    """Outward flux h(T_surface-T_bath); h=0 insulated, h=inf Dirichlet."""
    bath_temperature_K: float = 293.15
    conductance_W_m2K: float = 0.0
    def __post_init__(self):
        if not np.isfinite(self.bath_temperature_K) or self.bath_temperature_K<=0:
            raise ValueError('bath temperature must be finite and positive')
        h=self.conductance_W_m2K
        if np.isnan(h) or h<0: raise ValueError('boundary conductance must be >=0, optionally +inf')

@dataclass(frozen=True)
class DiskCooling:
    rear: ThermalBoundary = field(default_factory=lambda:ThermalBoundary(293.15,1e5))
    front: ThermalBoundary = field(default_factory=ThermalBoundary)
    rim: ThermalBoundary = field(default_factory=ThermalBoundary)

@dataclass
class ThermalSolution:
    temperature_K: np.ndarray
    input_heat_W: float
    boundary_heat_W: dict[str,float]
    stored_energy_change_J: float
    balance_error_W: float
    relative_balance_error: float

class DiskHeatSolver:
    """Linear conservative diffusion, including finite half-cell contact resistance.

    k may be a scalar or a positive field [W/(m K)]. rho*Cp uses SI units.
    Steady solutions require a heat-removing boundary. A completely insulated
    transient is allowed. Material coefficients are constant within each solve.
    """
    def __init__(self, mesh: DiskThermalMesh, conductivity_W_mK=14.0,
                 cooling: DiskCooling | None=None, *, density_kg_m3=4560.,
                 heat_capacity_J_kgK=680., front_heat_flux_W_m2=0., rear_heat_flux_W_m2=0.):
        self.mesh=mesh; self.cooling=cooling or DiskCooling()
        self.k=mesh.field(conductivity_W_mK,'conductivity')
        if np.any(self.k<=0): raise ValueError('conductivity must be positive')
        rho=mesh.field(density_kg_m3,'mass density');cp=mesh.field(heat_capacity_J_kgK,'heat capacity')
        if np.any(rho<=0) or np.any(cp<=0): raise ValueError('mass density and heat capacity must be positive')
        self.capacity_J_K=rho*cp*mesh.volumes_m3
        self.boundaries={}; self._boundary_rhs=np.zeros(mesh.shape); self._flux_rhs=np.zeros(mesh.shape)
        self._surface_flux={}; self._direct_surface_heat={}; self._surface_power_W=0.
        for name,flux,iz in (('front',front_heat_flux_W_m2,0),('rear',rear_heat_flux_W_m2,-1)):
            a=np.broadcast_to(np.asarray(flux,float),(mesh.nr,mesh.nphi))
            if not np.all(np.isfinite(a)): raise ValueError(f'{name} heat flux must be finite')
            self._surface_flux[name]=a.copy()
            self._surface_power_W+=float(np.sum(a*mesh.face_areas_m2))
        self.K=self._assemble()

    def _assemble(self):
        m=self.mesh;k=self.k; ids=np.arange(np.prod(m.shape)).reshape(m.shape)
        rows=[];cols=[];data=[]
        def edge(a,b,g):
            a,b,g=np.broadcast_arrays(a,b,g);a=a.ravel();b=b.ravel();g=g.ravel()
            rows.extend((a,b,a,b));cols.extend((a,b,b,a));data.extend((g,g,-g,-g))
        dr=np.diff(m.r_edges_m);dz=np.diff(m.z_edges_m);dphi=2*np.pi/m.nphi
        if m.nr>1:
            area=dz[:,None,None]*m.r_edges_m[None,1:-1,None]*dphi
            resistance=(dr[None,:-1,None]/(2*k[:,:-1]) + dr[None,1:,None]/(2*k[:,1:]))
            edge(ids[:,:-1],ids[:,1:],area/resistance)
        if m.nz>1:
            resistance=dz[:-1,None,None]/(2*k[:-1])+dz[1:,None,None]/(2*k[1:])
            edge(ids[:-1],ids[1:],m.face_areas_m2[None]/resistance)
        if m.nphi>1:
            other=np.roll(k,-1,axis=2)
            distance=m.r_m[None,:,None]*dphi
            area=dz[:,None,None]*dr[None,:,None]
            conductance=area/(distance/(2*k)+distance/(2*other))
            edge(ids,np.roll(ids,-1,axis=2),conductance)
        diag=np.zeros(m.shape)
        for name,sl,area,distance in (
            ('front',(0,slice(None),slice(None)),m.face_areas_m2,dz[0]/2),
            ('rear',(-1,slice(None),slice(None)),m.face_areas_m2,dz[-1]/2),
            ('rim',(slice(None),-1,slice(None)),np.broadcast_to((dz*m.r_edges_m[-1]*dphi)[:,None],(m.nz,m.nphi)),dr[-1]/2),
        ):
            bc=getattr(self.cooling,name);h=bc.conductance_W_m2K
            g=np.zeros_like(k[sl]) if h==0 else area/(distance/k[sl]+(0 if np.isinf(h) else 1/h))
            diag[sl]+=g;self._boundary_rhs[sl]+=g*bc.bath_temperature_K
            self.boundaries[name]=(sl,g,bc.bath_temperature_K)
            # Surface heat splits between the crystal and its cooling contact.
            # Eliminating the surface temperature gives this half-cell factor.
            flux=self._surface_flux.get(name,0.)
            fraction=0. if np.isinf(h) else 1/(1+h*distance/k[sl])
            self._flux_rhs[sl]+=flux*area*fraction
            self._direct_surface_heat[name]=float(np.sum(flux*area*(1-fraction)))
        rows.append(ids.ravel());cols.append(ids.ravel());data.append(diag.ravel())
        n=ids.size
        return coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n)).tocsc()

    def boundary_heat_flows(self, temperature_K):
        t=self.mesh.field(temperature_K,'temperature')
        return {name:float(np.sum(g*(t[sl]-bath)))+self._direct_surface_heat[name] for name,(sl,g,bath) in self.boundaries.items()}

    def _result(self,t,source,delta_energy=0.,dt=None):
        if not np.all(np.isfinite(t)) or np.min(t)<=0:
            raise FloatingPointError('thermal solution is non-finite or below absolute zero')
        pin=float(np.sum(source*self.mesh.volumes_m3)+self._surface_power_W)
        flows=self.boundary_heat_flows(t)
        error=pin-sum(flows.values())-(0 if dt is None else delta_energy/dt)
        return ThermalSolution(t,pin,flows,float(delta_energy),float(error),float(abs(error)/max(abs(pin),sum(abs(v) for v in flows.values()),1e-15)))

    def steady(self,source_W_m3) -> ThermalSolution:
        q=self.mesh.field(source_W_m3,'heat source')
        if not any(np.any(g>0) for _,g,_ in self.boundaries.values()):
            raise ValueError('steady insulated disk has no unique temperature; add cooling')
        reference=self.cooling.rear.bath_temperature_K
        boundary=np.zeros(self.mesh.shape)
        for sl,g,bath in self.boundaries.values(): boundary[sl]+=g*(bath-reference)
        b=q*self.mesh.volumes_m3+boundary+self._flux_rhs
        t=reference+spsolve(self.K,b.ravel()).reshape(self.mesh.shape)
        return self._result(t,q)

    def advance(self,temperature_K,source_W_m3,dt_s) -> ThermalSolution:
        """Backward Euler; budget uses new-time outward flux, matching the scheme."""
        if not np.isfinite(dt_s) or dt_s<=0: raise ValueError('dt_s must be finite and positive')
        t0=self.mesh.field(temperature_K,'initial temperature');q=self.mesh.field(source_W_m3,'heat source')
        if np.min(t0)<=0: raise ValueError('initial temperature must be positive')
        mass=self.capacity_J_K/dt_s
        matrix=self.K+diags(mass.ravel())
        reference=self.cooling.rear.bath_temperature_K
        boundary=np.zeros(self.mesh.shape)
        for sl,g,bath in self.boundaries.values(): boundary[sl]+=g*(bath-reference)
        b=q*self.mesh.volumes_m3+boundary+self._flux_rhs+mass*(t0-reference)
        t=reference+spsolve(matrix,b.ravel()).reshape(self.mesh.shape)
        delta=float(np.sum(self.capacity_J_K*(t-t0)))
        return self._result(t,q,delta,dt_s)


def nonlinear_steady(mesh,source_W_m3,conductivity_of_temperature,cooling=None,*,
                     initial_temperature_K=293.15,max_iterations=100,tolerance_K=1e-6,relaxation=.6):
    """Optional user-supplied k(T); no unverified Debye curve is silently used."""
    if not 0<relaxation<=1 or tolerance_K<=0 or max_iterations<1:
        raise ValueError('invalid nonlinear iteration settings')
    t=mesh.field(initial_temperature_K)
    for i in range(max_iterations):
        solver=DiskHeatSolver(mesh,conductivity_of_temperature(t),cooling)
        candidate=solver.steady(source_W_m3)
        error=float(np.max(abs(candidate.temperature_K-t)))
        t+=relaxation*(candidate.temperature_K-t)
        if error<tolerance_K:
            # Check the residual against conductivity at the returned temperature.
            final=DiskHeatSolver(mesh,conductivity_of_temperature(t),cooling)
            result=final._result(t,mesh.field(source_W_m3))
            return result,i+1
    raise RuntimeError('k(T) fixed point did not converge')
