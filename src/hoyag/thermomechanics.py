"""Stage 6: three-dimensional small-strain disk/plate thermoelasticity.

Linear tetrahedral FEM, two distinct elastic bodies and an explicitly compliant
bond. The interface can transmit tensile/compressive and shear traction. This is
NOT unilateral frictional contact: no opening, Coulomb slip, plasticity or damage
is inferred. Use measured interface stiffnesses; defaults are illustrative.
Coordinates: z=0 optical front, z=d rear HR/bond, plate at d<z<d+t_plate.
"""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.sparse import coo_matrix, block_diag, diags
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@dataclass(frozen=True)
class ElasticMaterial:
    young_Pa: float
    poisson: float
    expansion_K1: float
    stress_free_temperature_K: float = 293.15
    name: str = 'configured isotropic material'

    def __post_init__(self):
        positive(self.young_Pa, 'Young modulus')
        positive(self.stress_free_temperature_K, 'stress-free temperature')
        if not np.isfinite(self.poisson) or not -1 < self.poisson < .5:
            raise ValueError('Poisson ratio must lie strictly between -1 and 0.5')
        if not np.isfinite(self.expansion_K1):
            raise ValueError('expansion coefficient must be finite')

    @property
    def matrix_Pa(self):
        """Engineering strain order xx, yy, zz, 2exy, 2exz, 2eyz."""
        mu = self.young_Pa/(2*(1+self.poisson))
        lam = self.young_Pa*self.poisson/((1+self.poisson)*(1-2*self.poisson))
        c = np.zeros((6,6))
        c[:3,:3] = lam
        c[np.arange(3),np.arange(3)] += 2*mu
        c[3:,3:] = np.eye(3)*mu
        return c


YAG_ELASTIC = ElasticMaterial(310e9, .25, 7e-6, name='YAG: Stage 0 baseline; alpha provisional')
# CDA C10100: E=17000 ksi, G=6400 ksi; nu derived from E/(2G)-1.
COPPER_ELASTIC = ElasticMaterial(117.210873e9, .328125, 16.92e-6,
                                name='C10100 copper: room-temperature isotropic approximation')


@dataclass(frozen=True)
class BondedInterface:
    normal_stiffness_Pa_m: float = 1e14
    tangential_stiffness_Pa_m: float = 1e13

    def __post_init__(self):
        positive(self.normal_stiffness_Pa_m, 'normal stiffness')
        if not np.isfinite(self.tangential_stiffness_Pa_m) or self.tangential_stiffness_Pa_m < 0:
            raise ValueError('tangential stiffness must be finite and nonnegative')

    @property
    def stiffness(self):
        return np.array([self.tangential_stiffness_Pa_m,
                         self.tangential_stiffness_Pa_m,
                         self.normal_stiffness_Pa_m])


def _polar_triangles(radii, ntheta):
    """Conforming polygonal disk; common radial prefix makes matching contacts."""
    radii = np.asarray(radii, float)
    if radii.ndim != 1 or radii[0] != 0 or np.any(np.diff(radii) <= 0):
        raise ValueError('radii must increase strictly from zero')
    integer(ntheta, 'ntheta', 8)
    phi = 2*np.pi*np.arange(ntheta)/ntheta
    xy = np.vstack((np.zeros((1,2)),
                    np.stack([radii[1:,None]*np.cos(phi),
                              radii[1:,None]*np.sin(phi)], axis=-1).reshape(-1,2)))
    triangles = []
    for j in range(ntheta):
        triangles.append([0, 1+j, 1+(j+1)%ntheta])
    for ring in range(1, len(radii)-1):
        lower = 1+(ring-1)*ntheta
        upper = 1+ring*ntheta
        for j in range(ntheta):
            k = (j+1)%ntheta
            triangles.extend(([lower+j, upper+j, upper+k], [lower+j, upper+k, lower+k]))
    return xy, np.asarray(triangles, int)


@dataclass
class TetraBody:
    nodes_m: np.ndarray
    tetrahedra: np.ndarray
    xy_m: np.ndarray
    triangles: np.ndarray
    z_planes_m: np.ndarray

    @property
    def nxy(self): return len(self.xy_m)
    @property
    def top_nodes(self): return np.arange(self.nxy)
    @property
    def bottom_nodes(self): return np.arange(len(self.nodes_m)-self.nxy, len(self.nodes_m))
    @property
    def centers_m(self): return self.nodes_m[self.tetrahedra].mean(axis=1)

    def interpolate_layers(self, nodal_values, xy_query, z_query):
        """Continuous P1 nodal recovery, transverse linear then axial linear.

        At points between the polygonal rim and its circumscribed circle, nearest
        boundary-node extrapolation is used. Stress is recovered from elements,
        so interpolation is not an exact representation of elementwise jumps.
        """
        v = np.asarray(nodal_values)
        if v.shape[0] != len(self.nodes_m) or not np.all(np.isfinite(v)):
            raise ValueError('one finite value/vector per body node is required')
        xy = np.asarray(xy_query, float).reshape(-1,2)
        z = np.atleast_1d(np.asarray(z_query,float))
        if np.min(z) < self.z_planes_m[0]-1e-12 or np.max(z) > self.z_planes_m[-1]+1e-12:
            raise ValueError('z query outside body')
        component_shape = v.shape[1:]
        layers = v.reshape(len(self.z_planes_m),self.nxy,-1)
        transverse = layers.transpose(1,0,2).reshape(self.nxy,-1)
        q = LinearNDInterpolator(self.xy_m, transverse)(xy)
        bad = np.any(~np.isfinite(q), axis=1)
        if np.any(bad):
            q[bad] = NearestNDInterpolator(self.xy_m, transverse)(xy[bad])
        q = q.reshape(len(xy),len(self.z_planes_m),-1).transpose(1,0,2)
        return interp1d(self.z_planes_m,q,axis=0)(z).reshape((len(z),len(xy))+component_shape)


def _extrude(xy, triangles, planes):
    nxy = len(xy)
    nodes = np.vstack([np.column_stack((xy, np.full(nxy,z))) for z in planes])
    # Sorted prism vertices give a globally consistent shared-face diagonal.
    tri = np.sort(triangles, axis=1)
    tets = []
    for layer in range(len(planes)-1):
        a,b,c = (tri+layer*nxy).T
        aa,bb,cc = a+nxy,b+nxy,c+nxy
        tets.extend((np.column_stack((a,b,c,cc)),
                     np.column_stack((a,b,bb,cc)),
                     np.column_stack((a,aa,bb,cc))))
    return TetraBody(nodes,np.vstack(tets),xy,triangles,np.asarray(planes))


@dataclass
class DiskPlateMesh:
    disk: TetraBody
    plate: TetraBody
    radius_m: float
    disk_thickness_m: float
    plate_radius_m: float
    plate_thickness_m: float

    @classmethod
    def make(cls, radius_m=5e-3, disk_thickness_m=1e-3,
             plate_radius_m=10e-3, plate_thickness_m=3e-3,
             nr=8, outer_rings=4, ntheta=32, nz_disk=4, nz_plate=4,
             radial_exponent=1.6):
        for name,n in [('nr',nr),('outer_rings',outer_rings),('nz_disk',nz_disk),('nz_plate',nz_plate)]:
            integer(n,name)
        for name,v in [('radius',radius_m),('disk thickness',disk_thickness_m),
                       ('plate radius',plate_radius_m),('plate thickness',plate_thickness_m),
                       ('radial exponent',radial_exponent)]: positive(v,name)
        if plate_radius_m < radius_m: raise ValueError('plate must cover the disk face')
        rd = radius_m*np.linspace(0,1,nr+1)**radial_exponent
        rp = rd if plate_radius_m == radius_m else np.r_[rd,np.linspace(radius_m,plate_radius_m,outer_rings+1)[1:]]
        xd,td = _polar_triangles(rd,ntheta)
        xp,tp = _polar_triangles(rp,ntheta)
        d = _extrude(xd,td,np.linspace(0,disk_thickness_m,nz_disk+1))
        p = _extrude(xp,tp,disk_thickness_m+np.linspace(0,plate_thickness_m,nz_plate+1))
        return cls(d,p,radius_m,disk_thickness_m,plate_radius_m,plate_thickness_m)


def tetra_kinematics(body):
    xyz = body.nodes_m[body.tetrahedra]
    edges = xyz[:,1:]-xyz[:,:1]
    volume = np.abs(np.linalg.det(edges))/6
    if np.any(volume <= 0): raise ValueError('degenerate tetrahedron')
    # Gradients of barycentric functions; using edge matrices avoids mixing
    # SI lengths with a column of ones in an ill-scaled 4x4 inverse.
    inv = np.linalg.inv(edges)
    grad = np.empty((len(xyz),4,3))
    grad[:,1:,:] = inv.transpose(0,2,1)
    grad[:,0,:] = -grad[:,1:,:].sum(axis=1)
    b = np.zeros((len(xyz),6,12))
    for j in range(4):
        gx,gy,gz = grad[:,j].T
        b[:,0,3*j] = gx; b[:,1,3*j+1] = gy; b[:,2,3*j+2] = gz
        b[:,3,3*j] = gy; b[:,3,3*j+1] = gx
        b[:,4,3*j] = gz; b[:,4,3*j+2] = gx
        b[:,5,3*j+1] = gz; b[:,5,3*j+2] = gy
    return volume,b


def assemble_body(body, material, temperature_K):
    v,b = tetra_kinematics(body)
    temperature = np.broadcast_to(np.asarray(temperature_K,float),(len(v),))
    if not np.all(np.isfinite(temperature)) or np.any(temperature<=0):
        raise ValueError('element temperatures must be finite and positive')
    c = material.matrix_Pa
    thermal = np.zeros((len(v),6))
    thermal[:,:3] = (material.expansion_K1*(temperature-material.stress_free_temperature_K))[:,None]
    ke = np.einsum('eai,ab,ebj,e->eij',b,c,b,v,optimize=True)
    fe = np.einsum('eai,ab,eb,e->ei',b,c,thermal,v,optimize=True)
    dof = (3*body.tetrahedra[:,:,None]+np.arange(3)).reshape(-1,12)
    rows = np.broadcast_to(dof[:,:,None],ke.shape).ravel()
    cols = np.broadcast_to(dof[:,None,:],ke.shape).ravel()
    k = coo_matrix((ke.ravel(),(rows,cols)),shape=(3*len(body.nodes_m),)*2).tocsc()
    load = np.zeros(k.shape[0]); np.add.at(load,dof.ravel(),fe.ravel())
    return k,load,(v,b,thermal,dof)


def _contact_matrix(mesh, interface, quality=1.):
    nd = len(mesh.disk.nodes_m)
    n = 3*(nd+len(mesh.plate.nodes_m))
    tri = mesh.disk.triangles
    xy = mesh.disk.xy_m[tri]
    ab = xy[:,1]-xy[:,0]; ac = xy[:,2]-xy[:,0]
    area = .5*np.abs(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0])
    q = np.broadcast_to(np.asarray(quality,float),area.shape)
    if np.any(~np.isfinite(q)) or np.any(q<=0) or np.any(q>1):
        raise ValueError('bond quality must be in (0,1]; zero/open contact is not supported')
    mass = area[:,None,None]/12*(np.ones((3,3))+np.eye(3))[None]
    dn = tri+(len(mesh.disk.z_planes_m)-1)*mesh.disk.nxy
    pn = tri+nd  # matching XY prefix in the plate front face
    rows=[];cols=[];vals=[]
    for a,ka in enumerate(interface.stiffness):
        if ka==0: continue
        for first,second,sign in [(dn,dn,1),(pn,pn,1),(dn,pn,-1),(pn,dn,-1)]:
            rows.append(np.broadcast_to((3*first+a)[:,:,None],mass.shape).ravel())
            cols.append(np.broadcast_to((3*second+a)[:,None,:],mass.shape).ravel())
            vals.append((sign*ka*q[:,None,None]*mass).ravel())
    k = coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n)).tocsc()
    return k,area,q


def _gauge_dofs(body, offset=0):
    # Six minimally sufficient rigid-body gauges on the bottom plane. Uniform
    # thermal expansion relative to this origin satisfies all six constraints.
    root = int(body.bottom_nodes[0])+offset
    on_x = int(body.bottom_nodes[np.argmax(body.xy_m[:,0])])+offset
    on_y = int(body.bottom_nodes[np.argmax(body.xy_m[:,1])])+offset
    return [3*root,3*root+1,3*root+2,3*on_x+1,3*on_x+2,3*on_y+2]


def _recover(body, u, material, cache):
    vol,b,eth,dof = cache
    strain = np.einsum('eai,ei->ea',b,u.reshape(-1)[dof])
    stress = (strain-eth) @ material.matrix_Pa.T
    nodal = np.zeros((len(body.nodes_m),6)); weight=np.zeros(len(body.nodes_m))
    for i in range(4):
        np.add.at(nodal,body.tetrahedra[:,i],stress*vol[:,None])
        np.add.at(weight,body.tetrahedra[:,i],vol)
    nodal /= weight[:,None]
    return strain,stress,nodal


def stress_tensor(stress_voigt):
    s = np.asarray(stress_voigt,float)
    if s.shape[-1]!=6: raise ValueError('stress order is xx,yy,zz,xy,xz,yz')
    out=np.zeros(s.shape[:-1]+(3,3))
    out[...,0,0]=s[...,0];out[...,1,1]=s[...,1];out[...,2,2]=s[...,2]
    out[...,0,1]=out[...,1,0]=s[...,3]
    out[...,0,2]=out[...,2,0]=s[...,4]
    out[...,1,2]=out[...,2,1]=s[...,5]
    return out


def von_mises(stress_voigt):
    s=np.asarray(stress_voigt,float)
    return np.sqrt(.5*((s[...,0]-s[...,1])**2+(s[...,1]-s[...,2])**2+(s[...,2]-s[...,0])**2)
                   +3*np.sum(s[...,3:]**2,axis=-1))


@dataclass
class AssemblyDisplacement:
    disk_u_m: np.ndarray
    plate_u_m: np.ndarray
    disk_strain: np.ndarray
    plate_strain: np.ndarray
    disk_stress_Pa: np.ndarray
    plate_stress_Pa: np.ndarray
    disk_nodal_stress_Pa: np.ndarray
    plate_nodal_stress_Pa: np.ndarray
    interface_jump_m: np.ndarray
    interface_traction_on_disk_Pa: np.ndarray
    interface_force_on_disk_N: np.ndarray
    support_reaction_N: np.ndarray
    free_residual_relative: float
    thermal_load_work_J: float
    displacement_quadratic_energy_J: float
    strain_and_bond_energy_J: float


def solve_disk_plate(mesh: DiskPlateMesh, disk_temperature_K, plate_temperature_K, *,
                     disk_material=YAG_ELASTIC, plate_material=COPPER_ELASTIC,
                     interface=BondedInterface(), bond_quality=1.,
                     support='clamped', front_pressure_Pa=0.):
    """Static 3-D assembly. Temperatures: scalar or one value per tetrahedron.

    support='clamped': plate underside fixed in xyz.
    support='roller': underside uz=0, in-plane rigid-body gauges only.
    support='free': only six rigid-body gauges for the whole bonded assembly.
    kt=0 is maintained normal contact with in-plane slip, NOT Coulomb friction.
    """
    if support not in ('clamped','roller','free'): raise ValueError('unknown support')
    if not np.isfinite(front_pressure_Pa) or front_pressure_Pa<0: raise ValueError('pressure must be nonnegative')
    kd,fd,cd = assemble_body(mesh.disk,disk_material,disk_temperature_K)
    kp,fp,cp = assemble_body(mesh.plate,plate_material,plate_temperature_K)
    kc,face_area,q = _contact_matrix(mesh,interface,bond_quality)
    k = block_diag((kd,kp),format='csc')+kc
    f = np.r_[fd,fp]
    # Positive z points into the cooler; compressive front loading is +z.
    for j in range(3):
        np.add.at(f,3*mesh.disk.triangles[:,j]+2,front_pressure_Pa*face_area/3)
    nd=len(mesh.disk.nodes_m)
    if support=='clamped':
        fixed=(3*(mesh.plate.bottom_nodes+nd)[:,None]+np.arange(3)).ravel().tolist()
    elif support=='roller':
        fixed=(3*(mesh.plate.bottom_nodes+nd)+2).tolist()+_gauge_dofs(mesh.plate,nd)[:2]+[_gauge_dofs(mesh.plate,nd)[3]]
    else: fixed=_gauge_dofs(mesh.plate,nd)
    if interface.tangential_stiffness_Pa_m==0:
        # Remove only disk in-plane translations and yaw, which are otherwise
        # indeterminate for a frictionless, maintained-contact interface.
        g=_gauge_dofs(mesh.disk)
        fixed+=g[:2]+[g[3]]
    fixed=np.unique(fixed)
    free=np.setdiff1d(np.arange(len(f)),fixed)
    diagonal=k.diagonal()[free]
    if np.any(diagonal<=0): raise ValueError('unconstrained or invalid stiffness')
    scale=1/np.sqrt(diagonal)
    a=diags(scale)@k[free][:,free]@diags(scale)
    u=np.zeros(len(f))
    with warnings.catch_warnings():
        warnings.simplefilter('error',MatrixRankWarning)
        u[free]=scale*spsolve(a,scale*f[free])
    if not np.all(np.isfinite(u)): raise FloatingPointError('singular mechanical system')
    residual=k@u-f
    relative=float(np.linalg.norm(residual[free])/max(np.linalg.norm(f[free]),1e-30))
    if relative>1e-6: raise RuntimeError('mechanical residual too large')
    ud=u[:3*nd].reshape(-1,3);up=u[3*nd:].reshape(-1,3)
    ed,sd,nsd=_recover(mesh.disk,ud,disk_material,cd)
    ep,sp,nsp=_recover(mesh.plate,up,plate_material,cp)
    jump=ud[mesh.disk.bottom_nodes]-up[:mesh.disk.nxy]
    traction=-jump[mesh.disk.triangles].mean(axis=1)*interface.stiffness*q[:,None]
    net=(traction*face_area[:,None]).sum(axis=0)
    reaction=residual.reshape(-1,3).sum(axis=0)
    return AssemblyDisplacement(ud,up,ed,ep,sd,sp,nsd,nsp,jump,traction,net,reaction,relative,
                               float(u@f),float(.5*u@(k@u)),
                               float(.5*np.sum((ed-cd[2])*sd*cd[0][:,None])
                                     +.5*np.sum((ep-cp[2])*sp*cp[0][:,None])+.5*u@(kc@u)))
