"""Display gradients/fluxes on the cylindrical thermal mesh.

These cell-center derivatives are visualization diagnostics. Conservative face
fluxes from the heat solver remain authoritative for energy accounting.
"""
from __future__ import annotations
import numpy as np


def cylindrical_temperature_derivatives(temperature_K, r_edges_m, z_edges_m,
                                        conductivity_W_mK):
    t=np.asarray(temperature_K,float)
    r=np.asarray(r_edges_m,float);z=np.asarray(z_edges_m,float)
    if t.ndim!=3 or t.shape[:2]!=(len(z)-1,len(r)-1):
        raise ValueError('temperature must have (z,r,phi) cell shape')
    if t.shape[0]<2 or t.shape[1]<2 or (t.shape[2]!=1 and t.shape[2]<3):
        raise ValueError('mesh does not resolve requested temperature gradient')
    if np.any(np.diff(r)<=0) or np.any(np.diff(z)<=0) or np.any(~np.isfinite(t)):
        raise ValueError('invalid thermal mesh or temperature')
    rc=(r[:-1]+r[1:])/2;zc=(z[:-1]+z[1:])/2
    dTdz=np.gradient(t,zc,axis=0,edge_order=1)
    dTdr=np.gradient(t,rc,axis=1,edge_order=1)
    dTdphi=(np.zeros_like(t) if t.shape[2]==1 else
            (np.roll(t,-1,axis=2)-np.roll(t,1,axis=2))/(4*np.pi/t.shape[2]))
    grad=np.stack((dTdr,dTdphi/rc[None,:,None],dTdz),axis=-1)
    conductivity=np.asarray(conductivity_W_mK,float)
    if conductivity.shape==():
        if conductivity<=0:raise ValueError('conductivity must be positive')
        flux=-conductivity*grad
    elif conductivity.shape==(3,3):
        if np.linalg.eigvalsh(conductivity).min()<=0:
            raise ValueError('conductivity tensor must be positive definite')
        flux=-np.einsum('ij,...j->...i',conductivity,grad)
    else:
        raise ValueError('conductivity must be scalar or 3x3 cylindrical tensor')
    return {'grad_T_K_m':grad,'grad_T_magnitude_K_m':np.linalg.norm(grad,axis=-1),
            'heat_flux_W_m2':flux,'coordinate_order':'r,phi,z',
            'energy_accounting':'use conservative heat-solver face fluxes'}
