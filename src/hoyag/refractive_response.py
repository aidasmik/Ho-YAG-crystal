"""Optional measured Ho concentration and population index increments.

No Ho:YAG coefficient is bundled: YAG host n and dn/dT do not determine these
increments. Coefficients must be measured at the signal wavelength and have
units m^3/ion. This module does not replace thermal or photoelastic OPD.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .inhomogeneity import HoDensityField
from .populations import I7


@dataclass(frozen=True)
class HoIndexResponse:
    dn_dHo_m3: float | None = None
    dn_dExcited_m3: float | None = None
    provenance: str = ''

    def __post_init__(self):
        for name in ('dn_dHo_m3','dn_dExcited_m3'):
            value=getattr(self,name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if (self.dn_dHo_m3 is not None or self.dn_dExcited_m3 is not None) and not self.provenance.strip():
            raise ValueError('index coefficients require measured/source provenance')

    def single_pass_opd_m(self,density: HoDensityField,populations_before_signal=None):
        n=density.values_m3
        opd=np.zeros(n.shape[1:])
        if self.dn_dHo_m3 is not None:
            active=n>0
            reference=float(n[active].mean()) if np.any(active) else 0.
            opd+=self.dn_dHo_m3*np.sum(np.where(active,n-reference,0.),axis=0)*density.dz_m
        if self.dn_dExcited_m3 is not None:
            if populations_before_signal is None:
                raise ValueError('excited-state lens requires pre-signal populations')
            state=np.asarray(populations_before_signal,float)
            if state.shape!=(4,*n.shape):
                raise ValueError('population shape does not match Ho map')
            opd+=self.dn_dExcited_m3*np.sum(state[I7],axis=0)*density.dz_m
        return opd
