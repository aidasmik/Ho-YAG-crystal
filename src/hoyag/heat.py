"""Stage 5A: local energy accounting for the four Ho manifolds.

Q = P_pump,net - P_signal,net - dU_ions/dt - P_fluorescence.
All quantities are W/m^3. Pump/laser powers are *net* stimulated transfers.
No output-coupler power or quantum-defect heat is subtracted/added a second time.

Default energies are a flagged centroid surrogate from Stage 0.1, not measured
line-strength-weighted fluorescence energies. Radiation transport, reabsorption,
and thermalization of an individual Stark manifold are not resolved here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .populations import C0, H, I5, I6, I7, I8, HoYAGFourLevelParams, four_level_rhs

# Fetched Stage 0.1 legacy Stark-level list; order is the existing population API.
_LEGACY_LEVELS = (
    (11301,11311,11322,11328,11332,11355,11382,11391,11422,11477,11485),
    (8721,8726,8735,8741,8745,8763,8773,8819,8844,8854,8869,8940,8954),
    (5229,5232,5243,5249,5303,5312,5320,5341,5352,5375,5395,5404,5418,5455,5490),
    (0,4,41,51,138,145,151,160,399,418,448,457,498,506,520,531,536),
)
_CENTROIDS = np.array([np.mean(v) for v in _LEGACY_LEVELS])
_DEFAULT_E = tuple((_CENTROIDS - _CENTROIDS[I8]) * (100*H*C0))
BRANCHES = ((I5,I6), (I5,I7), (I5,I8), (I6,I7), (I6,I8), (I7,I8))

@dataclass(frozen=True)
class HeatSpectroscopy:
    """Effective manifold energies and mean spontaneous photon energies, in J.

    Field order: I5,I6,I7,I8; branch order: 56,57,58,67,68,78.
    Replacing branch energies is supported; negative heat is never clipped.
    The default assumes each photon escapes the ion subsystem (no radiation
    transport). A cavity-seeded photon is still radiation leaving that subsystem.
    """
    manifold_energy_J: tuple[float,...] = _DEFAULT_E
    branch_photon_energy_J: tuple[float,...] | None = None
    provenance: str = 'Stage0.1 legacy unweighted centroid surrogate; unvalidated fluorescence means'

    def __post_init__(self):
        e = np.asarray(self.manifold_energy_J, float)
        if e.shape != (4,) or not np.all(np.isfinite(e)) or not e[0]>e[1]>e[2]>e[3]>=0:
            raise ValueError('require finite E5 > E6 > E7 > E8 >= 0')
        b = np.array([e[i]-e[j] for i,j in BRANCHES]) if self.branch_photon_energy_J is None else np.asarray(self.branch_photon_energy_J,float)
        if b.shape != (6,) or not np.all(np.isfinite(b)) or np.any(b<=0):
            raise ValueError('six positive finite mean branch photon energies required')
        object.__setattr__(self,'manifold_energy_J',tuple(e))
        object.__setattr__(self,'branch_photon_energy_J',tuple(b))

    @classmethod
    def from_stark_levels(cls, data: dict) -> 'HeatSpectroscopy':
        levels = data.get('levels_cm_1', data)
        centroids = np.array([np.mean(levels[k]) for k in ('I5','I6','I7','I8')])
        return cls(tuple((centroids-centroids[-1])*100*H*C0))


def _population_array(populations):
    n=np.asarray(populations,float)
    if n.ndim<1 or n.shape[0]!=4 or not np.all(np.isfinite(n)):
        raise ValueError('populations must be finite with first axis I5,I6,I7,I8')
    # BDF interpolants can have tiny negative fractions even for positive states.
    # Do not clip them: doing so would break the energy ledger. Reject macroscopic ones.
    scale=np.maximum(np.sum(np.abs(n),axis=0),1.)
    if np.any(n < -2e-7*scale):
        raise ValueError('populations are materially negative')
    return n


def ion_energy_density(populations, spectroscopy: HeatSpectroscopy | None = None):
    """Stored electronic energy U [J/m^3], relative to the configured ground level."""
    s=spectroscopy or HeatSpectroscopy()
    return np.tensordot(np.asarray(s.manifold_energy_J),_population_array(populations),axes=(0,0))


def fluorescence_power_density(populations, params=None, spectroscopy=None):
    """Spontaneous radiative power; each branch uses its own upper-state lifetime."""
    p=params or HoYAGFourLevelParams(); s=spectroscopy or HeatSpectroscopy()
    n=_population_array(populations)
    rates=(p.beta56/p.tau5_s,p.beta57/p.tau5_s,p.beta58/p.tau5_s,
           p.beta67/p.tau6_s,p.beta68/p.tau6_s,p.beta78/p.tau7_s)
    return sum(n[i]*rate*energy for (i,_),rate,energy in zip(BRANCHES,rates,s.branch_photon_energy_J))


@dataclass
class HeatBudget:
    heat_W_m3: np.ndarray
    pump_net_W_m3: np.ndarray
    signal_net_W_m3: np.ndarray
    stored_rate_W_m3: np.ndarray
    fluorescence_W_m3: np.ndarray
    background_W_m3: np.ndarray

    @property
    def closure_W_m3(self):
        return (self.heat_W_m3+self.signal_net_W_m3+self.stored_rate_W_m3+
                self.fluorescence_W_m3-self.pump_net_W_m3-self.background_W_m3)


def heat_budget(populations, params=None, spectroscopy=None, *,
                pump_intensity_W_m2=0.0, signal_intensity_W_m2=0.0,
                pump_absorption_m2=None, background_W_m3=0.0) -> HeatBudget:
    """Energy-consistent source for CW, transient, single-pulse and dark dynamics.

    ETU/CR and multiphonon heating are contained in E dot dN/dt. Endothermic
    channels may cool locally; no max(Q,0) operation is used. The material model
    should have beta78=1, as the baseline four_level_rhs assumes that value.
    """
    p=params or HoYAGFourLevelParams(); s=spectroscopy or HeatSpectroscopy()
    if not np.isclose(p.beta78,1.,rtol=0,atol=1e-12):
        raise ValueError('the inherited rate equations require beta78=1')
    n=_population_array(populations); shape=n.shape[1:]
    ip=np.broadcast_to(np.asarray(pump_intensity_W_m2,float),shape)
    il=np.broadcast_to(np.asarray(signal_intensity_W_m2,float),shape)
    bg=np.broadcast_to(np.asarray(background_W_m3,float),shape)
    for name,a in (('pump intensity',ip),('signal intensity',il),('background heat',bg)):
        if not np.all(np.isfinite(a)) or np.any(a<0): raise ValueError(f'{name} must be finite and nonnegative')
    sa=p.sigma_abs_pump_m2 if pump_absorption_m2 is None else float(pump_absorption_m2)
    if not np.isfinite(sa) or sa<0: raise ValueError('invalid pump absorption cross section')
    ep=H*C0/p.pump_wavelength_m; es=H*C0/p.laser_wavelength_m
    wa=ip*sa/ep; we=ip*p.sigma_em_pump_m2/ep
    la=il*p.sigma_abs_laser_m2/es; le=il*p.sigma_em_laser_m2/es
    dn=four_level_rhs(n,p,wa,we,la,le)
    stored=np.tensordot(np.asarray(s.manifold_energy_J),dn,axes=(0,0))
    pump=ep*(wa*n[I8]-we*n[I7])
    signal=es*(le*n[I7]-la*n[I8])
    rad=fluorescence_power_density(n,p,s)
    return HeatBudget(pump-signal-stored-rad+bg,pump,signal,stored,rad,bg)


def pump_kick_budget(before, after, pump_photon_energy_J, spectroscopy=None):
    """Net optical energy, stored energy and heat of a short I8 <-> I7 kick.

    Returns maps in J/m^3. The kick must freeze I5/I6 and conserve total density.
    Reflected/escaped light and mirror loss are NOT disk bulk heat.
    """
    s=spectroscopy or HeatSpectroscopy()
    a=_population_array(before); b=_population_array(after)
    if a.shape!=b.shape: raise ValueError('before/after shape mismatch')
    if not np.isfinite(pump_photon_energy_J) or pump_photon_energy_J<=0:
        raise ValueError('invalid pump photon energy')
    scale=np.maximum(a.sum(axis=0),1.)
    if np.any(np.abs((b-a)[:2])>1e-10*scale) or np.any(np.abs(b.sum(axis=0)-a.sum(axis=0))>1e-10*scale):
        raise ValueError('pump kick must conserve density and freeze I5/I6')
    optical=(b[I7]-a[I7])*pump_photon_energy_J
    stored=np.tensordot(np.asarray(s.manifold_energy_J),b-a,axes=(0,0))
    return {'pump_J_m3':optical,'stored_J_m3':stored,'heat_J_m3':optical-stored}
