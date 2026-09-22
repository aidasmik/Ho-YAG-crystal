"""Stage 5A: signed lattice heat from the four Ho manifolds, in SI units.

Q = absorbed pump - stimulated signal extraction - spontaneous optical energy
    - d(ionic stored energy)/dt.

ETU, cross-relaxation and multiphonon terms are already in that identity. They
must NOT be added to it a second time. Positive Q heats the lattice.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .populations import H, C0, I5, I6, I7, I8, HoYAGFourLevelParams, four_level_rhs

# Means of the legacy Stark lists committed in data/ho_yag_stark_levels.json.
# These are bookkeeping energies, NOT measured, line-strength-weighted spectra.
LEGACY_CENTROIDS_CM1 = (
    125106.0 / 11, 114484.0 / 13, 80118.0 / 15, 5003.0 / 17,
)
BRANCHES = ((I5, I6), (I5, I7), (I5, I8), (I6, I7), (I6, I8), (I7, I8))


@dataclass(frozen=True)
class ManifoldEnergies:
    """I5/I6/I7/I8 energies; optional six independent mean photon energies.

    The default centroid approximation is explicitly provisional. Changing the
    common zero of all manifold energies cannot change heat or stored changes.
    """
    centers_cm1: tuple[float, ...] = LEGACY_CENTROIDS_CM1
    photon_wavenumbers_cm1: tuple[float, ...] | None = None
    provenance: str = "legacy unweighted Stark centroids; provisional"

    def __post_init__(self):
        e = np.asarray(self.centers_cm1, float)
        if e.shape != (4,) or not np.all(np.isfinite(e)) or not np.all(np.diff(e) < 0):
            raise ValueError("require finite ordered I5 > I6 > I7 > I8 energies")
        if self.photon_wavenumbers_cm1 is not None:
            v = np.asarray(self.photon_wavenumbers_cm1, float)
            if v.shape != (6,) or np.any(~np.isfinite(v)) or np.any(v <= 0):
                raise ValueError("six finite positive branch photon wavenumbers required")

    @classmethod
    def from_stark_levels(cls, levels: dict) -> ManifoldEnergies:
        return cls(tuple(float(np.mean(levels[k])) for k in ("I5", "I6", "I7", "I8")))

    @property
    def joules(self) -> np.ndarray:
        c = np.asarray(self.centers_cm1, float)
        return (c - c[I8]) * (100 * H * C0)

    @property
    def photons_J(self) -> np.ndarray:
        if self.photon_wavenumbers_cm1 is not None:
            return np.asarray(self.photon_wavenumbers_cm1) * (100 * H * C0)
        e = self.joules
        return np.array([e[i] - e[j] for i, j in BRANCHES])


def stored_energy_density(populations, energies: ManifoldEnergies | None = None):
    n = np.asarray(populations, float)
    if n.ndim < 1 or n.shape[0] != 4 or np.any(~np.isfinite(n)):
        raise ValueError("finite populations with leading I5/I6/I7/I8 axis required")
    return np.tensordot((energies or ManifoldEnergies()).joules, n, axes=(0, 0))


def heat_rates(populations, params: HoYAGFourLevelParams | None = None, *,
               pump_intensity_W_m2=0.0, signal_intensity_W_m2=0.0,
               pump_absorption_m2: float | None = None,
               energies: ManifoldEnergies | None = None,
               cavity_spontaneous_fraction: float = 0.0) -> dict[str, np.ndarray]:
    """Local signed heat and separate optical/storage terms, all in W/m^3.

    Intensity includes the SUM of directional waves, not their coherent fringe
    pattern. All non-cavity fluorescence escapes in this first implementation.
    cavity_spontaneous_fraction counts ALL selected modes together. That tiny
    share has photon energy h*nu_laser and is NOT also counted as escaping light.
    """
    p = params or HoYAGFourLevelParams()
    e = energies or ManifoldEnergies()
    if not np.isclose(p.beta78, 1.0):
        raise ValueError("the base four-manifold RHS requires beta78=1")
    n = np.asarray(populations, float)
    if n.ndim < 1 or n.shape[0] != 4 or np.any(~np.isfinite(n)):
        raise ValueError("invalid population array")
    if np.min(n) < -1e-10 * max(float(np.max(np.abs(n))), 1.0):
        raise ValueError("negative populations")
    ip, il = np.broadcast_arrays(np.asarray(pump_intensity_W_m2, float),
                                np.asarray(signal_intensity_W_m2, float))
    if np.any(~np.isfinite(ip)) or np.any(~np.isfinite(il)) or np.any(ip < 0) or np.any(il < 0):
        raise ValueError("intensities must be finite and nonnegative")
    if not np.isfinite(cavity_spontaneous_fraction) or not 0 <= cavity_spontaneous_fraction <= 1:
        raise ValueError("invalid spontaneous cavity fraction")
    sa = p.sigma_abs_pump_m2 if pump_absorption_m2 is None else pump_absorption_m2
    if not np.isfinite(sa) or sa < 0:
        raise ValueError("invalid pump cross section")
    ep, el = H*C0/p.pump_wavelength_m, H*C0/p.laser_wavelength_m
    e5, e6, e7, e8 = e.joules
    a, b, c, d = n
    rp = (sa*d - p.sigma_em_pump_m2*c) * ip / ep
    rl = (p.sigma_em_laser_m2*c - p.sigma_abs_laser_m2*d) * il / el
    transitions = (a*p.beta56/p.tau5_s, a*p.beta57/p.tau5_s,
                   a*p.beta58/p.tau5_s, b*p.beta67/p.tau6_s,
                   b*p.beta68/p.tau6_s, c*p.beta78/p.tau7_s)
    spontaneous = sum(r*energy for r, energy in zip(transitions, e.photons_J))
    cavity_sp = cavity_spontaneous_fraction * transitions[-1] * el
    escape = spontaneous - cavity_spontaneous_fraction * transitions[-1] * e.photons_J[-1]
    mp = (e5-e6)*p.M56_s1*a + (e6-e7)*p.M67_s1*b + (e7-e8)*p.M78_s1*c
    etu = (2*e7-e5-e8)*p.k75_m3_s*c*c + (2*e7-e6-e8)*p.k76_m3_s*c*c
    cr = (e5+e8-2*e7)*p.C57_m3_s*a*d + (e6+e8-2*e7)*p.C67_m3_s*b*d
    sp_defect = sum(r*(e.joules[i]-e.joules[j]-energy)
                    for r, (i, j), energy in zip(transitions, BRANCHES, e.photons_J))
    sp_defect += cavity_spontaneous_fraction * transitions[-1] * (e.photons_J[-1]-el)
    pump_defect = rp*(ep-e7+e8)
    laser_defect = rl*(e7-e8-el)
    q = mp + etu + cr + sp_defect + pump_defect + laser_defect
    dn = four_level_rhs(n, p, sa*ip/ep, p.sigma_em_pump_m2*ip/ep,
                        p.sigma_abs_laser_m2*il/el, p.sigma_em_laser_m2*il/el)
    du = stored_energy_density(dn, e)
    return {
        "heat": q, "pump_absorbed": ep*rp, "signal_extracted": el*rl,
        "fluorescence_escape": escape, "spontaneous_to_cavity": cavity_sp,
        "stored_rate": du, "multiphonon": mp, "etu": etu, "cross_relaxation": cr,
        "pump_quantum_defect": pump_defect, "laser_quantum_defect": laser_defect,
        "spontaneous_quantum_defect": sp_defect,
        "balance_residual": ep*rp-el*rl-escape-cavity_sp-du-q,
    }


def pulse_heat_from_states(before, after, absorbed_pump_J_m3,
                           extracted_signal_J_m3=0.0, spontaneous_J_m3=0.0, *,
                           energies: ManifoldEnergies | None = None):
    """Finite-event first law, J/m^3. Stored inversion is NOT prompt heat."""
    return (np.asarray(absorbed_pump_J_m3) - np.asarray(extracted_signal_J_m3)
            - np.asarray(spontaneous_J_m3)
            - stored_energy_density(np.asarray(after)-np.asarray(before), energies))
