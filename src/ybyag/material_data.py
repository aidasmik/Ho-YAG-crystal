"""Yb:YAG material inputs with explicit provenance and strict domain guards.

All temperatures are K, concentrations are at.% on Y sites. Source cross
sections use cm^2; transport helpers use SI. No simulator kernel is modified.
"""
from __future__ import annotations

import csv
import json
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "data"
META = json.loads((ROOT / 'material.json').read_text())
MECH = json.loads((ROOT / 'mechanical/tensors.json').read_text())
HC_J_M = 6.62607015e-34 * 299792458.0
KB_J_K = 1.380649e-23
HC_OVER_K_CM_K = HC_J_M / KB_J_K * 100.0


class ApproximationWarning(UserWarning):
    """The caller explicitly selected a non-calibrated approximation."""


def _array(value, name, low=None, high=None):
    x = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError(f'{name} must contain only finite numbers')
    if low is not None and np.any(x < low):
        raise ValueError(f'{name} must be >= {low}')
    if high is not None and np.any(x > high):
        raise ValueError(f'{name} must be <= {high}')
    return x


def _out(x):
    x = np.asarray(x)
    return x.item() if x.ndim == 0 else x


@lru_cache(maxsize=None)
def _csv(path):
    with (ROOT / path).open(newline='') as f:
        return tuple(csv.DictReader(f))


def site_density_m3():
    """300 K host-volume baseline. Not a concentration-dependent density fit."""
    s = META['structure']
    return s['Y_sites_per_formula'] * s['rho_host_300K_kg_m3'] / s['molar_mass_host_kg_mol'] * s['avogadro_mol-1']


def yb_number_density_m3(yb_at_percent):
    c = _array(yb_at_percent, 'Yb at.%', 0, 100)
    return _out(site_density_m3() * c / 100.0)


def n_yag(wavelength_nm):
    """Undoped-host room-temperature Sellmeier fit, 400-5000 nm."""
    spec = META['sellmeier']
    lam2 = (_array(wavelength_nm, 'wavelength_nm', *spec['range_nm']) / 1000.0)**2
    n2 = 1.0 + sum(a * lam2 / (lam2 - b) for a, b in zip(spec['A'], spec['B_um2']))
    return _out(np.sqrt(n2))


def partition_functions(temperature_K):
    """Boltzmann partitions relative to the bottom of each Stark manifold."""
    T = _array(temperature_K, 'temperature_K', np.finfo(float).tiny)
    rows = _csv('spectra/stark_levels.csv')
    Z = []
    for manifold in ('ground', 'excited'):
        rr = [r for r in rows if r['manifold'] == manifold]
        E = np.array([float(r['energy_cm-1']) for r in rr])
        g = np.array([float(r['degeneracy']) for r in rr])
        Z.append(np.sum(g * np.exp(-HC_OVER_K_CM_K * (E - E.min()) / T[..., None]), axis=-1))
    return tuple(_out(z) for z in Z)


def mccumber_emission_absorption_ratio(wavelength_nm, temperature_K):
    """sigma_e/sigma_a at the SAME temperature; cannot transfer spectra across T."""
    wl, T = np.broadcast_arrays(_array(wavelength_nm, 'wavelength_nm', 400, 5000),
                               _array(temperature_K, 'temperature_K', 80, 773))
    zl, zu = partition_functions(T)
    e0 = min(float(r['energy_cm-1']) for r in _csv('spectra/stark_levels.csv') if r['manifold'] == 'excited')
    exponent = HC_OVER_K_CM_K * (e0 - 1e7 / wl) / T
    return _out(np.asarray(zl) / np.asarray(zu) * np.exp(exponent))


def cross_sections_cm2(wavelength_nm, temperature_K=293.15, *, dataset='rt_legacy', allow_approximate=False):
    """Return (absorption, emission). Shapes follow NumPy broadcasting.

    rt_legacy: exact upstream arrays, 905-1095 nm, 293.15 K ONLY.
    laser_band_figure: manual figure nodes, 1020-1060 nm, 293.15-473.15 K.
    The latter derives absorption using McCumber, not independent measurements.
    No extrapolation, no silent clipping and no RT fallback at another T.
    """
    wl, T = np.broadcast_arrays(_array(wavelength_nm, 'wavelength_nm'),
                               _array(temperature_K, 'temperature_K'))
    if dataset == 'rt_legacy':
        _array(wl, 'wavelength_nm', 905, 1095)
        if np.any(np.abs(T - 293.15) > 1e-8):
            raise ValueError('rt_legacy has no temperature dependence; use it at 293.15 K only')
        rr = _csv('spectra/yb_yag_293K_legacy.csv')
        grid = np.array([float(r['wavelength_nm']) for r in rr])
        sa = np.interp(wl.ravel(), grid, [float(r['sigma_abs_cm2']) for r in rr]).reshape(wl.shape)
        se = np.interp(wl.ravel(), grid, [float(r['sigma_em_cm2']) for r in rr]).reshape(wl.shape)
    elif dataset == 'laser_band_figure':
        if not allow_approximate:
            raise ValueError('Manual figure dataset requires allow_approximate=True')
        _array(wl, 'wavelength_nm', 1020, 1060)
        _array(T, 'temperature_K', 293.15, 473.15)
        warnings.warn('Manual figure readings; inferred absorption; not calibrated ground truth.', ApproximationWarning, stacklevel=2)
        rr = _csv('spectra/korner2012_laser_band_manual.csv')
        grid = np.array([float(r['wavelength_nm']) for r in rr])
        temps = np.array([293.15, 353.15, 413.15, 473.15])
        curves = np.array([[float(r[f'sigma_em_{c}C_cm2']) for r in rr] for c in (20, 80, 140, 200)])
        interp = np.array([np.interp(wl.ravel(), grid, row) for row in curves])
        j = np.clip(np.searchsorted(temps, T.ravel(), side='right') - 1, 0, 2)
        f = (T.ravel() - temps[j]) / (temps[j + 1] - temps[j])
        col = np.arange(wl.size)
        se = ((1-f) * interp[j, col] + f * interp[j+1, col]).reshape(wl.shape)
        sa = se / mccumber_emission_absorption_ratio(wl, T)
    else:
        raise ValueError(f'Unknown spectral dataset {dataset!r}')
    return _out(sa), _out(se)


def cross_sections_m2(wavelength_nm, temperature_K=293.15, **kwargs):
    sa, se = cross_sections_cm2(wavelength_nm, temperature_K, **kwargs)
    return _out(np.asarray(sa) * 1e-4), _out(np.asarray(se) * 1e-4)


def hot_25at_figure_cross_sections_m2(wavelength_nm, temperature_K,
                                      *, yb_at_percent=25.0,
                                      allow_approximate=False):
    """Figure-traced 25 at.% spectra at 300-450 K, never a default YAG input.

    Figs. 4 and 6 of Esmaeilzadeh et al. (2012) were traced at 300 and
    450 K. Linear interpolation between the two digitized endpoint curves is
    an approximation, not an intermediate-temperature measurement. Fig. 6
    emission was derived by the paper's authors using reciprocity.
    """
    if not allow_approximate:
        raise ValueError('25 at.% figure spectra require allow_approximate=True')
    if float(yb_at_percent) != 25.0:
        raise ValueError('These spectra are from a 25 at.% sample only')
    wl, T = np.broadcast_arrays(_array(wavelength_nm, 'wavelength_nm', 920, 1040),
                               _array(temperature_K, 'temperature_K', 300, 450))
    warnings.warn('25 at.% figure tracing and temperature interpolation are approximate.',
                  ApproximationWarning, stacklevel=2)
    path = Path(__file__).resolve().parent / 'derived_spectra/esmaeilzadeh2012_25at_endpoints.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        rows = tuple(csv.DictReader(stream))
    grid = np.array([float(row['wavelength_nm']) for row in rows])
    fraction = (T - 300) / 150
    def curve(name):
        cold = np.interp(wl, grid, [float(row[f'{name}_300K_cm2']) for row in rows])
        hot = np.interp(wl, grid, [float(row[f'{name}_450K_cm2']) for row in rows])
        return ((1 - fraction) * cold + fraction * hot) * 1e-4
    return _out(curve('sigma_abs')), _out(curve('sigma_em'))


def tang2014_rt_absorption_coefficient_m1(wavelength_nm, *,
                                          yb_at_percent,
                                          allow_approximate=False):
    """Tang et al. Fig. 5: room-temperature ceramic alpha, not sigma_a.

    Concentration is restricted to an actually pictured specimen. This
    figure-derived coefficient includes that sample's optical losses and
    must not be combined with the repository's per-ion RT cross sections as
    an independent extra absorption term.
    """
    if not allow_approximate:
        raise ValueError('Tang 2014 figure readout requires allow_approximate=True')
    concentration = float(yb_at_percent)
    if concentration not in (5., 10., 15.):
        raise ValueError('Tang figure lookup requires one of 5, 10 or 15 at.%')
    wl = _array(wavelength_nm, 'wavelength_nm', 935, 1040)
    warnings.warn('Tang 2014 RT absorption is figure traced from individual ceramics.',
                  ApproximationWarning, stacklevel=2)
    path = Path(__file__).resolve().parent / 'derived_spectra/tang2014_5_10_15at_rt_absorption.csv'
    with path.open(newline='', encoding='utf-8') as stream:
        rows = tuple(csv.DictReader(stream))
    grid = np.array([float(row['wavelength_nm']) for row in rows])
    alpha_cm1 = np.interp(wl, grid,
                          [float(row[f'alpha_{int(concentration)}at_cm-1']) for row in rows])
    return _out(alpha_cm1 * 100)


def absorption_coefficient_m1(wavelength_nm, yb_at_percent, temperature_K=293.15, **kwargs):
    """Unexcited ground-state absorption coefficient; no saturation."""
    sa, _ = cross_sections_m2(wavelength_nm, temperature_K, **kwargs)
    return _out(yb_number_density_m3(yb_at_percent) * np.asarray(sa))


def gain_coefficient_m1(wavelength_nm, yb_at_percent, excited_fraction, temperature_K=293.15, **kwargs):
    beta = _array(excited_fraction, 'excited_fraction', 0, 1)
    sa, se = cross_sections_m2(wavelength_nm, temperature_K, **kwargs)
    return _out(yb_number_density_m3(yb_at_percent) * (beta * np.asarray(se) - (1-beta) * np.asarray(sa)))


def saturation_fluence_J_m2(wavelength_nm, temperature_K=293.15, **kwargs):
    """Effective two-manifold saturation fluence h*nu/(sigma_a+sigma_e)."""
    wl = _array(wavelength_nm, 'wavelength_nm', np.finfo(float).tiny)
    sa, se = cross_sections_m2(wl, temperature_K, **kwargs)
    total = np.asarray(sa) + np.asarray(se)
    if np.any(total <= 0):
        raise ValueError('Saturation fluence requires a positive total cross section')
    return _out(HC_J_M / (wl * 1e-9 * total))


def saturation_intensity_W_m2(wavelength_nm, temperature_K=293.15, *, lifetime_s=None, **kwargs):
    tau = META['lifetime']['default_s'] if lifetime_s is None else lifetime_s
    tau = _array(tau, 'lifetime_s', np.finfo(float).tiny)
    return _out(saturation_fluence_J_m2(wavelength_nm, temperature_K, **kwargs) / tau)


def quantum_defect_fraction(pump_nm, laser_nm):
    """Ideal laser-photon quantum defect; not total heat or extraction efficiency."""
    pump = _array(pump_nm, 'pump_nm', np.finfo(float).tiny)
    laser = _array(laser_nm, 'laser_nm', np.finfo(float).tiny)
    if np.any(pump >= laser):
        raise ValueError('This helper requires a Stokes laser transition: pump_nm < laser_nm')
    return _out(1.0 - pump / laser)


def thermal_conductivity_doped(T_K, yb_at_percent, *, family, interpolate_doping=False):
    """Cini2017 k0*(T/T0)^m fits. CT and HT are NOT automatically joined.

    Non-tabulated concentrations require explicit opt-in to linear interpolation
    in thermal resistivity 1/k. This interpolation is an engineering assumption.
    """
    rr = [r for r in _csv('thermal/cini2017_k_fits.csv') if r['family'] == family]
    if not rr:
        raise ValueError("family must be 'CT' or 'HT'")
    concentrations = np.array([float(r['Yb_at_percent']) for r in rr])
    T, c = np.broadcast_arrays(_array(T_K, 'T_K', float(rr[0]['Tmin_K']), float(rr[0]['Tmax_K'])),
                              _array(yb_at_percent, 'Yb_at_percent', concentrations[0], concentrations[-1]))
    exact = np.any(np.isclose(c[..., None], concentrations, rtol=0, atol=1e-10), axis=-1)
    if not np.all(exact):
        if not interpolate_doping:
            raise ValueError('Concentration not tabulated; set interpolate_doping=True explicitly')
        warnings.warn('Interpolating thermal resistivity across different dopings.', ApproximationWarning, stacklevel=2)
    kvals = np.array([float(r['k0_W_mK']) * (T.ravel()/float(r['T0_K']))**float(r['exponent']) for r in rr])
    out = [1/np.interp(cc, concentrations, 1/kvals[:, i]) for i, cc in enumerate(c.ravel())]
    return _out(np.array(out).reshape(T.shape))


def thermal_conductivity_host(T_K, sample_type='single_crystal'):
    s = META['sato_host_k']
    if sample_type not in ('single_crystal', 'ceramic'):
        raise ValueError('Unknown sample_type')
    T = _array(T_K, 'T_K', *s['range_K'])
    k0, k1, k2, T0 = s[sample_type]
    return _out(k0 + k1/T + k2*np.exp(-T/T0))


def thermal_diffusivity_host_m2_s(T_K, sample_type='single_crystal'):
    s = META['sato_host_diffusivity']
    if sample_type not in ('single_crystal', 'ceramic'):
        raise ValueError('Unknown sample_type')
    T = _array(T_K, 'T_K', *s['range_K'])
    d0, d1, d2, m = s[sample_type]
    return _out((d0 + d1/T + d2/T**m) * 1e-6)


def heat_capacity_host_J_kgK(T_K):
    """Linear interpolation of published host Cp table, not Yb-specific Cp(T)."""
    T = _array(T_K, 'T_K', 100, 500)
    rr = [r for r in _csv('thermal/sato2025_host_tables.csv') if r['sample_type'] == 'single_crystal']
    return _out(np.interp(T, [float(r['temperature_K']) for r in rr], [1000*float(r['Cp_J_gK']) for r in rr]))


def _thermo(T_K, dataset):
    if dataset not in META['host_thermooptic_polynomials']:
        raise ValueError('Unknown thermo-optic dataset')
    s = META['host_thermooptic_polynomials'][dataset]
    return _array(T_K, 'T_K', *s['range_K']), s


def thermal_expansion_per_K(T_K, *, dataset='aggarwal2005'):
    T, s = _thermo(T_K, dataset)
    return _out(np.polynomial.polynomial.polyval(T, s['alpha_coeff_per_K']))


def dn_dT_per_K(T_K, *, dataset='aggarwal2005', allow_apparent=False):
    T, s = _thermo(T_K, dataset)
    if dataset == 'sato2025':
        if not allow_apparent:
            raise ValueError('Sato2025 dn/dT may include mount stress; set allow_apparent=True explicitly')
        warnings.warn('Sato2025 apparent dn/dT can include mounting-stress photoelasticity.', ApproximationWarning, stacklevel=2)
    return _out(np.polynomial.polynomial.polyval(T, s['dn_dT_coeff_per_K']))


def integrated_expansion(T_K, reference_K=293.15, *, dataset='aggarwal2005'):
    T, s = _thermo(T_K, dataset)
    T0, _ = _thermo(reference_K, dataset)
    a = np.asarray(s['alpha_coeff_per_K']) / np.arange(1, 5)
    return _out(sum(a[i] * (T**(i+1) - T0**(i+1)) for i in range(4)))


def density_host_kg_m3(T_K, *, dataset='sato2025'):
    """Expansion-derived host density referenced to Sato2025 300 K, not measured rho(T)."""
    strain = integrated_expansion(T_K, 300.0, dataset=dataset)
    return _out(META['structure']['rho_host_300K_kg_m3'] * np.exp(-3*np.asarray(strain)))


def thermal_index_change(T_K, reference_K=293.15, *, wavelength_nm=1064.0,
                         dataset='aggarwal2005', allow_wavelength_proxy=False, allow_apparent=False):
    """Integrated host dn/dT. Using 1064-nm data at another wavelength is explicit."""
    wl = _array(wavelength_nm, 'wavelength_nm', 400, 5000)
    if np.any(np.abs(wl - 1064.0) > 1e-8):
        if not allow_wavelength_proxy:
            raise ValueError('dn/dT is for 1064 nm; another wavelength requires allow_wavelength_proxy=True')
        warnings.warn('Using 1064-nm thermo-optic data at another wavelength.', ApproximationWarning, stacklevel=2)
    T, s = _thermo(T_K, dataset)
    T0, _ = _thermo(reference_K, dataset)
    dn_dT_per_K(T, dataset=dataset, allow_apparent=allow_apparent)
    a = np.asarray(s['dn_dT_coeff_per_K']) / np.arange(1, 5)
    out = sum(a[i] * (T**(i+1) - T0**(i+1)) for i in range(4))
    return _out(out + np.zeros_like(wl))


def elastic_tensor_Pa():
    """Fourth-rank cubic tensor in [100]/[010]/[001] axes."""
    e = MECH['elastic']
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        C[i,i,i,i] = e['C11_Pa']
        for j in range(3):
            if i != j:
                C[i,i,j,j] = e['C12_Pa']
                C[i,j,i,j] = C[i,j,j,i] = e['C44_Pa']
    return C


def elastic_vrh_moduli():
    """Isotropic Voigt-Reuss-Hill averages, derived from the same cubic tensor."""
    e = MECH['elastic']; a = e['C11_Pa'] - e['C12_Pa']; c = e['C44_Pa']
    B = (e['C11_Pa'] + 2*e['C12_Pa']) / 3
    Gv = (a + 3*c)/5; Gr = 5*a*c/(4*c + 3*a); G = (Gv + Gr)/2
    return {'B_Pa': B, 'G_Pa': G, 'E_Pa': 9*B*G/(3*B+G), 'nu': (3*B-2*G)/(2*(3*B+G))}


def rotate_rank4(tensor, rotation):
    """Rotate crystal tensor into lab; columns of rotation are crystal axes in lab."""
    A = _array(tensor, 'tensor'); r = _array(rotation, 'rotation')
    if A.shape != (3,3,3,3) or r.shape != (3,3):
        raise ValueError('Expected tensor (3,3,3,3), rotation (3,3)')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-10) or not np.isclose(np.linalg.det(r), 1, atol=1e-10):
        raise ValueError('Rotation must be orthonormal and right handed')
    return np.einsum('ia,jb,kc,ld,abcd->ijkl', r,r,r,r,A)


def photoelastic_delta_B(mechanical_strain, *, tensor_set='dixon1967'):
    """Delta inverse dielectric tensor from MECHANICAL elastic strain in crystal axes.

    Subtract thermal eigenstrain before calling. This routine intentionally does
    not combine apparent/stress-free dn/dT or assume an optical propagation axis.
    """
    if tensor_set not in MECH['photoelastic_sets']:
        raise ValueError('Unknown tensor_set')
    p = MECH['photoelastic_sets'][tensor_set]
    if p['p44'] is None:
        raise ValueError('The selected photoelastic tensor is incomplete; p44 is unknown')
    e = _array(mechanical_strain, 'mechanical_strain')
    if e.shape[-2:] != (3,3) or not np.allclose(e, np.swapaxes(e,-1,-2), rtol=1e-10, atol=1e-14):
        raise ValueError('mechanical_strain must have symmetric final dimensions (3,3)')
    out = 2*p['p44'] * e.copy()
    tr = np.trace(e, axis1=-2, axis2=-1)
    for i in range(3):
        out[...,i,i] = p['p11']*e[...,i,i] + p['p12']*(tr-e[...,i,i])
    return out


def measured_thermal_conductivity_W_mK(T_K, yb_at_percent):
    """Interpolate Aggarwal2005 measured/derived k table at a tabulated doping."""
    c = float(_array(yb_at_percent, 'Yb_at_percent', 0, 15))
    rr = [r for r in _csv('thermal/aggarwal2005_yb_doping.csv') if abs(float(r['Yb_at_percent'])-c) < 1e-10]
    if not rr:
        raise ValueError('Measured table has only 0, 2, 4 and 15 at.%')
    rr.sort(key=lambda r: float(r['temperature_K']))
    ts = [float(r['temperature_K']) for r in rr]
    T = _array(T_K, 'T_K', ts[0], ts[-1])
    return _out(np.interp(T, ts, [float(r['k_W_mK']) for r in rr]))


def doped_RT_density_heat_capacity(yb_at_percent):
    """Return rho [kg/m^3], Cp [J/kg/K] from Aggarwal2005 RT table.

    Tabulated dopings only; Cp is volumetric heat capacity divided by density.
    Temperature-dependent Yb-specific heat capacities are not invented.
    """
    c = float(_array(yb_at_percent, 'Yb_at_percent', 0, 15))
    rr = [r for r in _csv('thermal/aggarwal2005_yb_doping.csv') if abs(float(r['Yb_at_percent'])-c) < 1e-10]
    if not rr:
        raise ValueError('Measured RT density/Cp table has only 0, 2, 4 and 15 at.%')
    rho = float(rr[0]['rho_RT_g_cm3']) * 1000
    cp = float(rr[0]['rhoCp_RT_J_cm3K']) * 1e6 / rho
    return {'rho_kg_m3':rho, 'Cp_J_kgK':cp}
