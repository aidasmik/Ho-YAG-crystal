from pathlib import Path
import io
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE_DENSITY_CM3 = 1.42e22
R = 8.31446261815324
THETA_D_K = 750.0
ATOMS_PER_FORMULA = 20.0
MOLAR_MASS_KG_MOL = 0.8517960925
CP_SCALE = 0.9622559386
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def _return_like_input(source, value):
    arr = np.asarray(value)
    if np.asarray(source).ndim == 0:
        return float(arr)
    return arr


def yb_number_density_cm3(yb_at_percent):
    return SITE_DENSITY_CM3 * np.asarray(yb_at_percent, dtype=float) / 100.0


def n_luag(wavelength_nm):
    wl = np.asarray(wavelength_nm, dtype=float)
    lam = wl / 1000.0
    n2 = 2.077 + 1.237 * lam**2 / (lam**2 - 0.1376**2) - 0.0104 * lam**2
    return _return_like_input(wavelength_nm, np.sqrt(n2))


def _poly(T_K, coeffs, Tmin=100.0, Tmax=300.0, extrapolate=False):
    T = np.asarray(T_K, dtype=float)
    if not extrapolate and np.any((T < Tmin) | (T > Tmax)):
        raise ValueError(f"T outside validated range {Tmin:g}-{Tmax:g} K")
    out = sum(c * T**i for i, c in enumerate(coeffs))
    return _return_like_input(T_K, out)


def thermal_expansion_per_K(T_K, extrapolate=False):
    ppm = _poly(T_K, (-1.4132, 0.045552, -6.798e-5, 0.0), extrapolate=extrapolate)
    return np.asarray(ppm) * 1e-6 if np.asarray(T_K).ndim else float(ppm) * 1e-6


def dn_dT_per_K(T_K, extrapolate=False):
    ppm = _poly(T_K, (-2.3926, 0.029863, 1.8715e-5, 0.0), extrapolate=extrapolate)
    return np.asarray(ppm) * 1e-6 if np.asarray(T_K).ndim else float(ppm) * 1e-6


def optical_path_temp_coeff_per_K(T_K, extrapolate=False):
    ppm = _poly(T_K, (-5.7998, 0.12344, -1.354e-4, 0.0), extrapolate=extrapolate)
    return np.asarray(ppm) * 1e-6 if np.asarray(T_K).ndim else float(ppm) * 1e-6


_K_T = np.array([101.0, 151.0, 201.0, 250.0, 298.0])
_K_V = np.array([25.4, 16.5, 12.2, 9.6, 8.3])


def thermal_conductivity_undoped_W_mK(T_K, extrapolate=False):
    T = np.asarray(T_K, dtype=float)
    if not extrapolate and np.any((T < _K_T.min()) | (T > _K_T.max())):
        raise ValueError("T outside measured undoped-LuAG range 101-298 K")
    out = np.interp(T, _K_T, _K_V)
    return _return_like_input(T_K, out)


def _debye_cp_scalar(T):
    xmax = THETA_D_K / T
    x = 0.5 * (_GL_X + 1.0) * xmax
    ex = np.exp(x)
    integ = np.sum(_GL_W * x**4 * ex / np.expm1(x)**2) * 0.5 * xmax
    cv_molar = 9.0 * ATOMS_PER_FORMULA * R * (T / THETA_D_K)**3 * integ
    return CP_SCALE * cv_molar / MOLAR_MASS_KG_MOL


def heat_capacity_luag_J_kgK(T_K, extrapolate=False):
    T = np.asarray(T_K, dtype=float)
    if not extrapolate and np.any((T < 80.0) | (T > 300.0)):
        raise ValueError("Engineering Cp model is intended for 80-300 K")
    out = np.array([_debye_cp_scalar(float(t)) for t in T.ravel()]).reshape(T.shape)
    return _return_like_input(T_K, out)


def _spectral_archive_bytes():
    spec = ROOT / "spectra"
    parts = [spec / f"yb_luag_cross_sections_20_200C_reconstructed.npz.part{i}" for i in range(5)]
    missing = [str(p) for p in parts if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing spectral archive part(s): " + ", ".join(missing))
    return b"".join(p.read_bytes() for p in parts)


def write_combined_spectral_npz(path=None):
    raw = _spectral_archive_bytes()
    out = Path(path) if path is not None else ROOT / "spectra" / "yb_luag_cross_sections_20_200C_reconstructed.npz"
    out.write_bytes(raw)
    return out


def _load_spectra():
    with np.load(io.BytesIO(_spectral_archive_bytes())) as d:
        return (d["wavelength_nm"].astype(float),
                d["temperature_C"].astype(float),
                d["sigma_abs_cm2"].astype(float),
                d["sigma_em_cm2"].astype(float))


def _sigma(kind, wavelength_nm, temperature_C, extrapolate=False):
    wl_grid, T_grid, sa, se = _load_spectra()
    wl = np.asarray(wavelength_nm, dtype=float)
    T = float(temperature_C)
    if not extrapolate:
        if np.any((wl < wl_grid[0]) | (wl > wl_grid[-1])):
            raise ValueError("wavelength outside reconstructed range 880-1150 nm")
        if T < T_grid[0] or T > T_grid[-1]:
            raise ValueError("temperature outside reconstructed range 20-200 C")
    table = sa if kind == "abs" else se
    spectral_at_Tnodes = np.vstack([np.interp(wl, wl_grid, row) for row in table])
    flat = spectral_at_Tnodes.reshape(len(T_grid), -1)
    out = np.array([np.interp(T, T_grid, flat[:, j]) for j in range(flat.shape[1])])
    out = out.reshape(wl.shape)
    return _return_like_input(wavelength_nm, out)


def sigma_abs_cm2(wavelength_nm, temperature_C=20.0, extrapolate=False):
    return _sigma("abs", wavelength_nm, temperature_C, extrapolate)


def sigma_em_cm2(wavelength_nm, temperature_C=20.0, extrapolate=False):
    return _sigma("em", wavelength_nm, temperature_C, extrapolate)


def absorption_coefficient_cm1(wavelength_nm, yb_at_percent, temperature_C=20.0):
    return yb_number_density_cm3(yb_at_percent) * sigma_abs_cm2(wavelength_nm, temperature_C)


def gain_coefficient_cm1(wavelength_nm, yb_at_percent, excited_fraction, temperature_C=20.0):
    N = yb_number_density_cm3(yb_at_percent)
    beta = np.asarray(excited_fraction, dtype=float)
    sa = sigma_abs_cm2(wavelength_nm, temperature_C)
    se = sigma_em_cm2(wavelength_nm, temperature_C)
    return N * (beta * se - (1.0 - beta) * sa)


def quantum_defect_fraction(pump_nm, laser_nm):
    return 1.0 - float(pump_nm) / float(laser_nm)


def saturation_intensity_W_cm2(wavelength_nm=1030.0, sigma_em_cm2_value=2.59e-20, lifetime_s=0.95e-3):
    h = 6.62607015e-34
    c = 299792458.0
    return h * c / ((wavelength_nm * 1e-9) * sigma_em_cm2_value * lifetime_s)
