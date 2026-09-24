from pathlib import Path
import csv
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))
import yb_luag_model as mat


def load():
    wl, T, sa, se = mat._load_spectra()
    return wl, T, sa, se


def export_cross_sections():
    wl, temps, sa, se = load()
    out = ROOT / "spectra" / "yb_luag_cross_sections_20_200C_reconstructed.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        ints = [int(x) for x in temps]
        w.writerow(["wavelength_nm", *[f"sigma_abs_{T}C_cm2" for T in ints],
                    *[f"sigma_em_{T}C_cm2" for T in ints], "data_quality"])
        for i, x in enumerate(wl):
            w.writerow([x, *sa[:, i], *se[:, i], "figure_guided_engineering_reconstruction"])
    return out


def export_absorption_coefficients():
    wl, temps, sa, _ = load()
    out = ROOT / "spectra" / "derived_absorption_coefficients.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wavelength_nm","temperature_C","alpha_5at_cm-1","alpha_10at_cm-1",
                    "alpha_12at_cm-1","alpha_15at_cm-1","data_quality"])
        for ti, T in enumerate(temps):
            for wi, x in enumerate(wl):
                sigma = sa[ti, wi]
                vals = [mat.SITE_DENSITY_CM3 * c/100.0 * sigma for c in (5,10,12,15)]
                w.writerow([x, T, *vals, "derived_from_reconstructed_cross_sections"])
    return out


def export_refractive_index():
    out = ROOT / "optical" / "luag_refractive_index_Hrabovsky_193_1690nm.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wavelength_nm","n_luag","source_id","data_quality"])
        for wl in range(193, 1691):
            w.writerow([wl, mat.n_luag(wl), "HRABOVSKY2021",
                        "derived_from_published_dispersion_fit"])
    return out


def export_heat_capacity():
    out = ROOT / "thermal" / "luag_heat_capacity_80_300K_engineering.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature_K","Cp_engineering_normalized_J_kgK","source_id","validity_note"])
        for T in range(80, 301):
            w.writerow([T, mat.heat_capacity_luag_J_kgK(T), "AGGARWAL2005+DERIVED",
                        "Debye engineering interpolation normalized to room-temperature volumetric heat capacity; not directly tabulated Cp(T)"])
    return out


if __name__ == "__main__":
    npz = mat.write_combined_spectral_npz()
    outputs = [npz, export_cross_sections(), export_absorption_coefficients(),
               export_refractive_index(), export_heat_capacity()]
    for p in outputs:
        print(p)
