"""Visual quality check of recovered original and figure-derived Yb:YAG data."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    with (HERE / "devido2020/devido_zpl_reconstructed.csv").open(newline="") as file:
        zpl = list(csv.DictReader(file))
    with (HERE / "high_temp_25at/endpoint_spectra.csv").open(newline="") as file:
        hot = list(csv.DictReader(file))
    with (HERE / "tang2014/rt_absorption_coefficients.csv").open(newline="") as file:
        tang = list(csv.DictReader(file))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for temp in (100, 150, 200, 250, 300):
        rows = [r for r in zpl if int(r["nominal_temperature_K"]) == temp and
                r["quality_flag"] == "valid"]
        x = np.array([float(r["wavelength_nm"]) for r in rows])
        y = np.array([float(r["sigma_abs_usable_cm2"]) for r in rows]) * 1e20
        order = np.argsort(x)
        axes[0].plot(x[order], y[order], lw=1, label=f"{temp} K")
    axes[0].set(xlim=(967.8, 970.3), ylim=(0, 30),
                xlabel="Wavelength (nm)", ylabel="Absorption ($10^{-20}$ cm$^2$)",
                title="1.1 at.% ZPL: reconstructed raw records")
    axes[0].legend(fontsize=8)
    x = np.array([float(r["wavelength_nm"]) for r in hot])
    for temp, color in ((300, "#1b347d"), (450, "#b93832")):
        axes[1].plot(x, [float(r[f"sigma_abs_{temp}K_cm2"]) * 1e21 for r in hot],
                     color=color, label=f"{temp} K")
        axes[2].plot(x, [float(r[f"sigma_em_{temp}K_cm2"]) * 1e20 for r in hot],
                     color=color, label=f"{temp} K")
    axes[1].set(xlim=(920, 1040), xlabel="Wavelength (nm)",
                ylabel="Absorption ($10^{-21}$ cm$^2$)",
                title="25 at.%: Fig. 4 traced endpoints")
    axes[2].set(xlim=(920, 1040), xlabel="Wavelength (nm)",
                ylabel="Emission ($10^{-20}$ cm$^2$)",
                title="25 at.%: Fig. 6 inferred emission")
    for ax in axes[1:3]:
        ax.legend()
    tang_wavelengths = np.array([float(r["wavelength_nm"]) for r in tang])
    for concentration in (5, 10, 15):
        axes[3].plot(tang_wavelengths,
                     [float(r[f"alpha_{concentration}at_cm-1"]) for r in tang],
                     label=f"{concentration} at.%")
    axes[3].set(xlim=(935, 1040), xlabel="Wavelength (nm)",
                ylabel="Absorption coefficient (cm$^{-1}$)",
                title="Separate RT ceramics: Tang Fig. 5")
    axes[3].legend()
    fig.text(.5, -.02, "Different samples; these curves cannot be combined into a calibrated 5/10/15 at.% hot model.",
             ha="center", fontsize=10)
    output = HERE / "recovered_spectra.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
