"""Reproduce bounded Yb:YAG README figures from packaged data and solver output.

Run through hoyag.local_supervisor as documented in Yb-YAG/README.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ybyag.model import YbYAGMaterial
from ybyag import material_data as md
from ybluag.gallery import YbGallerySettings, simulate_pulsed_seed

OUT = ROOT / "Yb-YAG" / "readme_figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def spectra():
    source = ROOT / "src/ybyag/data/spectra/yb_yag_293K_legacy.csv"
    data = np.genfromtxt(source, delimiter=",", names=True)
    wl = data["wavelength_nm"]
    sa = data["sigma_abs_cm2"] / 1e-20
    se = data["sigma_em_cm2"] / 1e-20
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    ax[0].plot(wl, sa, label="absorption", color="#b64d3d")
    ax[0].plot(wl, se, label="emission", color="#176e94")
    ax[0].axvline(969, ls=":", color="0.4")
    ax[0].axvline(1030, ls=":", color="0.4")
    ax[0].set(ylabel=r"cross section [$10^{-20}$ cm$^2$/ion]",
              title="Default 293.15 K Yb:YAG input: exact legacy software arrays")
    ax[0].legend()
    threshold = np.divide(sa, sa + se, out=np.full_like(sa, np.nan), where=sa + se > 0)
    ax[1].plot(wl, threshold, color="#306947")
    ax[1].set(xlabel="vacuum wavelength [nm]", ylabel=r"transparency fraction $\beta_{tr}$",
              ylim=(0, 1))
    ax[1].grid(alpha=.2)
    fig.text(.5, -.01, "Source: HASEonGPU legacy table; upstream experimental metadata and uncertainty are unspecified.",
             ha="center", fontsize=9)
    save(fig, "runtime_spectra.png")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def comparison_spectra():
    source = ROOT / "research/ybyag/tang2014/rt_absorption_coefficients.csv"
    with source.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    columns = rows[0].keys()
    wave_key = next(k for k in columns if "wavelength" in k)
    wl = np.array([float(r[wave_key]) for r in rows])
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    colors = {5: "#176e94", 10: "#c07422", 15: "#9d4488"}
    for concentration in (5, 10, 15):
        key = next(k for k in columns if k.startswith(f"alpha_{concentration}at"))
        measured = np.array([float(r[key]) for r in rows])
        model = md.absorption_coefficient_m1(wl, concentration) / 100
        ax.plot(wl, measured, color=colors[concentration],
                label=f"{concentration}% Tang figure readout")
        ax.plot(wl, model, ls="--", color=colors[concentration], alpha=.8,
                label=f"{concentration}% default $N\sigma_a$")
    ax.set(xlabel="vacuum wavelength [nm]", ylabel=r"absorption coefficient [cm$^{-1}$]",
           title="Independent room-temperature ceramic figure readouts vs default table")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2)
    save(fig, "rt_concentration_comparison.png")


def optional_temperature_spectra():
    korner = np.genfromtxt(ROOT / "Yb-YAG/spectra/korner2012_laser_band_manual.csv",
                          delimiter=",", names=True)
    hot = np.genfromtxt(ROOT / "research/ybyag/high_temp_25at/endpoint_spectra.csv",
                        delimiter=",", names=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for c in (20, 80, 140, 200):
        ax[0].plot(korner["wavelength_nm"],
                   korner[f"sigma_em_{c}C_cm2"] / 1e-20, label=f"{c} °C")
    ax[0].set(title="Körner laser band: manual figure readings",
              xlabel="wavelength [nm]", ylabel=r"emission [$10^{-20}$ cm$^2$]")
    ax[0].legend(fontsize=8)
    for T, style in ((300, "-"), (450, "--")):
        for name, color in (("abs", "#b64d3d"), ("em", "#176e94")):
            ax[1].plot(hot["wavelength_nm"],
                       hot[f"sigma_{name}_{T}K_cm2"] / 1e-20,
                       ls=style, color=color, label=f"{name}, {T} K")
    ax[1].set(title="25 at.% crystal: digitized figure endpoints",
              xlabel="wavelength [nm]", ylabel=r"cross section [$10^{-20}$ cm$^2$]")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.2)
    save(fig, "optional_temperature_spectra.png")


def simulate(concentration, pump_power_W, *, thermal=False):
    settings = YbGallerySettings(
        pump_power_W=pump_power_W, pump_radius_m=1e-3, thickness_m=100e-6,
        disk_radius_m=5e-3, assembly_property_model="yag_rt_proxy",
        grid_n=64, field_size_m=8e-3, z_steps=1,
        thermal_nr=4, thermal_nphi=6, thermal_nz=2,
        cluster_contrast=0, waist_m=.6e-3, post_disk_distance_m=0)
    result = simulate_pulsed_seed(
        YbYAGMaterial(yb_at_percent=concentration), settings, "Gaussian TEM00",
        10e-9, 10e-12, 1e4, 10, pump_passes=10,
        compute_thermal=thermal, operation_duration_s=0,
        thermal_optical_mode="lumped_phase" if thermal else "cold")
    return settings, result


def main():
    sha = spectra()
    comparison_spectra()
    optional_temperature_spectra()
    cases = {}
    outputs = {}
    for concentration, power, thermal in ((5, 40, False), (20, 40, False), (20, .1, True)):
        settings, result = simulate(concentration, power, thermal=thermal)
        key = f"{concentration}at_{power:g}W"
        outputs[key] = result
        cases[key] = {
            "yb_at_percent": concentration, "pump_incident_W": power,
            "seed_energy_J": 10e-9, "repetition_rate_Hz": 1e4,
            "pulse_fwhm_s": 10e-12, "pump_wavelength_nm": 969,
            "signal_wavelength_nm": 1030, "pump_passes": 10,
            "signal_traversals": 10, "grid_n": settings.grid_n,
            "z_steps": settings.z_steps, "output_energy_J": result["output_energy_J"],
            "pump_absorbed_W": result["cycle_average_pump_absorbed_W"],
            "signal_gain_W": result["cycle_average_signal_gain_W"],
            "heat_W": result["cycle_average_heat_W_upper_or_assumed"],
            "fluorescence_escape_W": result["cycle_average_escaping_fluorescence_W"],
            "mean_pre_pulse_beta": result["mean_excited_fraction_before_pulse"],
            "periodic_cycles": result["cycles"],
            "population_photon_balance_relative_L1": result["ideal_multipass"]["population_photon_balance_relative_L1"],
            "thermal_status": result["thermal"]["status"] if thermal else "not_run",
        }
        if thermal and result["thermal"]["status"] == "computed":
            cases[key]["disk_temperature_max_C"] = result["thermal"]["disk_temperature_max_C"]
            cases[key]["thermal_balance_error_W"] = result["thermal"]["balance_error_W"]
    r = outputs["20at_40W"]
    f, ax = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    extent = [-4, 4, -4, 4]
    for a, field, title in ((ax[0, 0], r["disk_input_fluence_J_m2"], "at disk, before gain"),
                            (ax[0, 1], r["output_fluence_J_m2"], "after ten ideal traversals")):
        im = a.imshow(field, extent=extent, origin="lower", cmap="inferno")
        a.set(title=title, xlabel="x [mm]", ylabel="y [mm]")
        a.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
        f.colorbar(im, ax=a, label=r"pulse fluence [J m$^{-2}$]")
    ax[1, 0].plot(r["time_ps"], r["input_power_trace_W"], label="disk input")
    ax[1, 0].plot(r["time_ps"], r["output_power_trace_W"], label="output")
    ax[1, 0].set(xlabel="retarded pulse time [ps]", ylabel="power [W]",
                 title="Time-sampled pulse transport")
    ax[1, 0].legend()
    ax[1, 1].plot(np.arange(1, 11), np.array(r["ideal_multipass"]["encounter_exit_energies_J"])*1e9,
                  marker="o")
    ax[1, 1].set(xlabel="signal traversal", ylabel="energy after disk [nJ]",
                 title="One shared inversion; ideal relays")
    f.suptitle("Yb:YAG solver example: 20 at.%, 40 W CW pump, cold 293.15 K spectra")
    save(f, "cold_amplifier_example.png")
    f, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    keys = ["5at_40W", "20at_40W"]
    ax.bar(["5 at.%", "20 at.%"], [cases[k]["output_energy_J"]*1e9 for k in keys],
           color=["#176e94", "#b64d3d"])
    ax.axhline(10, color="0.3", ls="--", label="10 nJ seed")
    ax.set(ylabel="output pulse energy [nJ]",
           title="Cold 40 W examples with identical optical settings")
    ax.legend()
    save(f, "doping_comparison.png")
    low = outputs["20at_0.1W"]
    thermal = low["thermal"]
    if thermal["status"] == "computed":
        f, ax = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
        temp = np.asarray(thermal["disk_temperature_K"])
        radial = np.mean(temp, axis=(0, 2)) - 273.15
        radius = (np.arange(len(radial)) + .5) * 5 / len(radial)
        ax[0].plot(radius, radial, marker="o")
        ax[0].set(xlabel="radius [mm]", ylabel="disk temperature [°C]",
                  title="azimuth/axial mean")
        for a, name, label in ((ax[1], "scalar_roundtrip_opd_nm", "round-trip OPD [nm]"),
                               (ax[2], "front_displacement_nm", "front displacement [nm]")):
            im = a.imshow(thermal[name], extent=extent, origin="lower", cmap="coolwarm")
            a.set(xlabel="x [mm]", ylabel="y [mm]", xlim=(-3, 3), ylim=(-3, 3))
            f.colorbar(im, ax=a, label=label)
        f.suptitle("20 at.% Yb:YAG, 0.1 W: steady generic cooler and scalar distortion")
        save(f, "near_rt_thermal_example.png")
    payload = {"source_revision": "6c384fbf7d18947db03f4902139e67bf698c6aac",
               "runtime_spectrum_sha256": sha,
               "scope": "Coarse 64x64, one optical axial slice examples; no spatial/temporal convergence claim.",
               "cases": cases}
    (OUT / "simulation_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
