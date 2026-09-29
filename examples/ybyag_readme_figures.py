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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ybyag.model import YbYAGMaterial
from ybyag import material_data as md
from ybluag.gallery import YbGallerySettings, simulate_pulsed_seed
from ybluag.camera_dataset import CameraSettings
from ybyag_dataset.distortions.material import sample_material
from ybyag_dataset.distortions.camera import capture, sample_camera_setup

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


def light_path():
    fig, ax = plt.subplots(figsize=(13, 4.7), layout="constrained")
    fig.patch.set_facecolor("#f8f7f3")
    ax.set_facecolor("#f8f7f3")
    ax.set(xlim=(0, 13), ylim=(0, 4.6))
    ax.axis("off")

    def box(x, y, w, h, title, detail, color):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.16",
                               fc=color, ec="#23313a", lw=1.4)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h*.67, title, ha="center", va="center", weight="bold", fontsize=11)
        ax.text(x+w/2, y+h*.31, detail, ha="center", va="center", fontsize=8.6)

    def arrow(x1, y1, x2, y2, color="#197299", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=17, lw=2, color=color))

    box(.2, 1.7, 1.7, 1.0, "Seed", "1030 nm / 10 ps", "#d9eaf3")
    box(2.4, 1.7, 2.0, 1.0, "SLM", "target + correction phase", "#e9dcf4")
    box(5.0, 1.7, 1.9, 1.0, "Free space", "angular-spectrum FFT", "#d9eaf3")
    box(7.55, 1.5, 1.8, 1.4, "Yb:YAG disk", "100 µm / shared β(x,y,z)", "#e9dcb5")
    box(10.2, 1.7, 2.4, 1.0, "Output", "beam + diagnostic camera", "#d9eaf3")
    for a, b in ((1.9, 2.4), (4.4, 5.0), (6.9, 7.55), (9.35, 10.2)):
        arrow(a, 2.2, b, 2.2)
    box(7.55, 3.42, 1.8, .70, "Pump", "969 nm CW / passes", "#f5d4c7")
    arrow(8.45, 3.42, 8.45, 2.93, "#b95740")
    box(7.48, .20, 1.95, .58, "Copper cooler", "heat → T → OPD", "#d5e7dc")
    arrow(8.45, 1.49, 8.45, .79, "#5a8266")
    ax.add_patch(FancyArrowPatch((9.32, 2.77), (7.61, 2.77),
                                 connectionstyle="arc3,rad=-.65", arrowstyle="-|>",
                                 mutation_scale=16, lw=1.8, color="#197299"))
    ax.text(10.35, 3.37, "ideal 1:1 signal relay returns the field\n(up to 10 disk traversals)",
            color="#175c7b", ha="center", fontsize=8.9)
    ax.text(.25, .72, "Schematic only: the pump and signal revisit one disk; the relay is idealized.",
            fontsize=9, color="#41515b")
    save(fig, "light_path_schematic.png")


def doping_map_and_transmission():
    ranges = json.loads((ROOT / "config/ybyag_nn_dataset.json").read_text(encoding="utf-8"))["ranges"]
    crystal = sample_material((128, 128), ranges, seed=314159, enabled=True)
    local_at = 10 * crystal["yb_concentration_scale"]
    sigma_p, _ = md.cross_sections_m2(969.)
    alpha = md.yb_number_density_m3(10) * crystal["yb_concentration_scale"] * sigma_p
    transmission = np.exp(-alpha * 100e-6)
    disk = np.hypot(*np.meshgrid(np.linspace(-6, 6, 128), np.linspace(-6, 6, 128))) <= 5
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0), layout="constrained")
    for ax, image, name, units, cmap in (
        (axes[0], local_at, "local Yb concentration", "Y-site at.%", "viridis"),
        (axes[1], alpha, "unpumped 969 nm loss", r"$\alpha_p$ [m$^{-1}$]", "magma"),
        (axes[2], transmission, "one 100 µm pass", r"$I_{out}/I_{in}$", "cividis"),
    ):
        im = ax.imshow(np.where(disk, image, np.nan), extent=(-6, 6, -6, 6),
                       origin="lower", cmap=cmap)
        ax.set(xlabel="x [mm]", ylabel="y [mm]", title=name,
               xlim=(-5.2, 5.2), ylim=(-5.2, 5.2))
        fig.colorbar(im, ax=ax, label=units)
    fig.suptitle("One seeded virtual 10 at.% crystal: 3% RMS Yb scale, frozen β = 0")
    save(fig, "doping_map_transport.png")
    return {"seed": 314159, "nominal_yb_at_percent": 10,
            "yb_scale_mean": float(np.mean(crystal["yb_concentration_scale"])),
            "yb_scale_rms_about_one": float(np.sqrt(np.mean((crystal["yb_concentration_scale"]-1)**2))),
            "local_at_percent_min_max": [float(np.min(local_at[disk])), float(np.max(local_at[disk]))],
            "single_pass_transmission_min_max": [float(np.min(transmission[disk])),
                                                  float(np.max(transmission[disk]))]}


def concentration_transport(cases):
    levels = np.arange(0, 21, .1)
    sigma, _ = md.cross_sections_m2(969.)
    alpha = md.yb_number_density_m3(levels) * sigma
    single = np.exp(-alpha * 100e-6)
    ten_frozen = np.exp(-alpha * 10 * 100e-6)
    keys = [f"{c}at_40W" for c in (5, 10, 15, 20)]
    concentration = np.array([cases[k]["yb_at_percent"] for k in keys])
    absorbed_fraction = np.array([cases[k]["pump_absorbed_W"] / 40 for k in keys])
    energy_gain = np.array([cases[k]["output_energy_J"] / 10e-9 for k in keys])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].plot(levels, single, label="one frozen pass")
    axes[0].plot(levels, ten_frozen, label="ten frozen passes")
    axes[0].set(xlabel="nominal Yb on Y sites [at.%]", ylabel="pump transmitted fraction",
                title="Analytic Beer–Lambert illustration, β = 0")
    axes[0].legend()
    axes[1].plot(concentration, absorbed_fraction, "o-", label="pump fraction absorbed")
    axes[1].plot(concentration, energy_gain, "s-", label="output / 10 nJ seed")
    axes[1].set(xlabel="nominal Yb on Y sites [at.%]", ylabel="fraction or energy ratio",
                title="Full cold periodic solver, 40 W pump")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, "concentration_transport.png")


def camera_noise_example(result):
    ranges = json.loads((ROOT / "config/ybyag_nn_dataset.json").read_text(encoding="utf-8"))["ranges"]
    settings = CameraSettings(width=384, height=216, object_fov_width_mm=6,
                              optical_throughput=1e-5, pulses_per_exposure=100,
                              exposure_s=.01, read_noise_e=1.6, background_e=2,
                              prnu_rms=.01, dsnu_rms_e=.5, psf_sigma_pixels=1)
    setup = sample_camera_setup(settings, ranges, seed=314159)
    grid = result["grid"]
    kw = dict(fluence_J_m2=result["output_fluence_J_m2"], x_m=grid.x, y_m=grid.y,
              wavelength_m=1030e-9, settings=settings, setup=setup, seed=314160)
    ideal, _, _ = capture(**kw, enabled=False)
    noisy, _, diagnostics = capture(**kw, enabled=True,
                                    pulse_variation={"energy_jitter_rms_fraction": .005,
                                                     "pointing_jitter_rms_pixels": .2})
    difference = noisy.astype(float) - ideal.astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), layout="constrained")
    peak = float(np.max(ideal))
    residual_scale = max(1., float(np.percentile(np.abs(difference), 99.9)))
    for ax, image, name, cmap, low, high in (
        (axes[0], ideal, "ideal camera response [ADU]", "inferno", 0, peak),
        (axes[1], noisy, "one noisy exposure [ADU]", "inferno", 0, peak),
        (axes[2], difference, "noisy − ideal [ADU]", "coolwarm", -residual_scale, residual_scale),
    ):
        im = ax.imshow(image, origin="lower", extent=(-3, 3, -1.6875, 1.6875),
                       cmap=cmap, vmin=low, vmax=high)
        ax.set(xlabel="x [mm]", ylabel="y [mm]", title=name,
               xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
        fig.colorbar(im, ax=ax, extend="both" if name.startswith("noisy") else "neither")
    fig.suptitle("Same simulated 20 at.% output field; fixed sensor map and one random exposure")
    save(fig, "camera_noise_example.png")
    return {"seed": 314159, "camera_pixels": [384, 216],
            "camera_peak_expected_electrons": diagnostics["expected_electron_peak"],
            "saturated_fraction": diagnostics["saturated_fraction"],
            "mean_absolute_adu_difference": float(np.mean(np.abs(difference)))}


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
    light_path()
    doping_summary = doping_map_and_transmission()
    cases = {}
    outputs = {}
    for concentration, power, thermal in ((5, 40, False), (10, 40, False),
                                          (15, 40, False), (20, 40, False),
                                          (20, .1, True)):
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
    concentration_transport(cases)
    camera_summary = camera_noise_example(r)
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
    keys = [f"{c}at_40W" for c in (5, 10, 15, 20)]
    ax.bar([f"{c} at.%" for c in (5, 10, 15, 20)],
           [cases[k]["output_energy_J"]*1e9 for k in keys],
           color=["#176e94", "#3f8b88", "#c07422", "#b64d3d"])
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
    payload = {"source_revision": "2cdf3cb67d803f5051c93ad5630c290ab8d499ba",
               "runtime_spectrum_sha256": sha,
               "scope": "Coarse 64x64, one optical axial slice examples; no spatial/temporal convergence claim.",
               "doping_map_example": doping_summary,
               "camera_example": camera_summary,
               "cases": cases}
    (OUT / "simulation_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
