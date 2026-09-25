"""Bounded diagnostic, not a complete grid-convergence qualification."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ybluag.field_metrics import coherent_overlap
from ybluag.gallery import YbGallerySettings, simulate_pulsed_seed
from ybluag.model import YbLuAGMaterial
from ybluag.research_export import source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results" / "ybluag_ideal_partial_validation"
SHAPES = ("Gaussian TEM00", "Helical LG(0,+1)",
          "Flattop super-Gaussian", "Needle Bessel-Gaussian")


def field_on(coarse, fine):
    src = coarse["grid"]
    dst = fine["grid"]
    field = coarse["output_complex_field_sqrt_J_m"]
    points = np.column_stack((dst.mesh[1].ravel(), dst.mesh[0].ravel()))
    real = RegularGridInterpolator((src.y, src.x), field.real,
                                   bounds_error=False, fill_value=0)(points)
    imag = RegularGridInterpolator((src.y, src.x), field.imag,
                                   bounds_error=False, fill_value=0)(points)
    return (real+1j*imag).reshape(dst.shape)


def radius(result):
    grid = result["grid"]
    x, y = grid.mesh
    f = result["output_fluence_J_m2"]
    total = np.sum(f)
    cx, cy = np.sum(x*f)/total, np.sum(y*f)/total
    return float(np.sqrt(2*np.sum(((x-cx)**2+(y-cy)**2)*f)/total))


def main():
    started = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.973e-3,
                              pump_wavelength_nm=938)
    base = YbGallerySettings(pump_power_W=.01, pump_radius_m=1e-3,
                             thickness_m=100e-6, assembly_property_model="proposal_12at",
                             cluster_contrast=0, waist_m=.6e-3,
                             post_disk_distance_m=0, z_steps=1,
                             thermal_nr=4, thermal_nphi=4, thermal_nz=1,
                             field_size_m=.012, grid_n=64)
    kwargs = dict(seed_energy_J=1e-9, seed_fwhm_s=10e-12,
                  repetition_rate_Hz=1e4, signal_traversals=2,
                  pump_passes=2, architecture="ideal_multipass")
    rows = []
    for shape in SHAPES:
        coarse = simulate_pulsed_seed(material, base, shape,
                                      compute_thermal=False, **kwargs)
        fine = simulate_pulsed_seed(material, replace(base, grid_n=128), shape,
                                    compute_thermal=False, **kwargs)
        overlap = coherent_overlap(field_on(coarse, fine),
                                   fine["output_complex_field_sqrt_J_m"])
        rows.append({"shape": shape, "coarse_n": 64, "fine_n": 128,
                     "window_mm": 12, "coherent_overlap": overlap,
                     "output_energy_relative_change": abs(
                         fine["output_energy_J"]-coarse["output_energy_J"])/
                         fine["output_energy_J"],
                     "beam_radius_relative_change": abs(radius(fine)-radius(coarse)) /
                         radius(fine)})
    cold = simulate_pulsed_seed(material, base, SHAPES[0],
                                compute_thermal=False, **kwargs)
    hot = simulate_pulsed_seed(material, base, SHAPES[0],
                               compute_thermal=True,
                               thermal_optical_mode="coupled_steady", **kwargs)
    before = cold["output_fluence_J_m2"]
    after = hot["output_fluence_J_m2"]
    phase_delta = np.angle(np.exp(1j*(hot["output_phase"]-cold["output_phase"])))
    phase_delta[before < .01*float(np.max(before))] = np.nan
    limit = max(float(np.max(before)), float(np.max(after)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, data, title in zip(axes, (before, after, phase_delta),
                               ("Cold output fluence", "Coupled output fluence",
                                "Coupled minus cold phase")):
        im = ax.imshow(data, origin="lower", extent=(-6, 6, -6, 6),
                       cmap="viridis" if "fluence" in title else "RdBu_r",
                       vmin=0 if "fluence" in title else None,
                       vmax=limit if "fluence" in title else None)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.add_patch(plt.Circle((0, 0), 5, fill=False, color="white",
                                linestyle="--", linewidth=.8))
        fig.colorbar(im, ax=ax, label="J/m²" if "fluence" in title else "rad")
    plot = DEST / "cold_vs_coupled.png"
    fig.savefig(plot, dpi=150)
    plt.close(fig)
    report = {
        "status": "partial_diagnostic_not_convergence_qualified",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=ROOT, text=True).strip(),
        "source_sha256": source_fingerprint(),
        "runtime_s": time.perf_counter()-started,
        "shape_pitch_comparison": rows,
        "coupled_gaussian": {
            "peak_disk_C": hot["steady_coupled_temperature_max_C"],
            "cold_energy_J": cold["output_energy_J"],
            "coupled_energy_J": hot["output_energy_J"],
            "max_phase_delta_rad": float(np.nanmax(np.abs(phase_delta))),
            "convergence_history": hot["coupled_steady_convergence"]["history"]},
        "unresolved_refinements": ["optical_window", "optical_depth",
                                   "thermal_nr", "thermal_nphi", "thermal_nz",
                                   "mechanical_mesh", "coupled_tolerance", "time_step"],
        "dataset_export_allowed": False,
    }
    (DEST / "partial_convergence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"status": report["status"], "runtime_s": report["runtime_s"],
                      "shapes": len(rows), "plot": str(plot)}))


if __name__ == "__main__":
    main()
