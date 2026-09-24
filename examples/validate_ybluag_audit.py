"""Compact deterministic Yb:LuAG audit plots and machine-readable evidence.

Run from the repository root with `.venv/bin/python examples/validate_ybluag_audit.py`.
This does not run a full coupled thermal-optical solve or reserve a coupled-run
budget. Unimplemented validation items are recorded as such.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from hoyag.propagation import Grid2D, gaussian_beam, angular_spectrum_propagate
from hoyag.resonator import fluence_transfer
from test_ybluag_cold_sampling import cold_roundtrip
from ybluag.model import C, H, YbLuAGMaterial, trapezoid
from ybluag.pulsed import propagate_pulse
from ybluag.diagnostics import gain_feasibility, hardware_validity
from ybluag.field_metrics import coherent_overlap


def main():
    output = ROOT / "results" / "ybluag_audit_validation"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for window_mm in (8, 12, 16):
        for n in (96, 192, 384):
            radius, analytic, overlap, edge = cold_roundtrip(
                Grid2D.square(n, window_mm*1e-3))
            rows.append({"window_mm": window_mm, "grid_n": n,
                         "radius_um": radius*1e6, "analytic_radius_um": analytic*1e6,
                         "coherent_overlap": overlap, "edge_energy_fraction": edge})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for window_mm in (8, 12, 16):
        cases = [r for r in rows if r["window_mm"] == window_mm]
        axes[0].plot([r["grid_n"] for r in cases],
                     [r["radius_um"] for r in cases], "o-", label=f"{window_mm} mm")
        axes[1].semilogy([r["grid_n"] for r in cases],
                         [max(1-r["coherent_overlap"], 1e-16) for r in cases], "o-",
                         label=f"{window_mm} mm")
    axes[0].axhline(rows[0]["analytic_radius_um"], color="black", linestyle="--",
                    label="ABCD/q")
    axes[0].set(xlabel="grid points per axis", ylabel="first-round-trip 1/e field radius (µm)")
    axes[1].set(xlabel="grid points per axis", ylabel="1 − coherent overlap")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(output / "cold_propagation.png", dpi=160)
    plt.close(fig)

    structured = {}
    for kind in ("vortex", "quadrant"):
        fields = {}
        for n in (96, 192, 384):
            grid = Grid2D.square(n, .012)
            x, y = grid.mesh
            phase = (np.angle(x+1j*y) if kind == "vortex" else
                     np.pi*((x > 0) ^ (y > 0)))
            field = angular_spectrum_propagate(
                gaussian_beam(grid, .6e-3)*np.exp(1j*phase),
                grid, 1030e-9, .25)
            fields[n] = grid, field
        fine_grid, fine = fields[384]
        x, y = fine_grid.mesh
        points = np.stack((y, x), axis=-1)
        inside = x*x+y*y < (.003)**2
        rows_for_kind = []
        for n in (96, 192):
            grid, field = fields[n]
            interpolated = (
                RegularGridInterpolator((grid.y, grid.x), field.real,
                                        bounds_error=False, fill_value=0)(points) +
                1j*RegularGridInterpolator((grid.y, grid.x), field.imag,
                                           bounds_error=False, fill_value=0)(points))
            rows_for_kind.append({"grid_n": n,
                "overlap_vs_384": coherent_overlap(np.where(inside, fine, 0),
                                                    np.where(inside, interpolated, 0))})
        structured[kind] = rows_for_kind
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for kind, cases in structured.items():
        ax.plot([r["grid_n"] for r in cases]+[384],
                [r["overlap_vs_384"] for r in cases]+[1.0], "o-", label=kind)
    ax.set(xlabel="grid points per axis (12 mm window)",
           ylabel="complex-field overlap against 384² reference")
    ax.set_ylim(.85, 1.005)
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output / "structured_sampling.png", dpi=160)
    plt.close(fig)

    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.000973,
                              pump_wavelength_nm=938, signal_wavelength_nm=1030)
    sa, se = material.cross_sections_m2(1030)
    fsat = H*C/(1030e-9)/(sa+se)
    g0 = material.number_density_m3*((sa+se)*.75-sa)*100e-6
    pulse_rows = []
    for fraction in (.001, .5, 5):
        expected = fluence_transfer(fraction*fsat, g0, fsat)
        for count, depth in ((41, 1), (161, 4), (641, 16)):
            time = np.linspace(-30e-12, 30e-12, count)
            signal = np.exp(-4*np.log(2)*(time/10e-12)**2)
            signal *= fraction*fsat/trapezoid(signal, time)
            result = propagate_pulse(material, time, np.zeros_like(signal),
                                     signal, 100e-6, depth,
                                     initial_excited_fraction=.75)
            actual = float(trapezoid(result.signal_out_W_m2, time))
            pulse_rows.append({"input_fsat": fraction, "time_points": count,
                               "axial_cells": depth, "relative_error": actual/expected-1,
                               "population_min": float(np.min(result.final_excited_fraction_by_slice)),
                               "population_max": float(np.max(result.final_excited_fraction_by_slice))})
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for fraction in (.001, .5, 5):
        cases = [r for r in pulse_rows if r["input_fsat"] == fraction]
        ax.loglog([r["time_points"] for r in cases],
                  [abs(r["relative_error"]) for r in cases], "o-",
                  label=f"input {fraction:g} Fsat")
    ax.set(xlabel="temporal samples (axial cells 1, 4, 16)",
           ylabel="|fluence error against Frantz–Nodvik|")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output / "pulse_convergence.png", dpi=160)
    plt.close(fig)

    bound = gain_feasibility(material, thickness_m=100e-6, pump_passes=10,
                             signal_traversals=10, input_energy_J=10e-9,
                             requested_output_energy_J=100e-6)
    field = np.ones((4, 4), complex)
    report = {
        "cold_propagation": rows,
        "structured_sampling": structured,
        "frantz_nodvik": pulse_rows,
        "gain_feasibility": bound,
        "coherent_global_phase_invariance": coherent_overlap(field, field*np.exp(1j*.7)),
        "hardware": hardware_validity(pump_nm=938, coating_band_nm=(940, 1090)),
        "temperature_validity": "Covered by tests/test_ybluag.py; coupled hot-cavity convergence is not established",
        "photon_and_cavity_balance": "Covered by tests/test_ybluag_regenerative.py; fluorescence transport is not calibrated",
        "coupled_thermal_optical_convergence": "not_calculated: iterative local-temperature and per-encounter hot-optic model unavailable",
        "material_provenance": "See docs/YBLUAG_SPECTRA_COOLING_COATINGS.md; figure-guided reconstruction, not author arrays",
    }
    (output / "validation_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
