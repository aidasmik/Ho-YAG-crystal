"""Tkinter desktop interface for the Ho:YAG and Yb:LuAG simulations.

Start from the repository root with ``.venv/bin/python examples/desktop_simulation.py``.
All calculations run in an owned worker process through the persistent local
budget ledger. The interface never starts a web server.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import uuid

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hoyag.local_supervisor import BudgetLedger, Limits, run_bounded
from hoyag.structured_beam_gallery import BEAM_NAMES, PHASE_MASKS, SOLVER_MODES
from structured_beam_app import (budget_status, latest_completed_run,
                                 run_calculation,
                                 validate_request)
from ybluag_desktop_views import YbResultPanel
from run_ybyag_control import run_bounded_episode
from scientific_style import apply_scientific_style, PAPER, MUTED, NAVY

NUMERIC_RANGES = {
    "pump_W": (.001, 1000), "radius_mm": (.01, 10),
    "thickness_um": (1, 2000), "disk_radius_mm": (1, 10),
    "yb_at_percent": (5, 15), "waist_mm": (.1, 2),
    "signal_W": (.001, 100), "seed_W": (0, 1000),
    "pump_nm": (880, 1150), "pulsed_pump_nm": (880, 1150), "signal_nm": (880, 1150),
    "phase_strength_rad": (-50, 50), "seed_energy_nj": (.001, 100000),
    "source_fwhm_fs": (50, 10000), "seed_fwhm_ps": (.1, 1000),
    "repetition_rate_kHz": (.01, 100), "pump_passes": (1, 48),
    "signal_traversals": (1, 10), "regen_round_trips": (1, 60),
    "cavity_length_m": (.01, 2), "mirror_radius_m": (.02, 10),
    "disk_hr_reflectivity": (.5, 1), "held_retention": (.01, 1),
    "injection_efficiency": (.01, 1), "extraction_efficiency": (.01, 1),
    "distance_m": (0, 2), "slm_to_disk_m": (.001, 2),
    "density_seed": (-2e9, 2e9), "cluster_count": (2, 64),
    "cluster_contrast": (0, 1), "escape_yield": (0, 1),
    "operation_duration_s": (0, 120), "cooling_target_C": (20, 250),
    "thermal_internal_max_step_s": (.001, 120),
    "probe_seed": (0, 2**31-1),
    "cooling_h_max_W_m2K": (10000, 200000),
    "ideal_relay_power_retention": (.001, 1),
    "grid_n": (32, 768), "field_size_mm": (8, 24),
    "optical_z_steps": (1, 16), "thermal_nr": (4, 48),
    "thermal_nphi": (4, 96), "thermal_nz": (1, 24),
}
INTEGER_KEYS = {"grid_n", "optical_z_steps", "thermal_nr", "thermal_nphi",
                "thermal_nz", "pump_passes", "signal_traversals", "regen_round_trips",
                "density_seed", "cluster_count", "probe_seed"}


def validate_yb_payload(kind: str, values: dict) -> dict:
    """Reject malformed desktop entries before reserving shared compute time."""
    if kind not in ("cw", "structured", "pulsed", "pump_sweep"):
        raise ValueError("unknown Yb calculation")
    payload = dict(values)
    yag = payload.get("material") == "Yb:YAG"
    for key, (low, high) in NUMERIC_RANGES.items():
        if key not in payload:
            continue
        if yag and key == "yb_at_percent":
            low, high = 1, 22.9
        try:
            value = float(payload[key])
        except ValueError as exc:
            raise ValueError(f"{key} must be numeric") from exc
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        if key in INTEGER_KEYS and not value.is_integer():
            raise ValueError(f"{key} must be an integer")
        payload[key] = int(value) if key in INTEGER_KEYS else value
    if kind != "cw":
        if payload["selected_beam"] not in BEAM_NAMES or payload["phase_mask"] not in PHASE_MASKS:
            raise ValueError("unknown beam or phase mask")
        if payload["assembly_property_model"] not in (("yag_rt_proxy",) if yag else ("reference_10at", "proposal_12at")):
            raise ValueError("unknown Yb assembly property assumption")
    if kind in ("pulsed", "pump_sweep") and payload["seed_fwhm_ps"] * 1000 < payload["source_fwhm_fs"]:
        raise ValueError("stretched pulse must be at least as long as source pulse")
    if kind in ("pulsed", "pump_sweep") and payload["thermal_optical_mode"] not in (
            "cold", "lumped_phase", "coupled_steady"):
        raise ValueError("unknown thermal-optical mode")
    if kind in ("pulsed", "pump_sweep") and payload["architecture"] not in (
            "ideal_multipass", "regenerative"):
        raise ValueError("unknown amplifier architecture")
    if kind == "structured" and not .1 <= payload["radius_mm"] <= 5:
        raise ValueError("structured pump radius must be 0.1–5 mm")
    if yag and payload.get("thermal_optical_mode") == "coupled_steady":
        raise ValueError("Yb:YAG coupled hot gain needs missing temperature-dependent pump spectra")
    if yag:
        for key in ("pump_nm", "pulsed_pump_nm", "signal_nm"):
            if key in payload and not 905 <= payload[key] <= 1095:
                raise ValueError(f"{key}: Yb:YAG spectra cover 905–1095 nm")
    return payload


def field(key, label, default, choices=None):
    return key, label, str(default), choices


YB_FIELDS = (
    field("kind", "Calculation", "pulsed", ("pulsed", "structured", "cw")),
    field("selected_beam", "Target beam", BEAM_NAMES[0], BEAM_NAMES),
    field("phase_mask", "Optional phase correction", "none", PHASE_MASKS),
    field("phase_strength_rad", "Correction strength (rad)", math.pi),
    field("architecture", "Amplifier architecture", "ideal_multipass",
          ("ideal_multipass", "regenerative")),
    field("thermal_optical_mode", "Thermal-optical calculation", "lumped_phase",
          ("cold", "lumped_phase", "coupled_steady")),
    field("solver_mode", "Structured CW solver", "saturated_cw",
          ("weak_probe", "saturated_cw", "modal_cw")),
    field("pump_W", "Pump power (W)", 40),
    field("radius_mm", "Pump 1/e² radius (mm)", 1),
    field("thickness_um", "Disk thickness (µm)", 100),
    field("yb_at_percent", "Yb concentration (at.%)", 12,
          ("5", "10", "12", "15")),
    field("disk_radius_mm", "Disk radius (mm)", 5),
    field("assembly_property_model", "Assembly property assumption", "proposal_12at",
          ("reference_10at", "proposal_12at")),
    field("grid_n", "Optical grid points per axis", 96),
    field("field_size_mm", "Optical window (mm)", 12),
    field("optical_z_steps", "Optical depth cells", 4),
    field("thermal_nr", "Thermal radial cells", 8),
    field("thermal_nphi", "Thermal angular cells", 12),
    field("thermal_nz", "Thermal depth cells", 4),
    field("thermal_internal_max_step_s", "Thermal internal max step (s)", 10),
    field("probe_seed", "Temperature-probe noise seed", 0),
    field("waist_mm", "Gaussian source 1/e² radius (mm)", 0.6),
    field("signal_W", "Structured input (W)", 1),
    field("seed_W", "CW input (W)", 1),
    field("pump_nm", "CW pump wavelength (nm)", 938),
    field("pulsed_pump_nm", "Pulsed amplifier pump wavelength (nm)", 969),
    field("signal_nm", "CW signal wavelength (nm)", 1030),
    field("seed_energy_nj", "Seed energy (nJ)", 10),
    field("source_fwhm_fs", "Source FWHM (fs)", 300),
    field("seed_fwhm_ps", "Stretched FWHM (ps)", 10),
    field("repetition_rate_kHz", "Repetition (kHz)", 10),
    field("pump_passes", "Pump passes", 10),
    field("signal_traversals", "Signal traversals", 10),
    field("regen_round_trips", "Cavity round trips", 10),
    field("ideal_relay_power_retention", "Ideal relay power retention", 1),
    field("cavity_length_m", "Disk to mirror (m)", 0.25),
    field("mirror_radius_m", "Mirror curvature radius (m)", 0.5),
    field("disk_hr_reflectivity", "Disk HR reflectivity", 0.9995),
    field("held_retention", "Held round trip retention", 0.98),
    field("injection_efficiency", "Injection efficiency", 0.9),
    field("extraction_efficiency", "Extraction efficiency", 0.9),
    field("distance_m", "Output distance (m)", 0),
    field("slm_to_disk_m", "SLM to disk (m)", 0.25),
    field("density_seed", "Yb cluster seed", 17),
    field("cluster_count", "Yb clusters", 24),
    field("cluster_contrast", "Cluster contrast", 0.27),
    field("escape_yield", "Fluorescence escape yield", 0),
    field("operation_duration_s", "Operating time (s)", 30),
    field("cooling_mode", "Cooler control", "feedback", ("feedback", "sensor_feedback", "fixed")),
    field("cooling_target_C", "Cooler target (°C)", 25),
    field("cooling_h_max_W_m2K", "Maximum cooler h (W/m²K)", 100000),
)

YAG_FIELDS = tuple(
    (key, label,
     {"yb_at_percent": "20", "pump_nm": "969",
                  "assembly_property_model": "yag_rt_proxy"}.get(key, default),
     ("cold", "lumped_phase") if key == "thermal_optical_mode" else
     ("yag_rt_proxy",) if key == "assembly_property_model" else
     ("5", "10", "15", "20") if key == "yb_at_percent" else choices)
    for key, label, default, choices in YB_FIELDS
)

HO_FIELDS = (
    field("solver_mode", "Solver", "weak_probe", SOLVER_MODES),
    field("selected_beam", "Input beam", BEAM_NAMES[0], BEAM_NAMES),
    field("phase_mask", "Phase mask", "none", PHASE_MASKS),
    field("phase_strength_rad", "Phase strength (rad)", math.pi),
    field("density_seed", "Ho cluster seed", 17),
    field("cluster_count", "Ho clusters", 24),
    field("cluster_contrast", "Cluster contrast", 0.27),
    field("cluster_min_radius_mm", "Smallest cluster (mm)", 0.20),
    field("cluster_max_radius_mm", "Largest cluster (mm)", 1.25),
    field("post_disk_distance_m", "Output distance (m)", 0.25),
    field("seed_energy_nj", "Seed energy (nJ)", 10),
    field("seed_fwhm_ps", "Seed FWHM (ps)", 10),
    field("signal_traversals", "Signal traversals", 10),
    field("relay_distance_m", "Relay distance (m)", 0),
    field("cavity_ejection_efficiency", "Cavity ejection fraction", 1),
    field("cpu_workers", "CPU workers", 4),
    field("dn_dHo_m3", "Measured dn/dHo (m³/ion)", ""),
    field("dn_dExcited_m3", "Measured dn/dExcited (m³/ion)", ""),
    field("index_provenance", "Index data source", ""),
)


def run_yb(kind: str, payload: dict) -> tuple[dict, Path]:
    """Run the local Yb solver under the shared bounded supervisor."""
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    directory = ROOT / "results" / "desktop_runs" / run_id
    directory.mkdir(parents=True)
    request = directory / "request.json"
    output = directory / "result.json"
    request.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    coupled = kind in ("pulsed", "pump_sweep") or (kind == "structured" and
                                   payload.get("solver_mode") == "modal_cw")
    category = "coupled" if coupled else "profile"
    record = run_bounded(
        [sys.executable, str(ROOT / "examples" / "desktop_worker.py"),
         kind, str(request), str(output)], cwd=ROOT,
        log_path=directory / "execution.log",
        summary_path=directory / "execution.json",
        ledger=BudgetLedger(ROOT / ".local_runtime" / "budget.json", Limits()),
        label=f"desktop_{'ybyag' if payload.get('material') == 'Yb:YAG' else 'ybluag'}_{kind}_{run_id}",
        configured_seconds=900 if coupled else 180, category=category)
    if record["status"] != "completed" or record["exit_code"] != 0:
        log = (directory / "execution.log").read_text(encoding="utf-8")
        raise RuntimeError(f"{record['status']}: {log[-1600:]}")
    return json.loads(output.read_text(encoding="utf-8")), directory


YB_CONTROL_GROUPS = {
    "Seed & phase": {"kind", "selected_beam", "phase_mask", "phase_strength_rad",
                     "waist_mm", "seed_energy_nj", "source_fwhm_fs", "seed_fwhm_ps",
                     "repetition_rate_kHz", "signal_W", "seed_W"},
    "Yb crystal": {"kind", "thickness_um", "yb_at_percent", "disk_radius_mm",
                   "assembly_property_model", "density_seed", "cluster_count", "cluster_contrast"},
    "Pump & cooling": {"kind", "pump_W", "radius_mm", "pump_nm", "pulsed_pump_nm", "pump_passes",
                       "escape_yield", "thermal_optical_mode", "operation_duration_s",
                       "cooling_mode", "cooling_target_C", "cooling_h_max_W_m2K"},
    "Amplifier & optics": {"kind", "architecture", "solver_mode", "signal_nm", "signal_traversals",
                           "regen_round_trips", "ideal_relay_power_retention", "cavity_length_m",
                           "mirror_radius_m", "disk_hr_reflectivity", "held_retention",
                           "injection_efficiency", "extraction_efficiency", "distance_m", "slm_to_disk_m"},
    "Numerical settings": {"kind", "grid_n", "field_size_mm", "optical_z_steps", "thermal_nr",
                           "thermal_nphi", "thermal_nz", "thermal_internal_max_step_s", "probe_seed"},
}


class InputPanel(ttk.Frame):
    def __init__(self, parent, fields):
        super().__init__(parent)
        self.grouped = fields is YB_FIELDS or fields is YAG_FIELDS
        self.active_keys = {f[0] for f in fields}
        self.vars, self.rows = {}, {}
        self.group = tk.StringVar(value="Seed & phase")
        if self.grouped:
            heading = ttk.Frame(self, padding=(12, 10, 12, 8))
            heading.pack(fill="x")
            ttk.Label(heading, text="MODEL PARAMETERS", style="Eyebrow.TLabel").pack(anchor="w")
            concentration = next((item for item in fields if item[0] == "yb_at_percent"), None)
            if concentration is not None:
                key, label, default, choices = concentration
                row = ttk.Frame(heading)
                row.pack(fill="x", pady=(9, 4))
                ttk.Label(row, text=label, style="Field.TLabel").pack(side="left")
                var = tk.StringVar(value=default)
                ttk.Combobox(row, textvariable=var, values=choices,
                             state="readonly", width=10).pack(side="right")
                self.vars[key] = var
            ttk.Label(heading, text="PARAMETER GROUP", style="Input.TLabel").pack(
                anchor="w", pady=(7, 2))
            selector = ttk.Combobox(heading, textvariable=self.group,
                                   values=tuple(YB_CONTROL_GROUPS), state="readonly", width=27)
            selector.pack(fill="x")
            selector.bind("<<ComboboxSelected>>", lambda *_: self.show_only(self.active_keys))
            ttk.Separator(heading, orient="horizontal").pack(fill="x", pady=(10, 0))
        self.canvas = canvas = tk.Canvas(self, width=320, highlightthickness=0,
                                         background=PAPER)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas, padding=(12, 4, 12, 12))
        self.inner.bind("<Configure>",
                        lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        for row, (key, label, default, choices) in enumerate(fields):
            if key in self.vars:
                continue
            holder = ttk.Frame(self.inner)
            holder.grid(row=row, column=0, sticky="ew", pady=(2, 5))
            holder.columnconfigure(0, weight=1)
            ttk.Label(holder, text=label, style="Field.TLabel", wraplength=178).grid(
                row=0, column=0, sticky="w", padx=(0, 7))
            var = tk.StringVar(value=default)
            widget = (ttk.Combobox(holder, textvariable=var, values=choices,
                                   state="readonly", width=11)
                      if choices else ttk.Entry(holder, textvariable=var, width=13))
            widget.grid(row=0, column=1, sticky="e")
            self.vars[key], self.rows[key] = var, holder
        self.inner.columnconfigure(0, weight=1)
        # Scope wheel scrolling to this panel, including its child controls.
        self.winfo_toplevel().bind("<MouseWheel>", self._wheel, add="+")

    def _wheel(self, event):
        widget = event.widget
        while widget is not None:
            if widget is self:
                self.canvas.yview_scroll(-int(event.delta/120), "units")
                return "break"
            widget = getattr(widget, "master", None)

    def values(self):
        return {key: variable.get().strip() for key, variable in self.vars.items()}

    def show_only(self, keys):
        self.active_keys = set(keys)
        visible = (self.active_keys & YB_CONTROL_GROUPS[self.group.get()]
                   if self.grouped else self.active_keys)
        for key, widget in self.rows.items():
            widget.grid() if key in visible else widget.grid_remove()
        self.canvas.yview_moveto(0)


class ResultPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.ho_choice = tk.StringVar(value="Input/output and x/y profiles")
        self.ho_selector = ttk.Combobox(self, textvariable=self.ho_choice, state="readonly",
            values=("Input/output and x/y profiles", "Beam on Ho concentration",
                    "Ho concentration", "Phase mask", "Profiles only"))
        self.ho_selector.bind("<<ComboboxSelected>>", lambda *_: self._draw_ho_image())
        self.figure = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        columns = ("pass", "seed", "log_gain", "gain", "inversion", "transfer", "stored", "temperature")
        self.pass_tree = ttk.Treeview(self, columns=columns, show="headings", height=7)
        headings = ("Pass", "Seed nJ", "Log gain", "Gain", "Beam inversion %", "Net transfer nJ",
                    "Stored µJ", "T K")
        for key, heading in zip(columns, headings):
            self.pass_tree.heading(key, text=heading)
            self.pass_tree.column(key, width=95 if key != "inversion" else 150, anchor="center")
        self.caption = tk.Text(self, height=6, wrap="word")
        self.caption.pack(fill="x")
        self.caption.configure(state="disabled")

    def note(self, value):
        self.caption.configure(state="normal")
        self.caption.delete("1.0", "end")
        self.caption.insert("1.0", value)
        self.caption.configure(state="disabled")

    def draw_maps(self, maps, note):
        self.ho_selector.pack_forget()
        self.pass_tree.pack_forget()
        self.figure.clear()
        axes = list(self.figure.subplots(2, 3).flat)
        for ax, (name, values) in zip(axes, maps):
            if isinstance(values, dict):
                for label, profile in values.items():
                    curve = np.asarray(profile, dtype=float)
                    peak = float(np.max(curve))
                    ax.plot(curve / peak if peak > 0 else curve, label=label)
                ax.legend(fontsize=8)
                ax.set_xlabel("Center-cut sample")
                ax.set_ylabel("Normalized intensity")
                ax.set_title(name, fontsize=10)
                continue
            array = np.asarray(values, dtype=float)
            if array.ndim == 2:
                ax.imshow(array, origin="lower", cmap="viridis", aspect="equal")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                ax.plot(array)
                ax.set_xlabel("Center-cut sample")
            ax.set_title(name, fontsize=10)
        for ax in axes[len(maps):]:
            ax.set_visible(False)
        self.canvas.draw_idle()
        self.note(note)

    def draw_cw(self, result, directory):
        self.ho_selector.pack_forget()
        self.pass_tree.pack_forget()
        self.figure.clear()
        left, right = self.figure.subplots(1, 2)
        x = result["wavelength_nm"]
        left.plot(x, result["absorption_cross_section_cm2"], label="Absorption")
        left.plot(x, result["emission_cross_section_cm2"], label="Emission")
        left.set(xlabel="Wavelength (nm)", ylabel="Cross section (cm²)",
                 title="Reconstructed 20 °C spectra")
        left.legend()
        right.plot(result["coating_transmission"],
                   result["coating_output_intensity_W_m2"], marker="o")
        right.set(xlabel="Output coupler transmission",
                  ylabel="Screened output intensity (W/m²)", title="Coating screen")
        self.canvas.draw_idle()
        self.note(f"Signal output {result['signal_out_W']:.4g} W; absorbed pump "
                  f"{result['absorbed_pump_W']:.4g} W; selected transmission "
                  f"{result['selected_coating_transmission']:.3g}.\n"
                  f"{result['scope']}\nSaved: {directory}")

    def draw_yb(self, kind, result, directory, selected_beam):
        if kind == "cw":
            self.draw_cw(result, directory)
        elif kind == "structured":
            mode = result["modes"][selected_beam]
            self.draw_maps([
                ("Phase mask", result["phase_mask"]),
                ("Yb concentration", result["yb_density_entrance_1e26_m3"]),
                ("Input beam", mode["input_intensity"]),
                ("At disk", mode["disk_input_intensity"]),
                ("Output beam", mode["output_intensity"]),
                ("Input/output center cuts", {"Input": mode["input_profile"],
                                               "Output": mode["output_profile"]}),
            ], f"{selected_beam}; output {mode['output_power_W']:.4g} W; "
               f"heat {mode['net_heat_W_upper_or_assumed']:.4g} W. "
               f"Thermal: {result['thermal'].get('status')}.\n"
               f"{result['scope']}\nSaved: {directory}")
        else:
            self.draw_maps([
                ("SLM phase mask", result["phase_mask"]),
                ("Yb concentration", np.asarray(result["yb_density_m3"])[0]),
                ("Gaussian source", result["input_fluence_J_m2"]),
                ("At disk", result["disk_input_fluence_J_m2"]),
                ("Amplified output", result["output_fluence_J_m2"]),
                ("Input/output center cuts", {
                    "At disk": np.asarray(result["disk_input_fluence_J_m2"])[len(result["y_mm"]) // 2],
                    "Output": np.asarray(result["output_fluence_J_m2"])[len(result["y_mm"]) // 2]}),
            ], f"{selected_beam}; {result['architecture']}; output "
               f"{result['output_energy_J']*1e6:.4g} µJ, gain "
               f"{result['net_energy_gain']:.4g}, average "
               f"{result['average_output_W']:.4g} W; extraction shape retention "
               f"{result['extraction_shape_retention']:.3f}.\n"
               f"Thermal: {result['thermal'].get('status')}; "
               f"feedback applied: {result['thermal_feedback_applied']}. "
               f"{result['spectral_scope']}\nSaved: {directory}")

    def draw_ho(self, result):
        self.ho_result = result
        self.ho_choice.set("Input/output and x/y profiles")
        self.ho_selector.pack(side="top", fill="x", padx=6, pady=3,
                              before=self.canvas.get_tk_widget())
        self._draw_ho_image()
        summary = result["summary"]
        directory = ROOT / "results" / "structured_beams" / "runs" / result["run_id"]
        diagnostics = summary.get("solver_diagnostics", {})
        extraction = diagnostics.get("gain_medium_extraction_efficiency")
        extraction_note = (f" Gain-medium extraction: {100*extraction:.4g}%; "
                           f"cavity ejection: {100*diagnostics['cavity_ejection_efficiency']:.3g}%; "
                           f"optical closure: {diagnostics['optical_energy_balance_residual_J']:.2e} J."
                           if extraction is not None else "")
        passes = diagnostics.get("pass_diagnostics") or []
        self.pass_tree.delete(*self.pass_tree.get_children())
        for p in passes:
            self.pass_tree.insert("", "end", values=(
                p["pass"], f"{p['seed_energy_J']*1e9:.4g}",
                f"{p['small_signal_log_gain']:.5g}",
                f"{p['saturated_energy_gain']:.5g}",
                f"{100*p['beam_weighted_inversion_before_fraction']:.4g}→"
                f"{100*p['beam_weighted_inversion_after_fraction']:.4g}",
                f"{p['net_stimulated_transfer_J']*1e9:.4g}",
                f"{p['stored_laser_energy_after_J']*1e6:.4g}",
                f"{p['beam_weighted_temperature_K']:.2f}"))
        if passes:
            self.pass_tree.pack(fill="x", before=self.caption)
        else:
            self.pass_tree.pack_forget()
        self.note(f"{summary.get('solver_mode', 'Ho:YAG')}; "
                  f"{summary.get('settings', {}).get('selected_beam', '')}. "
                  f"Status: {result['execution']['status']}.{extraction_note}\n"
                  f"Stored energy is I7 photon-equivalent energy over the whole disk; "
                  f"net transfer includes reabsorption. Temperature is the held "
                  f"steady thermal map.\nSaved: {directory}")

    def _draw_ho_image(self):
        result = getattr(self, "ho_result", None)
        if result is None:
            return
        self.figure.clear()
        directory = ROOT / "results" / "structured_beams" / "runs" / result["run_id"]
        options = {"Input/output and x/y profiles":"beams", "Beam on Ho concentration":"beam_density",
                   "Ho concentration":"density", "Phase mask":"phase", "Profiles only":"profiles"}
        key = options[self.ho_choice.get()]
        path = directory / Path(result["images"][key]).name
        ax = self.figure.subplots()
        if path.is_file():
            with Image.open(path) as image:
                ax.imshow(image)
        else:
            ax.text(.5, .5, "Image unavailable", ha="center", va="center")
        ax.axis("off")
        self.canvas.draw_idle()


class DesktopSimulation(tk.Tk):
    def __init__(self, initial_material="Ho:YAG"):
        super().__init__()
        self.title(f"Thin-disk laser simulator | {initial_material}")
        self.configure(background=PAPER)
        apply_scientific_style(self)
        self.geometry(f"{min(1550, self.winfo_screenwidth()-60)}x{min(960, self.winfo_screenheight()-100)}")
        header = tk.Frame(self, background=PAPER, padx=16, pady=10)
        header.pack(fill="x")
        tk.Label(header, text="THIN-DISK LASER  /  OPTICAL SIMULATION",
                 background=PAPER, foreground=NAVY,
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        tk.Label(header, text="SOURCE  →  SLM  →  DISK  →  DETECTOR",
                 background=PAPER, foreground=MUTED,
                 font=("Cascadia Mono", 9)).pack(side="right")
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        self.minsize(1000, 650)
        self.running = False
        self.control_progress_path = None
        self.control_progress_mtime = None
        self.last_yb_payload = None
        self.yb_pages = {}
        self.status = tk.StringVar(value="Ready. Calculations have per-run time and memory limits.")
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        for material, fields in (("Yb:LuAG", YB_FIELDS), ("Yb:YAG", YAG_FIELDS)):
            inputs, result, button = self.make_tab(
                notebook, material, fields,
                lambda name=material: self.calculate_yb(name), YbResultPanel)
            sweep = ttk.Button(inputs.master, text="Pump → output curve (5 points)",
                command=lambda name=material: self.calculate_yb_sweep(name), state="disabled")
            sweep.pack(fill="x", padx=8, pady=(0, 8))
            camera = ttk.Button(inputs.master, text="Export camera data (1080p)…",
                command=lambda name=material: self.camera_dialog(name), state="disabled")
            camera.pack(fill="x", padx=8, pady=(0, 8))
            dataset = None
            if material == "Yb:YAG":
                dataset = ttk.Button(inputs.master, text="Generate NN dataset…",
                    command=self.dataset_dialog)
                dataset.pack(fill="x", padx=8, pady=(0, 8))
                control = ttk.Button(inputs.master, text="Closed-loop correction…",
                    command=self.control_dialog)
                control.pack(fill="x", padx=8, pady=(0, 8))
            else:
                control = None
            self.yb_pages[material] = dict(input=inputs, result=result, button=button,
                                           sweep=sweep, camera=camera, dataset=dataset,
                                           control=control,
                                           last_payload=None)
            for key in ("kind", "architecture"):
                inputs.vars[key].trace_add("write", lambda *_, name=material: self.update_yb_fields(name))
            self.update_yb_fields(material)
            if material == "Yb:YAG":
                result.model_note.set("Yb:YAG repository spectra at 20 °C · heated pump spectra missing · lumped phase is an approximation.")
        luag = self.yb_pages["Yb:LuAG"]
        self.yb_input, self.yb_result, self.yb_button = luag['input'], luag['result'], luag['button']
        self.yb_sweep_button = luag['sweep']
        self.ho_input, self.ho_result, self.ho_button = self.make_tab(
            notebook, "Ho:YAG", HO_FIELDS, self.calculate_ho)
        ttk.Separator(self, orient="horizontal").pack(fill="x")
        bar = ttk.Frame(self, padding=(12, 6))
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=95)
        self.progress.pack(side="left", padx=(0, 10))
        bar.pack(fill="x")
        ttk.Label(bar, textvariable=self.status, style="Info.TLabel").pack(
            side="left", fill="x", expand=True)
        self.cancel_button = ttk.Button(bar, text="Stop correction", command=self.cancel_control,
                                        state="disabled")
        self.cancel_button.pack(side="right", padx=(8, 0))
        ttk.Button(bar, text="Run limits", command=self.show_budget).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.close)
        latest = latest_completed_run()
        if latest:
            self.ho_result.draw_ho(latest)
        self.restore_yb_result("Yb:LuAG")
        self.restore_yb_result("Yb:YAG")
        self.material_tabs = notebook
        notebook.select({"Yb:LuAG": 0, "Yb:YAG": 1, "Ho:YAG": 2}[initial_material])

    def make_tab(self, notebook, title, fields, command, result_class=ResultPanel):
        page = ttk.Panedwindow(notebook, orient="horizontal")
        notebook.add(page, text=title)
        left = ttk.Frame(page)
        inputs = InputPanel(left, fields)
        inputs.pack(fill="both", expand=True)
        button = ttk.Button(left, text="Run simulation", style="Accent.TButton", command=command)
        button.pack(fill="x", padx=8, pady=8)
        page.add(left, weight=0)
        result = result_class(page)
        page.add(result, weight=1)
        return inputs, result, button

    def update_yb_fields(self, material="Yb:LuAG"):
        page = self.yb_pages[material]
        inputs = page["input"]
        kind = inputs.vars["kind"].get()
        common = {"kind", "pump_W", "radius_mm", "thickness_um", "disk_radius_mm",
                  "yb_at_percent",
                  "assembly_property_model", "grid_n",
                  "field_size_mm", "optical_z_steps", "thermal_nr",
                  "thermal_nphi", "thermal_nz"}
        if kind == "cw":
            common -= {"grid_n", "field_size_mm", "optical_z_steps",
                       "thermal_nr", "thermal_nphi", "thermal_nz",
                       "disk_radius_mm", "assembly_property_model"}
        shaped = {"selected_beam", "phase_mask", "phase_strength_rad", "waist_mm",
                  "distance_m", "slm_to_disk_m", "density_seed", "cluster_count",
                  "cluster_contrast", "escape_yield"}
        if kind == "cw":
            keys = common | {"pump_nm", "signal_nm", "seed_W"}
        elif kind == "structured":
            keys = common | shaped | {"solver_mode", "signal_W"}
        else:
            keys = common | shaped | {"architecture", "seed_energy_nj",
                   "thermal_optical_mode",
                   "source_fwhm_fs", "seed_fwhm_ps", "repetition_rate_kHz",
                   "pump_passes", "pulsed_pump_nm", "signal_traversals", "operation_duration_s",
                   "thermal_internal_max_step_s",
                   "probe_seed",
                   "cooling_mode", "cooling_target_C", "cooling_h_max_W_m2K"}
            if inputs.vars["architecture"].get() == "ideal_multipass":
                keys.add("ideal_relay_power_retention")
            if inputs.vars["architecture"].get() == "regenerative":
                keys |= {"regen_round_trips", "cavity_length_m", "mirror_radius_m",
                         "disk_hr_reflectivity", "held_retention",
                         "injection_efficiency", "extraction_efficiency"}
        if material == "Yb:YAG":
            keys.add("yb_at_percent")
            if kind == "structured":
                keys.add("pulsed_pump_nm")
        page["keys"] = keys
        inputs.show_only(keys)
        if material == "Yb:LuAG":
            self.active_yb_keys = keys

    def calculate_yb(self, material="Yb:LuAG"):
        if self.running:
            return
        page = self.yb_pages[material]
        values = {key: value for key, value in page['input'].values().items() if key in page['keys']}
        kind = values.pop("kind")
        values['material'] = material
        try:
            values = validate_yb_payload(kind, values)
        except ValueError as exc:
            messagebox.showerror(f"Invalid {material} input", str(exc))
            return
        beam = values.get("selected_beam", "Gaussian TEM00")
        self.start(lambda: ("yb", material, kind, beam, values, *run_yb(kind, values)))

    def calculate_yb_sweep(self, material="Yb:LuAG"):
        page = self.yb_pages[material]
        if self.running or page['last_payload'] is None:
            return
        payload = dict(page['last_payload'])
        try:
            validate_yb_payload("pump_sweep", payload)
        except ValueError as exc:
            messagebox.showerror("Invalid pump sweep", str(exc))
            return
        self.start(lambda: ("yb_sweep", material, *run_yb("pump_sweep", payload)))

    def restore_yb_result(self, material="Yb:LuAG"):
        panel = self.yb_pages[material]["result"]
        root = ROOT / "results" / "desktop_runs"
        if not root.is_dir():
            return
        for directory in sorted(root.iterdir(), reverse=True):
            request_file, result_file = directory / "request.json", directory / "result.json"
            if not request_file.is_file() or not result_file.is_file():
                continue
            try:
                request = json.loads(request_file.read_text(encoding="utf-8"))
                result = json.loads(result_file.read_text(encoding="utf-8"))
                if request.get("material", "Yb:LuAG") != material:
                    continue
                if "points" in result:
                    continue
                kind = ("pulsed" if "architecture" in result else
                        "structured" if "modes" in result else "cw")
                beam = request.get("selected_beam", "Gaussian TEM00")
                result.setdefault("disk_radius_mm", float(request.get("disk_radius_mm", 5)))
                panel.draw(kind, result, directory, beam)
                if kind == "pulsed" and "output_fluence_J_m2" in result:
                    self.yb_pages[material]["camera"].configure(state="normal")
                panel.set_provenance("SAVED RESULT · inputs may differ; run to update")
                self.status.set("Showing a saved Yb result; input controls use current defaults.")
                return
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                continue

    def calculate_ho(self):
        if self.running:
            return
        try:
            values = validate_request(self.ho_input.values())
        except ValueError as exc:
            messagebox.showerror("Invalid Ho:YAG input", str(exc))
            return
        self.start(lambda: ("ho", run_calculation(values)))

    def start(self, task, *, cancellable=False):
        self.running = True
        self.cancel_event = threading.Event()
        self.cancel_button.configure(state="normal" if cancellable else "disabled")
        self.progress.start(12)
        for page in self.yb_pages.values():
            page["button"].configure(state="disabled")
            page["sweep"].configure(state="disabled")
            page["camera"].configure(state="disabled")
            if page["dataset"] is not None:
                page["dataset"].configure(state="disabled")
            if page["control"] is not None:
                page["control"].configure(state="disabled")
        self.ho_button.configure(state="disabled")
        self.status.set("Calculating in a bounded worker…")

        def work():
            try:
                result = task()
                self.after(0, lambda: self.complete(result))
            except Exception as exc:
                self.after(0, lambda error=str(exc): self.failed(error))

        threading.Thread(target=work, daemon=True).start()

    def _enable_controls(self):
        self.running = False
        self.control_progress_path = None
        self.progress.stop()
        self.cancel_button.configure(state="disabled")
        self.ho_button.configure(state="normal")
        for page in self.yb_pages.values():
            page['button'].configure(state="normal")
            page['sweep'].configure(state="normal" if page['last_payload'] else "disabled")
            panel = page['result']
            page['camera'].configure(state="normal" if panel.kind == "pulsed" and
                                     panel.directory is not None and
                                     panel.result is not None and
                                     "output_fluence_J_m2" in panel.result else "disabled")
            if page['dataset'] is not None:
                page['dataset'].configure(state="normal")
            if page['control'] is not None:
                page['control'].configure(state="normal")

    def complete(self, value):
        if value[0] == "yb":
            _, material, kind, beam, payload, result, directory = value
            page = self.yb_pages[material]
            page['result'].draw(kind, result, directory, beam)
            page['last_payload'] = payload if kind == "pulsed" else None
            if material == "Yb:LuAG":
                self.last_yb_payload = page['last_payload']
            self.status.set(f"{material} {kind} completed; saved in {directory}")
        elif value[0] == "yb_sweep":
            _, material, result, directory = value
            self.yb_pages[material]['result'].update_sweep(result['points'])
            self.status.set(f"{material} cold pump curve completed; saved in {directory}")
        elif value[0] == "camera":
            _, material, output = value
            self.status.set(f"{material} exploratory camera frames saved in {output}")
            messagebox.showinfo("Camera frames exported",
                                f"Saved {output}\n\nSensor parameters are illustrative. "
                                "Metadata marks this export as unqualified for training ground truth.")
        elif value[0] == "dataset":
            self.status.set(f"Yb:YAG grouped dataset generated: {value[1]}")
            messagebox.showinfo("Dataset generation complete", str(value[1]))
        elif value[0] == "control":
            _, directory, result = value
            self.yb_pages["Yb:YAG"]["result"].load_control(result)
            self.status.set(f"Correction episode: {result['status']}; "
                            f"{result['evaluations']} full solves; saved in {directory}")
        else:
            self.ho_result.draw_ho(value[1])
            self.status.set(f"Ho:YAG calculation completed: {value[1]['run_id']}")
        self._enable_controls()

    def camera_dialog(self, material):
        page = self.yb_pages[material]
        panel = page["result"]
        if self.running or panel.directory is None or panel.kind != "pulsed":
            return
        from ybluag.camera_dataset import CameraSettings, suggest_optical_throughput
        try:
            suggested_throughput = suggest_optical_throughput(panel.result, CameraSettings())
        except (ValueError, KeyError):
            suggested_throughput = 1e-8
        dialog = tk.Toplevel(self)
        dialog.title(f"{material} camera export")
        dialog.transient(self)
        dialog.resizable(False, False)
        body = ttk.Frame(dialog, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="1080p monochrome camera · exploratory data",
                  style="Eyebrow.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        defaults = (
            ("frames", "Frames per solved state", "2"),
            ("seed", "Random seed", "17"),
            ("fov", "Object field width (mm)", "10"),
            ("qe", "QE at signal wavelength", "0.05"),
            ("throughput", "Optical throughput (suggested)", f"{suggested_throughput:.4g}"),
            ("pulses", "Pulses per exposure", "100"),
            ("read", "Read noise (electrons RMS)", "3"),
            ("temp", "Detector temperature jitter (°C)", "0.1"),
            ("other", "Other solved run dirs (; separated)", ""),
        )
        variables = {}
        for row, (key, label, default) in enumerate(defaults, 1):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
            var = tk.StringVar(value=default)
            ttk.Entry(body, textvariable=var, width=40).grid(row=row, column=1, sticky="ew", pady=3)
            variables[key] = var
        ttk.Label(body, text="Extra runs add independent physical states; repeated frames add sensor noise.",
                  wraplength=530).grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 2))
        ttk.Label(body, text="Camera values need calibration. Yb:YAG hot gain remains approximate.",
                  wraplength=530).grid(row=11, column=0, columnspan=2, sticky="w")

        def submit():
            try:
                frames = int(variables["frames"].get())
                seed = int(variables["seed"].get())
                fov = float(variables["fov"].get())
                qe = float(variables["qe"].get())
                throughput = float(variables["throughput"].get())
                pulses = int(variables["pulses"].get())
                read = float(variables["read"].get())
                temp = float(variables["temp"].get())
                others = [Path(s.strip()) for s in variables["other"].get().split(";") if s.strip()]
                CameraSettings(object_fov_width_mm=fov, qe_at_signal=qe,
                               optical_throughput=throughput, pulses_per_exposure=pulses,
                               read_noise_e=read, sensor_temperature_jitter_C=temp)
                if not 1 <= frames <= 8 or seed < 0 or (len(others)+1)*frames > 8:
                    raise ValueError("Use 1–8 total frames and a nonnegative random seed")
                runs = [panel.directory, *others]
                if any(not (path/"request.json").is_file() or
                       not (path/"result.json").is_file() for path in runs):
                    raise ValueError("Each run directory needs request.json and result.json")
            except (ValueError, OSError) as exc:
                messagebox.showerror("Camera settings", str(exc), parent=dialog)
                return
            dialog.destroy()
            self.start(lambda: ("camera", material, self.export_camera(
                material, runs, frames, seed, fov, qe, throughput, pulses, read, temp)))

        ttk.Button(body, text="Export frames", style="Accent.TButton",
                   command=submit).grid(row=12, column=1, sticky="e", pady=(14, 0))

    @staticmethod
    def export_camera(material, runs, frames, seed, fov, qe, throughput, pulses, read, temp):
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        directory = ROOT / "results" / "camera_datasets" / run_id
        directory.mkdir(parents=True)
        output = directory / "camera"
        command = [sys.executable, str(ROOT / "examples" / "yb_camera_dataset.py"),
                   "--output", str(output), "--frames-per-state", str(frames),
                   "--seed", str(seed), "--fov-mm", str(fov), "--qe", str(qe),
                   "--throughput", str(throughput), "--pulses-per-exposure", str(pulses),
                   "--read-noise-e", str(read), "--sensor-temp-jitter-C", str(temp),
                   "--runs", *(str(path) for path in runs)]
        record = run_bounded(
            command, cwd=ROOT, log_path=directory/"execution.log",
            summary_path=directory/"execution.json",
            ledger=BudgetLedger(ROOT/".local_runtime"/"budget.json", Limits()),
            label=f"desktop_{material.lower().replace(':', '')}_camera_{run_id}",
            configured_seconds=180, category="profile")
        if record["status"] != "completed" or record["exit_code"] != 0:
            log = (directory/"execution.log").read_text(encoding="utf-8")
            raise RuntimeError(f"{record['status']}: {log[-1600:]}")
        return output.with_suffix(".npz")

    def dataset_dialog(self):
        if self.running:
            return
        dialog = tk.Toplevel(self)
        dialog.title("Yb:YAG grouped NN dataset")
        dialog.transient(self)
        body = ttk.Frame(dialog,padding=18)
        body.pack(fill="both",expand=True)
        ttk.Label(body,text="Physical states → phase-diverse cameras → grouped splits",
                  style="Eyebrow.TLabel").grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,10))
        run_id=time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
        defaults=(
            ("config","Configuration JSON",str(ROOT/"config"/"ybyag_nn_dataset.json")),
            ("output","Output directory",str(ROOT/"results"/"ybyag_nn_dataset"/run_id)),
            ("points","Operating points per setup","1"),
        )
        vars_={}
        for row,(key,label,default) in enumerate(defaults,1):
            ttk.Label(body,text=label).grid(row=row,column=0,sticky="w",padx=(0,12),pady=3)
            vars_[key]=tk.StringVar(value=default)
            ttk.Entry(body,textvariable=vars_[key],width=62).grid(row=row,column=1,sticky="ew",pady=3)
        ttk.Label(body,text="Edit the JSON ranges and nominal Yb:YAG point before generating. "
                  "Only solver-valid near-room-temperature states are exported.",
                  wraplength=600).grid(row=4,column=0,columnspan=2,sticky="w",pady=(10,3))

        def submit():
            try:
                config=Path(vars_["config"].get())
                output=Path(vars_["output"].get())
                points=int(vars_["points"].get())
                if not config.is_file() or not 1<=points<=2:
                    raise ValueError("Choose an existing config and 1–2 operating points")
                json.loads(config.read_text(encoding="utf-8"))
            except (OSError,ValueError) as exc:
                messagebox.showerror("Dataset configuration",str(exc),parent=dialog)
                return
            dialog.destroy()
            self.start(lambda:("dataset",self.run_dataset(config,output,points)))

        ttk.Button(body,text="Generate bounded dataset",style="Accent.TButton",
                   command=submit).grid(row=5,column=1,sticky="e",pady=(12,0))

    @staticmethod
    def run_dataset(config, output, points):
        command=[sys.executable,str(ROOT/"examples"/"ybyag_nn_dataset.py"),
                 "--config",str(config),"--output",str(output),
                 "--points-per-setup",str(points)]
        run=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=960)
        if run.returncode:
            raise RuntimeError((run.stderr or run.stdout)[-3000:])
        return output/"manifest.json"

    def cancel_control(self):
        if self.running and hasattr(self, "cancel_event"):
            self.cancel_event.set()
            self.status.set("Stopping the supervised correction worker…")

    def _poll_control_progress(self):
        path=self.control_progress_path
        if not self.running or path is None:
            return
        try:
            if path.is_file():
                changed=path.stat().st_mtime_ns
                if changed!=self.control_progress_mtime:
                    progress=json.loads(path.read_text(encoding="utf-8"))
                    if progress.get("steps"):
                        panel=self.yb_pages["Yb:YAG"]["result"]
                        panel.load_control(progress)
                        panel.control_view.show(len(progress["steps"])-1)
                        self.control_progress_mtime=changed
                        live=progress.get("latest_observation") or {}
                        self.status.set(f"Correction running: physical iteration {live.get('evaluation', 0)} "
                                        f"· optimization cycle {progress['steps'][-1]['iteration']} "
                                        "shown with active camera noise · Stop correction to cancel")
        except (OSError,ValueError,KeyError,TypeError):
            pass  # An atomic snapshot may not be available yet.
        self.after(400,self._poll_control_progress)

    def control_dialog(self):
        if self.running:
            return
        dialog=tk.Toplevel(self)
        dialog.title("Yb:YAG | correction experiment")
        dialog.transient(self)
        dialog.geometry(f"{min(1340, self.winfo_screenwidth()-80)}x"
                        f"{min(740, self.winfo_screenheight()-100)}")
        body=ttk.Frame(dialog,padding=(14,12))
        body.pack(fill="both",expand=True)
        ttk.Label(body,text="CLOSED-LOOP / EXPERIMENT SETUP",
                  style="Eyebrow.TLabel").grid(
                  row=0,column=0,columnspan=3,sticky="w",pady=(0,3))
        ttk.Label(body,text="Gaussian seed → SLM → thin disk → two diagnostic cameras → phase update",
                  style="Info.TLabel").grid(
                  row=1,column=0,columnspan=3,sticky="w",pady=(0,10))
        current=self.yb_pages["Yb:YAG"]["input"].values()
        specifications=(
            ("target","Structured target",current["selected_beam"],BEAM_NAMES),
            ("correction_enabled","Correction enabled","yes",("yes","no")),
            ("mode","Episode mode","in_situ",("snapshot","in_situ")),
            ("method","Controller method","hybrid",("hybrid","interferometric","response_matrix","spgd")),
            ("enable_material","Crystal/contact variation","yes",("yes","no")),
            ("enable_slm_error","Imperfect SLM","yes",("yes","no")),
            ("enable_external_optics","External optical phase","yes",("yes","no")),
            ("enable_camera_noise","Camera and probe noise","yes",("yes","no")),
            ("enable_thermal_variation","Changing coolant/pump","yes",("yes","no")),
            ("coolant_setpoint_C","Coolant setpoint (°C)","20.2",None),
            ("coolant_jitter_K","Coolant jitter (K RMS)","0.03",None),
            ("pump_jitter_fraction","Pump power jitter (fraction)","0.005",None),
            ("pump_radius_jitter_fraction","Pump radius jitter (fraction)","0.003",None),
            ("pump_pointing_jitter_um","Pump pointing jitter (µm RMS)","5",None),
            ("camera_read_noise_e","Camera read noise (e⁻ RMS)","3",None),
            ("camera_background_e","Camera background (e⁻)","2",None),
            ("camera_gain_jitter_fraction","Exposure gain jitter (fraction)","0.005",None),
            ("camera_gain_random_walk_per_sqrt_s","Gain drift (fraction/√s)","0.001",None),
            ("probe_noise_scale","Probe noise multiplier","1",None),
            ("photodiode_noise_fraction","Photodiode noise (fraction RMS)","0.005",None),
            ("mode_count","Modal modes (hybrid/SPGD/matrix)","14",None),
            ("perturbation_rad","Probe step (rad)","0.08",None),
            ("spgd_gain","SPGD update gain","0.25",None),
            ("phase_gain_rad","Interferometric update (rad RMS)","0.12",None),
            ("phase_smoothing_pixels","Phase-map smoothing (pixels)","0.7",None),
            ("phase_target_rms_rad","Measured phase target (rad RMS)","0.5",None),
            ("improvement_tolerance","Camera-loss allowance","0.002",None),
            ("restore_best_at_end","Recheck best command at end","yes",("yes","no")),
            ("max_update_rad","Max phase update (rad)","0.4",None),
            ("iterations","Optimization cycle limit","30",None),
            ("evaluation_limit","Full-solver evaluation limit","400",None),
            ("diagnostic_astigmatism_waves","Diagnostic astigmatism (waves)","0.25",None),
            ("slm_delay_s","SLM delay (s)","0.01",None),
            ("slm_settle_s","SLM settle (s)","0.02",None),
            ("control_period_s","Measurement period (s)","0.1",None),
            ("exposure_s","Exposure (s)","0.01",None),
            ("pump_W","Pump power (W; near-RT start)",
             str(min(float(current["pump_W"]),.1)),None),
            ("yb_at_percent","Yb concentration (at.%)",current["yb_at_percent"],None),
            ("grid_n","Optical grid",current["grid_n"],None),
            ("operation_duration_s","Thermal warm-up (s; per solve in snapshot)",
             current["operation_duration_s"],None),
        )
        sections=(
            ("OPTICAL / THERMAL STATE", {
                "target","mode","enable_material","enable_slm_error",
                "enable_external_optics","enable_thermal_variation",
                "coolant_setpoint_C","pump_W","yb_at_percent","grid_n",
                "operation_duration_s","diagnostic_astigmatism_waves"}),
            ("MEASUREMENT / DRIFT", {
                "enable_camera_noise","coolant_jitter_K","pump_jitter_fraction",
                "pump_radius_jitter_fraction","pump_pointing_jitter_um",
                "camera_read_noise_e","camera_background_e",
                "camera_gain_jitter_fraction","camera_gain_random_walk_per_sqrt_s",
                "probe_noise_scale","photodiode_noise_fraction","exposure_s",
                "slm_delay_s","slm_settle_s","control_period_s"}),
            ("CONTROL / LIMITS", {
                "correction_enabled","method","mode_count","perturbation_rad",
                "spgd_gain","phase_gain_rad","phase_smoothing_pixels",
                "phase_target_rms_rad",
                "improvement_tolerance",
                "restore_best_at_end","max_update_rad",
                "iterations","evaluation_limit"}),
        )
        variables={}
        for column,(title,keys) in enumerate(sections):
            group=ttk.LabelFrame(body,text=title,padding=(10,9))
            group.grid(row=2,column=column,sticky="nsew",padx=(0,10 if column<2 else 0))
            group.columnconfigure(0,weight=1)
            body.columnconfigure(column,weight=1,uniform="experiment")
            row=0
            for key,label,default,choices in specifications:
                if key not in keys:
                    continue
                ttk.Label(group,text=label,style="Field.TLabel",wraplength=220).grid(
                    row=row,column=0,sticky="w",padx=(0,8),pady=4)
                var=tk.StringVar(value=default)
                variables[key]=var
                control=(ttk.Combobox(group,textvariable=var,values=choices,
                                      state="readonly",width=16)
                         if choices else ttk.Entry(group,textvariable=var,width=18))
                control.grid(row=row,column=1,sticky="e",pady=4)
                row+=1
        def controller_method_changed(*_):
            if variables["method"].get()=="spgd":
                variables["iterations"].set("60")
                variables["evaluation_limit"].set("400")
                variables["control_period_s"].set("0.1")
            elif variables["method"].get() in ("interferometric", "hybrid"):
                variables["iterations"].set("30")
                variables["evaluation_limit"].set("400")
                variables["control_period_s"].set("0.1")
            else:
                variables["iterations"].set("3")
                variables["evaluation_limit"].set("100")
                variables["control_period_s"].set("0")
        variables["method"].trace_add("write",controller_method_changed)
        ttk.Label(body,text="Current Yb:YAG spectra and thermal properties permit only near-room-temperature states. "
                  "Changing coolant/pump requires in-situ mode. The worker rejects unsupported hot runs. "
                  "Hybrid mode uses measured interferometric phase updates and camera-based shape recovery when phase steps stall. "
                  "Pure interferometric, SPGD and response-matrix modes remain selectable. "
                  "At the selected grid the Nyquist spatial frequency is grid_n/(2 × field_size_mm); "
                  "the proposal's 40 mm⁻¹ needs a finer grid.",
                  wraplength=1250,style="Info.TLabel").grid(
                      row=3,column=0,columnspan=3,sticky="w",pady=(10,4))
        def submit():
            try:
                config=json.loads((ROOT/"config/ybyag_control.json").read_text(encoding="utf-8"))
                ep=config["episode"];ctrl=config["controller"]
                ep.update(target=variables["target"].get(),mode=variables["mode"].get(),
                    correction_enabled=variables["correction_enabled"].get()=="yes",
                    enable_material=variables["enable_material"].get()=="yes",
                    enable_slm_error=variables["enable_slm_error"].get()=="yes",
                    enable_external_optics=variables["enable_external_optics"].get()=="yes",
                    enable_camera_noise=variables["enable_camera_noise"].get()=="yes",
                    enable_thermal_variation=variables["enable_thermal_variation"].get()=="yes",
                    coolant_setpoint_C=float(variables["coolant_setpoint_C"].get()),
                    coolant_jitter_K=float(variables["coolant_jitter_K"].get()),
                    pump_jitter_fraction=float(variables["pump_jitter_fraction"].get()),
                    pump_radius_jitter_fraction=float(variables[
                        "pump_radius_jitter_fraction"].get()),
                    pump_pointing_jitter_um=float(variables["pump_pointing_jitter_um"].get()),
                    camera_read_noise_e=float(variables["camera_read_noise_e"].get()),
                    camera_background_e=float(variables["camera_background_e"].get()),
                    camera_gain_jitter_fraction=float(variables["camera_gain_jitter_fraction"].get()),
                    camera_gain_random_walk_per_sqrt_s=float(
                        variables["camera_gain_random_walk_per_sqrt_s"].get()),
                    probe_noise_scale=float(variables["probe_noise_scale"].get()),
                    photodiode_noise_fraction=float(variables["photodiode_noise_fraction"].get()),
                    pump_W=float(variables["pump_W"].get()),
                    yb_at_percent=float(variables["yb_at_percent"].get()),
                    grid_n=int(variables["grid_n"].get()),
                    operation_duration_s=float(variables["operation_duration_s"].get()),
                    diagnostic_astigmatism_waves=float(variables["diagnostic_astigmatism_waves"].get()),
                    slm_delay_s=float(variables["slm_delay_s"].get()),
                    slm_settle_s=float(variables["slm_settle_s"].get()),
                    control_period_s=float(variables["control_period_s"].get()),
                    slm_drift_fraction_per_sqrt_s=(0.0005 if variables["method"].get() in ("spgd","interferometric","hybrid")
                                                  else 0.),
                    exposure_s=float(variables["exposure_s"].get()),
                    thickness_um=float(current["thickness_um"]),
                    pump_radius_mm=float(current["radius_mm"]),
                    disk_radius_mm=float(current["disk_radius_mm"]),
                    waist_mm=float(current["waist_mm"]),
                    seed_energy_nj=float(current["seed_energy_nj"]),
                    seed_fwhm_ps=float(current["seed_fwhm_ps"]),
                    repetition_rate_kHz=float(current["repetition_rate_kHz"]),
                    pump_passes=int(current["pump_passes"]),
                    signal_traversals=int(current["signal_traversals"]),
                    field_size_mm=float(current["field_size_mm"]))
                ctrl.update(method=variables["method"].get(),
                    mode_count=int(variables["mode_count"].get()),
                    perturbation_rad=float(variables["perturbation_rad"].get()),
                    spgd_gain=float(variables["spgd_gain"].get()),
                    phase_gain_rad=float(variables["phase_gain_rad"].get()),
                    phase_smoothing_pixels=float(variables["phase_smoothing_pixels"].get()),
                    phase_target_rms_rad=float(variables["phase_target_rms_rad"].get()),
                    improvement_tolerance=float(variables["improvement_tolerance"].get()),
                    restore_best_at_end=variables["restore_best_at_end"].get()=="yes",
                    max_update_rad=float(variables["max_update_rad"].get()),
                    iterations=int(variables["iterations"].get()),
                    evaluation_limit=int(variables["evaluation_limit"].get()))
                from ybyag_control.adapter import EpisodeConfig
                from ybyag_control import ControllerConfig
                EpisodeConfig(**ep);ControllerConfig(**ctrl)
            except (OSError,ValueError,KeyError) as exc:
                messagebox.showerror("Invalid correction settings",str(exc),parent=dialog)
                return
            run_id=time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
            directory=ROOT/"results/ybyag_control"/run_id
            directory.mkdir(parents=True,exist_ok=True)
            request=directory/"controller_config.json"
            request.write_text(json.dumps(config,indent=2),encoding="utf-8")
            dialog.destroy()
            self.control_progress_path=directory/"progress.json"
            self.control_progress_mtime=None
            def task():
                result_path=run_bounded_episode(request,directory,cancel=self.cancel_event)
                return "control",directory,json.loads(result_path.read_text(encoding="utf-8"))
            self.start(task,cancellable=True)
            self.after(400,self._poll_control_progress)
        ttk.Button(body,text="Run bounded correction",style="Accent.TButton",
                   command=submit).grid(row=4,column=2,sticky="e",pady=(12,0))

    def failed(self, error):
        self._enable_controls()
        self.status.set(f"Calculation failed: {error[:180]}")
        messagebox.showerror("Calculation failed", error)

    def show_budget(self):
        status = budget_status()
        messagebox.showinfo("Calculation limits",
                            "No cumulative attempt or time limit.\n"
                            f"Per coupled run: {status['per_run_seconds']:.0f} s\n"
                            f"Memory ceiling: {status['memory_bytes'] / 1024**3:.1f} GiB\n"
                            f"Past coupled runs: {status['coupled_attempts_used']}\n"
                            f"Active: {status['active']}")

    def close(self):
        if self.running:
            messagebox.showinfo("Calculation running", "Wait for the bounded calculation to finish.")
            return
        self.destroy()


if __name__ == "__main__":
    DesktopSimulation().mainloop()
