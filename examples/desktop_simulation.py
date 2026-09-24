"""Tkinter desktop interface for the Ho:YAG and Yb:LuAG simulations.

Start from the repository root with ``.venv/bin/python examples/desktop_simulation.py``.
All calculations run in an owned worker process through the persistent local
budget ledger. The interface never starts a web server.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
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
                                 run_calculation, start_new_budget,
                                 validate_request)
from ybluag_desktop_views import YbResultPanel

NUMERIC_RANGES = {
    "pump_W": (.001, 1000), "radius_mm": (.01, 10),
    "thickness_um": (1, 2000), "waist_mm": (.1, 2),
    "signal_W": (.001, 100), "seed_W": (0, 1000),
    "pump_nm": (880, 1150), "signal_nm": (880, 1150),
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
    "cooling_h_max_W_m2K": (10000, 200000),
    "grid_n": (32, 768), "field_size_mm": (8, 24),
    "optical_z_steps": (1, 16), "thermal_nr": (4, 48),
    "thermal_nphi": (4, 96), "thermal_nz": (1, 24),
}
INTEGER_KEYS = {"grid_n", "optical_z_steps", "thermal_nr", "thermal_nphi",
                "thermal_nz", "pump_passes", "signal_traversals", "regen_round_trips",
                "density_seed", "cluster_count"}


def validate_yb_payload(kind: str, values: dict) -> dict:
    """Reject malformed desktop entries before reserving shared compute time."""
    if kind not in ("cw", "structured", "pulsed", "pump_sweep"):
        raise ValueError("unknown Yb calculation")
    payload = dict(values)
    for key, (low, high) in NUMERIC_RANGES.items():
        if key not in payload:
            continue
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
    if kind in ("pulsed", "pump_sweep") and payload["seed_fwhm_ps"] * 1000 < payload["source_fwhm_fs"]:
        raise ValueError("stretched pulse must be at least as long as source pulse")
    if kind == "structured" and not .1 <= payload["radius_mm"] <= 5:
        raise ValueError("structured pump radius must be 0.1–5 mm")
    return payload


def field(key, label, default, choices=None):
    return key, label, str(default), choices


YB_FIELDS = (
    field("kind", "Calculation", "pulsed", ("pulsed", "structured", "cw")),
    field("selected_beam", "Target beam", BEAM_NAMES[0], BEAM_NAMES),
    field("phase_mask", "Added phase mask", "none", PHASE_MASKS),
    field("phase_strength_rad", "Added phase (rad)", math.pi),
    field("architecture", "Amplifier architecture", "regenerative",
          ("regenerative", "ideal_multipass")),
    field("solver_mode", "Structured CW solver", "saturated_cw",
          ("weak_probe", "saturated_cw", "modal_cw")),
    field("pump_W", "Pump power (W)", 40),
    field("radius_mm", "Pump radius (mm)", 1),
    field("thickness_um", "Disk thickness (µm)", 100),
    field("grid_n", "Optical grid points per axis", 96),
    field("field_size_mm", "Optical window (mm)", 12),
    field("optical_z_steps", "Optical depth cells", 4),
    field("thermal_nr", "Thermal radial cells", 8),
    field("thermal_nphi", "Thermal angular cells", 12),
    field("thermal_nz", "Thermal depth cells", 4),
    field("waist_mm", "Signal waist (mm)", 0.6),
    field("signal_W", "Structured input (W)", 1),
    field("seed_W", "CW input (W)", 1),
    field("pump_nm", "CW pump wavelength (nm)", 938),
    field("signal_nm", "CW signal wavelength (nm)", 1030),
    field("seed_energy_nj", "Seed energy (nJ)", 10),
    field("source_fwhm_fs", "Source FWHM (fs)", 300),
    field("seed_fwhm_ps", "Stretched FWHM (ps)", 10),
    field("repetition_rate_kHz", "Repetition (kHz)", 10),
    field("pump_passes", "Pump passes", 10),
    field("signal_traversals", "Signal traversals", 10),
    field("regen_round_trips", "Cavity round trips", 10),
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
    field("cooling_mode", "Cooler control", "feedback", ("feedback", "fixed")),
    field("cooling_target_C", "Cooler target (°C)", 40),
    field("cooling_h_max_W_m2K", "Maximum cooler h (W/m²K)", 100000),
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
    """Run the latest PR's Yb backend under the shared bounded supervisor."""
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
        label=f"desktop_ybluag_{kind}_{run_id}",
        configured_seconds=900 if coupled else 180, category=category)
    if record["status"] != "completed" or record["exit_code"] != 0:
        log = (directory / "execution.log").read_text(encoding="utf-8")
        raise RuntimeError(f"{record['status']}: {log[-1600:]}")
    return json.loads(output.read_text(encoding="utf-8")), directory


class InputPanel(ttk.Frame):
    def __init__(self, parent, fields):
        super().__init__(parent)
        canvas = tk.Canvas(self, width=320, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas, padding=8)
        self.inner.bind("<Configure>",
                        lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.vars = {}
        self.rows = {}
        for row, (key, label, default, choices) in enumerate(fields):
            holder = ttk.Frame(self.inner)
            holder.grid(row=row, column=0, sticky="ew", pady=3)
            ttk.Label(holder, text=label).pack(anchor="w")
            var = tk.StringVar(value=default)
            widget = (ttk.Combobox(holder, textvariable=var, values=choices,
                                   state="readonly", width=34)
                      if choices else ttk.Entry(holder, textvariable=var, width=37))
            widget.pack(fill="x")
            self.vars[key] = var
            self.rows[key] = holder
        self.inner.columnconfigure(0, weight=1)

    def values(self):
        return {key: variable.get().strip() for key, variable in self.vars.items()}

    def show_only(self, keys):
        for key, widget in self.rows.items():
            if key in keys:
                widget.grid()
            else:
                widget.grid_remove()


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
        self.title("Ho:YAG and Yb:LuAG simulation")
        self.geometry("1450x900")
        self.minsize(1000, 650)
        self.running = False
        self.last_yb_payload = None
        self.status = tk.StringVar(value="Ready. Calculations use the shared local budget.")
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        self.yb_input, self.yb_result, self.yb_button = self.make_tab(
            notebook, "Yb:LuAG", YB_FIELDS, self.calculate_yb, YbResultPanel)
        self.ho_input, self.ho_result, self.ho_button = self.make_tab(
            notebook, "Ho:YAG", HO_FIELDS, self.calculate_ho)
        self.yb_input.vars["kind"].trace_add("write", lambda *_: self.update_yb_fields())
        self.yb_input.vars["architecture"].trace_add("write", lambda *_: self.update_yb_fields())
        self.update_yb_fields()
        self.yb_sweep_button = ttk.Button(self.yb_input.master,
            text="Calculate five-point pump curve", command=self.calculate_yb_sweep,
            state="disabled")
        self.yb_sweep_button.pack(fill="x", padx=8, pady=(0, 8))
        bar = ttk.Frame(self, padding=6)
        bar.pack(fill="x")
        ttk.Label(bar, textvariable=self.status).pack(side="left", fill="x", expand=True)
        ttk.Button(bar, text="Budget status", command=self.show_budget).pack(side="right")
        ttk.Button(bar, text="New bounded budget", command=self.new_budget).pack(side="right", padx=5)
        self.protocol("WM_DELETE_WINDOW", self.close)
        latest = latest_completed_run()
        if latest:
            self.ho_result.draw_ho(latest)
        self.restore_yb_result()
        notebook.select(0 if initial_material == "Yb:LuAG" else 1)

    def make_tab(self, notebook, title, fields, command, result_class=ResultPanel):
        page = ttk.Panedwindow(notebook, orient="horizontal")
        notebook.add(page, text=title)
        left = ttk.Frame(page)
        inputs = InputPanel(left, fields)
        inputs.pack(fill="both", expand=True)
        button = ttk.Button(left, text="Calculate", command=command)
        button.pack(fill="x", padx=8, pady=8)
        page.add(left, weight=0)
        result = result_class(page)
        page.add(result, weight=1)
        return inputs, result, button

    def update_yb_fields(self):
        kind = self.yb_input.vars["kind"].get()
        previous = getattr(self, "previous_yb_kind", None)
        if kind != previous:
            thickness = self.yb_input.vars["thickness_um"]
            if kind == "structured" and thickness.get() == "100":
                thickness.set("150")
            elif kind == "pulsed" and thickness.get() == "150":
                thickness.set("100")
            self.previous_yb_kind = kind
        common = {"kind", "pump_W", "radius_mm", "thickness_um", "grid_n",
                  "field_size_mm", "optical_z_steps", "thermal_nr",
                  "thermal_nphi", "thermal_nz"}
        if kind == "cw":
            common -= {"grid_n", "field_size_mm", "optical_z_steps",
                       "thermal_nr", "thermal_nphi", "thermal_nz"}
        shaped = {"selected_beam", "phase_mask", "phase_strength_rad", "waist_mm",
                  "distance_m", "slm_to_disk_m", "density_seed", "cluster_count",
                  "cluster_contrast", "escape_yield"}
        if kind == "cw":
            keys = common | {"pump_nm", "signal_nm", "seed_W"}
        elif kind == "structured":
            keys = common | shaped | {"solver_mode", "signal_W"}
        else:
            keys = common | shaped | {"architecture", "seed_energy_nj",
                   "source_fwhm_fs", "seed_fwhm_ps", "repetition_rate_kHz",
                   "pump_passes", "signal_traversals", "operation_duration_s",
                   "cooling_mode", "cooling_target_C", "cooling_h_max_W_m2K"}
            if self.yb_input.vars["architecture"].get() == "regenerative":
                keys |= {"regen_round_trips", "cavity_length_m", "mirror_radius_m",
                         "disk_hr_reflectivity", "held_retention",
                         "injection_efficiency", "extraction_efficiency"}
        self.yb_input.show_only(keys)
        self.active_yb_keys = keys

    def calculate_yb(self):
        if self.running:
            return
        values = {key: value for key, value in self.yb_input.values().items()
                  if key in self.active_yb_keys}
        kind = values.pop("kind")
        try:
            values = validate_yb_payload(kind, values)
        except ValueError as exc:
            messagebox.showerror("Invalid Yb:LuAG input", str(exc))
            return
        beam = values.get("selected_beam", "Gaussian TEM00")
        self.start(lambda: ("yb", kind, beam, values, *run_yb(kind, values)))

    def calculate_yb_sweep(self):
        if self.running or self.last_yb_payload is None:
            return
        payload = dict(self.last_yb_payload)
        try:
            validate_yb_payload("pump_sweep", payload)
        except ValueError as exc:
            messagebox.showerror("Invalid pump sweep", str(exc))
            return
        self.start(lambda: ("yb_sweep", *run_yb("pump_sweep", payload)))

    def restore_yb_result(self):
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
                if "points" in result:
                    continue
                kind = ("pulsed" if "architecture" in result else
                        "structured" if "modes" in result else "cw")
                beam = request.get("selected_beam", "Gaussian TEM00")
                self.yb_input.vars["kind"].set(kind)
                for key, value in request.items():
                    if key in self.yb_input.vars:
                        self.yb_input.vars[key].set(str(value))
                self.update_yb_fields()
                self.yb_result.draw(kind, result, directory, beam)
                if kind == "pulsed":
                    self.last_yb_payload = request
                    self.yb_sweep_button.configure(state="normal")
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

    def start(self, task):
        self.running = True
        self.yb_button.configure(state="disabled")
        self.yb_sweep_button.configure(state="disabled")
        self.ho_button.configure(state="disabled")
        self.status.set("Calculating in a bounded worker…")

        def work():
            try:
                result = task()
                self.after(0, lambda: self.complete(result))
            except Exception as exc:
                self.after(0, lambda error=str(exc): self.failed(error))

        threading.Thread(target=work, daemon=True).start()

    def complete(self, value):
        self.running = False
        self.yb_button.configure(state="normal")
        self.ho_button.configure(state="normal")
        if value[0] == "yb":
            _, kind, beam, payload, result, directory = value
            self.yb_result.draw(kind, result, directory, beam)
            self.last_yb_payload = payload if kind == "pulsed" else None
            self.yb_sweep_button.configure(state="normal" if kind == "pulsed" else "disabled")
            self.status.set(f"Yb:LuAG {kind} calculation completed; saved in {directory}")
        elif value[0] == "yb_sweep":
            _, result, directory = value
            self.yb_result.update_sweep(result["points"])
            self.yb_sweep_button.configure(state="normal")
            self.status.set(f"Yb:LuAG five-point pump curve completed; saved in {directory}")
        else:
            self.ho_result.draw_ho(value[1])
            self.yb_sweep_button.configure(state="normal" if self.last_yb_payload else "disabled")
            self.status.set(f"Ho:YAG calculation completed: {value[1]['run_id']}")

    def failed(self, error):
        self.running = False
        self.yb_button.configure(state="normal")
        self.ho_button.configure(state="normal")
        self.yb_sweep_button.configure(state="normal" if self.last_yb_payload else "disabled")
        self.status.set(f"Calculation failed: {error[:180]}")
        messagebox.showerror("Calculation failed", error)

    def show_budget(self):
        status = budget_status()
        messagebox.showinfo("Shared compute budget",
                            f"Coupled attempts: {status['coupled_attempts_used']} / "
                            f"{status['coupled_attempts_limit']}\n"
                            f"Remaining: {status['remaining_seconds']:.0f} s\n"
                            f"Active: {status['active']}")

    def new_budget(self):
        status = budget_status()
        if not status["coupled_exhausted"]:
            messagebox.showinfo("Shared compute budget", "The current budget is available.")
            return
        if not messagebox.askyesno("Archive exhausted budget",
                                  "Archive the exhausted ledger and start one new bounded budget?"):
            return
        try:
            result = start_new_budget()
            self.status.set(f"New budget ready; old ledger: {result['archive']}")
        except Exception as exc:
            messagebox.showerror("Budget error", str(exc))

    def close(self):
        if self.running:
            messagebox.showinfo("Calculation running", "Wait for the bounded calculation to finish.")
            return
        self.destroy()


if __name__ == "__main__":
    DesktopSimulation().mainloop()
