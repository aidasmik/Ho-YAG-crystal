"""Native Tkinter result views for the Yb:LuAG calculator."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import numpy as np


def _array(value):
    return np.asarray(value, dtype=float)


def _line_pair(ax, x, y, label, style="-"):
    peak = float(np.max(y))
    ax.plot(x, y / peak if peak > 0 else y, style, label=label, linewidth=1.5)


class YbResultPanel(ttk.Frame):
    """Solver output organized as native plots and readable scientific metrics."""

    NAMES = ("Overview", "Beam and profiles", "Phase and Yb", "Gain and pulse",
             "Cooling timeline", "Thermal surfaces")

    def __init__(self, parent):
        super().__init__(parent)
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.frames = {}
        self.figures = {}
        self.canvases = {}
        self.overview = None
        for name in self.NAMES:
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=name)
            self.frames[name] = frame
            if name == "Overview":
                text = tk.Text(frame, wrap="word", font=("TkDefaultFont", 11), padx=14, pady=12)
                scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
                text.configure(yscrollcommand=scroll.set)
                text.pack(side="left", fill="both", expand=True)
                scroll.pack(side="right", fill="y")
                self.overview = text
            else:
                fig = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                self.figures[name] = fig
                self.canvases[name] = canvas
        self.kind = None
        self.result = None
        self.directory = None
        self.sweep_points = None

    def _visible(self, names):
        for name, frame in self.frames.items():
            self.tabs.tab(frame, state="normal" if name in names else "hidden")
        self.tabs.select(self.frames["Overview"])

    def _text(self, value):
        self.overview.configure(state="normal")
        self.overview.delete("1.0", "end")
        self.overview.insert("1.0", value)
        self.overview.configure(state="disabled")

    def _finish(self, name):
        self.canvases[name].draw_idle()

    def draw(self, kind, result, directory, beam):
        self.kind, self.result, self.directory, self.beam = kind, result, directory, beam
        self.sweep_points = None
        if kind == "cw":
            self._visible(("Overview", "Gain and pulse"))
            self._cw()
        elif kind == "structured":
            self._visible(("Overview", "Beam and profiles", "Phase and Yb",
                           "Gain and pulse", "Thermal surfaces"))
            self._structured()
        else:
            self._visible(self.NAMES)
            self._pulsed()

    def _map(self, ax, values, title, x=None, y=None, *, cmap="viridis", vmin=None, vmax=None,
             disk=False):
        data = np.ma.asarray(values, dtype=float)
        extent = ((float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
                  if x is not None and y is not None else None)
        image = ax.imshow(data, origin="lower", extent=extent, cmap=cmap,
                          vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        if extent:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            if disk:
                ax.add_patch(Circle((0, 0), 5, fill=False, color="white",
                                    linewidth=1, linestyle="--"))
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        return image

    def _beams(self, source, incident, output, reference, x, y, unit, title):
        name = "Beam and profiles"
        fig = self.figures[name]
        fig.clear()
        axes = fig.subplots(3, 2)
        cases = ((source, "Gaussian source"), (incident, "Shaped disk input"),
                 (output, "Amplified output"))
        for row, (values, label) in enumerate(cases):
            values = _array(values)
            image = self._map(axes[row, 0], values, label, x, y, cmap="inferno", disk=True)
            fig.colorbar(image, ax=axes[row, 0], shrink=.7, label=unit)
            ax = axes[row, 1]
            iy, ix = len(y) // 2, len(x) // 2
            _line_pair(ax, x, values[iy], "x cut")
            _line_pair(ax, y, values[:, ix], "y cut", "--")
            if row == 2 and reference is not None:
                ref = _array(reference)
                _line_pair(ax, x, ref[iy], "uniform Yb x", ":")
                _line_pair(ax, y, ref[:, ix], "uniform Yb y", "-.")
            ax.set(xlabel="position (mm)", ylabel="normalized center-cut fluence/irradiance",
                   title=f"{label} · top and side profiles")
            ax.set_xlim(-2, 2)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc="upper right")
        fig.suptitle(title, fontsize=12)
        self._finish(name)

    def _phase_density(self, result, x, y, structured=False):
        name = "Phase and Yb"
        fig = self.figures[name]
        fig.clear()
        axes = fig.subplots(2, 4).flat
        if structured:
            density = (result["yb_density_entrance_1e26_m3"],
                       result["yb_density_middle_1e26_m3"],
                       result["yb_density_xz_1e26_m3"])
            mode = result["modes"][self.beam]
            maps = ((result["target_phase_mask"], "Target SLM phase", "twilight"),
                    (result["aberration_phase_mask"], "Added phase", "twilight"),
                    (result["phase_mask"], "Applied phase", "twilight"),
                    (mode["output_phase"], "Output phase", "twilight"),
                    (density[0], "Yb entrance (10²⁶/m³)", "viridis"),
                    (density[1], "Yb mid-depth (10²⁶/m³)", "viridis"),
                    (density[2], "Yb x–z cut (10²⁶/m³)", "viridis"),
                    (result["pump_intensity_W_m2"], "Pump irradiance (W/m²)", "inferno"))
        else:
            density = _array(result["yb_density_m3"])[0] / 1e26
            maps = ((result["target_phase_mask"], "Target SLM phase", "twilight"),
                    (result["aberration_phase_mask"], "Added phase", "twilight"),
                    (result["phase_mask"], "Applied phase", "twilight"),
                    (result["phase_residual_rad"] if result.get("thermal_feedback_applied")
                     else result.get("cold_density_phase_residual_rad", result["phase_residual_rad"]),
                     "Output residual phase (rad)" if result.get("thermal_feedback_applied")
                     else "Cold density residual only (rad)", "RdBu_r"),
                    (density, "Yb entrance (10²⁶/m³)", "viridis"),
                    (result["input_phase"], "Source phase", "twilight"),
                    (result["disk_input_phase"], "Disk input phase", "twilight"),
                    (result["output_phase"], "Output phase", "twilight"))
        for ax, (data, title, cmap) in zip(axes, maps):
            if "phase" in title.lower():
                values = _array(data)
                if not structured and "residual" in title.lower():
                    weights = _array(result["output_fluence_J_m2"])
                    data = np.ma.masked_where(weights < .01 * weights.max(), values)
                elif not structured and title == "Output phase":
                    weights = _array(result["output_fluence_J_m2"])
                    data = np.ma.masked_where(weights < .01 * weights.max(), values)
                elif structured and title == "Output phase":
                    weights = _array(result["modes"][self.beam]["output_intensity"])
                    data = np.ma.masked_where(weights < .01 * weights.max(), values)
            if "x–z" in title:
                image = self._map(ax, data, title, cmap=cmap)
            else:
                image = self._map(ax, data, title, x, y, cmap=cmap,
                                  disk=title.startswith("Yb"))
            fig.colorbar(image, ax=ax, shrink=.62)
        fig.suptitle("Phase-only shaping and synthetic Yb concentration", fontsize=12)
        self._finish(name)

    def _thermal_surfaces(self, thermal, x):
        name = "Thermal surfaces"
        fig = self.figures[name]
        fig.clear()
        axes = fig.subplots(2, 2).flat
        available = thermal.get("status") in ("computed", "design_reference")
        if not available:
            for ax in axes:
                ax.axis("off")
            axes[0].text(.02, .7, thermal.get("reason", "No thermal result"),
                         transform=axes[0].transAxes, fontsize=12, wrap=True)
            self._finish(name)
            return
        label = ("5 K design reference; selected heat is outside calibrated range"
                 if thermal["status"] == "design_reference" else
                 "Calculated in supported 20–26.85 °C parameter range")
        shown = thermal.get("design_reference_maps", thermal)
        for ax, key, title in zip(axes[:3],
                                  ("scalar_roundtrip_opd_nm", "front_displacement_nm",
                                   "rear_displacement_nm"),
                                  ("Round-trip OPD (nm)", "Front displacement (nm)",
                                   "Rear displacement (nm)")):
            image = self._map(ax, shown[key], title, x, x, cmap="coolwarm", disk=True)
            fig.colorbar(image, ax=ax, shrink=.7)
        front = _array(shown["front_displacement_nm"])
        rear = _array(shown["rear_displacement_nm"])
        mid = len(x) // 2
        axes[3].plot(x, front[mid] - front[mid, mid], label="front")
        axes[3].plot(x, rear[mid] - rear[mid, mid], label="rear")
        axes[3].set(xlabel="x (mm)", ylabel="relative displacement (nm)",
                    title="Center cuts · own-center reference")
        axes[3].legend()
        axes[3].grid(alpha=.2)
        fig.suptitle(label, fontsize=12)
        self._finish(name)

    def _pulse_gain(self, result):
        name = "Gain and pulse"
        fig = self.figures[name]
        fig.clear()
        axes = fig.subplots(2, 2).flat
        regen = result.get("regenerative")
        if regen:
            rows = regen["roundtrip_energy_J"]
            energy = [result["input_energy_J"] * regen["injection_efficiency"]]
            energy += [row[2] for row in rows]
            energy += [result["output_energy_J"]]
            axes[0].plot(range(len(energy)), _array(energy) * 1e9, marker="o")
            axes[0].set(xlabel="round trip; last point extracted", ylabel="pulse energy (nJ)",
                        title="Regenerative storage and ejection")
        else:
            axes[0].plot((0, 1, 2), _array((result["input_energy_J"],
                         result["disk_output_energy_J"], result["output_energy_J"])) * 1e9,
                         marker="o")
            axes[0].set_xticks((0, 1, 2), ("seed", "disk exit", "output"))
            axes[0].set(ylabel="pulse energy (nJ)", title="Ideal multipass extraction")
        axes[0].grid(alpha=.2)
        if self.sweep_points:
            xs = [p["incident_pump_W"] for p in self.sweep_points]
            ys = [p["average_output_W"] for p in self.sweep_points]
            axes[1].plot(xs, ys, marker="o", color="tab:orange")
            axes[1].set_title("Five recalculated optical points")
        else:
            axes[1].plot([result["incident_pump_W"]], [result["average_output_W"]],
                         "o", color="tab:orange")
            axes[1].set_title("Selected pump/output point · sweep available")
        axes[1].set(xlabel="incident pump (W)", ylabel="average output (W)")
        axes[1].grid(alpha=.2)
        axes[2].plot(result["time_ps"], result["input_power_trace_W"], label="input")
        if result["output_power_trace_W"] is not None:
            axes[2].plot(result["time_ps"], result["output_power_trace_W"], label="output")
        axes[2].set(xlabel="time (ps)", ylabel="power (W)",
                    title=("Output temporal envelope unavailable (fluence-only)"
                           if result["output_power_trace_W"] is None else
                           "Time-sampled intensity transport"))
        axes[2].legend()
        axes[2].grid(alpha=.2)
        screen = result.get("spectral_gain_screen", {})
        if screen.get("status") == "small_signal_screen_only":
            wl = screen["wavelength_nm"]
            axes[3].plot(wl, screen["input_spectral_weights"], label="source weight")
            axes[3].set(xlabel="wavelength (nm)", ylabel="normalized source weight",
                        title="Shared-inversion small-signal screen")
            other = axes[3].twinx()
            other.plot(wl, screen["unsaturated_gain"], color="tab:orange",
                       label="unsaturated gain")
            other.set_ylabel("unsaturated gain ceiling")
        else:
            axes[3].axis("off")
            axes[3].text(.03, .95, screen.get("reason", "No spectral screen"),
                         transform=axes[3].transAxes, va="top", fontsize=10)
        fig.suptitle("Pulse gain and limited spectral diagnostics", fontsize=12)
        self._finish(name)

    def _cooling(self, result):
        name = "Cooling timeline"
        fig = self.figures[name]
        fig.clear()
        axes = fig.subplots(2, 2).flat
        timeline = result.get("thermal_timeline")
        if not timeline:
            for ax in axes:
                ax.axis("off")
            axes[0].text(.02, .7, "No transient cooler calculation requested.",
                         transform=axes[0].transAxes)
            self._finish(name)
            return
        t = _array(timeline["time_s"])
        x = np.maximum(t, 1e-8)
        valid = np.asarray(timeline["material_range_valid"], bool)
        series = (("disk_max_C", "Disk maximum (°C)", 1),
                  ("roundtrip_opd_pv_nm", "Round-trip OPD PV (nm)", 1),
                  ("front_displacement_pv_nm", "Front displacement PV (nm)", 1),
                  ("coolant_conductance_W_m2K", "Cooler h (kW/m²K)", 1e-3))
        for ax, (key, label, factor) in zip(axes, series):
            values = _array(timeline[key]) * factor
            ax.plot(x, np.ma.masked_where(~valid, values), color="tab:green",
                    marker="o", markersize=2, label="supported range")
            ax.plot(x, np.ma.masked_where(valid, values), color="tab:orange",
                    linestyle="--", marker="o", markersize=2,
                    label="extrapolated")
            ax.axvline(max(timeline["requested_time_s"], 1e-8),
                       color="tab:blue", linestyle=":", label="selected time")
            ax.set_xscale("log")
            ax.set(xlabel="time (s; 0 shown at left edge)", ylabel=label)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        status = ("Thermal phase applied" if timeline["requested_material_range_valid"]
                  else "Selected temperature outside calibrated 20–26.85 °C range; phase withheld")
        fig.suptitle(f"{status}; steady screen {timeline['steady_disk_max_C']:.2f} °C",
                     fontsize=12)
        self._finish(name)

    def _cw(self):
        result = self.result
        self._text(
            f"Yb:LuAG · 10 at.% · 20 °C CW material and coating screen\n\n"
            f"Incident pump: {result['absorbed_pump_W'] + result['pump_out_W']:.4f} W\n"
            f"Absorbed pump: {result['absorbed_pump_W']:.4f} W\n"
            f"Signal out: {result['signal_out_W']:.6f} W\n"
            f"Signal change: {result['signal_change_W']:.6f} W\n"
            f"Mean excited fraction: {100*result['mean_excited_fraction']:.4f}%\n"
            f"Selected output-coupler transmission: {100*result['selected_coating_transmission']:.3f}%\n\n"
            f"{result['scope']}\n\nSaved: {self.directory}")
        fig = self.figures["Gain and pulse"]
        fig.clear()
        left, right = fig.subplots(1, 2)
        left.plot(result["wavelength_nm"], result["absorption_cross_section_cm2"],
                  label="absorption")
        left.plot(result["wavelength_nm"], result["emission_cross_section_cm2"],
                  label="emission")
        left.set(xlabel="wavelength (nm)", ylabel="cross section (cm²)",
                 title="Reconstructed 20 °C cross sections")
        left.legend()
        right.plot(result["coating_transmission"],
                   result["coating_output_intensity_W_m2"], marker="o")
        right.set(xlabel="output-coupler transmission",
                  ylabel="screened output intensity (W/m²)",
                  title="Approximate coating screen")
        self._finish("Gain and pulse")

    def _structured(self):
        r = self.result
        mode = r["modes"][self.beam]
        thermal = r["thermal"]
        resonator = r.get("resonator")
        self._text(
            f"Yb:LuAG · 10 at.% · {r['solver_mode']}\n"
            f"Selected target: {self.beam}\n\n"
            f"Input: {mode['input_power_W']:.5f} W\n"
            f"Disk exit: {mode['disk_output_power_W']:.5f} W\n"
            f"Output: {mode['output_power_W']:.5f} W\n"
            f"Pump absorbed: {mode['pump_absorbed_W']:.5f} W\n"
            f"Heat: {mode['net_heat_W_upper_or_assumed']:.5f} W\n"
            f"Mean excitation: {100*mode['mean_excited_fraction']:.3f}%\n"
            + (f"Target vs cold coherent overlap: {mode['target_vs_cold_coherent_overlap']:.4f}\n"
               if 'target_vs_cold_coherent_overlap' in mode else "")
            + (f"Optical sampling: {r['sampling']['status']}, {r['sampling']['pixel_um']:.1f} µm/pixel\n"
               if 'sampling' in r else "")
            + f"Thermal status: {thermal['status']}\n"
            + (f"{'5 K design peak' if thermal['status'] == 'design_reference' else 'Disk peak'}: {thermal['disk_temperature_max_C']:.2f} °C\n"
               if thermal["status"] in ("computed", "design_reference") else "")
            + (f"Fixed cavity output coupler: {resonator['output_coupler_power_W']:.4f} W\n"
               if resonator else "")
            + f"\n{r['scope']}\n\nSaved: {self.directory}")
        x, y = _array(r["x_mm"]), _array(r["y_mm"])
        self._beams(mode["input_intensity"], mode["disk_input_intensity"],
                    mode["output_intensity"], r["uniform_reference_intensity"],
                    x, y, "W/m²", f"{self.beam} · {r['solver_mode']}")
        self._phase_density(r, x, y, structured=True)
        self._thermal_surfaces(thermal, x)
        fig = self.figures["Gain and pulse"]
        fig.clear()
        axes = fig.subplots(1, 2)
        axes[0].plot(x, mode["input_profile"], label="input")
        axes[0].plot(x, mode["output_profile"], label="output")
        axes[0].plot(x, r["uniform_reference_profiles"][self.beam],
                     linestyle="--", label="uniform Yb reference")
        axes[0].set(xlabel="x (mm)", ylabel="irradiance (W/m²)",
                    title="Horizontal center cuts")
        axes[0].legend()
        axes[1].axis("off")
        axes[1].text(.03, .95,
                     f"Pump absorbed {mode['pump_absorbed_W']:.4f} W\n"
                     f"Signal output {mode['output_power_W']:.4f} W\n"
                     f"Heat {mode['net_heat_W_upper_or_assumed']:.4f} W\n"
                     f"Thermal status {thermal['status']}",
                     transform=axes[1].transAxes, va="top", fontsize=11)
        self._finish("Gain and pulse")

    def _pulsed(self):
        r = self.result
        thermal = r["thermal"]
        timeline = r.get("thermal_timeline")
        targets = r["proposal_targets"]
        valid = bool(timeline and timeline["requested_material_range_valid"])
        phase_target = (
            f"Phase residual target <{targets['roundtrip_phase_distortion_rad_max']:.2f} rad: "
            f"{r['phase_residual_rms_rad']:.4f} rad calculated with current scope"
            if r.get("thermal_feedback_applied") and r["phase_residual_rms_rad"] is not None else
            "Hot phase distortion: unavailable")
        target_lines = (
            f"Output energy target >{targets['output_energy_uJ_min']:.0f} µJ: "
            f"{r['output_energy_J']*1e6:.4f} µJ calculated\n"
            f"Energy gain target >{targets['energy_gain_min']:.0f}×: "
            f"{r['net_energy_gain']:.3f}× calculated\n"
            f"Pattern overlap reference >{targets['pattern_fidelity_min']:.2f}: "
            f"{r['extraction_shape_retention']:.3f} input-to-exit intensity overlap\n"
            f"{phase_target}")
        self._text(
            f"Yb:LuAG · 12 at.% proposal pulse · {r['architecture']}\n"
            f"Target beam: {self.beam}\n\n"
            f"Input: {r['input_energy_J']*1e9:.4f} nJ\n"
            f"Intracavity/disk exit: {r['disk_output_energy_J']*1e9:.4f} nJ\n"
            f"Extracted output: {r['output_energy_J']*1e9:.4f} nJ\n"
            f"Net energy gain: {r['net_energy_gain']:.3f}×\n"
            f"Average output: {r['average_output_W']:.5f} W\n"
            f"Input-to-extraction overlap: {r['extraction_shape_retention']:.4f}\n"
            + (f"Target vs cold coherent overlap: {r['field_fidelity']['target_vs_cold_coherent']:.4f}\n"
               f"Cold vs hot coherent overlap: "
               f"{r['field_fidelity']['cold_vs_hot_coherent'] if r['field_fidelity']['cold_vs_hot_coherent'] is not None else 'unavailable'}\n"
               if "field_fidelity" in r else "")
            + (f"Pump-asymptotic gain ceiling: {r['gain_feasibility']['pump_asymptotic_unsaturated_gain_ceiling']:.3f}× "
               f"before configured losses; {r['gain_feasibility']['material_traversals']} material traversals\n"
               if "gain_feasibility" in r else "")
            + (f"Pump coating: {r['hardware_validity']['pump_coating_status']}\n"
               if "hardware_validity" in r else "")
            + (f"Sampling: {r['sampling']['status']}, {r['sampling']['pixel_um']:.1f} µm/pixel\n"
               if "sampling" in r else "")
            + (f"Effective fluorescence escape heat screen, 0→1: "
               f"{r['fluorescence_effective_escape_sensitivity']['rows'][0]['heat_W']:.3f}→"
               f"{r['fluorescence_effective_escape_sensitivity']['rows'][-1]['heat_W']:.3f} W\n"
               if 'fluorescence_effective_escape_sensitivity' in r else "")
            + (f"Fixed-heat contact 0.5×/2× disk peaks: "
               f"{r['cooler_conductance_sensitivity']['rows'][1]['disk_max_C']:.2f}/"
               f"{r['cooler_conductance_sensitivity']['rows'][2]['disk_max_C']:.2f} °C\n"
               if r.get('cooler_conductance_sensitivity') else "")
            + f"Pump absorbed: {r['cycle_average_pump_absorbed_W']:.4f} W\n"
            f"Cycle-average signal gain: {r['cycle_average_signal_gain_W']:.5f} W\n"
            f"Heat: {r['cycle_average_heat_W_upper_or_assumed']:.4f} W\n"
            + (f"Excitation storage change: {r['cycle_average_excitation_storage_change_W']:.3e} W\n"
               if 'cycle_average_excitation_storage_change_W' in r else "")
            + f"Population convergence: {r['cycles']} cycles; residual {r['residual']:.2e}\n"
            f"Thermal: {thermal['status']}; requested-time range valid: {valid}; "
            f"beam thermal phase applied: {r['thermal_feedback_applied']}\n"
            f"Hot phase validity: {r.get('hot_phase_validity', 'legacy result; check source')}; "
            f"{r.get('hot_phase_reason') or 'within the supported material parameter range only'}\n"
            + (f"Requested disk maximum: {timeline['requested_disk_max_C']:.2f} °C\n"
               if timeline else "")
            + f"\nProposal comparisons (targets are not measured performance):\n{target_lines}\n\n"
            + f"{r['spectral_scope']}\n\n{r['scope']}\n\nSaved: {self.directory}")
        x, y = _array(r["x_mm"]), _array(r["y_mm"])
        self._beams(r["input_fluence_J_m2"], r["disk_input_fluence_J_m2"],
                    r["output_fluence_J_m2"],
                    r["uniform_isothermal_output_fluence_J_m2"],
                    x, y, "J/m²",
                    f"{self.beam} · output {r['output_distance_m']:.3f} m after extraction")
        self._phase_density(r, x, y)
        self._pulse_gain(r)
        self._cooling(r)
        self._thermal_surfaces(thermal, x)

    def update_sweep(self, points):
        if self.kind != "pulsed":
            raise ValueError("pump sweep requires a pulsed result")
        self.sweep_points = points
        self._pulse_gain(self.result)
        self.tabs.select(self.frames["Gain and pulse"])
