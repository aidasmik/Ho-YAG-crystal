"""Native Tkinter result views for the Yb:LuAG calculator."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from ybluag_camera_preview_view import CameraPreviewPanel


def _array(value):
    return np.asarray(value, dtype=float)


def _line_pair(ax, x, y, label, style="-"):
    peak = float(np.max(y))
    ax.plot(x, y / peak if peak > 0 else y, style, label=label, linewidth=1.5)


class YbResultPanel(ttk.Frame):
    """Solver output organized as native plots and readable scientific metrics."""

    NAMES = ("Overview", "Beam and profiles", "Beam on crystal", "Phase and Yb", "Gain and pulse",
             "Cooling timeline", "Thermal surfaces", "Camera preview")

    def __init__(self, parent):
        super().__init__(parent)
        self.provenance = tk.StringVar(value="Run a simulation to display calculated fields")
        self.model_note = tk.StringVar(value="Spectra reconstructed from figures; generic copper assembly is uncalibrated.")
        top = ttk.Frame(self, padding=(12, 12, 12, 4))
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.provenance, style="Eyebrow.TLabel").pack(anchor="w")
        cards = ttk.Frame(top)
        cards.pack(fill="x", pady=(8, 7))
        self.metrics = {}
        for column, label in enumerate(("OUTPUT ENERGY", "ENERGY GAIN", "AVERAGE OUTPUT", "DISK HEAT")):
            card = ttk.Frame(cards, style="Card.TFrame", padding=(12, 9))
            card.grid(row=0, column=column, sticky="nsew", padx=(0, 7))
            cards.columnconfigure(column, weight=1, uniform="metric")
            ttk.Label(card, text=label, style="Card.TLabel").pack(anchor="w")
            variable = tk.StringVar(value="—")
            ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(anchor="w")
            self.metrics[label] = variable
        ttk.Label(top, textvariable=self.model_note, wraplength=1000,
                  foreground="#855515").pack(anchor="w", pady=(0, 5))
        viewbar = ttk.Frame(top)
        viewbar.pack(fill="x")
        ttk.Label(viewbar, text="Field of view", style="Input.TLabel").pack(side="left")
        self.zoom = tk.StringVar(value="Beam detail")
        zoom = ttk.Combobox(viewbar, textvariable=self.zoom,
                           values=("Beam detail", "Full crystal"), state="readonly", width=14)
        zoom.pack(side="left", padx=8)
        zoom.bind("<<ComboboxSelected>>", lambda *_: self.refresh_views())
        ttk.Label(viewbar, text="Shared x/y limits; color bars show absolute values",
                  style="Input.TLabel").pack(side="left")
        self.disk_radius_mm = 5.0
        self.phase_view = tk.StringVar(value="Output comparison")
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        self.frames = {}
        self.figures = {}
        self.canvases = {}
        self.toolbars = {}
        self.overview = None
        self.camera_preview = None
        for name in self.NAMES:
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=name)
            self.frames[name] = frame
            if name == "Overview":
                text = tk.Text(frame, wrap="word", font=("Segoe UI", 11), padx=22, pady=18, relief="flat",
                               background="white", foreground="#213247", spacing3=4)
                scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
                text.configure(yscrollcommand=scroll.set)
                text.pack(side="left", fill="both", expand=True)
                scroll.pack(side="right", fill="y")
                self.overview = text
            elif name == "Camera preview":
                self.camera_preview = CameraPreviewPanel(frame)
                self.camera_preview.pack(fill="both", expand=True)
            else:
                if name == "Phase and Yb":
                    choice = ttk.Combobox(frame, textvariable=self.phase_view,
                        values=("Output comparison", "SLM masks"), state="readonly", width=24)
                    choice.pack(anchor="w", padx=12, pady=8)
                    choice.bind("<<ComboboxSelected>>", lambda *_: self.refresh_views())
                fig = Figure(figsize=(11, 7), dpi=100, constrained_layout=True)
                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
                toolbar.update()
                toolbar.pack(fill="x")
                self.figures[name] = fig
                self.canvases[name] = canvas
                self.toolbars[name] = toolbar
        self.kind = None
        self.result = None
        self.directory = None
        self.sweep_points = None

    def set_provenance(self, text):
        self.provenance.set(text)

    def _tab_changed(self, _event):
        if (self.camera_preview is not None and self.kind == "pulsed" and
                self.tabs.select() == str(self.frames["Camera preview"]) and
                self.camera_preview.rendered_generation != self.camera_preview.generation):
            self.camera_preview.render()

    def refresh_views(self):
        if self.result is None:
            return
        if self.kind == "pulsed":
            self._pulsed()
        elif self.kind == "structured":
            self._structured()
        else:
            self._cw()

    def _summary(self):
        r = self.result
        for value in self.metrics.values():
            value.set("—")
        if self.kind == "pulsed":
            self.metrics["OUTPUT ENERGY"].set(f"{r['output_energy_J']*1e6:.4g} µJ")
            self.metrics["ENERGY GAIN"].set(f"{r['net_energy_gain']:.4g} ×")
            self.metrics["AVERAGE OUTPUT"].set(f"{r['average_output_W']:.4g} W")
            self.metrics["DISK HEAT"].set(f"{r['cycle_average_heat_W_upper_or_assumed']:.4g} W")
            mode = r.get("thermal_optical_mode", "unknown")
            status = ("Coupled steady optics + temperature" if mode == "coupled_steady" and r.get("thermal_feedback_applied")
                      else "Lumped phase after amplification" if r.get("thermal_feedback_applied")
                      else "Cold optical output · thermal feedback unavailable or disabled")
            ledger = r.get("regenerative") or r.get("ideal_multipass") or {}
            error = ledger.get("population_photon_balance_relative_L1")
            diagnostic = f" · photon balance error {error:.2e}" if error is not None else ""
            bound = r.get("gain_feasibility", {})
            ceiling = bound.get("pump_ceiling_after_configured_losses")
            warning = (f" · Target exceeds cold uniform gain bound ({ceiling:.3g}×)"
                       if ceiling is not None and bound.get("requested_energy_exceeds_pump_ceiling") else "")
            if r.get("material") == "Yb:YAG":
                status += " · YAG gain at 20 °C; hot pump spectra missing"
            self.model_note.set(status + diagnostic + warning + " · Assumptions in Overview.")
        else:
            case = r['modes'][self.beam] if self.kind == "structured" else r
            self.metrics["AVERAGE OUTPUT"].set(f"{case.get('output_power_W', case.get('signal_out_W', 0)):.4g} W")
            self.model_note.set("CW engineering model · no pulsed thermal feedback · assumptions in Overview")

    def _field_half_width(self, cases, x, y):
        if self.zoom.get() == "Full crystal":
            return 1.05*self.disk_radius_mm
        xx, yy = np.meshgrid(x, y)
        radius = np.maximum(abs(xx), abs(yy)).ravel()
        order = np.argsort(radius)
        width = max(abs(x[1]-x[0]), abs(y[1]-y[0]))*3
        for case in cases:
            if case is None:
                continue
            weights = np.maximum(_array(case), 0).ravel()[order]
            total = weights.sum()
            if total > 0:
                idx = min(np.searchsorted(np.cumsum(weights), .995*total), len(order)-1)
                width = max(width, radius[order[idx]])
        return min(width*1.12, max(abs(x[0]), abs(x[-1]), abs(y[0]), abs(y[-1])))

    def _disk_outline(self, ax):
        ring = Circle((0, 0), self.disk_radius_mm, fill=False, color="white", linewidth=1.2)
        ring.set_path_effects([path_effects.Stroke(linewidth=2.4, foreground="#233448"),
                               path_effects.Normal()])
        ax.add_patch(ring)

    def _visible(self, names):
        for name, frame in self.frames.items():
            self.tabs.tab(frame, state="normal" if name in names else "hidden")
        self.tabs.select(self.frames["Beam and profiles"] if "Beam and profiles" in names
                         else self.frames["Overview"])

    def _text(self, value):
        self.overview.configure(state="normal")
        self.overview.delete("1.0", "end")
        self.overview.insert("1.0", value)
        self.overview.configure(state="disabled")

    def _finish(self, name):
        self.toolbars[name].update()
        self.canvases[name].draw_idle()

    def draw(self, kind, result, directory, beam):
        self.kind, self.result, self.directory, self.beam = kind, result, directory, beam
        self.sweep_points = None
        self.disk_radius_mm = float(result.get("disk_radius_mm", 5))
        self.set_provenance(f"{result.get('material', 'Yb:LuAG')} CALCULATED RESULT  ·  {beam}  ·  {directory.name}")
        self._summary()
        if kind == "cw":
            self._visible(("Overview", "Gain and pulse"))
            self._cw()
        elif kind == "structured":
            self._visible(("Overview", "Beam and profiles", "Beam on crystal", "Phase and Yb",
                           "Gain and pulse", "Thermal surfaces"))
            self._structured()
        else:
            self._visible(self.NAMES)
            self._pulsed()
            self.camera_preview.set_result(result)

    def _map(self, ax, values, title, x=None, y=None, *, cmap="viridis", vmin=None, vmax=None,
             disk=False):
        data = np.ma.asarray(values, dtype=float)
        extent = ((float(x[0]-(x[1]-x[0])/2), float(x[-1]+(x[1]-x[0])/2),
                   float(y[0]-(y[1]-y[0])/2), float(y[-1]+(y[1]-y[0])/2))
                  if x is not None and y is not None else None)
        image = ax.imshow(data, origin="lower", extent=extent, cmap=cmap,
                          vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        if extent:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            if disk:
                self._disk_outline(ax)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        return image

    def _beams(self, source, incident, output, reference, x, y, unit, title):
        name = "Beam and profiles"
        fig = self.figures[name]
        fig.clear()
        fig.set_layout_engine(None)
        beam_axes = fig.subplots(1, 3)
        fig.subplots_adjust(left=.06, right=.95, bottom=.10, top=.86, wspace=.40)
        cases = ((source, "Gaussian source"), (incident, "Shaped disk input"),
                 (output, "Amplified output"))
        half_width = self._field_half_width((source, incident, output, reference), x, y)
        for column, (values, label) in enumerate(cases):
            values = _array(values)
            image_ax = beam_axes[column]
            divider = make_axes_locatable(image_ax)
            top_ax = divider.append_axes("top", size="28%", pad=.10, sharex=image_ax)
            side_ax = divider.append_axes("left", size="20%", pad=.12, sharey=image_ax)
            color_ax = divider.append_axes("right", size="5%", pad=.10)
            image = self._map(image_ax, values, "", x, y, cmap="inferno", disk=True)
            fig.colorbar(image, cax=color_ax)
            color_ax.tick_params(labelsize=8, pad=2)
            top_ax.set_title(label, fontsize=11, fontweight="bold", pad=12)
            iy, ix = int(np.argmin(abs(_array(y)))), int(np.argmin(abs(_array(x))))
            norm = max(float(values.max()), 1e-300)
            top_ax.plot(x, values[iy]/norm, color="#173b52", linewidth=1.7)
            side_ax.plot(values[:, ix]/norm, y, color="#173b52", linewidth=1.7)
            if column == 2 and reference is not None:
                ref = _array(reference)
                ref_norm = max(float(ref.max()), 1e-300)
                top_ax.plot(x, ref[iy]/ref_norm, color="#087f8c", linestyle="--", linewidth=1.6)
                side_ax.plot(ref[:, ix]/ref_norm, y, color="#087f8c", linestyle="--", linewidth=1.6)
            image_ax.set_xlim(-half_width, half_width)
            image_ax.set_ylim(-half_width, half_width)
            image_ax.set_aspect("equal", adjustable="box")
            top_ax.set_ylim(0, 1.1)
            side_ax.set_xlim(1.1, 0)
            side_ax.set_xticks([0, 1])
            top_ax.set_yticks([0, 1])
            top_ax.tick_params(axis="x", labelbottom=False)
            image_ax.tick_params(axis="y", labelleft=False)
            image_ax.set_ylabel("")
            side_ax.set_ylabel("y (mm)", fontsize=9)
            side_ax.tick_params(axis="y", labelsize=8)
            top_ax.grid(alpha=.18)
            side_ax.grid(alpha=.18)
        fig.suptitle(f"{title} · {unit} · center cuts normalized to each map's peak\n"
                     "Dashed teal: uniform Yb, cold reference · outline: crystal boundary (visible in Full crystal)",
                     fontsize=10)
        self._finish(name)

    @staticmethod
    def _footprint_contours(ax, beam, x, y):
        beam = _array(beam)
        peak = float(np.max(beam))
        if peak > 0:
            normalized = beam / peak
            ax.contour(x, y, normalized, levels=(np.exp(-2), .5),
                       colors=("white", "#ff5c4d"), linewidths=(1.8, 1.5))
        return peak

    def _beam_on_crystal(self, density, beam, x, y, unit):
        """Show the calculated disk-entrance beam contours on the Yb map."""
        name = "Beam on crystal"
        fig = self.figures[name]
        fig.clear()
        density = _array(density)
        beam = _array(beam)
        if density.shape != beam.shape or density.shape != (len(y), len(x)):
            raise ValueError("crystal and disk-entrance beam grids disagree")
        active = density > 0
        shown = np.ma.masked_where(~active, density)
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad("#111b2a")
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        extent = (x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2)
        axes = fig.subplots(1, 2)
        peak = float(np.max(beam))
        if peak > 0:
            footprint = beam >= .01 * peak
            yy, xx = np.where(footprint)
            radius = max(float(np.max(np.abs(x[xx]))),
                         float(np.max(np.abs(y[yy])))) * 1.15
            zoom_radius = min(5.5, max(1.5, radius))
        else:
            zoom_radius = 2.0
        for ax, radius, title in zip(axes, (5.5, zoom_radius),
                                     ("Full 10 mm crystal", "Beam footprint detail")):
            image = ax.imshow(shown, origin="lower", extent=extent, cmap=cmap)
            self._disk_outline(ax)
            self._footprint_contours(ax, beam, x, y)
            ax.set(xlim=(-radius, radius), ylim=(-radius, radius),
                   xlabel="x (mm)", ylabel="y (mm)", title=title)
            ax.set_aspect("equal")
        fig.colorbar(image, ax=axes, shrink=.72, label="Yb density (10²⁶ ions/m³)")
        handles = (Line2D([], [], color="white", linestyle="--", label="Crystal edge"),
                   Line2D([], [], color="white", linewidth=2,
                          label="Incident beam: 1/e² of peak"),
                   Line2D([], [], color="#ff5c4d", linewidth=2,
                          label="Incident beam: 50% of peak"))
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
        captured = float(np.sum(beam[active]) / np.sum(beam)) if np.sum(beam) > 0 else 0.0
        fig.suptitle(f"Calculated disk-entrance {unit} contours on Yb concentration · "
                     f"{captured:.1%} of sampled beam within crystal", fontsize=12)
        self._finish(name)

    def _phase_density(self, result, x, y, structured=False):
        name = "Phase and Yb"
        fig = self.figures[name]
        fig.clear()
        axes = list(fig.subplots(2, 2).flat)
        if self.phase_view.get() == "SLM masks" or structured:
            maps = ((result["target_phase_mask"], "Target mask", "twilight", None),
                    (result["aberration_phase_mask"], "Optional phase term", "twilight", None),
                    (result["phase_mask"], "Total applied SLM mask", "twilight", None),
                    ((result["modes"][self.beam]["output_phase"] if structured else result["output_phase"]),
                     "Output phase", "twilight", (result["modes"][self.beam]["output_intensity"]
                                                  if structured else result["output_fluence_J_m2"])))
        else:
            residual = result.get("phase_residual_rad")
            if residual is None:
                residual = result.get("cold_density_phase_residual_rad")
            maps = ((result["disk_input_phase"], "Shaped disk input phase", "twilight", result["disk_input_fluence_J_m2"]),
                    (result["output_phase"], "Amplified output phase", "twilight", result["output_fluence_J_m2"]),
                    (result["uniform_isothermal_output_phase"], "Uniform Yb · cold output phase", "twilight", result["uniform_isothermal_output_fluence_J_m2"]),
                    (residual, ("Output − reference · piston removed" if result.get("thermal_feedback_applied")
                                else "Cold output − reference · hot residual unavailable"), "RdBu_r", result["output_fluence_J_m2"]))
        for ax, (data, title, cmap, weights) in zip(axes, maps):
            if data is None:
                ax.axis("off")
                ax.text(.1, .5, "Phase unavailable for this state", transform=ax.transAxes)
                continue
            values = _array(data)
            if weights is not None:
                w = _array(weights)
                values = np.ma.masked_where(w < .01*float(w.max()), values)
            limit = (max(float(np.ma.max(abs(values))), 1e-9) if cmap == "RdBu_r" else np.pi)
            if cmap == "twilight":
                values = np.ma.array(np.angle(np.exp(1j*values)), mask=np.ma.getmaskarray(values))
            im = self._map(ax, values, title, x, y, cmap=cmap, vmin=-limit, vmax=limit, disk=True)
            if weights is not None:
                width = self._field_half_width((weights,), x, y)
                ax.set_xlim(-width, width)
                ax.set_ylim(-width, width)
            fig.colorbar(im, ax=ax, shrink=.85, label="rad")
        fig.suptitle("Phase at calculated planes · low-signal phase masked below 1% of peak", fontsize=11)
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
        label = ("5 K design reference; selected heat is outside supported range"
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
            ideal = result.get("ideal_multipass") or {}
            visits = ideal.get("encounter_exit_energies_J")
            if visits:
                energy = [result["input_energy_J"], *visits]
                axes[0].plot(range(len(energy)), _array(energy)*1e9, marker="o")
                axes[0].set(xlabel="disk encounter", ylabel="pulse energy (nJ)",
                            title="Shared-disk ideal multipass")
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
            axes[1].set_title("Cold optical sweep · thermal feedback excluded")
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
        if screen.get("status") in ("small_signal_screen_only",
                                    "spatial_small_signal_screen_only"):
            wl = screen["wavelength_nm"]
            axes[3].plot(wl, screen["input_spectral_weights"], label="source weight")
            axes[3].set(xlabel="wavelength (nm)", ylabel="normalized source weight",
                        title=("Spatial small-signal screen" if
                               screen["status"].startswith("spatial") else
                               "Homogeneous small-signal screen"))
            other = axes[3].twinx()
            other.plot(wl, screen["unsaturated_gain"], color="tab:orange",
                       label="unsaturated gain")
            other.set_ylabel("unsaturated gain ceiling")
        else:
            axes[3].axis("off")
            axes[3].text(.02, .65, screen.get("reason", "Spectral result unavailable"),
                         transform=axes[3].transAxes, wrap=True)
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
        status = ("Fixed-heat thermal replay · in-range material assumptions" if timeline["requested_material_range_valid"]
                  else "Selected temperature outside supported 20–26.85 °C range; phase withheld")
        fig.suptitle(f"{status}; steady screen {timeline['steady_disk_max_C']:.2f} °C",
                     fontsize=12)
        self._finish(name)

    def _cw(self):
        result = self.result
        self._text(
            f"{result.get('material', 'Yb:LuAG')} · {result.get('yb_at_percent', 10):g} at.% · 20 °C CW material and coating screen\n\n"
            f"Incident pump: {result['absorbed_pump_W'] + result['pump_out_W']:.4f} W\n"
            f"Absorbed pump: {result['absorbed_pump_W']:.4f} W\n"
            f"Signal out: {result['signal_out_W']:.6f} W\n"
            f"Signal change: {result['signal_change_W']:.6f} W\n"
            f"Mean excited fraction: {100*result['mean_excited_fraction']:.4f}%\n"
            f"Selected output-coupler transmission: {100*result['selected_coating_transmission']:.3f}%\n\n"
            f"{result.get('material_status', '')}\n{result['scope']}\n\nSaved: {self.directory}")
        fig = self.figures["Gain and pulse"]
        fig.clear()
        left, right = fig.subplots(1, 2)
        left.plot(result["wavelength_nm"], result["absorption_cross_section_cm2"],
                  label="absorption")
        left.plot(result["wavelength_nm"], result["emission_cross_section_cm2"],
                  label="emission")
        left.set(xlabel="wavelength (nm)", ylabel="cross section (cm²)",
                 title="20 °C source cross sections")
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
            f"{r.get('material', 'Yb:LuAG')} · {r.get('yb_at_percent', 10):g} at.% · {r['solver_mode']}\n"
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
            + f"\n{r.get('material_status', '')}\n{r['scope']}\n\nSaved: {self.directory}")
        x, y = _array(r["x_mm"]), _array(r["y_mm"])
        self._beams(mode["input_intensity"], mode["disk_input_intensity"],
                    mode["output_intensity"], r["uniform_reference_intensity"],
                    x, y, "W/m²", f"{self.beam} · {r['solver_mode']}")
        self._beam_on_crystal(r["yb_density_entrance_1e26_m3"],
                              mode["disk_input_intensity"], x, y, "irradiance")
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
            f"{r.get('material', 'Yb:LuAG')} · {r.get('yb_at_percent', 12):g} at.% pulse · {r['architecture']}\n"
            f"Thermal-optical mode: {r.get('thermal_optical_mode', 'unknown')}\n"
            f"Pump wavelength: {r.get('pump_wavelength_nm', 'not recorded in saved result')} nm\n"
            f"Target beam: {self.beam}\n\n"
            f"Input: {r['input_energy_J']*1e9:.4f} nJ\n"
            f"Intracavity/disk exit: {r['disk_output_energy_J']*1e9:.4f} nJ\n"
            f"Extracted output: {r['output_energy_J']*1e9:.4f} nJ\n"
            f"Net energy gain: {r['net_energy_gain']:.3f}×\n"
            f"Average output: {r['average_output_W']:.5f} W\n"
            + (f"Ideal relay retention: {r['ideal_multipass']['ideal_relay_power_retention']:.4f}; "
               f"loss: {r['ideal_multipass']['ideal_relay_loss_J']*1e9:.4g} nJ\n"
               if r.get("ideal_multipass") else "")
            + f"Input-to-extraction overlap: {r['extraction_shape_retention']:.4f}\n"
            + (f"Cold homogeneous pump-asymptotic gain ceiling: {r['gain_feasibility']['pump_ceiling_after_configured_losses']:.4g}× (unsaturated bound, not achieved gain)\n"
               if r.get('gain_feasibility') else "")
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
            + (f"Photon balance relative L1 error: {(r.get('regenerative') or r.get('ideal_multipass'))['population_photon_balance_relative_L1']:.3e}\n"
               if 'population_photon_balance_relative_L1' in (r.get('regenerative') or r.get('ideal_multipass') or {}) else "")
            + f"{r.get('material_status', 'Material and generic copper/contact properties need calibration.')}\n"
            "Assembly assumes near-room-temperature constants (20–26.85 °C).\n"
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
        self._beam_on_crystal(_array(r["yb_density_m3"])[0] / 1e26,
                              r["disk_input_fluence_J_m2"], x, y, "fluence")
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
