"""Tkinter camera preview driven by the saved pulsed optical result."""
from __future__ import annotations

from dataclasses import fields
from queue import Empty, Queue
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ybluag.camera_preview import PreviewSettings, preview_frame
from ybluag.camera_dataset import CameraSettings, suggest_optical_throughput
from beam_profile_view import add_camera_profiles
from scientific_style import PAPER


class CameraPreviewPanel(ttk.Frame):
    LABELS = {
        "tip_x_waves": "Tip x (waves)",
        "tip_y_waves": "Tilt y (waves)",
        "astigmatism_0_waves": "Astigmatism 0° (waves)",
        "astigmatism_45_waves": "Astigmatism 45° (waves)",
        "coma_x_waves": "Coma x (waves)",
        "spherical_waves": "Spherical (waves)",
        "defocus_waves": "Optics defocus (waves)",
        "residual_rms_waves": "Correlated phase RMS (waves)",
        "focus_offset_mm": "Camera plane offset (mm)",
        "fov_width_mm": "Object field width (mm)",
        "optical_throughput": "Optical throughput",
        "pulses_per_exposure": "Pulses per exposure",
        "qe": "Quantum efficiency",
        "read_noise_e": "Read noise (e⁻ RMS)",
        "prnu_rms": "Pixel sensitivity RMS",
        "background_e": "Background (e⁻)",
        "dark_current_e_s": "Dark current (e⁻/s)",
        "dsnu_rms_e": "Pixel offset RMS (e⁻)",
        "psf_sigma_pixels": "Blur (camera pixels)",
        "hot_pixel_fraction": "Hot pixel fraction",
        "dead_pixel_fraction": "Dead pixel fraction",
        "camera_shift_pixels": "Camera x shift (pixels)",
        "camera_rotation_deg": "Camera rotation (degrees)",
        "seed": "Noise/session seed",
    }
    GROUPS = (
        ("External optics", ("tip_x_waves", "tip_y_waves", "astigmatism_0_waves", "astigmatism_45_waves",
                             "coma_x_waves", "spherical_waves", "defocus_waves",
                             "residual_rms_waves", "focus_offset_mm")),
        ("Sensor and alignment", ("fov_width_mm", "optical_throughput",
                                  "pulses_per_exposure", "qe", "read_noise_e",
                                  "prnu_rms", "background_e", "dark_current_e_s",
                                  "dsnu_rms_e", "psf_sigma_pixels",
                                  "hot_pixel_fraction", "dead_pixel_fraction",
                                  "camera_shift_pixels", "camera_rotation_deg", "seed")),
    )

    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        self.running = False
        self.completed = Queue()
        self.generation = 0
        self.rendered_generation = -1
        self.values = {f.name: tk.StringVar(value=str(getattr(PreviewSettings(), f.name)))
                       for f in fields(PreviewSettings)}
        self.status = tk.StringVar(value="Run a pulsed simulation to preview its camera image.")
        left = ttk.Frame(self, width=290)
        left.pack(side="left", fill="y", padx=(8, 0), pady=8)
        left.pack_propagate(False)
        ttk.Label(left, text="DETECTOR / OPTICS", style="Eyebrow.TLabel").pack(anchor="w")
        ttk.Label(left, text="Adjust downstream optics and sensor response; the laser solution stays fixed.",
                  wraplength=265, style="Info.TLabel").pack(anchor="w", pady=(5, 8))
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=(0, 8))
        buttons = ttk.Frame(left)
        buttons.pack(fill="x")
        self.render_button = ttk.Button(buttons, text="Render camera",
                                        style="Accent.TButton", command=self.render)
        self.render_button.pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="New noise", command=self.new_noise).pack(side="left", padx=(4, 0))
        canvas = tk.Canvas(left, highlightthickness=0, width=274, background=PAPER)
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas)
        body.bind("<Configure>", lambda *_: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True, pady=(8, 0))
        scroll.pack(side="right", fill="y", pady=(8, 0))
        for heading, keys in self.GROUPS:
            group = ttk.LabelFrame(body, text=heading.upper(), padding=(8, 7))
            group.pack(fill="x", pady=(0, 10))
            group.columnconfigure(0, weight=1)
            for row, key in enumerate(keys):
                ttk.Label(group, text=self.LABELS[key], style="Field.TLabel",
                          wraplength=160).grid(row=row, column=0, sticky="w",
                                               padx=(0, 6), pady=3)
                ttk.Entry(group, textvariable=self.values[key], width=11).grid(
                    row=row, column=1, sticky="e", pady=3)
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        ttk.Label(right, textvariable=self.status, style="Info.TLabel",
                  wraplength=900).pack(fill="x", pady=(0, 5))
        self.figure = Figure(figsize=(15, 6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        ax = self.figure.subplots()
        ax.text(.5, .5, "Camera preview will appear here", ha="center", va="center")
        ax.axis("off")
        self.canvas.draw_idle()

    def set_result(self, result):
        self.result = result
        self.generation += 1
        try:
            fov = min(6.0, .7*float(result["x_mm"][-1]-result["x_mm"][0]))
            self.values["fov_width_mm"].set(f"{fov:.3g}")
            camera = CameraSettings(object_fov_width_mm=fov)
            throughput = suggest_optical_throughput(result, camera)
            self.values["optical_throughput"].set(f"{throughput:.4g}")
        except (ValueError, KeyError, TypeError, IndexError):
            pass
        self.status.set("Saved physical output loaded. Press Render camera to update the observation.")

    def new_noise(self):
        try:
            self.values["seed"].set(str(int(self.values["seed"].get())+1))
        except ValueError:
            self.values["seed"].set("1")
        self.render()

    def render(self):
        if self.result is None or self.running:
            return
        try:
            values = {f.name: (int(self.values[f.name].get()) if
                              f.name in ("seed", "pulses_per_exposure") else
                              float(self.values[f.name].get()))
                      for f in fields(PreviewSettings)}
            settings = PreviewSettings(**values)
        except ValueError as exc:
            self.status.set(f"Invalid camera setting: {exc}")
            return
        self.running = True
        self.render_button.configure(state="disabled")
        self.status.set("Forming the camera image from the saved field…")
        generation = self.generation
        result = self.result

        def work():
            try:
                frame = preview_frame(result, settings)
                self.completed.put((frame, settings, generation, None))
            except Exception as exc:
                self.completed.put((None, settings, generation, str(exc)))
        threading.Thread(target=work, daemon=True).start()
        self.after(50, self._poll)

    def _poll(self):
        try:
            frame, settings, generation, error = self.completed.get_nowait()
        except Empty:
            if self.running:
                self.after(50, self._poll)
            return
        if error is not None:
            self._error(error)
        else:
            self._show(frame, settings, generation)

    def _error(self, error):
        self.running = False
        self.render_button.configure(state="normal")
        self.status.set(f"Camera preview: {error}")

    def _show(self, frame, settings, generation):
        self.running = False
        self.render_button.configure(state="normal")
        if generation != self.generation:
            return
        adu, clean, diagnostics = frame
        self.figure.clear()
        slots=self.figure.add_gridspec(1,3,wspace=.27)
        add_camera_profiles(self.figure,slots[0,0],clean,
            "Clean camera-plane fluence",fov_width_mm=settings.fov_width_mm,
            unit="J/m²",cmap="inferno",display_stride=2)
        add_camera_profiles(self.figure,slots[0,1],adu,
            "CCD observation (12-bit ADU)",fov_width_mm=settings.fov_width_mm,
            baseline=32.,unit="ADU",cmap="gray",vmax=2**12-1,
            display_stride=2)
        phase = np.asarray(diagnostics["residual_phase_rad"], float)
        phase_ax = self.figure.add_subplot(slots[0,2])
        phase_image = phase_ax.imshow(
            phase, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi,
            extent=diagnostics["residual_phase_extent_mm"], interpolation="nearest",
        )
        camera_height_mm = settings.fov_width_mm * adu.shape[0] / adu.shape[1]
        phase_ax.set_xlim(-settings.fov_width_mm/2, settings.fov_width_mm/2)
        phase_ax.set_ylim(-camera_height_mm/2, camera_height_mm/2)
        phase_ax.set_xlabel("x (mm)")
        phase_ax.set_ylabel("y (mm)")
        phase_ax.set_title("Added phase residual (simulation truth)", fontsize=11)
        self.figure.colorbar(phase_image, ax=phase_ax, label="Phase error (rad)",
                             shrink=.7)
        self.canvas.draw_idle()
        self.rendered_generation = generation
        self.status.set(
            f"1920 × 1080 detector · {settings.pulses_per_exposure} pulses · "
            f"peak {diagnostics['expected_electron_peak']:.0f} e⁻ · "
            f"saturated {100*diagnostics['saturated_fraction']:.3f}% · "
            f"added phase RMS {diagnostics['residual_phase_rms_rad']:.3g} rad · "
            f"saved output energy {diagnostics['output_energy_J']*1e9:.3g} nJ. "
            "Top/right curves are mean x/y profiles (camera black level removed). "
            "Residual phase compares the camera-plane field with the saved field "
            "propagated to the same plane; the CCD does not measure phase. "
            "Laser state is unchanged."
            + (" Saved hot phase is unavailable; this camera image uses the cold optical output."
               if self.result.get("hot_phase_validity") == "outside_supported_conditions" else ""))
