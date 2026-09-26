"""Playback of accepted physical correction steps in the Tkinter application."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from beam_profile_view import add_camera_profiles
from ybluag.phase_diagnostics import (
    compensation_residual, piston_removed_residual,
)


class ControlResultPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.result = None
        self.playing = False
        self.index = tk.IntVar(value=0)
        self.info = tk.StringVar(
            value="Run a closed-loop episode to see the correction history."
        )
        bar = ttk.Frame(self, padding=(12, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="CORRECTION SEQUENCE", style="Eyebrow.TLabel").pack(
            side="left", padx=(0, 12))
        ttk.Button(bar, text="Play", style="Accent.TButton",
                   command=self.play).pack(side="left")
        self.slider = ttk.Scale(
            bar, from_=0, to=0, orient="horizontal", command=self._slide
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=10)
        ttk.Label(bar, textvariable=self.info, style="Info.TLabel",
                  wraplength=780).pack(side="left")
        self.views = ttk.Notebook(self)
        self.views.pack(fill="both", expand=True)
        measured = ttk.Frame(self.views)
        compensation_view = ttk.Frame(self.views)
        diagnostics = ttk.Frame(self.views)
        validation = ttk.Frame(self.views)
        self.views.add(measured, text="Measured correction")
        self.views.add(compensation_view, text="Phase compensation")
        self.views.add(diagnostics, text="Cameras + residual phase")
        self.views.add(validation, text="Simulation validation (truth)")
        self.figure = Figure(figsize=(11, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=measured)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.compensation_figure = Figure(figsize=(15, 8), dpi=100, constrained_layout=True)
        self.compensation_canvas = FigureCanvasTkAgg(
            self.compensation_figure, master=compensation_view
        )
        self.compensation_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.camera_figure = Figure(figsize=(15, 6), dpi=100, constrained_layout=True)
        self.camera_canvas = FigureCanvasTkAgg(self.camera_figure, master=diagnostics)
        self.camera_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.truth_figure = Figure(figsize=(11, 8), dpi=100, constrained_layout=True)
        self.truth_canvas = FigureCanvasTkAgg(self.truth_figure, master=validation)
        self.truth_canvas.get_tk_widget().pack(fill="both", expand=True)
        ax = self.figure.subplots()
        ax.text(
            0.5, 0.5, "Correction history will appear here", ha="center", va="center"
        )
        ax.axis("off")
        self.canvas.draw_idle()

    def load(self, result):
        self.result = result
        self.playing = False
        self.slider.configure(to=max(0, len(result["steps"]) - 1))
        self.show(0)

    def _slide(self, value):
        if self.result is not None:
            self.show(
                min(len(self.result["steps"]) - 1, max(0, int(round(float(value)))))
            )

    def play(self):
        if self.result is None:
            return
        self.playing = True
        self.slider.set(0)
        self.after(700, self._advance)

    def _advance(self):
        if not self.playing or self.result is None:
            return
        next_step = int(round(self.slider.get())) + 1
        if next_step >= len(self.result["steps"]):
            self.playing = False
            return
        self.slider.set(next_step)
        self.after(700, self._advance)

    def show(self, index):
        result = self.result
        if result is None:
            return
        steps = result["steps"]
        step = steps[index]
        hardware = result.get("controller", {}).get("method") in ("spgd", "interferometric")
        update_kind = "command updates" if hardware else "accepted updates"
        reference = np.asarray(result["reference_fluence_J_m2"], float)
        initial = np.asarray(steps[0]["output_fluence_J_m2"], float)
        current = np.asarray(step["output_fluence_J_m2"], float)
        correction = np.asarray(step["correction_phase_rad"], float)
        live = (
            result.get("latest_observation")
            if result.get("status") == "running"
            else None
        )
        show_live = live is not None and index == len(steps) - 1
        camera_source = live if show_live else step
        self._draw_phase_compensation(camera_source, step, result)
        camera_frames = np.asarray(camera_source["camera_adu"])
        camera = camera_frames[0]
        reference_camera = np.asarray(result["reference_camera_adu"])

        def profiles(source, arm):
            data = source.get("camera_profiles")
            return (
                None
                if data is None
                else (
                    np.asarray(data["x_mean_adu"])[arm],
                    np.asarray(data["y_mean_adu"])[arm],
                )
            )

        reference_profile_source = {
            "camera_profiles": result.get("reference_camera_profiles")
        }
        camera_fov = float(result["episode"]["camera_fov_mm"])
        self.figure.clear()
        slots = self.figure.add_gridspec(2, 3)
        common = max(
            float(reference.max()), float(initial.max()), float(current.max()), 1e-30
        )
        panels = (
            (reference, "Target at output plane", "inferno", common),
            (initial, "Uncorrected output", "inferno", common),
            (
                current,
                f"Output after {step['iteration']} {update_kind}",
                "inferno",
                common,
            ),
            (correction, "Input SLM correction (rad)", "twilight", None),
        )
        for slot, (values, title, cmap, vmax) in zip(
            (slots[0, 0], slots[0, 1], slots[0, 2], slots[1, 0]), panels
        ):
            ax = self.figure.add_subplot(slot)
            ax.imshow(
                values,
                origin="lower",
                cmap=cmap,
                vmin=0 if vmax else None,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        add_camera_profiles(
            self.figure,
            slots[1, 1],
            camera,
            (
                f"Live probe #{live['evaluation']} · focus camera"
                if show_live
                else "Measured focus camera"
            ),
            fov_width_mm=camera_fov,
            baseline=32.0,
            unit="ADU",
            compact=True,
            reference_image=reference_camera[0],
            profiles=profiles(camera_source, 0),
            reference_profiles=profiles(reference_profile_source, 0),
        )
        history = self.figure.add_subplot(slots[1, 2])
        iterations = [s["iteration"] for s in steps]
        losses = [s["camera_loss"] for s in steps]
        energies = [s["measured_energy_J"] * 1e9 for s in steps]
        if all(v is not None for v in losses):
            history.plot(iterations, losses, "o-", label="Camera loss")
            if hardware:
                colors = {
                    "improved": "#1a8c6a",
                    "regressed": "#c66335",
                    "energy_rollback": "#8b3a77",
                    "rejected_restore": "#9b3f45",
                    "initial": "#116d83",
                    "best_restore": "#1769aa",
                    "best_rejected": "#707070",
                }
                history.scatter(
                    iterations,
                    losses,
                    c=[colors.get(s.get("update_status"), "#116d83") for s in steps],
                    s=18,
                    zorder=3,
                )
                rejected = [
                    s for s in steps
                    if s.get("rejected_trial_camera_loss") is not None
                ]
                if rejected:
                    history.scatter(
                        [s["iteration"] for s in rejected],
                        [s["rejected_trial_camera_loss"] for s in rejected],
                        marker="x", color="#b54040", s=42,
                        label="Rejected trial", zorder=4,
                    )
                    history.legend(fontsize=7, loc="best")
            history.set_ylabel("Camera loss")
        twin = history.twinx()
        twin.plot(iterations, energies, "s--", color="#087f8c", label="Measured energy")
        twin.set_ylabel("Measured energy (nJ)")
        history.axvline(step["iteration"], color="#cc5a37", alpha=0.6)
        history.set_xlabel(
            "Physical command update" if hardware else "Accepted correction iteration"
        )
        history.set_title("Progress through the physical solver", fontsize=10)
        self.canvas.draw_idle()
        self.camera_figure.clear()
        has_measured_phase = (
            camera_source.get("measured_phase_rad") is not None
            and result.get("reference_measured_phase_rad") is not None
        )
        camera_slots = self.camera_figure.add_gridspec(
            1, 4 if has_measured_phase else 3, wspace=0.25
        )
        for arm, title in enumerate(("Focus camera", "Astigmatic diagnostic camera")):
            if arm >= len(camera_frames):
                break
            add_camera_profiles(
                self.camera_figure,
                camera_slots[0, arm],
                camera_frames[arm],
                f"{title} · solid measured / dashed target",
                fov_width_mm=camera_fov,
                baseline=32.0,
                unit="ADU",
                reference_image=reference_camera[arm],
                profiles=profiles(camera_source, arm),
                reference_profiles=profiles(reference_profile_source, arm),
            )
        phase_source = camera_source if "output_phase_rad" in camera_source else step
        phase_intensity = np.asarray(phase_source["output_fluence_J_m2"], float)
        phase_angle = np.asarray(phase_source["output_phase_rad"], float)
        target_phase = np.asarray(result["reference_phase_rad"], float)
        actual_field = np.sqrt(phase_intensity) * np.exp(1j * phase_angle)
        target_field = np.sqrt(reference) * np.exp(1j * target_phase)
        if has_measured_phase:
            measured_amp = np.asarray(camera_source["measured_phase_amplitude"])
            measured_angle = np.asarray(camera_source["measured_phase_rad"])
            reference_amp = np.asarray(result["reference_measured_phase_amplitude"])
            reference_angle = np.asarray(result["reference_measured_phase_rad"])
            measured_ax = self.camera_figure.add_subplot(camera_slots[0, 2])
            try:
                measured_residual, measured_rms = piston_removed_residual(
                    measured_amp * np.exp(1j * measured_angle),
                    reference_amp * np.exp(1j * reference_angle),
                )
            except ValueError:
                measured_ax.text(.5, .5, "No measured phase overlap",
                                 ha="center", va="center", transform=measured_ax.transAxes)
                measured_ax.set_axis_off()
            else:
                field_size = float(result["episode"]["field_size_mm"])
                measured_ax.imshow(
                    measured_residual, origin="lower", cmap="twilight",
                    vmin=-np.pi, vmax=np.pi,
                    extent=(-field_size/2, field_size/2, -field_size/2, field_size/2),
                    interpolation="nearest",
                )
                measured_ax.set_title(
                    f"Interferometer residual · {measured_rms:.2g} rad RMS", fontsize=10
                )
                measured_ax.set_xlabel("x (mm)")
                measured_ax.set_ylabel("y (mm)")
        phase_ax = self.camera_figure.add_subplot(camera_slots[0, 3 if has_measured_phase else 2])
        try:
            residual, residual_rms = piston_removed_residual(
                actual_field, target_field)
        except ValueError:
            phase_ax.text(.5, .5, "No shared illuminated pupil",
                          ha="center", va="center", transform=phase_ax.transAxes)
            phase_ax.set_axis_off()
        else:
            field_size = float(result["episode"]["field_size_mm"])
            phase_image = phase_ax.imshow(
                residual, origin="lower", cmap="twilight",
                vmin=-np.pi, vmax=np.pi,
                extent=(-field_size/2, field_size/2, -field_size/2, field_size/2),
                interpolation="nearest",
            )
            camera_height = camera_fov * camera_frames.shape[1] / camera_frames.shape[2]
            phase_ax.set_xlim(-camera_fov/2, camera_fov/2)
            phase_ax.set_ylim(-camera_height/2, camera_height/2)
            phase_ax.set_xlabel("x (mm)")
            phase_ax.set_ylabel("y (mm)")
            phase_ax.set_title(
                f"Output phase residual (truth) · {residual_rms:.2g} rad RMS",
                fontsize=10,
            )
            self.camera_figure.colorbar(phase_image, ax=phase_ax,
                                         label="Phase error (rad)", shrink=.7)
        self.camera_canvas.draw_idle()
        live_note = f" · live probe {live['evaluation']}" if show_live else ""
        status_note = (
            f" · {step.get('update_status', 'applied')}" if hardware and index else ""
        )
        self.info.set(
            f"Update {step['iteration']}/{steps[-1]['iteration']} · {step['full_solves']} full solves{live_note}{status_note} · "
            f"measured {step['measured_energy_J']*1e9:.3g} nJ · "
            f"camera loss {step['camera_loss'] if step['camera_loss'] is not None else 'off'}."
        )
        self.truth_figure.clear()
        truth_axes = self.truth_figure.subplots(2, 3)
        phase_axes = truth_axes[0]
        target_phase = np.asarray(result["reference_phase_rad"])
        actual_phase = np.asarray(step["output_phase_rad"])
        target_mask = reference > 0.01 * reference.max()
        actual_mask = current > 0.01 * current.max()
        phase_axes[0].imshow(
            np.where(target_mask, target_phase, np.nan),
            origin="lower",
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        phase_axes[1].imshow(
            np.where(actual_mask, actual_phase, np.nan),
            origin="lower",
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        phase_axes[0].set_title("Target complex-field phase")
        phase_axes[1].set_title("Simulated output phase")
        overlaps = [s["validation_truth"]["coherent_overlap"] for s in steps]
        useful = [
            s["validation_truth"]["useful_target_mode_energy_J"] * 1e9 for s in steps
        ]
        phase_axes[2].plot(iterations, overlaps, "o-", label="Coherent overlap")
        phase_axes[2].plot(iterations, useful, "s--", label="Useful target energy (nJ)")
        phase_axes[2].axvline(step["iteration"], color="#cc5a37", alpha=0.6)
        phase_axes[2].legend(fontsize=8)
        phase_axes[2].set_xlabel(
            "Physical command update" if hardware else "Accepted correction iteration"
        )
        phase_axes[2].set_title("Truth-assisted validation only")
        trace = result.get("observation_trace", [])
        if trace and "disk_peak_temperature_C" in trace[0]:
            chronological = result["episode"]["mode"] == "in_situ"
            xx = np.asarray(
                [r["time_s"] if chronological else r["evaluation"] for r in trace],
                float,
            )
            x_label = "Simulated time (s)" if chronological else "Probe # (snapshot)"
            disk = np.asarray([r["disk_peak_temperature_C"] for r in trace], float)
            coolant = np.asarray(
                [r["coolant_temperature_K"] - 273.15 for r in trace], float
            )
            probes = np.asarray(
                [
                    max(
                        (v for v in r["measured_probe_temperature_K"] if v is not None),
                        default=np.nan,
                    )
                    - 273.15
                    for r in trace
                ],
                float,
            )
            truth_axes[1, 0].plot(xx, disk, label="Disk peak, true")
            truth_axes[1, 0].plot(xx, coolant, label="Coolant setting")
            truth_axes[1, 0].plot(
                xx, probes, ".", ms=3, label="Highest probe, measured"
            )
            truth_axes[1, 0].set_ylabel("Temperature (°C)")
            truth_axes[1, 0].set_title("Thermal evolution")
            truth_axes[1, 0].legend(fontsize=7)
            truth_axes[1, 1].plot(
                xx, [r["roundtrip_opd_pv_nm"] for r in trace], color="#8b3a77"
            )
            truth_axes[1, 1].set_ylabel("OPD peak-to-valley (nm)")
            truth_axes[1, 1].set_title("Thermal optical path")
            truth_axes[1, 2].plot(
                xx, [r["pump_power_W"] for r in trace], color="#c66335"
            )
            truth_axes[1, 2].set_ylabel("Pump power (W)")
            truth_axes[1, 2].set_title("Applied pump condition")
            for ax in truth_axes[1]:
                ax.set_xlabel(x_label)
        else:
            for ax in truth_axes[1]:
                ax.text(
                    0.5, 0.5, "Thermal history unavailable", ha="center", va="center"
                )
                ax.axis("off")
        self.truth_canvas.draw_idle()

    def _draw_phase_compensation(self, source, step, result):
        """Compare the truth-based phase-only oracle with requested and delivered SLM maps."""
        figure = self.compensation_figure
        figure.clear()
        if source.get("ideal_compensation_rad") is None:
            ax = figure.subplots()
            ax.text(.5, .5, "Phase-compensation maps were not saved in this run.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            self.compensation_canvas.draw_idle()
            return
        ideal = np.asarray(source["ideal_compensation_rad"], dtype=float)
        requested = np.asarray(
            source.get("requested_compensation_rad", step["correction_phase_rad"]),
            dtype=float,
        )
        delivered = np.asarray(source["delivered_slm_compensation_rad"], dtype=float)
        if ideal.shape != requested.shape or ideal.shape != delivered.shape:
            raise ValueError("phase-compensation maps must share the optical grid")
        pupil = np.isfinite(ideal) & np.isfinite(delivered)
        field_width_mm = float(result["episode"]["field_size_mm"])
        if result.get("reference_slm_illumination") is not None:
            illumination = np.asarray(result["reference_slm_illumination"], float)
        else:
            # Earlier run files did not save the nominal Gaussian SLM field.
            x = (np.arange(ideal.shape[1]) + .5 - ideal.shape[1] / 2) * (
                field_width_mm / ideal.shape[1]
            )
            y = (np.arange(ideal.shape[0]) + .5 - ideal.shape[0] / 2) * (
                field_width_mm / ideal.shape[0]
            )
            xx, yy = np.meshgrid(x, y)
            waist_mm = float(result["episode"]["waist_mm"])
            illumination = np.exp(-2 * (xx**2 + yy**2) / waist_mm**2)
        requested_difference, requested_rms = compensation_residual(
            requested, ideal, illumination
        )
        delivered_difference, delivered_rms = compensation_residual(
            delivered, ideal, illumination
        )
        panels = (
            (ideal, "Ideal phase-only compensation · truth"),
            (np.where(pupil, requested, np.nan), "Controller-requested correction"),
            (delivered, "SLM-delivered correction"),
            (requested_difference, "Requested − ideal · piston removed"),
            (delivered_difference, "Delivered − ideal · piston removed"),
        )
        half_width = field_width_mm / 2
        if np.any(pupil):
            rows, columns = np.nonzero(pupil)
            pixel_mm = 2 * half_width / ideal.shape[1]
            x_edge = np.max(np.abs((columns + .5) * pixel_mm - half_width))
            y_edge = np.max(np.abs((rows + .5) * pixel_mm - half_width))
            view_half_width = min(
                half_width,
                max(1.5 * float(result["episode"]["waist_mm"]),
                    1.25 * float(max(x_edge, y_edge))),
            )
        else:
            view_half_width = half_width
        axes = figure.subplots(2, 3)
        for ax, (values, title) in zip(axes.flat[:5], panels):
            image = ax.imshow(
                values, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi,
                extent=(-half_width, half_width, -half_width, half_width),
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_aspect("equal")
            ax.set_xlim(-view_half_width, view_half_width)
            ax.set_ylim(-view_half_width, view_half_width)
        trend = axes.flat[5]
        update_indices, requested_history, delivered_history = [], [], []
        for saved in result["steps"]:
            if saved.get("ideal_compensation_rad") is None:
                continue
            saved_ideal = np.asarray(saved["ideal_compensation_rad"], float)
            saved_requested = np.asarray(saved["correction_phase_rad"], float)
            saved_delivered = np.asarray(saved["delivered_slm_compensation_rad"], float)
            update_indices.append(saved["iteration"])
            requested_history.append(
                compensation_residual(saved_requested, saved_ideal, illumination)[1]
            )
            delivered_history.append(
                compensation_residual(saved_delivered, saved_ideal, illumination)[1]
            )
        trend.plot(update_indices, requested_history, "o-", label="Requested − ideal")
        trend.plot(update_indices, delivered_history, "s--", label="Delivered − ideal")
        trend.axvline(step["iteration"], color="#cc5a37", alpha=.6)
        trend.set_xlabel("Correction update")
        trend.set_ylabel("Seed-fluence-weighted RMS (rad)")
        trend.set_title("Correction residual through the run", fontsize=10)
        trend.legend(fontsize=8)
        figure.colorbar(image, ax=list(axes.flat[:5]), label="Phase (rad)", shrink=.78)
        figure.suptitle(
            f"Update {step['iteration']} · requested/ideal {requested_rms:.2f} rad RMS"
            f" · delivered/ideal {delivered_rms:.2f} rad RMS",
            fontsize=11,
        )
        self.compensation_canvas.draw_idle()
