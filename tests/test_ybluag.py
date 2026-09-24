"""Independent Yb:LuAG rate, gain and energy checks."""

import unittest
import json
import csv
import sys
from pathlib import Path

import numpy as np

from hoyag.propagation import (Grid2D, angular_spectrum_propagate,
                               angular_spectrum_transfer,
                               _cached_angular_spectrum_transfer)
from hoyag.thermal import DiskThermalMesh
from ybluag import (YbLuAGMaterial, propagate_cw,
                    propagate_structured_small_signal, propagate_pulse,
                    periodic_pump_state, periodic_pulse_heat,
                    fluorescence_spectrum, scan_output_coupler)
from ybluag import solve_yb_assembly
from ybluag import YbGallerySettings, simulate_structured_gallery, simulate_pulsed_seed
from ybluag.multipass_pump import steady_multipass_pump, transport_multipass_pump
from ybluag.beam_shaping import gaussian_seed_and_target_mask


class YbLuAGPhysicsTests(unittest.TestCase):
    def test_phase_only_mask_shapes_gaussian_after_propagation(self):
        grid = Grid2D.square(96, 0.012)
        x, y = grid.mesh
        radius = np.hypot(x, y)
        waist = 0.6e-3
        center = radius < 0.12e-3
        ring = (radius > 0.25e-3) & (radius < 0.38e-3)
        core = radius < 0.45e-3
        fields = {}
        for target in ("Gaussian TEM00", "Helical LG(0,+1)",
                       "Double helix LG(0,+2)", "Hermite-Gaussian HG(1,1)",
                       "Needle Bessel-Gaussian", "Flattop super-Gaussian"):
            source, mask = gaussian_seed_and_target_mask(
                grid, waist, 1.0, target, 1030e-9, 0.25)
            np.testing.assert_allclose(abs(source * np.exp(1j * mask))**2,
                                       abs(source)**2, rtol=1e-14, atol=1e-9)
            fields[target] = abs(angular_spectrum_propagate(
                source * np.exp(1j * mask), grid, 1030e-9, 0.25))**2
            self.assertAlmostEqual(float(np.sum(fields[target]) * grid.dx * grid.dy),
                                   1.0, places=12)
        gaussian = fields["Gaussian TEM00"]
        vortex = fields["Helical LG(0,+1)"]
        needle = fields["Needle Bessel-Gaussian"]
        flat = fields["Flattop super-Gaussian"]
        self.assertGreater(float(gaussian[center].mean() / gaussian[ring].mean()), 1)
        self.assertLess(float(vortex[center].mean() / vortex[ring].mean()), 0.3)
        self.assertGreater(float(needle[center].mean() / needle[ring].mean()), 5)
        self.assertLess(float(flat[core].std() / flat[core].mean()),
                        0.5 * float(gaussian[core].std() / gaussian[core].mean()))

    def test_proposal_pulse_defaults_and_stretch_validation(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
        from ybluag_app import calculate_pulsed
        result = calculate_pulsed({})
        self.assertEqual(result["yb_at_percent"], 12)
        self.assertEqual(result["pump_passes"], 10)
        self.assertAlmostEqual(result["input_energy_J"], 10e-9)
        self.assertAlmostEqual(result["stretch_factor"], 10_000 / 300)
        self.assertLess(result["peak_pump_intensity_kW_cm2"], 10)
        np.testing.assert_allclose(result["uniform_isothermal_output_fluence_J_m2"],
                                   result["output_fluence_J_m2"])
        self.assertEqual(len(result["x_mm"]), len(result["output_fluence_J_m2"][0]))
        self.assertEqual(len(result["y_mm"]), len(result["output_fluence_J_m2"]))
        thermal = result["thermal"]
        self.assertEqual(thermal["status"], "design_reference")
        self.assertLessEqual(thermal["disk_temperature_max_C"], 25.01)
        self.assertGreater(thermal["actual_constant_property_max_C"], 26.85)
        self.assertAlmostEqual(thermal["actual_heat_W"],
                               result["cycle_average_heat_W_upper_or_assumed"], places=7)
        self.assertEqual(np.asarray(thermal["front_displacement_nm"]).shape,
                         np.asarray(thermal["rear_displacement_nm"]).shape)
        timeline = result["thermal_timeline"]
        self.assertEqual(timeline["time_s"][-1], 30.0)
        self.assertTrue(timeline["stabilized"])
        self.assertAlmostEqual(timeline["requested_time_s"], 30.0)
        self.assertLess(abs(timeline["disk_max_C"][-1] - timeline["steady_disk_max_C"]),
                        timeline["stabilization_tolerance_K"])
        self.assertEqual(timeline["cooling_mode"], "feedback")
        self.assertGreater(timeline["coolant_conductance_W_m2K"][-1],
                           timeline["coolant_conductance_W_m2K"][0])
        self.assertFalse(timeline["material_range_valid"][-1])
        self.assertFalse(result["thermal_feedback_applied"])
        self.assertLess(result["phase_residual_rms_rad"], 1e-12)
        self.assertLess(timeline["energy_balance_relative_max"], 1e-7)
        with self.assertRaisesRegex(ValueError, "at least as long"):
            calculate_pulsed({"source_fwhm_fs": 500, "seed_fwhm_ps": 0.3})

    def test_multipass_pump_conserves_energy_and_saturates(self):
        material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=0.000973)
        pump = np.full((2, 2), 2e7)
        density = np.ones((4, 2, 2))
        one = steady_multipass_pump(material, pump, density, 100e-6, 1)
        ten = steady_multipass_pump(material, pump, density, 100e-6, 10)
        self.assertGreater(float(np.mean(ten.excited_fraction_by_slice)),
                           float(np.mean(one.excited_fraction_by_slice)))
        self.assertTrue(np.all((ten.excited_fraction_by_slice >= 0) &
                               (ten.excited_fraction_by_slice <= 1)))
        midpoint, absorbed, final = transport_multipass_pump(
            material, pump, density, 100e-6, 10, ten.excited_fraction_by_slice)
        self.assertTrue(np.all(midpoint > 0))
        np.testing.assert_allclose(np.sum(absorbed, axis=0) + final, pump,
                                   rtol=1e-12)

    def test_nonuniform_pulse_reference_changes_output(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
        from ybluag_app import calculate_pulsed
        result = calculate_pulsed({"cluster_contrast": 0.1,
                                   "operation_duration_s": 0})
        actual = np.asarray(result["output_fluence_J_m2"])
        uniform = np.asarray(result["uniform_isothermal_output_fluence_J_m2"])
        self.assertEqual(actual.shape, uniform.shape)
        self.assertGreater(float(np.max(abs(actual - uniform))), 0)
        self.assertGreater(result["phase_residual_rms_rad"], 0)

    def test_valid_transient_opd_changes_output_phase_not_pulse_energy(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
        from ybluag_app import calculate_pulsed
        hot = calculate_pulsed({"pump_W": 0.2, "operation_duration_s": 1.0})
        cold = calculate_pulsed({"pump_W": 0.2, "operation_duration_s": 0})
        self.assertTrue(hot["thermal_timeline"]["material_range_valid"][-1])
        self.assertTrue(hot["thermal_feedback_applied"])
        self.assertGreater(hot["phase_residual_rms_rad"], 0)
        self.assertAlmostEqual(hot["output_energy_J"], cold["output_energy_J"], places=12)
        self.assertGreater(float(np.max(np.abs(
            np.asarray(hot["output_phase"]) - np.asarray(cold["output_phase"])))), 0)

    def test_selected_cw_shape_includes_uniform_reference_and_cooler(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
        from ybluag_app import calculate_structured
        result = calculate_structured({
            "solver_mode": "weak_probe", "selected_beam": "Helical LG(0,+1)",
            "pump_W": 40, "radius_mm": 0.6,
            "thickness_um": 150, "signal_W": 1, "waist_mm": 0.408,
            "distance_m": 0.25, "phase_mask": "none",
            "phase_strength_rad": np.pi, "density_seed": 17,
            "cluster_count": 24, "cluster_contrast": 0.27,
            "escape_yield": 0})
        self.assertEqual(list(result["modes"]), ["Helical LG(0,+1)"])
        self.assertGreater(max(result["x_mm"]), 5)
        self.assertLess(min(result["x_mm"]), -5)
        self.assertIn(result["thermal"]["status"], ("computed", "design_reference"))
        for name, case in result["modes"].items():
            self.assertEqual(len(case["output_profile"]),
                             len(result["uniform_reference_profiles"][name]))
        self.assertGreater(float(np.max(np.abs(
            np.asarray(result["modes"]["Helical LG(0,+1)"]["output_profile"]) -
            np.asarray(result["uniform_reference_profiles"]["Helical LG(0,+1)"])))), 0)
        self.assertEqual(np.asarray(result["uniform_reference_intensity"]).shape,
                         np.asarray(result["modes"]["Helical LG(0,+1)"]["output_intensity"]).shape)

    def test_yb_gallery_phase_power_and_saturation(self):
        material = YbLuAGMaterial()
        base = dict(pump_power_W=40, input_power_W=1, grid_n=64,
                    cluster_count=4, z_steps=2, post_disk_distance_m=0.25)
        plain = simulate_structured_gallery(material, YbGallerySettings(**base))
        vortex = simulate_structured_gallery(material, YbGallerySettings(
            **base, phase_mask_name="vortex+1"))
        a = plain["outcomes"]["Gaussian TEM00"]
        b = vortex["outcomes"]["Gaussian TEM00"]
        np.testing.assert_allclose(a["input_intensity"], b["input_intensity"], rtol=1e-13)
        self.assertGreater(float(np.max(abs(a["disk_input_intensity"] -
                                            b["disk_input_intensity"]))), 0)
        self.assertAlmostEqual(a["disk_output_power_W"], a["output_power_W"], places=10)
        self.assertGreater(a["net_heat_W_upper_or_assumed"], 0)
        saturated = simulate_structured_gallery(material, YbGallerySettings(
            **base, solver_mode="saturated_cw"))
        self.assertLess(saturated["outcomes"]["Gaussian TEM00"]["disk_output_power_W"],
                        a["disk_output_power_W"])

    def test_yb_periodic_seed_energy_integral(self):
        material = YbLuAGMaterial()
        result = simulate_pulsed_seed(
            material, YbGallerySettings(pump_power_W=40, grid_n=64,
                                        cluster_count=4, z_steps=2),
            "Gaussian TEM00", 10e-9, 10e-12, 10_000, 1)
        self.assertAlmostEqual(
            float(np.trapezoid(result["output_power_trace_W"], result["time_ps"] * 1e-12)) /
            result["disk_output_energy_J"], 1, places=10)
        self.assertAlmostEqual(result["output_energy_J"] / result["disk_output_energy_J"],
                               1, places=10)
        self.assertLess(result["residual"], 1e-6)
        self.assertAlmostEqual(
            result["cycle_average_heat_W_upper_or_assumed"],
            result["cycle_average_pump_absorbed_W"] -
            result["cycle_average_signal_gain_W"] -
            result["cycle_average_escaping_fluorescence_W"], places=10)
        self.assertAlmostEqual(
            result["cycle_average_signal_gain_W"],
            (result["disk_output_energy_J"] - result["input_energy_J"]) * 10_000,
            places=7)

    def test_pulsed_target_uses_gaussian_source_and_additive_correction(self):
        result = simulate_pulsed_seed(
            YbLuAGMaterial(),
            YbGallerySettings(pump_power_W=40, grid_n=64, z_steps=2,
                              cluster_count=4, phase_mask_name="defocus"),
            "Helical LG(0,+1)", 10e-9, 10e-12, 10_000, 1,
            compute_thermal=False)
        source = result["input_fluence_J_m2"]
        disk = result["disk_input_fluence_J_m2"]
        self.assertGreater(float(np.max(abs(source - disk))), 0)
        np.testing.assert_allclose(result["phase_mask"],
                                   np.mod(result["target_phase_mask"] +
                                          result["aberration_phase_mask"], 2 * np.pi),
                                   atol=1e-12)
        self.assertAlmostEqual(float(np.sum(source)), float(np.sum(disk)), places=9)

    def test_yb_modal_cw_roundtrip_balance(self):
        result = simulate_structured_gallery(
            YbLuAGMaterial(), YbGallerySettings(
                pump_power_W=40, solver_mode="modal_cw", grid_n=64,
                cluster_count=4, z_steps=2))
        resonator = result["resonator"]
        self.assertGreater(resonator["small_signal_roundtrip_log_margin"], 0)
        self.assertGreater(resonator["output_coupler_power_W"], 0)
        self.assertLess(abs(resonator["roundtrip_log_residual"]), 1e-8)
        self.assertGreater(result["outcomes"]["Gaussian TEM00"]["net_heat_W_upper_or_assumed"], 0)

    def test_pulse_density_scaling_changes_gain(self):
        material = YbLuAGMaterial()
        time = np.linspace(-20e-12, 20e-12, 21)
        signal = 1e9 * np.exp(-4 * np.log(2) * (time / 10e-12)**2)
        beta = np.array([0.2, 0.2])
        full = propagate_pulse(material, time, np.zeros_like(signal), signal,
                               150e-6, 2, initial_excited_fraction=beta)
        dilute = propagate_pulse(material, time, np.zeros_like(signal), signal,
                                 150e-6, 2, initial_excited_fraction=beta,
                                 density_scale_by_slice=np.array([0.5, 0.5]))
        self.assertGreater(float(np.trapezoid(full.signal_out_W_m2, time)),
                           float(np.trapezoid(dilute.signal_out_W_m2, time)))

    def test_material_anchors_and_units(self):
        material = YbLuAGMaterial()
        self.assertAlmostEqual(material.number_density_m3 / 1.42e27, 1.0)
        absorption, _ = material.cross_sections_m2(940)
        _, emission = material.cross_sections_m2(1030)
        self.assertAlmostEqual(absorption / 7.22e-25, 1.0, places=5)
        self.assertAlmostEqual(emission / 3.062e-24, 1.0, places=3)
        self.assertAlmostEqual(material.lifetime_s / 0.965e-3, 1.0)
        self.assertAlmostEqual(material.coefficients_m1(0)[0] / 1025.24, 1.0, places=5)

    def test_two_manifold_rates_and_transparency(self):
        material = YbLuAGMaterial()
        beta = material.excited_fraction_cw(1e8, 1e6)
        upward, downward = material.rates_s1(1e8, 1e6)
        self.assertAlmostEqual(float((1 - beta) * upward - beta * (downward + 1 / material.lifetime_s)), 0, places=9)
        self.assertAlmostEqual(float(material.coefficients_m1(material.transparency_fraction())[1]), 0, places=11)
        self.assertEqual(material.excited_fraction_cw(0, 0), 0)

    def test_mccumber_reciprocity_over_temperature_and_wavelength(self):
        from ybluag.model import C, H, K_B, GROUND_STARK_CM1, EXCITED_STARK_CM1
        for temperature in (293.15, 353.15, 473.15):
            material = YbLuAGMaterial(temperature_K=temperature, lifetime_s=0.965e-3)
            thermal_cm1 = K_B * temperature / (H * C) / 100
            zg = sum(np.exp(-e / thermal_cm1) for e in GROUND_STARK_CM1)
            ze = sum(np.exp(-(e - EXCITED_STARK_CM1[0]) / thermal_cm1)
                     for e in EXCITED_STARK_CM1)
            for wavelength in (940.0, 1030.0, 1100.0):
                absorption, emission = material.cross_sections_m2(wavelength)
                expected_ratio = zg / ze * np.exp((EXCITED_STARK_CM1[0] - 1e7 / wavelength) / thermal_cm1)
                self.assertAlmostEqual(emission / absorption / expected_ratio, 1.0, places=12)

    def test_beer_limit_and_saturation(self):
        material = YbLuAGMaterial()
        thickness = 100e-6
        weak = propagate_cw(material, thickness, 128, 1.0)
        alpha, _ = material.coefficients_m1(0)
        self.assertAlmostEqual(float(weak.pump_out_W_m2 / np.exp(-alpha * thickness)), 1.0, places=7)
        strong = propagate_cw(material, thickness, 128, 1e9)
        self.assertGreater(float(strong.pump_out_W_m2 / 1e9), float(weak.pump_out_W_m2))
        self.assertTrue(np.all((strong.excited_fraction_by_step >= 0) & (strong.excited_fraction_by_step <= 1)))

    def test_energy_ledger_and_step_convergence(self):
        material = YbLuAGMaterial()
        args = dict(fluorescence_quantum_yield=0.9,
                    mean_fluorescence_wavelength_nm=1030.0)
        coarse = propagate_cw(material, 150e-6, 50, 1e8, 1e5, **args)
        fine = propagate_cw(material, 150e-6, 200, 1e8, 1e5, **args)
        self.assertAlmostEqual(float(fine.heat_W_m2 - fine.absorbed_pump_W_m2 + fine.signal_change_W_m2 + fine.fluorescence_W_m2), 0, places=7)
        self.assertGreater(float(fine.heat_W_m2), 0)
        self.assertAlmostEqual(float(np.sum(fine.heat_W_m3_by_step) * 150e-6 / 200),
                               float(fine.heat_W_m2), places=7)
        self.assertLess(abs(float(fine.pump_out_W_m2 / coarse.pump_out_W_m2) - 1), 2e-5)
        self.assertLess(abs(float(fine.signal_out_W_m2 / coarse.signal_out_W_m2) - 1), 2e-5)

    def test_bounds_and_sample_specific_lifetime(self):
        with self.assertRaisesRegex(ValueError, "lifetime"):
            YbLuAGMaterial(yb_at_percent=15)
        with self.assertRaisesRegex(ValueError, "temperature"):
            YbLuAGMaterial(temperature_K=290)
        with self.assertRaisesRegex(ValueError, "lifetime"):
            YbLuAGMaterial(temperature_K=353.15)
        with self.assertRaisesRegex(ValueError, "wavelength"):
            YbLuAGMaterial(pump_wavelength_nm=1907.7)
        material = YbLuAGMaterial(yb_at_percent=15, lifetime_s=0.985e-3)
        with self.assertRaisesRegex(ValueError, "fluorescence_quantum_yield"):
            propagate_cw(material, 150e-6, 10, 1e8,
                         mean_fluorescence_wavelength_nm=1030)

    def test_derived_fluorescence_spectrum_and_heat(self):
        material = YbLuAGMaterial()
        spectrum = fluorescence_spectrum(material)
        self.assertAlmostEqual(float(np.trapezoid(
            spectrum.photon_probability_per_nm, spectrum.wavelength_nm)), 1.0)
        self.assertAlmostEqual(spectrum.energy_equivalent_wavelength_nm,
                               1012.5969786847127, places=6)
        inferred = propagate_cw(material, 150e-6, 16, 1e7,
                                fluorescence_quantum_yield=0.9)
        explicit = propagate_cw(
            material, 150e-6, 16, 1e7,
            fluorescence_quantum_yield=0.9,
            mean_fluorescence_wavelength_nm=spectrum.energy_equivalent_wavelength_nm)
        np.testing.assert_allclose(inferred.heat_W_m3_by_step,
                                   explicit.heat_W_m3_by_step, rtol=0, atol=0)
        self.assertEqual(inferred.fluorescence_wavelength_nm_used,
                         spectrum.energy_equivalent_wavelength_nm)

    def test_spectral_export_is_numeric_and_traceable(self):
        path = (Path(__file__).resolve().parents[1] / "Yb-LuAG" / "spectra" /
                "yb_luag_model_spectra_20_200C.csv")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4 * 541)
        row = next(r for r in rows if float(r["temperature_C"]) == 20 and
                   float(r["wavelength_nm"]) == 1030)
        material = YbLuAGMaterial()
        absorption, emission = material.cross_sections_m2(1030)
        self.assertAlmostEqual(float(row["absorption_cm2"]) / (absorption * 1e4), 1)
        self.assertAlmostEqual(float(row["mccumber_emission_cm2"]) / (emission * 1e4), 1)

    def test_output_coupler_screen_and_copper_source_config(self):
        root = Path(__file__).resolve().parents[1]
        coatings = json.loads((root / "config/ybluag_10at_coatings.json").read_text())
        oc = coatings["output_coupler"]
        pump_intensity = oc["screening_assumed_pump_W"] / (
            np.pi * oc["screening_assumed_pump_radius_m"]**2)
        result = scan_output_coupler(
            YbLuAGMaterial(pump_wavelength_nm=coatings["wavelengths_nm"]["pump"]),
            pump_intensity,
            oc["screening_disk_thickness_m"], 8,
            oc["screening_candidates"],
            disk_hr_reflectivity=coatings["disk_rear"]["target_min_reflectance_laser"],
            other_roundtrip_survival=oc["screening_other_roundtrip_survival"])
        self.assertEqual(result.selected_transmission,
                         oc["screening_selected_transmission"])
        self.assertGreater(result.selected_output_intensity_W_m2, 0)
        assembly = json.loads((root / "config/ybluag_10at_assembly.json").read_text())
        self.assertEqual(assembly["thermal"]["plate"]["conductivity_W_mK"], 394)

    def test_structured_field_gain_and_phase(self):
        material = YbLuAGMaterial()
        grid = Grid2D.square(16, 2e-3)
        x, y = grid.mesh
        field = np.exp(-(x*x + y*y) / (0.4e-3)**2) * np.exp(1j * np.arctan2(y, x))
        pump = np.full(grid.shape, 1e8)
        result = propagate_structured_small_signal(material, field, grid, pump,
                                                   150e-6, 80, include_diffraction=False)
        beta = result.excited_fraction_by_step[:, 0, 0]
        gain = np.array([material.coefficients_m1(b)[1] for b in beta])
        expected_ratio = np.exp(np.sum(gain) * 150e-6 / 80)
        self.assertAlmostEqual(result.output_power_W / result.input_power_W / expected_ratio, 1, places=10)
        self.assertLess(np.max(abs(np.angle(result.field_out * np.conj(field)))), 1e-12)
        self.assertAlmostEqual(material.refractive_index(), 1.82358010361, places=8)

    def test_shared_diffraction_cache_preserves_both_materials(self):
        grid = Grid2D.square(16, 2e-3)
        x, y = grid.mesh
        field = np.exp(-(x*x + y*y) / (0.3e-3)**2).astype(complex)
        _cached_angular_spectrum_transfer.cache_clear()
        for wavelength, index in ((2.0903e-6, 1.7991),
                                  (1.030e-6, YbLuAGMaterial().refractive_index())):
            transfer = angular_spectrum_transfer(grid, wavelength, 150e-6,
                                                  refractive_index=index)
            expected = np.fft.ifft2(np.fft.fft2(field) * transfer)
            actual = angular_spectrum_propagate(field, grid, wavelength,
                                                150e-6, refractive_index=index)
            self.assertTrue(np.allclose(actual, expected, rtol=1e-13, atol=1e-13))
            repeat = angular_spectrum_propagate(field, grid, wavelength,
                                                150e-6, refractive_index=index)
            self.assertTrue(np.array_equal(repeat, actual))
        self.assertEqual(_cached_angular_spectrum_transfer.cache_info().hits, 2)
        self.assertEqual(_cached_angular_spectrum_transfer.cache_info().currsize, 2)

    def test_pulse_dark_decay_and_population_bounds(self):
        material = YbLuAGMaterial()
        time = np.linspace(0, 1e-4, 11)
        dark = np.zeros_like(time)
        result = propagate_pulse(material, time, dark, dark, 150e-6, 3,
                                 initial_excited_fraction=0.4)
        expected = 0.4 * np.exp(-(time[-1] - time[0]) / material.lifetime_s)
        self.assertTrue(np.allclose(result.final_excited_fraction_by_slice, expected,
                                    rtol=0, atol=1e-14))
        self.assertEqual(float(result.absorbed_pump_fluence_J_m2), 0)
        pulse_time = np.linspace(-30e-12, 30e-12, 121)
        pulse = 1e12 * np.exp(-4*np.log(2)*(pulse_time/10e-12)**2)
        pumped = propagate_pulse(material, pulse_time, pulse, np.zeros_like(pulse),
                                  150e-6, 4)
        self.assertTrue(np.all((pumped.final_excited_fraction_by_slice >= 0) &
                               (pumped.final_excited_fraction_by_slice <= 1)))
        self.assertGreater(float(pumped.absorbed_pump_fluence_J_m2), 0)
        self.assertLess(float(pumped.absorbed_pump_fluence_J_m2),
                        float(np.sum(0.5*(pulse[:-1]+pulse[1:])*np.diff(pulse_time))))

    def test_periodic_pump_converges_without_copying_crystal(self):
        material = YbLuAGMaterial()
        time = np.linspace(-15e-12, 15e-12, 41)
        pump = 5e10 * np.exp(-4*np.log(2)*(time/10e-12)**2)
        before, cycles, error = periodic_pump_state(material, time, pump,
                                                      150e-6, 2, 10_000,
                                                      tolerance=1e-7)
        self.assertGreater(cycles, 1)
        self.assertLess(error, 1e-7)
        self.assertTrue(np.all((before > 0) & (before < 1)))

    def test_periodic_heat_has_local_first_law_balance(self):
        material = YbLuAGMaterial()
        time = np.linspace(-20e-12, 20e-12, 41)
        pump = 1e12 * np.exp(-4*np.log(2)*(time/10e-12)**2)
        signal = 1e8 * np.exp(-4*np.log(2)*(time/10e-12)**2)
        result = periodic_pulse_heat(
            material, time, pump, signal, 150e-6, 2, 1000,
            fluorescence_quantum_yield=0.9,
            mean_fluorescence_wavelength_nm=1030.0)
        dz = 150e-6 / 2
        np.testing.assert_allclose(
            result.heat_W_m3_by_slice * dz,
            result.pump_absorbed_W_m2_by_slice -
            result.signal_change_W_m2_by_slice -
            result.escaping_fluorescence_W_m2_by_slice,
            rtol=1e-12, atol=1e-9)
        self.assertTrue(np.all(result.heat_W_m3_by_slice > 0))
        self.assertLess(result.residual, 1e-8)

    def test_yb_assembly_heat_balance_without_photoelastic_proxy(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config/ybluag_10at_assembly.json").read_text())
        mesh = DiskThermalMesh.disk(3, 2, radius_m=0.005, thickness_m=0.00015)
        grid = Grid2D.square(16, 0.012)
        pump = np.full((mesh.nr, mesh.nphi), 1e3)
        optical = propagate_cw(YbLuAGMaterial(), 150e-6, mesh.nz,
                               pump, np.zeros_like(pump),
                               fluorescence_quantum_yield=0.9,
                               mean_fluorescence_wavelength_nm=1030.0)
        heat = optical.heat_W_m3_by_step
        result = solve_yb_assembly(mesh, heat, grid, config)
        self.assertAlmostEqual(result.temperature.input_heat_W,
                               float(np.sum(heat * mesh.volumes_m3)), places=10)
        self.assertLess(abs(result.temperature.balance_error_W), 1e-8)
        self.assertGreater(result.temperature.disk_temperature_K.max(), 293.15)
        self.assertLessEqual(result.temperature.disk_temperature_K.max(), 300.0)
        self.assertTrue(np.all(result.screens.photoelastic_retardance_bound_rad == 0))
        self.assertTrue(np.all(result.screens.photoelastic_mean_single_pass_opd_m == 0))
        self.assertTrue(np.allclose(result.screens.outward_jones[..., 0, 1], 0))


if __name__ == "__main__":
    unittest.main()
