"""Independent Yb:LuAG rate, gain and energy checks."""

import unittest
import json
from pathlib import Path

import numpy as np

from hoyag.propagation import (Grid2D, angular_spectrum_propagate,
                               angular_spectrum_transfer,
                               _cached_angular_spectrum_transfer)
from hoyag.thermal import DiskThermalMesh
from ybluag import (YbLuAGMaterial, propagate_cw,
                    propagate_structured_small_signal, propagate_pulse,
                    periodic_pump_state, periodic_pulse_heat)
from ybluag import solve_yb_assembly


class YbLuAGPhysicsTests(unittest.TestCase):
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
        with self.assertRaisesRegex(ValueError, "both"):
            propagate_cw(material, 150e-6, 10, 1e8, fluorescence_quantum_yield=0.9)

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
