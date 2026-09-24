"""Independent Yb:LuAG rate, gain and energy checks."""

import unittest

import numpy as np

from hoyag.propagation import Grid2D
from ybluag import YbLuAGMaterial, propagate_cw, propagate_structured_small_signal


class YbLuAGPhysicsTests(unittest.TestCase):
    def test_material_anchors_and_units(self):
        material = YbLuAGMaterial()
        self.assertAlmostEqual(material.number_density_m3 / 1.42e27, 1.0)
        absorption, _ = material.cross_sections_m2(940)
        _, emission = material.cross_sections_m2(1030)
        self.assertAlmostEqual(absorption / 7.22e-25, 1.0, places=5)
        self.assertAlmostEqual(emission / 2.78e-24, 1.0, places=5)
        self.assertAlmostEqual(material.lifetime_s / 0.965e-3, 1.0)
        self.assertAlmostEqual(material.coefficients_m1(0)[0] / 1025.24, 1.0, places=5)

    def test_two_manifold_rates_and_transparency(self):
        material = YbLuAGMaterial()
        beta = material.excited_fraction_cw(1e8, 1e6)
        upward, downward = material.rates_s1(1e8, 1e6)
        self.assertAlmostEqual(float((1 - beta) * upward - beta * (downward + 1 / material.lifetime_s)), 0, places=9)
        self.assertAlmostEqual(float(material.coefficients_m1(material.transparency_fraction())[1]), 0, places=11)
        self.assertEqual(material.excited_fraction_cw(0, 0), 0)

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
        self.assertLess(abs(float(fine.pump_out_W_m2 / coarse.pump_out_W_m2) - 1), 2e-5)
        self.assertLess(abs(float(fine.signal_out_W_m2 / coarse.signal_out_W_m2) - 1), 2e-5)

    def test_bounds_and_sample_specific_lifetime(self):
        with self.assertRaisesRegex(ValueError, "lifetime"):
            YbLuAGMaterial(yb_at_percent=15)
        with self.assertRaisesRegex(ValueError, "temperature"):
            YbLuAGMaterial(temperature_K=290)
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


if __name__ == "__main__":
    unittest.main()
