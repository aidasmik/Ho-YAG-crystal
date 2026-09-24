"""Short-pulse benchmarks in the domain of the two-manifold transport model."""

import unittest

import numpy as np

from hoyag.resonator import fluence_transfer
from ybluag.model import YbLuAGMaterial, H, C, trapezoid
from ybluag.pulsed import propagate_pulse
from ybluag.diagnostics import spectral_gain_screen


class PulseConvergenceTests(unittest.TestCase):
    def test_frantz_nodvik_weak_saturated_and_depleted(self):
        material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.000973,
                                  pump_wavelength_nm=938, signal_wavelength_nm=1030)
        sigma_a, sigma_e = material.cross_sections_m2(1030)
        fsat = H*C/(1030e-9)/(sigma_a+sigma_e)
        beta = .75
        thickness = 100e-6
        g0 = material.number_density_m3*((sigma_a+sigma_e)*beta-sigma_a)*thickness
        for fraction, tolerance in ((.001, 1e-4), (.5, .002), (5, .002)):
            expected = fluence_transfer(fraction*fsat, g0, fsat)
            estimates = []
            for count, steps in ((41, 1), (161, 4), (641, 16)):
                time = np.linspace(-30e-12, 30e-12, count)
                pulse = np.exp(-4*np.log(2)*(time/10e-12)**2)
                pulse *= fraction*fsat/trapezoid(pulse, time)
                result = propagate_pulse(material, time, np.zeros_like(pulse),
                                         pulse, thickness, steps,
                                         initial_excited_fraction=beta)
                self.assertTrue(np.all((result.final_excited_fraction_by_slice >= 0) &
                                       (result.final_excited_fraction_by_slice <= 1)))
                estimates.append(trapezoid(result.signal_out_W_m2, time))
            self.assertLess(abs(estimates[-1]/expected-1), tolerance)
            self.assertLess(abs(estimates[-1]/expected-1),
                            abs(estimates[0]/expected-1)+1e-8)

    def test_spectral_screen_preserves_source_bandwidth_assumption(self):
        material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.000973,
                                  pump_wavelength_nm=938, signal_wavelength_nm=1030)
        early = spectral_gain_screen(material, source_fwhm_fs=300,
                                     stretched_fwhm_ps=10, shared_inversion=.5,
                                     material_traversals=10, thickness_m=100e-6)
        later = spectral_gain_screen(material, source_fwhm_fs=300,
                                     stretched_fwhm_ps=100, shared_inversion=.5,
                                     material_traversals=10, thickness_m=100e-6)
        self.assertEqual(early["status"], "small_signal_screen_only")
        np.testing.assert_allclose(early["input_spectral_weights"],
                                   later["input_spectral_weights"])
        self.assertIsNone(early["compressed_duration_s"])
        self.assertIsNone(early["spectral_phase"])
