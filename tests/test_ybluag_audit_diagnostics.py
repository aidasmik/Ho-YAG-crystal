"""Deterministic audit checks independent of expensive coupled runs."""

import sys
from pathlib import Path
import unittest
import warnings

import numpy as np

from ybluag.model import (YbLuAGMaterial, spectral_cross_sections_m2,
                          H, C, K_B, GROUND_STARK_CM1, EXCITED_STARK_CM1)
from ybluag.diagnostics import gain_feasibility, hardware_validity
from ybluag.field_metrics import coherent_overlap, intensity_overlap
from ybluag.gallery import _conservative_heat_polar
from ybluag.gallery import YbGallerySettings
from hoyag.propagation import Grid2D
from hoyag.thermal import DiskThermalMesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Yb-LuAG" / "models"))
import yb_luag_model as legacy


class AuditDiagnosticsTests(unittest.TestCase):
    def test_legacy_spectra_delegate_and_archived_alternative_is_named(self):
        for temperature_C in (20, 80, 140, 200):
            for wavelength_nm in (938, 1030):
                absorption, emission = spectral_cross_sections_m2(
                    wavelength_nm, temperature_C + 273.15)
                self.assertAlmostEqual(legacy.sigma_abs_cm2(wavelength_nm, temperature_C),
                                       absorption*1e4, delta=absorption*1e-8)
                self.assertAlmostEqual(legacy.sigma_em_cm2(wavelength_nm, temperature_C),
                                       emission*1e4, delta=emission*1e-8)
        self.assertNotAlmostEqual(
            legacy.sigma_em_cm2(1030),
            legacy.sigma_em_cm2(1030, dataset="archived_reconstruction"),
            delta=1e-22)

    def test_reciprocity_and_explicit_extrapolation(self):
        wl = 1030.0
        temperature = 293.15
        absorption, emission = spectral_cross_sections_m2(wl, temperature)
        kbt = K_B*temperature/(H*C)/100
        zg = sum(np.exp(-e/kbt) for e in GROUND_STARK_CM1)
        ze = sum(np.exp(-(e-EXCITED_STARK_CM1[0])/kbt) for e in EXCITED_STARK_CM1)
        expected = zg/ze*np.exp((EXCITED_STARK_CM1[0]-1e7/wl)/kbt)
        self.assertAlmostEqual(emission/absorption, expected, delta=expected*1e-12)
        with self.assertRaises(ValueError):
            legacy.sigma_abs_cm2(870)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outside = legacy.sigma_abs_cm2(870, extrapolate=True)
        self.assertNotEqual(outside, legacy.sigma_abs_cm2(880))
        self.assertTrue(any("unvalidated" in str(item.message) for item in caught))
        with self.assertRaises(ValueError):
            legacy.thermal_conductivity_undoped_W_mK(320)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            outside_k = legacy.thermal_conductivity_undoped_W_mK(320, extrapolate=True)
        self.assertNotEqual(outside_k, legacy.thermal_conductivity_undoped_W_mK(298))
        self.assertTrue(caught)

    def test_gain_bounds_and_hardware_unknowns(self):
        material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=0.000973,
                                  pump_wavelength_nm=938, signal_wavelength_nm=1030)
        bound = gain_feasibility(material, thickness_m=100e-6, pump_passes=10,
                                 signal_traversals=10, input_energy_J=10e-9,
                                 requested_output_energy_J=100e-6)
        self.assertAlmostEqual(bound["full_inversion_unsaturated_gain_ceiling"],
                               184.499, delta=0.1)
        self.assertAlmostEqual(bound["pump_asymptotic_unsaturated_gain_ceiling"],
                               81.896, delta=0.1)
        self.assertTrue(bound["requested_energy_exceeds_pump_ceiling"])
        self.assertEqual(bound["material_traversals"], 10)
        regen = gain_feasibility(material, thickness_m=100e-6, pump_passes=10,
                                 signal_traversals=10, regenerative_round_trips=10)
        self.assertEqual(regen["material_traversals"], 20)
        hardware = hardware_validity(pump_nm=938, coating_band_nm=(940, 1090))
        self.assertEqual(hardware["pump_coating_status"], "outside_specified_target_band")
        self.assertIsNone(hardware["damage_margin"])
        self.assertIsNone(hardware["nonlinear_B_integral"])

    def test_coherent_fidelity_distinguishes_phase_from_intensity(self):
        x = np.linspace(-1, 1, 31)
        xx, yy = np.meshgrid(x, x)
        gaussian = np.exp(-(xx*xx+yy*yy))
        opposite_phase = gaussian*np.exp(1j*np.pi*(xx > 0))
        self.assertAlmostEqual(intensity_overlap(gaussian, opposite_phase), 1)
        self.assertLess(coherent_overlap(gaussian, opposite_phase), 0.01)
        self.assertAlmostEqual(coherent_overlap(gaussian,
                                                gaussian*np.exp(1j*0.87)), 1)

    def test_independent_thermal_depth_grid_preserves_heat_and_profile(self):
        settings = YbGallerySettings(grid_n=192, field_size_m=.012,
                                     z_steps=2, thermal_nz=5)
        self.assertEqual((settings.grid_n, settings.thermal_nz), (192, 5))
        grid = Grid2D.square(96, .012)
        x, y = grid.mesh
        radial = np.exp(-2*(x*x+y*y)/(.001)**2)
        source = np.stack((radial, 2*radial)) * 1e8
        mesh = DiskThermalMesh.disk(nr=16, nz=5, nphi=24,
                                    radius_m=.005, thickness_m=100e-6)
        mapped = _conservative_heat_polar(source, grid, mesh)
        input_W = float(source.sum() * grid.dx*grid.dy*100e-6/2)
        output_W = float(np.sum(mapped*mesh.volumes_m3))
        self.assertAlmostEqual(output_W/input_W, 1, delta=1e-12)
        radial_profile = np.mean(mapped, axis=(0, 2))
        self.assertGreater(np.corrcoef(radial_profile,
                                       np.exp(-2*(mesh.r_m/.001)**2))[0, 1], .95)
