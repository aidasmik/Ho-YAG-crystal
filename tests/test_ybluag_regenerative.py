import numpy as np
import unittest

from hoyag.propagation import Grid2D, gaussian_beam
from ybluag.model import YbLuAGMaterial
from ybluag.regenerative import RegenerativeCavity, amplify_regenerative


class RegenerativeTests(unittest.TestCase):
 def test_regenerative_cavity_tracks_periodic_gain_and_extraction(self):
    grid = Grid2D.square(32, 12e-3)
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=0.965e-3,
                             pump_wavelength_nm=938, signal_wavelength_nm=1030)
    seed = gaussian_beam(grid, 0.6e-3)
    x, y = grid.mesh
    pump = np.exp(-2*(x*x+y*y)/(1e-3)**2)
    pump *= 5/(pump.sum()*grid.dx*grid.dy)
    scale = np.ones((2, *grid.shape))
    cavity = RegenerativeCavity(round_trips=3, air_gap_m=0.25,
                                mirror_radius_m=0.5)
    result = amplify_regenerative(material, grid, seed, 10e-9, pump, scale,
                                   100e-6, 10, 1e4, cavity, 1.9e-19, 0)
    self.assertEqual(result["roundtrip_energy_J"].shape, (3, 3))
    self.assertLess(result["cycles"], 100)
    self.assertLessEqual(result["residual"], 1e-6)
    self.assertAlmostEqual(result["output_energy_J"],
                           result["stored_energy_J"]*cavity.extraction_efficiency,
                           delta=result["output_energy_J"]*1e-10)
    self.assertLess(result["storage_time_s"], 1e-4)
    self.assertTrue(np.isfinite(result["heat_W_m3_by_slice"]).all())
    self.assertGreater(result["pump_absorbed_W_m2_by_slice"].sum(), 0)
    dz = 100e-6/2
    np.testing.assert_allclose(
        result["heat_W_m3_by_slice"]*dz,
        result["pump_absorbed_W_m2_by_slice"] -
        result["signal_gain_W_m2_by_slice"] -
        result["escaping_fluorescence_W_m2_by_slice"] -
        result["excitation_storage_change_W_m2_by_slice"], rtol=1e-12)
    self.assertLess(abs(result["cavity_energy_balance_residual_J"]), 1e-16)
    self.assertLess(result["population_photon_balance_relative_L1"], 1e-3)
    self.assertGreater(result["cavity_losses_J"]["unextracted_ejection_J"], 0)
    self.assertGreater(result["cavity_losses_J"]["held_optics_J"], 0)


 def test_regenerative_geometry_rejects_unstable_cavity(self):
    grid = Grid2D.square(32, 12e-3)
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=0.965e-3)
    with self.assertRaisesRegex(ValueError, "unstable"):
        RegenerativeCavity(air_gap_m=0.6, mirror_radius_m=0.5).validate(
            material, 100e-6, grid, 1e4)

 def test_unpumped_cavity_cannot_create_signal_energy(self):
    grid = Grid2D.square(32, 12e-3)
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=0.965e-3)
    seed = gaussian_beam(grid, 0.6e-3)
    result = amplify_regenerative(
        material, grid, seed, 10e-9, np.zeros(grid.shape),
        np.ones((2, *grid.shape)), 100e-6, 10, 1e4,
        RegenerativeCavity(round_trips=2), 1.9e-19, 0)
    self.assertLess(result["output_energy_J"], 10e-9)

 def test_recovery_step_refinement(self):
    grid = Grid2D.square(32, 12e-3)
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=0.965e-3)
    seed = gaussian_beam(grid, 0.6e-3)
    x, y = grid.mesh
    pump = np.exp(-2*(x*x+y*y)/(1e-3)**2)
    pump *= 5/(pump.sum()*grid.dx*grid.dy)
    outputs = []
    for steps in (2, 4, 8):
        result = amplify_regenerative(
            material, grid, seed, 10e-6, pump,
            np.ones((2, *grid.shape)), 100e-6, 10, 1e4,
            RegenerativeCavity(round_trips=3, recovery_substeps=steps),
            1.9e-19, 0)
        outputs.append(result["output_energy_J"])
        self.assertLess(result["population_photon_balance_relative_L1"], 1e-3)
    coarse = abs(outputs[0]-outputs[1])
    fine = abs(outputs[1]-outputs[2])
    self.assertLess(fine, coarse)
    self.assertLess(fine/outputs[2], 0.01)
