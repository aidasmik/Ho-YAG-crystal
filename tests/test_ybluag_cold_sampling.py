"""Independent paraxial Gaussian/q check of the regenerative FFT path.

The Gaussian starts at a 0.6 mm 1/e field radius.  A mirror reflection has
phase exp(-ik r²/R).  No gain, aperture, or disk material is included here.
"""

import math
import unittest

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from hoyag.propagation import (Grid2D, angular_spectrum_propagate,
                               gaussian_beam, normalize_power)
from ybluag.field_metrics import coherent_overlap


WAVELENGTH_M = 1030e-9
AIR_GAP_M = 0.25
MIRROR_RADIUS_M = 0.5
INPUT_RADIUS_M = 0.6e-3


def cold_roundtrip(grid):
    x, y = grid.mesh
    r2 = x*x + y*y
    k = 2*math.pi/WAVELENGTH_M
    numeric = angular_spectrum_propagate(gaussian_beam(grid, INPUT_RADIUS_M),
                                          grid, WAVELENGTH_M, AIR_GAP_M)
    numeric *= np.exp(-1j*k*r2/MIRROR_RADIUS_M)
    numeric = angular_spectrum_propagate(numeric, grid, WAVELENGTH_M, AIR_GAP_M)
    # Gaussian coefficient a in exp(-a r²), equivalent to the ABCD q map.
    a = 1/INPUT_RADIUS_M**2
    a = a/(1+1j*2*a*AIR_GAP_M/k)
    a += 1j*k/MIRROR_RADIUS_M
    a = a/(1+1j*2*a*AIR_GAP_M/k)
    analytic = normalize_power(np.exp(-a*r2), grid)
    intensity = abs(numeric)**2
    radius = math.sqrt(2*np.sum(intensity*r2)/np.sum(intensity))
    overlap = abs(np.vdot(numeric, analytic))**2 / (
        np.vdot(numeric, numeric).real * np.vdot(analytic, analytic).real)
    edge = (abs(x) > 0.4*grid.nx*grid.dx) | (abs(y) > 0.4*grid.ny*grid.dy)
    return radius, 1/math.sqrt(a.real), overlap, float(intensity[edge].sum()/intensity.sum())


class ColdSamplingTests(unittest.TestCase):
    def test_resolution_and_window_are_independent(self):
        results = {}
        for window_mm in (8, 12, 16):
            for n in (96, 192, 384):
                results[window_mm, n] = cold_roundtrip(Grid2D.square(n, window_mm*1e-3))
        # Reproduces the aliasing: power/edge loss can look fine while the
        # focused radius at 96 pixels over 12 mm is wrong.
        coarse = results[12, 96]
        self.assertGreater(abs(coarse[0]/coarse[1]-1), 0.1)
        self.assertLess(coarse[3], 1e-6)
        for window_mm in (8, 12, 16):
            for n in (192, 384):
                radius, analytic, overlap, edge = results[window_mm, n]
                self.assertLess(abs(radius/analytic-1), 5e-4)
                self.assertGreater(overlap, 0.99999)
                self.assertLess(edge, 1e-6)
        self.assertAlmostEqual(results[12, 192][1]*1e6, 136.608, delta=0.1)

    def test_vortex_and_quadrant_need_separate_convergence(self):
        for kind, minimum_192 in (("vortex", .99), ("quadrant", .95)):
            fields = {}
            for n in (96, 192, 384):
                grid = Grid2D.square(n, .012)
                x, y = grid.mesh
                phase = (np.angle(x+1j*y) if kind == "vortex" else
                         np.pi*((x > 0) ^ (y > 0)))
                field = angular_spectrum_propagate(
                    gaussian_beam(grid, INPUT_RADIUS_M)*np.exp(1j*phase),
                    grid, WAVELENGTH_M, AIR_GAP_M)
                fields[n] = (grid, field)
            fine_grid, fine = fields[384]
            x, y = fine_grid.mesh
            points = np.stack((y, x), axis=-1)
            inside = x*x+y*y < (.003)**2
            overlaps = []
            for n in (96, 192):
                grid, field = fields[n]
                real = RegularGridInterpolator(
                    (grid.y, grid.x), field.real,
                    bounds_error=False, fill_value=0)(points)
                imag = RegularGridInterpolator(
                    (grid.y, grid.x), field.imag,
                    bounds_error=False, fill_value=0)(points)
                overlaps.append(coherent_overlap(np.where(inside, fine, 0),
                                                 np.where(inside, real+1j*imag, 0)))
            self.assertGreater(overlaps[1], overlaps[0])
            self.assertGreater(overlaps[1], minimum_192)
            # In particular, Gaussian radius convergence at 192² did not
            # establish even 99.9% coherent convergence for these masks.
            self.assertLess(overlaps[1], .999)
