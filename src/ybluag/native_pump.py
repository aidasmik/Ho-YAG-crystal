"""Optional native loop for fixed-temperature, scalar-cross-section pump transport.

Numba compiles this numerical loop to machine code. Cross sections and ion
density are supplied by the material model; no spectral data are duplicated.
The general NumPy path remains available for spatially varying temperature.
"""

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # The simulator works without its optional speed extra.
    njit = None


if njit is not None:

    @njit(cache=True)
    def transport_fixed_temperature(
        pump,
        scale,
        beta,
        thickness_m,
        passes,
        relay_efficiency,
        ion_density_m3,
        sigma_abs_m2,
        sigma_em_m2,
    ):
        """Return midpoint pump intensity, absorbed power and final pump map."""
        nz, ny, nx = scale.shape
        dz = thickness_m / nz
        midpoint = np.zeros_like(scale)
        absorbed = np.zeros_like(scale)
        current = pump.copy()

        for pump_pass in range(passes):
            for position in range(nz):
                iz = position if pump_pass % 2 == 0 else nz - 1 - position
                for iy in range(ny):
                    for ix in range(nx):
                        population = beta[iz, iy, ix]
                        alpha = ion_density_m3 * (
                            sigma_abs_m2 * (1.0 - population) - sigma_em_m2 * population
                        )
                        depth = -alpha * scale[iz, iy, ix] * dz
                        incoming = current[iy, ix]
                        outgoing = incoming * math.exp(depth)
                        cell_average = (
                            incoming * math.expm1(depth) / depth
                            if depth != 0.0
                            else incoming
                        )
                        midpoint[iz, iy, ix] += cell_average
                        absorbed[iz, iy, ix] += incoming - outgoing
                        current[iy, ix] = outgoing
            if pump_pass < passes - 1:
                current *= relay_efficiency
        return midpoint, absorbed, current

else:
    transport_fixed_temperature = None
