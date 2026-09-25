"""Contact map fixed per crystal and operating-point pump/coolant samples."""
import numpy as np
from .common import correlated_unit_map


def sample_contact(shape, ranges, seed, *, enabled=True, stress=1.0):
    if not enabled:
        return np.ones(shape)
    field = correlated_unit_map(shape, max(1, min(shape)/4), seed)
    return np.maximum(.1, 1+stress*ranges["thermal_contact_fraction"]*field)


def sample_operating_point(nominal, ranges, seed, *, enabled=True, stress=1.0,
                           elapsed_s=0.0):
    rng = np.random.default_rng(seed)
    if not enabled:
        return dict(pump_W=nominal["pump_W"], radius_mm=nominal["radius_mm"],
                    coolant_temperature_K=nominal["coolant_temperature_C"]+273.15,
                    pump_center_m=(0., 0.))
    power = nominal["pump_W"]*(1+stress*rng.uniform(-1, 1)*ranges["pump_power_fraction"])
    radius = nominal["radius_mm"]*(1+stress*rng.uniform(-1, 1)*ranges["pump_radius_fraction"])
    displacement = stress*rng.uniform(0, ranges["pump_offset_radius_fraction"])*radius*1e-3
    angle = rng.uniform(0, 2*np.pi)
    coolant = (nominal["coolant_temperature_C"]+273.15+
               stress*rng.uniform(-1, 1)*ranges["coolant_temperature_K"]+
               rng.normal(0, ranges["thermal_drift_K_per_s"]*elapsed_s))
    return dict(pump_W=power, radius_mm=radius, coolant_temperature_K=coolant,
                pump_center_m=(displacement*np.cos(angle), displacement*np.sin(angle)))
