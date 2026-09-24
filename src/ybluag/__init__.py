"""Yb:LuAG two-manifold material and CW propagation model."""

from .model import YbLuAGMaterial, CWResult, propagate_cw
from .structured import StructuredSignalResult, propagate_structured_small_signal
from .pulsed import (PulseResult, PeriodicHeatResult, propagate_pulse,
                     periodic_pump_state, periodic_pulse_heat)
from .assembly import YbAssemblyResult, scalar_yb_screens, solve_yb_assembly
from .fluorescence import FluorescenceSpectrum, fluorescence_spectrum
from .coating import OutputCouplerScan, scan_output_coupler
from .gallery import YbGallerySettings, simulate_structured_gallery, simulate_pulsed_seed

__all__ = ["YbLuAGMaterial", "CWResult", "propagate_cw",
           "StructuredSignalResult", "propagate_structured_small_signal",
           "PulseResult", "PeriodicHeatResult", "propagate_pulse",
           "periodic_pump_state", "periodic_pulse_heat",
           "YbAssemblyResult", "scalar_yb_screens", "solve_yb_assembly",
           "FluorescenceSpectrum", "fluorescence_spectrum",
           "OutputCouplerScan", "scan_output_coupler",
           "YbGallerySettings", "simulate_structured_gallery", "simulate_pulsed_seed"]
