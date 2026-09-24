"""Export auditable Yb:LuAG model spectra from the bundled reconstruction.

Run with PYTHONPATH=src. This does not claim a new trace of the source figures.
"""

import csv
from pathlib import Path

from ybluag import YbLuAGMaterial, fluorescence_spectrum
from ybluag.model import _spectra


def main():
    root = Path(__file__).resolve().parents[1] / "spectra"
    wavelength, temperatures, absorption, archived_emission = _spectra()
    path = root / "yb_luag_model_spectra_20_200C.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("temperature_C", "wavelength_nm", "absorption_cm2",
                         "archived_emission_reconstruction_cm2",
                         "mccumber_emission_cm2", "fluorescence_photon_probability_per_nm",
                         "energy_equivalent_fluorescence_wavelength_nm"))
        for index, temperature in enumerate(temperatures):
            # The lifetime is not used in cross-section or normalized-shape export.
            # Higher-temperature measured lifetimes are not implied here.
            material = YbLuAGMaterial(temperature_K=float(temperature),
                                     lifetime_s=0.965e-3)
            fluorescence = fluorescence_spectrum(material)
            for j, value in enumerate(wavelength):
                writer.writerow((float(temperature - 273.15), float(value),
                                 float(absorption[index, j] * 1e4),
                                 float(archived_emission[index, j] * 1e4),
                                 material.cross_sections_m2(float(value))[1] * 1e4,
                                 float(fluorescence.photon_probability_per_nm[j]),
                                 fluorescence.energy_equivalent_wavelength_nm))
    print(path)


if __name__ == "__main__":
    main()
