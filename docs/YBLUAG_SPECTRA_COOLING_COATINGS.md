# Yb:LuAG spectroscopy, coatings, and generic cooling inputs

## Numerical spectra and provenance

[`Yb-LuAG/spectra/yb_luag_model_spectra_20_200C.csv`](../Yb-LuAG/spectra/yb_luag_model_spectra_20_200C.csv)
contains 0.5 nm samples from 880 to 1150 nm at 20, 80, 140, and 200 °C.
Each row gives the existing figure-guided absorption reconstruction, the
archived figure-guided emission reconstruction, emission recalculated from
the Stark-level McCumber relation, and a normalized spontaneous-fluorescence
photon density. Regenerate it with
`PYTHONPATH=src python Yb-LuAG/tools/export_model_spectra.py`.

These are **not newly digitized primary measurements**. Körner et al.'s
[published article](https://doi.org/10.1364/JOSAB.29.002493) provides the
Yb:LuAG temperature plots, but its downloadable figure files are not publicly
accessible from the publisher page in this environment. The accessible
ResearchGate graph link did not yield a usable source image. The existing
archive lacks pixel coordinates and trace uncertainty, so a defensible
independent retrace cannot be claimed. The CSV makes the current reconstruction
auditable and replaceable if a source PDF or high-resolution figure is supplied.

The intrinsic fluorescence shape is inferred from the emission cross section:

    p_photon(lambda) ∝ sigma_em(lambda) / lambda^4.

This follows the Einstein A/B relation when host-index dispersion is neglected.
It is normalized over the available 880–1150 nm interval. The resulting
energy-equivalent fluorescence wavelengths `hc/<E_photon>` are approximately
1012.60, 1005.62, 1000.96, and 997.26 nm at 20, 80, 140, and 200 °C.
They are derived values, not measured means. The 10 at.% intrinsic luminescence
quantum efficiency of about 0.90 comes from
[Brenier et al.](https://doi.org/10.1364/JOSAB.23.000676); its sample and
lifetime method differ from Beil's 0.965 ms pinhole measurement.
Radiation trapping and escape depend on disk geometry and coating spectra, so
the **effective escaping yield remains an input**. A 0.90 value in example
tests assumes all radiative photons escape and therefore gives optimistic
fluorescence removal and a lower heat estimate.

`propagate_cw` and `periodic_pulse_heat` now compute the energy-equivalent
fluorescence wavelength automatically when the effective escaping yield is
provided without a wavelength. An explicitly measured mean may override it.

## Coating design screen

[`config/ybluag_10at_coatings.json`](../config/ybluag_10at_coatings.json)
sets front dual-band AR and rear dual-band HR **design targets** around 938 nm
pumping and 1030 nm lasing. This layout is the one used by
[Beil et al.](https://doi.org/10.1364/OE.18.020712), with 24 pump passes.
Their 10 at.% roughly 220 µm disk had its best measured 40 W/1.2 mm-pump-spot
slope efficiency at 2.2% output coupling; 1.5–3.5% performed well. Front
reflectance targets of at most 0.2% in each band and rear reflectance targets
of at least 99.9% at pump and 99.95% at laser are engineering targets, **not
measurements or a multilayer coating recipe**. Actual angle, polarization,
absorptance, damage threshold, phase, and group-delay dispersion require vendor
specifications and measurement.

`scan_output_coupler` evaluates an explicit simplified CW two-direction gain
screen with the existing Yb rate model and coating losses. For the example
150 µm disk, 40 W over a 0.6 mm-radius top-hat pump spot, and listed candidate
transmissions, it selects 1.5%. The empirical 2.2% value remains the stronger
reference for its tested 220 µm hardware. The scan assumes one effective pump
intensity and equal counterpropagating signal intensities; it does not include
the 24-pass pump recycler, a spatial cavity mode, thermal feedback or measured
coating loss. It cannot establish the final device optimum.

## Generic copper cooler

[`config/ybluag_10at_assembly.json`](../config/ybluag_10at_assembly.json)
uses a water-cooled C10100 oxygen-free copper plate. The
[KME manufacturer sheet](https://www.kme.com/fileadmin/DOWNLOADCENTER/COPPER%20DIVISON/4%20Industrial%20Rolled/3_Datasheets/Datasheets_NEW_2021/Cu-OFE_01_2021_e.pdf)
gives at room temperature `k=394 W/(m K)`, `rho=8930 kg/m3`,
`cp=390 J/(kg K)`, `E=130 GPa`, and mean expansion
`17.7e-6/K` over 20–300 °C. The plate's Poisson ratio is still a generic
copper proxy.

The chosen disk/plate thermal contact conductance is `15,000 W/(m2 K)`,
within the [10,000–20,000 W/(m2 K) guidance](https://medsi.lbl.gov/engineering-design-guide/heat-transfer)
for a clamped indium-foil crystal/copper interface. That guidance is for
silicon/copper, so this is a **generic starting value**, not a Yb:LuAG
measurement. The water-side `10,000 W/(m2 K)` is also a design assumption
requiring a specified channel geometry and flow before quantitative prediction.
The bonded elastic interface remains an approximation to a clamped foil.
