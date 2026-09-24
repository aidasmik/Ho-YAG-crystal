# Yb:LuAG simulation material package

For issue-by-issue audit status, reproduced numerical comparisons and missing
measurements, see [Yb:LuAG audit fixes](../docs/YBLUAG_AUDIT_FIXES.md).

## Native Tkinter simulator

From the repository root, run `python examples/ybluag_desktop.py` using a
Python environment with Tk, Matplotlib, Pillow and the project dependencies.
The desktop window exposes the CW, structured CW, and proposal pulse solvers,
including regenerative cavity controls. It restores the latest saved Yb result
and displays beam maps with a horizontal center cut above and a vertical
center cut beside each image, phase and synthetic dopant maps, gain curves,
cooler timeline, and thermal surfaces. The **Beam on crystal** tab overlays
calculated disk-entrance 1/e² and 50%-of-peak beam contours on the Yb entrance
concentration, with full-disk and footprint-detail views. The toolbar on every
plot tab provides zoom, pan, and reset controls. "Added phase for selected beam"
is the shaping phase generated from the selected target; optional correction
phase is shown separately.
The five-point pump curve is calculated on demand from the displayed pulse
configuration. Results and logs are saved in `results/desktop_runs/` under the
shared bounded-run ledger. The physics scope and assumptions below apply to
the desktop simulator as well.

## Local browser calculator

On Windows, from the repository root in Command Prompt:

    cd /d "F:\BAKALAUSKARAS\YbYag Studeis\Ho-YAG-crystal"
    "C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe" examples\ybluag_app.py

In PowerShell use `cd 'F:\BAKALAUSKARAS\YbYag Studeis\Ho-YAG-crystal'` and
`& 'C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe' .\examples\ybluag_app.py`.

Open <http://127.0.0.1:8781/>. This is a separate Yb:LuAG app from the
Ho:YAG structured-beam calculator on port 8780. It provides the same six
structured target shapes, phase-only SLM masks, synthetic clustered dopant maps,
irradiance/phase/side-profile views, and output-plane diffraction. Three CW
options are available: pump-only weak probe, a fixed Gaussian cavity-mode
background with separate weak probes, and a signal-saturated single pass.
The proposal pulse section defaults to 12 at.% Yb:LuAG, a 100 µm disk, a
2 mm Gaussian pump diameter, ten alternating pump traversals, a 10 nJ seed,
and 10 kHz repetition. The femtosecond laser seed is stretched before the
disk: 300 fs at the source and 10 ps at the amplifier are editable assumptions,
since the proposal only specifies a picosecond seed there. Forty watts incident
pump, ten ideal relayed signal traversals, 938/1030 nm centers, and a 0.6 mm
signal waist are also modeling assumptions. The 12 at.% lifetime (0.973 ms)
is interpolated from the 10 and 15 at.% pinhole data; it is not a measured
12 at.% value. The exact inputs and their status are in
`config/ybslam_proposal_luag.json`.

The source seed is always a Gaussian TEM00. The selected target determines a
phase-only mask at an assumed SLM/phase-plate plane; an optional aberration or
correction is added to that mask. The field propagates over an editable
SLM-to-disk distance (default 0.25 m) before it enters the amplifier. Spiral
phases make the two vortex targets, a quadrant phase makes the HG-like target,
a converging axicon makes the finite-aperture Bessel-like needle, and a
48-iteration scalar alternating-projection hologram makes the approximate
flat-top target. A phase-only mask does not change intensity in its own plane;
the disk-input and outgoing views show the structure formed by propagation.
These targets are approximate spatial fields, not pure-mode guarantees. The
coherent source is propagated at the 1030 nm center wavelength; chromatic SLM
response and broadband shaping of the femtosecond seed remain unmodeled.
The flat-top design follows the phase-only Gaussian reshaping approach of
[Gerchberg–Saxton holograms](https://www.sciencedirect.com/science/article/pii/S0030401818308654);
the needle uses the established [Gaussian-plus-axicon mechanism](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-22-11-2542).

The pulse solver evolves the two-manifold population over a periodic seed
train, supports ideal relayed signal traversals, and reports a cycle-average
first-law heat estimate using a fixed multipass pump profile and declared
fluorescence escape yield. It applies the 1030 nm center cross sections to
the pulse energy. It does not propagate femtosecond spectral bandwidth,
chirp, gain narrowing, dispersion, or nonlinear phase, so its pulse energy
and phase cannot verify the proposal's >100 µJ, >10,000 gain, or pattern
fidelity targets. The uniform-dopant 40 W run gives about 39 nJ at the disk
exit, well below the energy target; the UI's synthetic nonuniform map changes
that result. The UI keeps the existing 10 at.% CW
comparison separate from this proposal pulse setup. The enlarged pulse
figures show the Gaussian source, shaped disk input, and outgoing transverse
fluence with horizontal and vertical center cuts. Dashed output cuts come
from a separate uniform-dopant, isothermal solve with the same pump, Gaussian
source, and phase mask. The pulsed UI defaults to a
synthetic 0.27 cluster contrast to expose that comparison. The temperature
timeline advances one disk, contact and finite copper plate from a uniform
20 °C start for the selected operating duration (30 s by default), then
continues toward a constant-heat steady solution. A bounded water-side
conductance controller can increase an assumed coolant-flow proxy from
10 to at most 100 kW/m²K when disk temperature exceeds a selected target;
fixed conductance is also available. The controller slope (3 kW/m²K per K),
instantaneous response, and constant-temperature coolant bath are assumptions,
not a measured cooling-system design. Short pulses at 10 kHz deposit average
heat continuously, while the finite coolant conductance removes increasing
power as the plate warms, so this model approaches a steady temperature.
Each sampled temperature drives a bonded elastic solve and round-trip OPD.
Thermal integration has a separate configurable maximum internal timestep.
Five synthetic crystal/plate probes sample the same field with response time,
sampling, latency, bias, noise and missing status. Cooling can use delivered
disk-probe measurements, the exact disk maximum as an ideal full-state
benchmark, or fixed conductance. The first
pump/population startup interval is approximated. Cold, lumped
post-extraction phase, and coupled steady regenerative modes are explicit
choices. Coupled steady mode iterates local temperature-dependent gain, heat,
temperature, deformation and a scalar phase at each disk encounter, stopping
outside the 293.15–300 K assembly property range. It does not resolve
transient optical/population feedback. The output phase residual is the wrapped,
fluence-weighted piston-removed difference from an otherwise identical
uniform-Yb, isothermal run. Photoelastic birefringence, measured
concentration-dependent index and a calibrated actual relay geometry are
omitted.
The app simulates one selected pulse shape at a time.
Every CW and pulsed shape can overlay a dashed uniform-Yb output profile
computed with the same pump and seed settings. Yb concentration maps use a
dark-blue/teal/yellow scale. Thermal heat interpolation is normalized to
preserve integrated deposited power. For a selected heat load above the
measured 293.15–300 K material range, the copper-cooler view reports the
out-of-range constant-property temperature and OPD timeline only as an
explicitly dashed extrapolation, not an operational prediction. It
then shows front/rear disk displacement and optical-path maps for the same
heat pattern scaled to the proposal's 5 K design rise. Those maps are a
separate in-range design reference, not a deformation prediction for the
selected pump case. The generic C10100 copper plate, indium-contact and
coolant conductances remain assumed hardware parameters in
`config/ybluag_10at_assembly.json`.
The 26.85 °C cutoff is a parameter-calibration limit, not crystal failure.
Published [Yb:LuAG absorption/emission spectra](https://opg.optica.org/josab/abstract.cfm?uri=josab-29-9-2493) reach 200 °C, while the current
high-doping conductivity and thermo-optic/elastic inputs do not support an
accurate 250 °C coupled calculation. The UI marks 250 °C as a user-supplied
crystal reference. The modeled indium interface would also cease to be solid
near [156.6 °C](https://www.nist.gov/publications/standard-reference-material-1745-indium-freezing-point-standard-and-standard-0); a 250 °C assembly requires a different bond/contact design.
The separate CW comparison retains the reconstructed cross-section plot and approximate
output-coupler design screen. All spectra are figure-guided reconstructions,
and cavity geometry and cooling boundary values are assumptions, not a
validated device prediction. This is a Yb:LuAG, not a Yb:YAG, material model.

Physics-based Yb:LuAG material data for thin-disk laser, amplifier, resonator, thermal, elastic-deformation, and phase-propagation simulations.

## Included

- Temperature-dependent Yb absorption and emission cross sections:
  - 880-1150 nm
  - 0.5 nm spacing
  - 20, 80, 140, and 200 C
- Yb concentration to ion-number-density conversion.
- Stark levels, lifetime and quantum-efficiency data.
- LuAG refractive-index dispersion equation over 193-1690 nm.
- Thermal conductivity, diffusivity, thermal expansion, dn/dT, and optical-path-temperature data.
- Debye-based heat-capacity engineering model normalized to the measured room-temperature volumetric heat capacity.
- Cubic LuAG elastic constants C11, C12, C44 and engineering moduli.
- A published Yb:LuAG thin-disk reference case.
- Python interpolation/model helpers and a table-export utility.
- Source references and an explicit gaps/assumptions table.

## Spectral storage

The compact spectral NPZ was split into five binary parts only to fit the GitHub connector's upload size limits:

    spectra/yb_luag_cross_sections_20_200C_reconstructed.npz.part0
    ...
    spectra/yb_luag_cross_sections_20_200C_reconstructed.npz.part4

models/yb_luag_model.py concatenates these parts in memory automatically. tools/export_spectra.py can reconstruct a normal NPZ and export human-readable CSV tables.

The stored arrays contain every one of the 541 wavelength samples at each of the four temperatures. Arrays are float32; this changes only numerical storage precision, not the wavelength sampling or curve structure used by the simulation.

`spectra/yb_luag_model_spectra_20_200C.csv` exposes the current reconstructed
absorption/emission arrays, the McCumber-consistent emission used by `ybluag`,
and its inferred normalized fluorescence photon spectrum. Regenerate it with
`PYTHONPATH=src python Yb-LuAG/tools/export_model_spectra.py`. This is a model
export from the existing reconstruction, not an independently retraced figure.

## Critical spectroscopy warning

The temperature-dependent absorption/emission curves are a figure-guided engineering reconstruction of Körner et al. (2012), not raw numerical arrays supplied by the authors. They are appropriate for model development, sensitivity studies, and integration. For publication-grade quantitative spectroscopy, replace them with raw measured arrays if those become available.

## Main relationships

Yb ion number density:

    N_Yb = 1.42e22 * (Yb at.% / 100)  [cm^-3]

Absorption coefficient:

    alpha(lambda,T) = N_Yb * sigma_a(lambda,T)

Small-signal net gain for excited-state fraction beta:

    g = N_Yb * [beta*sigma_e - (1-beta)*sigma_a]

Quantum-defect heat fraction:

    eta_q = 1 - lambda_p/lambda_l

LuAG host dispersion, lambda in micrometres:

    n^2 = 2.077 + 1.237*lambda^2/(lambda^2 - 0.1376^2) - 0.0104*lambda^2

## Usage

Install:

    pip install -r requirements.txt

Example:

    from models.yb_luag_model import sigma_abs_cm2, sigma_em_cm2, n_luag

    sa = sigma_abs_cm2(940.0, temperature_C=20)
    se = sigma_em_cm2(1030.0, temperature_C=20)
    n = n_luag(1030.0)

To reconstruct normal files:

    python tools/export_spectra.py

This creates the combined NPZ plus CSV tables for spectra, absorption coefficient, host refractive index, and heat capacity.

## Known model gap

A trustworthy LuAG-specific full photoelastic tensor p11, p12, p44 was not established. Do not silently substitute YAG values. Scalar thermo-optic phase distortion and elastic disk deformation are parameterized; rigorous stress-induced birefringence/depolarization remains underdetermined.

See references.csv and gaps_and_assumptions.csv for provenance and limitations.
