# Yb:LuAG simulation material package

## Local browser calculator

On Windows, from the repository root in Command Prompt:

    cd /d "F:\BAKALAUSKARAS\YbYag Studeis\Ho-YAG-crystal"
    "C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe" examples\ybluag_app.py

In PowerShell use `cd 'F:\BAKALAUSKARAS\YbYag Studeis\Ho-YAG-crystal'` and
`& 'C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe' .\examples\ybluag_app.py`.

Open <http://127.0.0.1:8781/>. This is a separate Yb:LuAG app from the
Ho:YAG structured-beam calculator on port 8780. It provides the same six
structured seed shapes, ideal phase masks, synthetic clustered dopant maps,
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

The pulse solver evolves the two-manifold population over a periodic seed
train, supports ideal relayed signal traversals, and reports a cycle-average
first-law heat estimate using a fixed multipass pump profile and declared
fluorescence escape yield. It applies the 1030 nm center cross sections to
the pulse energy. It does not propagate femtosecond spectral bandwidth,
chirp, gain narrowing, dispersion, or nonlinear phase, so its pulse energy
and phase cannot verify the proposal's >100 µJ, >10,000 gain, or pattern
fidelity targets. The proposal-default 40 W run gives about 39 nJ at the disk
exit, well below the energy target. The UI keeps the existing 10 at.% CW
comparison separate from this proposal pulse setup. A separate room-temperature copper-cooler and scalar
thermoelastic reference is calculated for CW or pulsed heat only when the disk
stays within the material data's 293.15–300 K range. It omits LuAG
photoelastic birefringence, which lacks a verified tensor.
The top section retains the reconstructed cross-section plot and approximate
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
