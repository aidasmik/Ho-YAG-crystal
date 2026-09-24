# Yb:LuAG simulation material package

## Local browser calculator

On Windows, from the repository root in Command Prompt:

    cd /d "F:\BAKALAUSKARAS\YbYag Studeis\Ho-YAG-crystal"
    "C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe" examples\ybluag_app.py

Open <http://127.0.0.1:8781/>. This is a separate Yb:LuAG app from the
Ho:YAG structured-beam calculator on port 8780. It shows a fixed 20 C,
10 at.% CW single-pass calculation, reconstructed model cross sections, and
an approximate output-coupler design screen. It does not represent a coupled
thermal/resonator prediction or a Yb:YAG material model.

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
