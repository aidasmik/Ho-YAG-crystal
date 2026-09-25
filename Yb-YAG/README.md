# Yb:YAG material dataset

Standalone material inputs for laser/amplifier simulation. **This is a sourced development dataset, not a fully calibrated experimental digital twin or validated neural-network training dataset.** It does not modify the existing Ho:YAG or Yb:LuAG simulator.

## Coverage and evidence

| Quantity | Packaged coverage | Evidence and limitation |
|---|---|---|
| Absorption and emission cross sections | 905-1095 nm, 191 points, 293.15 K | Exact numerical arrays from HASEonGPU. Original experimental provenance, sample doping and uncertainty are not supplied upstream. |
| Temperature-dependent laser-band spectra | 1020-1060 nm; 20, 80, 140, 200 C | 21 coarse manual readings per emission curve from the Korner2012 figure. Absorption is derived by same-temperature McCumber reciprocity. Explicit opt-in required. |
| Temperature-dependent pump spectra | **Not packaged above 293.15 K** | The software refuses to substitute a room-temperature pump spectrum at another temperature. Original numerical arrays remain needed. |
| Thermal conductivity versus temperature and Yb doping | Aggarwal2005: 0, 2, 4, 15 at.%, approximately 100-300 K; Cini2017 fits: CT 80-300 K and HT 300-475 K | Tabulated measurements/derived conductivities and published empirical fits. CT and HT are separate source families, not automatically stitched. |
| Host thermal properties | Sato2025 tables: 100-500 K for Cp, expansion and apparent dn/dT; 160-500 K for conductivity/diffusivity | Undoped-host values, not concentration-specific Yb:YAG measurements. Conductivity fits extend to 773 K. |
| Host optical dispersion | 400-5000 nm | Zelmon1998 room-temperature Sellmeier fit. No doping-dependent dispersion or KK-consistent resonant dielectric model is claimed. |
| Elasticity and photoelasticity | Cubic host room-temperature elastic tensor; legacy complete photoelastic set and conflicting incomplete alternative | Host proxies. No temperature/doping-dependent tensor calibration. Missing p44 in the alternative set stays missing. |
| Stark levels, lifetime, ion density, gain and saturation | Source scalar anchors and explicit formulas | Lifetime is not a calibrated tau(T,c) law. Scaling ion density does not create concentration-resolved cross sections. |

Source identities, locations and qualifications are in `provenance/references.json`. All missing quantities and integration risks are listed in `gaps_and_assumptions.csv`.

## Quick start

From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/verify_manifest.py
python tools/export_dataset.py
```

The exporter needs no network access. It writes **ordinary CSV and compressed NPZ files** to `generated/`, including source-temperature spectra in SI units, absorption coefficients for 2/5/10/12/15 at.% and sampled host optical/thermal curves. There are no split binary archives. Generated files are reproducible outputs and are not required to use the module.

```python
from models import yb_yag as yag

# Exact upstream RT tabulation: cm^2 per Yb ion.
sigma_a, sigma_e = yag.cross_sections_cm2(1030.0, temperature_K=293.15)
alpha = yag.absorption_coefficient_m1(940.0, yb_at_percent=5.0)
gain = yag.gain_coefficient_m1(1030.0, yb_at_percent=5.0,
                              excited_fraction=0.20)
n_host = yag.n_yag(1030.0)

# Published high-temperature conductivity fit at a tabulated doping.
k_doped = yag.thermal_conductivity_doped(373.15, 5.0, family='HT')

# Optional, COARSE laser-band sensitivity input; not measured raw arrays.
sa_est, se_est = yag.cross_sections_cm2(
    1030.0, 373.15, dataset='laser_band_figure', allow_approximate=True)
```

A request for `cross_sections_cm2(940, 373.15)` raises an error because the required temperature-dependent pump data are missing. It does not extrapolate, clamp temperature, invent peaks or fall back to RT data. Non-tabulated concentrations in the conductivity fits likewise require explicit `interpolate_doping=True`; that option interpolates thermal resistivity and is an engineering assumption.

## Units and definitions

Temperatures in the API are kelvin. Wavelengths are nm. Spectral source files use cm^2 per Yb ion; transport helpers use m^2, m^-1, W m^-1 K^-1, J kg^-1 K^-1 and Pa. The source thermal tables retain their explicitly named units.

Yb at.% means substitution on the **Y sublattice**, not the fraction of all atoms. The number-density baseline is

```
N_Yb = (3 * rho_YAG / M_YAG) * N_A * Yb_at_percent / 100
```

It uses the host volume at 300 K. This is not a measured concentration-dependent lattice/density relation. RT doped densities and heat capacities at 0/2/4/15 at.% are supplied separately by `doped_RT_density_heat_capacity()`.

Net small-signal gain is `N * [beta*sigma_e - (1-beta)*sigma_a]`. The effective two-manifold saturation fluence is `h*nu/(sigma_a+sigma_e)`, not the emission-only expression. Saturation intensity additionally depends on the chosen lifetime. These quantities do not by themselves specify extraction efficiency, ASE losses, or total heat load.

## Thermal and stress-optic cautions

`thermal_conductivity_host()` and `heat_capacity_host_J_kgK()` describe the YAG **host**. Do not silently use undoped-host conductivity for highly doped material. The Cini CT and HT families can disagree near their boundary because they fit different samples/data; choose a family rather than inventing continuity.

Sato2025 explicitly notes that its apparent dn/dT may include mounting-stress photoelasticity. Access requires `allow_apparent=True`. It is not automatically combined with a separate photoelastic calculation. The older Aggarwal thermo-optic baseline is restricted to 100-300 K. Both dn/dT sources are near 1064 nm; use at 1030 nm requires `allow_wavelength_proxy=True`.

For a stress-free thermo-optic index, apply photoelasticity to **mechanical elastic strain = total strain - thermal eigenstrain**, not to total strain. Otherwise free thermal dilation is counted twice. `photoelastic_delta_B()` accepts mechanical strain in [100]/[010]/[001] axes and uses

```
Delta B_ii = p11*e_ii + p12*(trace(e)-e_ii)
Delta B_ij = 2*p44*e_ij, i != j
```

`B` is the inverse relative dielectric tensor. The tensor choice and coordinate rotation must remain explicit. The alternative Johnson/Olson set has no verified p44 here and is rejected as incomplete.

## Data quality and validation

The four original HASEonGPU text files are preserved in `spectra/haseongpu_original/`; their Git blob hashes are checked by the tests. The merged RT CSV reproduces their values exactly. They are **not** relabeled as Korner2012 raw measurements.

The optional temperature curves are deliberately separate: manual readings from a reproduced authors' figure, rounded values, limited laser-band coverage, no calibrated confidence intervals. The digitization record specifies the viewed page, axis calibration and suggested sensitivity perturbations. Dense interpolation does not create additional experimental resolution.

`QA_REPORT.json` records a local software test and export round-trip. Tests cover source hashes, units, broadcasting, domain failures, gain/transparency, saturation, thermal anchors, elastic symmetries, tensor rotations and photoelastic conventions. **Passing tests does not establish agreement with a real laser.** Before generating quantitative training data, replace the approximate spectra and missing high-T pump data, calibrate the sample/mount and compare gain, temperature and phase maps against measurements.

## Layout

- `material.json`: machine-readable definitions, fit coefficients and domain limits.
- `spectra/`: source RT arrays, merged CSV, coarse temperature nodes, Stark/lifetime anchors.
- `thermal/`: host and doped tables plus separate conductivity-fit families.
- `mechanical/`: elastic and photoelastic tensors with conventions and competing sets.
- `models/`: standalone NumPy material helpers.
- `examples/`: Korner2012 illustrative thin-disk configuration, not an experimental benchmark.
- `provenance/`: source references, digitization record, original-data notice and license.
- `tests/`, `tools/`: bounded validation, manifest verification and export.

No article PDFs, hidden credentials, existing simulator changes, automatic training, or remote compute are included. The upstream HASEonGPU data retain their license; see `provenance/NOTICE.md` and `provenance/COPYING.GPL-3`.
