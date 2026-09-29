# Yb:YAG data and thin-disk simulator

This guide covers the **Yb:YAG** material package and the Yb:YAG paths in the
[Ho-YAG-crystal repository](https://github.com/aidasmik/Ho-YAG-crystal). It
explains the source spectra, the physics the code actually solves, how to
reproduce example runs, and what remains unvalidated. The Yb:YAG solver reuses
generic two-manifold Yb and thin-disk kernels but supplies its own YAG
spectroscopy, density, index and thermal property choices. Ho:YAG's four-manifold
laser and Yb:LuAG's material spectra are separate models.

**Status:** this is a development and sensitivity model. Neither the material
table nor the simulated output is a calibrated prediction for a particular
crystal, coating, mount or laser. The example figures below were regenerated at
repository revision `6c384fb` on 29 September 2026. Their settings and numerical
outputs are in [simulation_summary.json](readme_figures/simulation_summary.json).

## Reproduce the data and figures

From the repository root, with Python 3.10+:

```powershell
python -m pip install -e '.[plots,dev]'
python Yb-YAG/tools/verify_manifest.py
python Yb-YAG/tools/export_dataset.py --output Yb-YAG/readme_figures/exported_data
python examples/run_ybyag_readme_supervised.py
python -m pytest Yb-YAG/tests tests/test_ybyag_simulation.py -q
```

The source files are already in `Yb-YAG/` and byte-identical runtime copies
are in `src/ybyag/data/`. The first command installs the project; the manifest
command checks all **28** source-file hashes. The exporter makes SI-unit CSV/NPZ
files, without acquiring new measurements. The figure runner uses
`hoyag.local_supervisor`, the shared persistent budget ledger, a 180 s run
limit and a memory cap. It writes figures, a JSON summary, and execution
records to `Yb-YAG/readme_figures/`. The saved run completed in **5.26 s**
with **211 MB** peak process RSS. These numbers are machine-specific.

For the native calculator, run `python examples/ybyag_desktop.py` from the
repository root with a Python installation that includes Tkinter. The Yb:YAG
tab offers ideal multipass, regenerative and CW calculations and shows the
available cold and near-room-temperature lumped-phase modes. Inputs in the UI
are engineering choices; the saved result is not a fresh calculation. See
[native simulation details](../docs/YBYAG_SIMULATION.md),
[closed-loop control](../docs/YBYAG_CLOSED_LOOP.md) and
[the NN dataset workflow](../docs/YBYAG_NN_DATASET.md).

## What data are used now?

The **default numerical amplifier** reads the 191-point, 1 nm spaced,
293.15 K absorption and emission cross sections from
[`yb_yag_293K_legacy.csv`](spectra/yb_yag_293K_legacy.csv). These are exact
arrays from [HASEonGPU](https://github.com/ComputationalRadiationPhysics/haseongpu/tree/5af022b40635b6cf22178252e243467d3d7aed90/material_library/data/legacy_yb_yag),
but their original experimental sample, doping, errors and processing history
are unspecified. Exact copying does not make them a calibrated measurement for
the proposed disk. Cross sections are per ion in cm² in the source CSV; the
runtime converts to m².

![The default Yb:YAG absorption and emission arrays and the excited fraction required for transparency](readme_figures/runtime_spectra.png)

The 969 nm pump and 1030 nm signal are *sample points on these same curves*.
The lower panel shows the local small-signal transparency threshold
`beta_tr = sigma_a/(sigma_a + sigma_e)`; it is not a measured excited-state
population. Narrow-band pump performance depends strongly on actual diode
linewidth and temperature-dependent zero-phonon-line shape, neither of which
the current amplifier integrates.

Additional source families are available for **comparison or explicit opt-in**:

| Source | Coverage and use | Why it is separate |
|---|---|---|
| [Tang et al. 2014](../research/ybyag/tang2014/README.md) | Figure-read room-temperature absorption coefficients for 5, 10 and 15 at.% ceramics, 935–1040 nm | Sample-level `alpha`, not an independent per-ion cross-section table; not added to the default loss. |
| [De Vido et al. 2020](../research/ybyag/devido2020/README.md) | Original 80–300 K scans and quality-flagged 969 nm reconstruction, 1.1 at.% ceramic | Strong ZPL comparison; different concentration and limited spectral span. |
| [Körner et al. 2012](spectra/korner2012_laser_band_manual.csv) | Coarse manual laser-band readings, 1020–1060 nm at 20/80/140/200 °C | Emission is figure-read and absorption is derived by same-temperature McCumber reciprocity; no hot pump band. |
| [Esmaeilzadeh et al. 2012](../research/ybyag/high_temp_25at/README.md) | 300/450 K figure readouts for a 25 at.% crystal | Guarded exploratory 25 at.% lookup; not a 5–20 at.% hot calibration. |

![Optional figure-derived temperature spectra, kept separate from the default simulator input](readme_figures/optional_temperature_spectra.png)

The left panel has only the 1020–1060 nm emission band, so it cannot supply
the hot 969 nm pump coefficient. The right panel comes from a different,
25 at.% specimen. Neither plot is substituted for the default 5–20 at.%
room-temperature table in the amplifier examples.

![Tang ceramic absorption readouts compared with the default cross-section model scaled by Yb density](readme_figures/rt_concentration_comparison.png)

This comparison is a useful scale and shape check. The dashed curves are
`N_Yb sigma_a` from the default table, while the solid curves include the
particular Tang ceramics and figure-reading uncertainty. Their differences
should not be fitted away by silently adding another loss term.

## Physical model, from source beam to output

The simulation is an envelope and population model, not an optical-cycle
Maxwell solver. In the pulsed path, it follows one physical Yb:YAG disk through
repeated pump and signal encounters. `YbGallerySettings` defines the Gaussian
seed, SLM phase mask, disk, pump, relay and numerical grids;
`YbYAGMaterial` dispatches YAG properties. `simulate_pulsed_seed` then builds
the source field, propagates to the disk, converges the periodic population,
passes the seed through the amplifier and computes cycle-average heat. A
separate finite disk/copper model can produce temperature, displacement and a
scalar optical path screen.

The Yb concentration is atomic percent on the **Y sites**. The runtime uses a
fixed host-volume conversion

```text
N_Yb = (3 rho_YAG / M_YAG) N_A (Yb_at_percent / 100).
```

This is a density scaling, not a concentration-dependent spectral model. For
ground and upper-manifold fractions `1-beta` and `beta`, respectively, the
local intensity coefficients are

```text
alpha_p = N [(1-beta) sigma_a(lambda_p) - beta sigma_e(lambda_p)]
g_s     = N [beta sigma_e(lambda_s) - (1-beta) sigma_a(lambda_s)].
```

Thus `dI_p/dz = -alpha_p I_p` and `dI_s/dz = g_s I_s` in a frozen cell.
Each pump pass is propagated through the longitudinal cells, with alternating
direction and explicit relay retention. The cell-average intensity is the
exact Beer–Lambert spatial mean for its frozen coefficient, which makes
absorbed photons consistent with the cell-face flux difference. A spatially
resolved continuous-wave pump drives an effective two-manifold rate equation:

```text
d beta/dt = (1-beta) W_up - beta W_down - beta/tau,
W_up   = sigma_a,p I_p/(h nu_p) + sigma_a,s I_s/(h nu_s),
W_down = sigma_e,p I_p/(h nu_p) + sigma_e,s I_s/(h nu_s).
```

The source calls the signal part zero during inter-pulse recovery. Recovery
recomputes pump bleaching over eight substeps and iterates successive seed
cycles to a periodic pre-pulse inversion. The short seed is transported on a
retarded-time intensity grid; pump evolution during the picosecond seed window
is neglected. Gain and reabsorption update the same population after each
disk traversal. In the **ideal multipass** architecture, a 1:1 phase-preserving
relay returns the field for up to ten signal traversals. `pump_passes` and
`signal_traversals` count different paths; their values need not match.

The **regenerative** option follows a different optical map. Each round trip
visits the same disk twice, propagates a complex field to a curved mirror and
back with an angular-spectrum FFT, and applies the finite aperture, HR,
hold, injection and extraction losses. Its short-pulse cell extraction uses
the two-manifold Frantz–Nodvik relation

```text
F_s   = h nu_s/(sigma_a,s + sigma_e,s)
g_0   = N [beta sigma_e,s - (1-beta) sigma_a,s] Delta z
F_out = F_s ln{1 + exp(g_0)[exp(F_in/F_s) - 1]}.
```

The implementation evaluates this stably at large fluence, then changes
`beta` from the photon transfer. This branch resolves fluence and round trips;
it does **not** return a computed output pulse temporal shape. The ideal
multipass branch does return a time-sampled intensity trace. A shaped SLM
field changes intensity only after propagation or filtering when the mask is
phase-only.

![Calculated disk fluence, pulse power and per-traversal energy for the cold 20 at.% example](readme_figures/cold_amplifier_example.png)

In this **cold** example, 40 W of 969 nm continuous-wave pump illuminates a
100 µm disk through ten pump passes. A 10 nJ, 10 ps seed at 1030 nm traverses
the 20 at.% disk ten times. The 64×64, one-axial-cell calculation returns
**30.04 nJ** at the output plane. The pulse-power curve is a calculated
retarded-time intensity trace, not optical carrier oscillations or a chirp
prediction. The energy curve shows repeated use of the same population, not
ten independent amplifiers.

![Cold amplifier output with 5 and 20 at.% Yb using otherwise identical coarse settings](readme_figures/doping_comparison.png)

At 5 at.% the same setup returns **14.01 nJ**. At 20 at.% it returns
**30.04 nJ**. These comparisons hold the per-ion spectrum fixed as
concentration changes; they do not include concentration quenching, hot
gain, damage, ASE or a calibrated relay. More doping cannot be interpreted
as a proven device improvement from this plot.

## Heat, cooler and phase

The code forms a **cycle-average lattice heat ledger** from the pump photons
absorbed in the disk, energy transferred to or absorbed from the signal,
escaping fluorescence and change in stored excitation:

```text
P_heat = P_pump,absorbed - P_signal,gain
         - P_fluorescence,escaped - dU_excitation/dt.
```

`fluorescence_escape_yield` is an effective assumption; the README examples
use zero. At the cold 20 at.%, 40 W operating point the code reports about
**27.87 W pump absorption** and **30.04 nJ output energy**. It does not run a
valid hot-gain calculation at that heat load. Pump absorption is an optical
ledger item, not proof that the generic mount can remove that power.

For the thermal calculation, the finite disk and copper plate solve a heat
diffusion/conduction problem of the form
`rho C_p dT/dt = div(k grad T) + Q`, with finite disk/contact and
plate/coolant conductances. The mechanical model solves linear thermoelastic
equilibrium `div(sigma)=0` with thermal eigenstrain. The scalar round-trip
optical path combines temperature-dependent index and front/rear surface
movement. In `lumped_phase` mode that phase is applied **after** optical
amplification; gain still uses 293.15 K spectra. The code does not apply the
available cubic photoelastic tensor in this scalar Yb:YAG amplifier.

![Steady thermal and scalar-distortion output for the 0.1 W example](readme_figures/near_rt_thermal_example.png)

This separate **0.1 W**, 20 at.% case gives about **0.0827 W** absorbed pump,
a **23.08 °C** maximum crystal cell temperature, and a calculated scalar OPD
map for the generic cooler. At this very low pump the output is **7.28 nJ**:
the seed is partly reabsorbed. This demonstrates that a gain coefficient can
be negative below transparency. The thermal example uses the generic 20 °C
coolant boundary and near-room-temperature property proxies; it is not the
steady temperature of the 40 W case.

## What the current model cannot establish

1. **Hot pump and gain:** the required concentration-resolved pump spectra
   above 293.15 K are absent. `YbYAGMaterial` rejects hot cross-section
   queries and `coupled_steady` Yb:YAG runs. The allowed near-room-temperature
   lumped phase does not feed temperature back into gain.
2. **Sample-specific spectroscopy:** the default RT arrays lack upstream
   sample and uncertainty metadata. Scaling ion density does not account for
   concentration-dependent cross sections, lifetime, defects or quenching.
   The 969 nm pump is modeled monochromatically; diode linewidth and ZPL
   temperature shift can change absorption materially.
3. **Thermal and optical hardware:** doped conductivity is selected from
   published fits or concentration interpolation; host heat capacity,
   expansion, index and thermo-optic values are proxies at most dopings.
   Contact, copper cooler and coating curves need measurements of the real
   assembly. The stated 293.15–300 K thermo-mechanical range is a model
   validity window, not a crystal survival limit.
4. **Missing device physics:** vector photoelasticity and depolarization,
   ASE and parasitic lasing, nonlinear phase, damage, finite switching,
   pulse chirp/compression, spectral gain narrowing and coherent diffraction
   within each picosecond disk pass are not solved. The ideal relay geometry
   is a boundary condition, not a ray-traced multipass layout.
5. **Numerical qualification:** the README runs use 64² transverse pixels,
   one optical axial slice and a coarse thermal mesh. Photon balance checks
   and source hashes are useful software evidence, but a mesh/time-step
   convergence study and experimental comparison are required before using
   output energy, temperature or phase maps for design limits or NN ground
   truth. The newer controller reports offline modal errors; see its
   [explicit qualification limits](../docs/YBYAG_MODAL_MODEL.md) and
   [V2 status](../docs/YBYAG_CONTROLLER_V2.md).

The most direct next measurements are wavelength-resolved pump and emission
spectra versus temperature **and concentration**, lifetime for the selected
sample, absorbed pump power, disk and cooler temperature maps, calibrated
surface/phase distortion, coating loss and measured output versus pump. See
the [missing-data inventory](../docs/YBYAG_MISSING_DATA_SEARCH.md).

## Code and output map

| Path | Role |
|---|---|
| `Yb-YAG/spectra/`, `thermal/`, `mechanical/` | Attributed material inputs. |
| `src/ybyag/material_data.py`, `src/ybyag/model.py`, `src/ybyag/assembly.py` | YAG-specific lookups, runtime material class and near-RT assembly properties. |
| `src/ybluag/multipass_pump.py`, `pulsed.py`, `regenerative.py`, `gallery.py` | Shared generic Yb pump, population, pulse, cavity and orchestration kernels. Several source docstrings retain LuAG wording; material dispatch selects YAG values. |
| `examples/ybyag_readme_figures.py` | Exact settings and plot-generation script used above. |
| `examples/run_ybyag_readme_supervised.py` | Bounded runner and persistent run record. |
| `Yb-YAG/readme_figures/` | Generated figures, SI exports, JSON values and execution log. |

The existing [data provenance notice](provenance/NOTICE.md) and bundled
[GPL text](provenance/COPYING.GPL-3) govern redistribution of the upstream
HASEonGPU arrays and their transformations. The rest of this README describes
the standalone material API in more detail.

## Material API reference

Standalone material inputs for laser/amplifier simulation. **This is a sourced development dataset, not a fully calibrated experimental digital twin or validated neural-network training dataset.** The Yb:YAG runtime uses these data while keeping the Ho:YAG and Yb:LuAG material inputs separate.

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
