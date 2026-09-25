# Yb:YAG native simulation

Launch from the repository directory in PowerShell:

```powershell
& 'C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe' .\examples\ybyag_desktop.py
```

The Tkinter window has separate **Yb:YAG**, **Yb:LuAG** and **Ho:YAG** tabs.
Each Yb tab has independent inputs, results and pump curves. Saved results
are restored only to their matching material. No HTTP server is started.

## Repository data

The data were fetched from
[`Yb-YAG/` at revision 7aa99048aa79f0f00a9a7f4efd21de50c19a6e00](https://github.com/aidasmik/Ho-YAG-crystal/tree/7aa99048aa79f0f00a9a7f4efd21de50c19a6e00/Yb-YAG).
All 28 source checksums in its manifest were verified. The required runtime
data in `src/ybyag/data` are byte-identical copies; a regression checks this.
The original data notices and GPL license remain with both copies.

| Input | Simulator implementation | Limitation |
|---|---|---|
| Absorption/emission | Exact repository RT arrays, 905–1095 nm, cm² converted to m² per ion | Upstream experimental concentration and uncertainty are unspecified. These are not labeled as raw Körner measurements. |
| Concentration | Default 20 at.% on Y sites; `3*rho/M * N_A * fraction` | Fixed host-volume approximation; concentration-dependent cross sections and lattice volume are unknown. |
| Lifetime | 0.95 ms | Literature model/normalization parameter, not measured tau(T,20 at.%). |
| Index | YAG-host Zelmon Sellmeier; group index derived from its wavelength derivative | Undoped-host proxy. No LuAG index is used in the YAG cavity. |
| Conductivity | Cini2017 doped fits; explicit interpolation of thermal resistivity | At default 20 at.% use HT-family 300 K conductivity held constant in the near-RT assembly; no invented CT extrapolation. At ≤15 at.% use CT-family 293.15 K value. |
| Heat capacity/density | Sato2025 YAG-host values near RT | Explicit host proxies, not measured 20 at.% properties. |
| Elasticity | YAG cubic tensor reduced to isotropic Voigt–Reuss–Hill moduli | Crystal anisotropy and orientation are not propagated by this scalar model. |
| Expansion and dn/dT | Aggarwal host values at 293.15 K | 1064 nm dn/dT is an explicit 1030 nm proxy. No apparent high-T stress contribution is added. |
| Copper/contact | Existing finite copper plate, generic contact and coolant boundary | Same configurable assembly assumptions as the other Yb solver; needs hardware calibration. |

The published conductivity fits and their concentration dependence are
described by [Cini and Mackenzie (2017)](https://link.springer.com/article/10.1007/s00340-017-6848-y).
Using one 300 K value in the near-RT constant-property assembly is our explicit
engineering approximation; it is not a temperature-dependent thermal solve.

## Available calculations

- **Pulsed ideal multipass:** one shared disk population, time-sampled pulse
  depletion, pump recovery, relay retention, energy/photon ledgers and gain.
- **Pulsed regenerative:** the same YAG data and shared disk, Frantz–Nodvik
  extraction, diffraction, cavity mirror and aperture, discrete hold/ejection,
  and pumping between round trips and seed pulses.
- **Structured CW:** Gaussian seed, target SLM mask and propagation to the
  disk; one selected beam with clustered doping, pump absorption and gain.
- **CW material screen:** one collinear pass and an explicitly generic output
  coupler screen, with no imported LuAG experimental optimum.
- **Thermal/deformation:** finite disk/contact/copper solution, fixed-optical-heat
  startup timeline, front/rear displacement, scalar OPD, and post-amplifier
  lumped phase within the stated near-RT range.
- **Native views:** incoming/output fluence and profiles, Yb map, phase and
  uniform-cold residual, pulse gain and cold pump/output curve.

Pulsed defaults follow the user's target: 20 at.% Yb:YAG, 100 µm disk, 969 nm
pump, 1030 nm signal, ten pump passes, 10 nJ seed at 10 kHz, and 10 ps stretched
pulse. The 300 fs source duration, 10 ps stretched duration, 40 W pump,
2 mm pump diameter, 0.6 mm source radius and ideal relay layout remain
editable engineering choices; they are not measured device parameters.

## Deliberate physical limits

**Fully coupled hot YAG gain is unavailable for the proposed 5/10/15/20 at.%
disks.** The original runtime data include only RT pump spectra. A new
quality-flagged 80–300 K ZPL reconstruction belongs to a 1.1 at.% ceramic;
the newly digitized 300/450 K pump and emission endpoints belong to a 25 at.%
crystal and have a guarded opt-in lookup. Neither calibrates the proposed
concentrations at high temperature, and neither supports 250 C operation.
The coarse earlier laser-band figure readings cover 1020–1060 nm only.
Programmatic `coupled_steady` requests retain their guard; the YAG desktop
offers `cold` and `lumped_phase`.

In lumped-phase mode, optical gain/populations use 293.15 K spectra. Their
cycle-average heat drives the thermal/mechanical calculation; a scalar phase
screen is applied after amplification when the assembly is within
293.15–300 K. This is **not** hot gain feedback or a synchronized transient
optical/thermal simulation. Outside that range, hot phase is withheld and
any design-reference geometry is explicitly separate from actual operation.
The range is a model validity limit, not the crystal damage temperature.

The YAG dataset contains photoelastic tensors, but the amplifier does not yet
apply orientation-dependent vector photoelastic propagation. Tensor availability
alone does not establish depolarization predictions. ASE, nonlinear phase,
coating spectral absorption/phase, finite switch waveform, pulse compression
and measured high-temperature sample properties remain unimplemented.

Run artifacts include the request, result, execution time/memory record, and
configuration/numerical-source hashes. Material identity accompanies every
new result. The shared numerical kernels ensure later conservation fixes
apply to both Yb hosts while their material inputs remain separate.

## Verification

The YAG integration and imported material tests passed (43 tests). The shared
LuAG suite initially passed 74 tests with one spectral-dispatch failure; that
failure was fixed, and all 16 focused regression tests then passed. A bounded
native-worker smoke calculation at 0.01 W pump completed in 1.437 s, produced
YAG thermal/deformation results and source hashes, and rendered the native
beam, phase and thermal figures. This deliberately coarse 32-grid check is
not a device-performance or convergence claim. The 28 imported source-file
checksums were reverified.

See [the subsequent missing-data search](YBYAG_MISSING_DATA_SEARCH.md) for
newly obtained raw temperature-dependent ZPL measurements and hot-spectrum
sources. They are research inputs; they have not yet changed the model's
validity range.
