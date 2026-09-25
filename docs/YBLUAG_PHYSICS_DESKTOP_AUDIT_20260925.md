# Yb:LuAG physics and Tkinter audit — 25 September 2026

The model is a synthetic engineering calculation with uncalibrated material
and assembly inputs. Passing software checks does not certify a device.

## Findings and corrections

| Area | Finding | Result |
|---|---|---|
| Pump absorption and population rates | Arithmetic means of cell-face intensities disagreed with integrated Beer–Lambert photon absorption, especially in optically thick cells. | Replaced with the exact spatial exponential mean. Used by the common multipass pump and time-domain pulse transport. Both amplifier architectures benefit. |
| Ideal multipass recovery | Recovery used fixed pump-only equilibrium rates after signal depletion; evaluating pump absorption at mean inversion does not equal integrating its time-dependent value. | Shares the regenerative exponential-midpoint recovery kernel. Pump bleaching is recalculated as the shared inversion recovers. All signal-pass population integrals enter fluorescence, and all pulse windows are excluded from inter-pulse recovery time. |
| Independent conservation evidence | Ideal multipass lacked the regenerative branch's independent photon diagnostic. | Publishes local cycle photon residual and signal/relay energy residual, separately from heat defined by subtraction. Pump-recovery time discretization still requires refinement. |
| Comparison reference | An imported SLM correction could be omitted from the uniform cold reference. A uniform disk with static cold phase could incorrectly reuse its actual output as the ideal reference. | Reference retains the same SLM correction and removes the static disk phase. Regression checks compare the resulting phase arrays. |
| Wavelength | Pulsed default was 938 nm, inconsistent with the user's 969 nm target and below the nominal 940–1090 nm coating band. | Default is now 969 nm; Tkinter exposes the pulsed pump wavelength. The separate CW default remains 938 nm. Neither wavelength has measured coating reflectance in this project. |
| Gain bound | Ideal relay retention was absent from the configured-loss bound. | Includes one relay retention factor between successive traversals. |
| Crystal outline | Native plots hard-coded a 5 mm radius and used sample centers as image edges. | Uses the configured radius and half-pixel image edges. Archived results recover radius from their request. |
| Scientific labels | Cold sweep points and in-range thermal results could be read as hot or calibrated predictions. | Cold optical sweeps, fixed-heat thermal replay, uncalibrated material assumptions, and unavailable hot phase are explicitly identified. |

## Quantitative checks

Recorded in `results/ybluag_desktop_audit/physics_checks.json`, including the
numerical source fingerprint and configuration:

- Thick-cell pump rate versus boundary-photon relative error: **4.44e-16**.
- Strongly depleted recovery photon relative L1 error at 4, 8 and 16 substeps:
  **1.079e-3, 2.753e-4, 6.951e-5**. This demonstrates refinement on one fixture;
  it does not establish convergence for every user setting.
- At 12 at.% uniform Yb:LuAG, 100 µm thickness, 1030 nm signal, 969 nm pump,
  ten signal traversals, 20 °C and lossless relays, the current reconstructed
  spectra give a pump-asymptotic unsaturated gain bound of **13.15×**.
  Even the full-inversion unsaturated bound is **184.50×**. The **10,000×**
  target cannot be reached by this configured homogeneous model. These are
  optimistic analytical bounds, not an achieved simulation or a measured limit
  for all possible geometries and materials. A regenerative architecture has
  a separately counted number of material traversals and optical losses.
- 10 ps at 10 kHz has a FWHM duty fraction of **1e-7**. The modeled pump is CW;
  short seed pulses do not eliminate cycle-average pump heat. The lattice
  balance subtracts signal extraction, escaped fluorescence and changing
  stored excitation from absorbed pump power. Thermal stabilization depends
  on heat removal through disk/contact/copper/coolant and their heat capacities.

Initial Yb regression: **73 passed in 87.62 s**, peak process RSS about 148 MB.
After wavelength/reference/relay-bound changes: **18 focused tests passed in
5.42 s**, peak RSS about 107 MB. Both used the persistent local supervisor.
The focused checks include the new default wavelength, configured disk radius,
SLM-reference phase, coupled ideal ledger and optical energy conservation.

## Remaining physical limits

| Item | Current limit |
|---|---|
| Spectroscopy | Figure-guided absorption and McCumber-derived emission, not raw author data with uncertainties. The source study covers 20–200 °C. Pump linewidth and wavelength-resolved saturated amplification are absent. |
| Lifetime | 0.973 ms at 12 at.% is interpolated from two other concentrations; temperature dependence is not measured here. |
| Assembly | The parameterization permits coupled mechanics/OPD only at 293.15–300 K. This is a model domain, **not a crystal survival limit**. The doped-crystal and copper/contact properties at 250 °C are not established by this implementation. |
| Heat | Fluorescence escape is an effective assumed fraction; its zero default is a conservative heat scenario. Coating absorption and an experimentally calibrated contact/cooler are missing. |
| Thermal time | The plotted startup is a physical thermal integration with fixed periodic optical heat. It is not a synchronized transient optical/population/thermal solver. A steady coupled output must not be labeled as the field at every plotted startup time. |
| Pump sweep | Five points recalculate cold optical states. They do not recalculate hot-cavity performance versus pump. |
| Ideal relays | Unit-magnification, phase-preserving boundary condition with configurable power loss. No measured relay geometry or aberration is supplied. |
| Temporal coherence | Ideal output fluence carries a common spatial phase. A full spatiotemporal complex waveform, chirp evolution, Kerr phase and compression are absent. Regenerative output temporal shape is unavailable. |
| Other device physics | LuAG photoelastic tensor, ASE/parasitic lasing, finite switch waveform, measured coating phase/absorption and damage margin remain unavailable. |

The retrieved primary spectroscopy source is
[Körner et al., JOSA B 29, 2493–2502 (2012)](https://opg.optica.org/josab/abstract.cfm?uri=josab-29-9-2493).
The importance of pump evolution and spontaneous decay in multipass pulse
models is also discussed in
[Park, Jeong and Yu, Optimization of the pulse width and injection time in a double-pass laser amplifier](https://arxiv.org/abs/1805.01235).
Neither source provides calibration for this user's assembled amplifier.

## Tkinter changes

The native app groups seed/phase, Yb distribution, pump/cooling, amplifier
optics and numerical controls. A result header shows output energy, gain,
average power and heat. Saved-result status is separate from current inputs.
Beam maps have aligned horizontal and vertical center cuts, an adaptive
99.5%-energy field of view, and a full-crystal view. Absolute color bars and
separately normalized profile cuts are labeled. The phase view offers four
large comparison panels or the SLM masks. A progress indicator remains active
while the bounded worker runs. No HTTP server is started.

Native widget creation, all five input groups, plot rendering and both field
of view settings were exercised. Plots were inspected as exported figures.
Desktop screen capture was unavailable in this execution environment, so the
full-window visual appearance was checked through Tk layout measurements
rather than a captured desktop screenshot.
