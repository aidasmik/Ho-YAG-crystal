# Yb:LuAG audit implementation and remaining limits

Baseline: `d536064a547c984b7074ddc526f9a4a0b6a59624`. This audit changes
local files only. The 12 at.% proposal setup and 10 at.% CW example are
distinct. None of the proposal performance targets is used to fit parameters.

## Issue disposition

| Audit item | Status | Implemented and verified | Remaining limit |
|---|---|---|---|
| 1. Spatial propagation | Partial | Optical grid and window are independent inputs; cold Gaussian FFT is compared with an independent ABCD/q field at 96, 192 and 384 points and 8, 12 and 16 mm windows. Map sampling diagnostics label each app run as a preview. | No full structured-mode, vortex-core or cavity gain convergence certification. High-resolution coupled runs were not launched. |
| 2. Thermal-phase validity | Partial | Unsupported actual-operation phase, OPD and deformation are null with a reason. Scaled 5 K maps are in `design_reference_maps`, separate from actual maps. The output is labeled cold-only if thermal phase cannot be applied. | The in-range hot screen is still a post-amplifier approximation with an uncalibrated assembly. No hot-cavity prediction is certified. |
| 3. Material data | Partial | Runtime and legacy helpers now default to the same absorption reconstruction and McCumber-derived emission; historical emission requires `dataset="archived_reconstruction"`. Out-of-range requests reject by default. Explicit linear extrapolation warns that it is unvalidated. Raw archives are unchanged. Reciprocity is tested. | Figure-guided spectra have no published per-point uncertainty or raw author arrays. Independent Füchtbauer–Ladenburg/radiative-lifetime closure is unavailable. Local lifetime versus temperature is not measured. |
| 4. Gain feasibility | Fixed as an upper-bound diagnostic | Reports transparency, pump asymptote, saturation fluence, full-inversion and pump-asymptotic unsaturated ceilings, distinct pump/traversal/round-trip counts and configured optical retention. Flags targets above that bound. | It is not an achieved saturated gain or a guarantee of feasibility below the bound. |
| 5. Thermal–optical closure | Partial | Optical depth cells and thermal radial, angular and depth cells can be chosen separately; depth transfer preserves integrated heat. In-range temperature status is explicit. | Local-temperature spectroscopy, iterative heat/optics convergence, per-encounter thermal cavity screens, fully coupled transients and mesh convergence of bending/OPD remain unimplemented. The app does not claim a calibrated hot cavity. |
| 6. Heat and fluorescence | Partial | Regenerative output has separate aperture, disk-HR, held-optics, injection and unextracted-ejection ledgers; a boundary-photon/population residual is computed independently of the heat formula. Fixed-population effective-fluorescence and fixed-heat contact/coolant sensitivity screens are available. | Fluorescence escape combines intrinsic efficiency, transport and reabsorption because those inputs are not measured separately. The thermal transient starts from a converged periodic optical state; excitation-energy startup storage is not modeled. |
| 7. Pulse and saturation | Partial | Weak, saturated and depleted fluence transport is benchmarked against Frantz–Nodvik with temporal and axial refinement. The regenerative fluence-only mode no longer presents a fabricated output waveform. A wavelength-resolved small-signal screen uses one shared inversion and preserves source spectral magnitude under stretching. Inter-pulse recovery substeps are configurable and a deterministic 2/4/8-step refinement check passes. | Wavelength-resolved **saturated** transport, spectral phase, compressed pulse and finite-time switching are not calculated. Recovery convergence is established only for the tested configuration. |
| 8. Structured fields | Partial | Same-plane coherent and intensity overlaps distinguish target-to-cold from cold-to-modeled-hot. Global phase is removed by the complex overlap; the target is the mask-generated field, not a pure LG/HG claim. Concentration-dependent index is explicitly unavailable. | No calibrated `dn/dYb`, full modal/OAM decomposition or experimentally measured dopant map. |
| 9. Hardware validity | Partial | 938 nm is flagged outside the proposal's 940–1090 nm HR **target** band. Measured reflectance, switch time, ASE, B-integral, coating heat and damage margin return unknown/null where inputs are absent. | Vendor coating spectra, geometry-specific ASE data, switching waveform, nonlinear coefficient and damage data are needed. |

## Reproduced numerical checks

Cold first round trip: 1030 nm, 0.6 mm initial **1/e field radius** (also
1/e² intensity radius), 0.25 m disk-to-mirror air gap, 0.5 m mirror curvature,
and a passive Gaussian. The analytic ABCD/q result is **136.608 µm**. A
12 mm FFT window gives **171.044 µm at 96²**, **136.608 µm at 192²**, and
**136.608 µm at 384²**, using a second-moment radius. The older audit's
183.745 µm for 96² was not reproduced; its radius estimator and exact field
path were not specified, so the numeric discrepancy cannot be reconciled
further. Both checks identify the coarse grid as inadequate. At 96², the 16 mm window is
still worse (about 346 µm) because pixel pitch increases. Edge energy can be
near zero despite a bad radius and coherent overlap, so energy conservation
alone is not a sampling test. These Gaussian results do not certify vortices
or sharp masks. Against the 384² field on the same 12 mm window, the
vortex-mask coherent overlap is **0.9885 at 96²** and **0.9977 at 192²**;
the sharp quadrant mask gives **0.8796** and **0.9613** respectively. Thus the
192² Gaussian result does not certify the structured fields; 384² itself is
only a comparison reference, not a converged solution.

From the canonical 20 °C 12 at.% reconstruction at 938/1030 nm and a 100 µm
disk, the transparency excited fraction is about **0.05113**, the infinite
pump-rate asymptote is **0.85230**, and the signal saturation fluence is about
**59,764 J/m²**. Ten material traversals yield unsaturated ceilings of
**184.499×** at full inversion and **81.896×** at the pump asymptote, before
loss. These ceilings are optimistic and cannot establish achieved gain. The
100 µJ target from a 10 nJ seed requires 10,000×, above the ten-traversal
pump-asymptotic ceiling.

At 1030 nm and 20 °C, the archived emission is **2.78×10⁻²⁰ cm²** while the
canonical McCumber-derived emission is **3.062×10⁻²⁰ cm²** (archived value
about 9.2% lower). The legacy helper now defaults to the latter. The archive
remains selectable for research comparison.

The deterministic 32² regenerative fixture closed its cavity optical ledger
to machine precision and gave a boundary photon/population relative L1
residual of about **1.75×10⁻⁴**. This is a software balance check, not a
measurement of device efficiency.

For a separate 10 µJ injected-seed recovery fixture, 2, 4 and 8 inter-pulse
recovery substeps gave **7.3128174108**, **7.3128173793** and
**7.3128173638 µJ** output, respectively. The 4-to-8-step change was
**2.13×10⁻⁹** of the 8-step output. This checks one numerical setting, not
the unmodeled thermal-optical feedback or all operating points.

## Provenance, units and validity

Spectra are figure-guided reconstructions over **880–1150 nm** and
**293.15–473.15 K**, with cross sections returned in **m²** by
`src/ybluag/model.py` and **cm²** by the legacy helper. The concentration is
the fraction of Lu sites, using **1.42×10²⁸ Lu sites/m³**. Wavelength uses
linear interpolation between 0.5 nm archived samples; temperature uses
piecewise linear interpolation between 20, 80, 140 and 200 °C. Emission is
derived from absorption and published Stark levels by the McCumber relation.
The archived emission is an independent figure-guided reconstruction, not raw
measured arrays. No per-point spectral uncertainty is available, so absolute
gain predictions are not experimentally calibrated.

The copper assembly has a narrower **293.15–300 K** material-property range.
Its contact and water-side conductances are assumptions. The 5 K design map
is a separate lower-heat calculation, not an actual-operation result. A
material-data boundary is not a damage threshold. Photoelastic coefficients
for the actual 12 at.% bulk disk, concentration-dependent index, measured
coating spectra and calibrated contact/cooler data remain absent.

## Reproduction

```bash
.venv/bin/python -m pytest -q tests/test_ybluag.py tests/test_ybluag_regenerative.py tests/test_ybluag_audit_diagnostics.py tests/test_ybluag_cold_sampling.py tests/test_ybluag_pulse_convergence.py
.venv/bin/python -m pytest -q tests/test_structured_beam_gallery.py tests/test_seeded_periodic.py
.venv/bin/python examples/validate_ybluag_audit.py
```

The first group passed **39** tests and the selected Ho regression group
passed **16** tests. The validation script generated
`results/ybluag_audit_validation/cold_propagation.png`,
`structured_sampling.png`, `pulse_convergence.png`, and
`validation_summary.json`. No new high-resolution
coupled thermal-optical campaign was run. The persistent local coupled-run
budget was not restarted.
