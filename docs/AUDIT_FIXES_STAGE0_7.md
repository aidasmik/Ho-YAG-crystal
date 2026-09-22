# Stage 0–7 audit corrections (API 0.8)

These corrections address the independently reproduced defects in `docs/AUDIT_STAGE0_7_20260922.md`. That original report remains historical evidence, not a claim that the old source passed. The correction run reruns the actual source, all regression tests and the independent probes before publishing a source commit. Executed counts and outcomes are recorded in `results/audit_fixes/verification.json`.

## Canonical population arrays

All spatial population outputs now use **(manifold,z,y,x)**, with I5/I6/I7/I8 manifold order. Single-pulse pump APIs have changed from `(z,manifold,y,x)`; repetitive-pump and signal layouts are unchanged. Result objects expose `population_axes`. `PopulationField` supports validated immutable tagged storage.

```python
from hoyag import PopulationField, convert_population_layout
canonical = convert_population_layout(old_array, source_layout='z,manifold,y,x')
field = PopulationField.from_legacy_z_major(old_array, local_density_m3)
```

No shape-based guessing is performed, particularly when both leading dimensions are four. Local populations must sum to the supplied local density before roundoff-only correction. Invalid totals, NaNs and negative populations are rejected instead of being turned into artificial inversion. End-to-end tests cover nz=3,4,5 with actual pump results passed to the signal API.

## Phase-preserving reference

`propagate_structured_signal_frozen` propagates the original complex E(t,y,x) through the same diffraction, GVD and gain, freezing populations only. The historical result field `small_signal_power_gain_reference` now stores this correct pulse-energy ratio. It no longer constructs a zero-phase field from sqrt(fluence). Vortex and nonseparable space/time tests check the weak-signal limit.

## Physical source and frequency convention

The physical convention is A(t) exp(+ikz-i omega0 t). With NumPy fft(A), optical frequency is **nu0 - fftfreq**. The even quadratic GDD phase is unchanged. Positive and negative detuning are tested independently.

`PumpSource` is shared by modal/radial Stage 4R, Stage 5 and Stage 7. A transform-limited Gaussian uses the exact intensity TBP 2 ln(2)/pi. Duration changes now reach the absorption cross section passed to the material constructor. A chirped Gaussian requires explicit bandwidth and consistent actual duration/GDD: duration alone does not determine its spectrum. Custom spectral energy quadratures and provenance-labeled scalar overrides are supported. Used source metadata and effective cross sections are saved. The low-level material API may still use explicit fixed cross sections without inferring a spectrum.

## Validation and masked propagation

The parameter object validates finite positive density, lifetimes and wavelengths; finite nonnegative rates and cross sections; and all branching-ratio contracts. The existing four-manifold RHS requires beta78=1. Very long finite lifetimes and zero rates remain available for isolated-limit tests. Grid sizes must be integers and spacings finite/positive.

Masked angular-spectrum propagation computes only propagating phase factors rather than evaluating a growing evanescent exponential first. Forward evanescent decay remains supported when explicitly enabled. Backward unmasked evanescent continuation is rejected. Stage 1P reuses the same transfer helper.

## Density statistics

Small perturbations retain the original affine smoothed-Gaussian map. When the lower bound would be violated, a smooth monotone positive transform is fitted to satisfy mean, RMS and floor together. Its histogram is no longer Gaussian; this is stated explicitly. Infeasible finite-map specifications are rejected. `ho_density_statistics` reports realized values and seed reproducibility is retained. `correlation_fraction` remains a spectral cutoff, not a calibrated physical correlation length.

## Accuracy diagnostics, not invented physics

`apply_frozen_spectral_absorption` resolves linear ground-state attenuation by frequency. It is not a saturated broadband population solver. The spectral diagnostic compares <exp(-column*sigma)> against exp(-column*<sigma>); policies support warning, error or report. Spectrum-aware modal runs warn when scalar absorption produces substantial ground-state transmission bias over the chosen path.

Stage 7 records whole-aperture mirror-phase sampling and explicitly marks mesh convergence and dataset validation as unestablished. A tiny eigensolver or outer-loop residual is not proof of optical/material/mechanical grid or mode-count convergence. The independent Frantz–Nodvik depth-refinement probe remains active, without looser tolerances.

No new material measurements, temperature-dependent spectra, radiation trapping, nonlinear contact or coherent mode beating are introduced. The finite copper plate and bonded interface are preserved. This patch repairs software and contracts, not experimental calibration or a completed coupled mesh-refinement campaign.

## Reproduce

```bash
python -m pip install -e '.[dev,plots]'
python -m pytest -q
python audit/deep_check.py
python examples/stage7_hot_cavity.py --quick --require-converged
```

The audit now exits unsuccessfully if any independent probe fails, after saving all outcomes. A green workflow must not be used to hide red probes. NumPy transform-convention reference: https://numpy.org/doc/stable/reference/routines.fft.html. Existing Stage 0–7 documentation retains material-parameter provenance.
