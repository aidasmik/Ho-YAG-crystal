# Stage 7V — Numerical qualification, seeded vortices and mount sensitivity

Stage 7V is a validation runner, not an extra laser-physics model. It calls the corrected Stage 7 hot-cavity and Stage 6 finite-assembly solvers. It never replaces them with a fitted curve, an assumed inversion or an independently rewritten RHS.

## Two profiles and an explicit unfinished-work gate

`pilot` defines 12 cases: full coupled calculations on 256², 384² and 512² optical grids; a two-retained-mode coupled calculation; six fixed-source mounting cases; and two further mechanical refinements. It probes injected LG azimuthal orders -3 through +3 in completed hot states. It deliberately does NOT qualify the entire model for training data.

`full` adds optical-window, material/thermal, coupled mechanical, joint, integration-tolerance, retained-mode-count and initial-field comparisons. All required groups are enumerated even in the pilot report. A missing case or required group is INCOMPLETE. Failed physics comparisons remain FAIL; a completed worker is not evidence of convergence. ERROR, TIMEOUT and NONCONVERGED states cannot become passing numerical qualifications.

The selected cavity is the existing 250 mm air-gap, 500 mm-curvature, 2% output-coupler configuration. The reference pump is 1 mJ, 10 ps at 10 kHz; crystal is 10 mm diameter by 1 mm thickness, with the finite Stage 6 copper plate and bonded interface. Mount properties remain assumptions, not experimental measurements.

## Commands

From the repository root on Linux/POSIX (subprocess deadlines terminate a process group):

```bash
python -m pip install -e '.[dev,plots]'
python -m pytest -q
python examples/stage7v.py plan --profile full

# Independent coupled optical-grid cases, then mount cases using base256 heat:
python examples/stage7v.py run --profile pilot --cases base256 opt384 opt512
python examples/stage7v.py run --profile pilot --cases modes2
python examples/stage7v.py run --profile pilot --cases \
  mount_base contact_low contact_high shear_soft roller coolant_low fem_mid fem_fine
python examples/stage7v.py report --profile pilot

# All defined full-profile cases, with independent per-case deadlines:
python examples/stage7v.py run --profile full
python examples/stage7v.py report --profile full --require-qualified
```

The final command exits unsuccessfully until the numerical gate passes. Merely writing a report is not a scientific success condition. `--require-completed` makes run mode fail if any selected computation errors, times out or does not converge. Partial progress is retained. `--timeout` overrides a per-case wall-clock limit without changing the optical equations or requested numerical tolerances.

Outputs: `results/stage7v/generated/<case>/summary.json`, full `state.npz`, input case JSON, unaltered solver progress and log. The top-level `report.json` contains per-comparison checks, incomplete groups and numerical/physical qualification flags. A Python API, `make_plan`, supplies explicit editable dictionaries; `execute_case` runs a custom case with its own parameter/provenance hash.

## Same specimen across grids

`ContinuousDopant` defines a seeded bounded Fourier field in physical metres. Refining an optical/material mesh does not regenerate its random phases, change its physical spatial frequencies, or renormalize each sampled specimen to a new mean. Cell averages use analytic axial integration and adaptively checked radial/azimuthal quadrature. The default reference is uniform. `amplitude_bound` is a bound, NOT an advertised RMS or an experimentally calibrated correlation function. `mean_m3` is the nominal density scale; nonzero perturbations need not have exactly zero volume-mean over this finite disk.

Physical configuration fingerprints exclude numerical grids. A numerical refinement is rejected if the physical pump, material/dopant definition, mirror setup, plate or interface changed. Sensitivity comparisons intentionally change physics and therefore are labeled MEASURED, not PASS/FAIL convergence tests.

## Comparing optical fields and optical path

Fields are interpolated in their real and imaginary components onto a common physical-coordinate grid covering the union of both windows. Wrapped phase is never interpolated. No recentering, shift, rotation, tilt or defocus correction is used to improve the agreement. Normalized vector overlap includes both laboratory polarization components. Global complex phase is irrelevant.

For multiple modes, branch permutation is matched by overlap. Subspace capture is reported separately: an arbitrary rotation of a degenerate polarization basis is not claimed to be identical individual eigenbranches. Different mode counts use subspace capture for the field component of their comparison; modal output powers are also recorded. This remains a finite-mode adiabatic model, not a proof of coherent multimode stability or stable vortex lasing.

OPD differences use the unwrapped length field from the thermo-mechanical Jones screens. The scalar mean OPD is compared over a 1 mm-radius pupil weighted with a 0.408 mm Gaussian radius. Only the difference's weighted piston is removed. Vector field overlap remains a separate check for birefringence/polarization changes. Interpolation has finite accuracy and is itself covered by analytic LG tests.

## Seeded amplifier probe versus a free-running oscillator

The weak-probe diagnostic injects a specified complex LG field at the disk front, applies two traversals of the frozen material gain and the ordered inward/outward Jones operators, geometric reflected OPD, the circular disk mask and rear mirror reflectivity. It does NOT include the curved output coupler or an eigensolver. The injected phase is not discarded.

This is a **collapsed-disk, frozen cycle-averaged population probe**, not a new independently pumped amplifier equilibrium. The seed does not extract enough energy to alter populations, and no claim of finite-energy saturation is made. Its background is the solved oscillator population map; it is not the population of an amplifier with its circulating oscillator removed. Existing Stage 4 saturated-pulse functions remain available for a separately specified state.

OAM diagnostics compute the azimuthal Fourier power at every sampled radius, integrate over radius, and sum both polarizations. The denominator includes all sampled azimuthal Fourier bins, not just the charges shown in the table. Reports include unreported-order power and the pupil energy relative to Cartesian energy. A single phase-winding contour is not used as a proxy for mode purity.

## Cooling-plate and bond sensitivity

`mount_base`, `contact_low`, `contact_high`, `shear_soft`, `roller` and `coolant_low` all use the identical baseline heat-state hash. They recompute the finite crystal AND plate temperatures, bonded-interface forces and deformation. They isolate mounting uncertainty without silently changing the source. These are fixed-heat sensitivity calculations, not new output-power predictions.

Similarly, `fem_mid` and `fem_fine` isolate mechanical discretization at fixed heat. They cannot satisfy the separate coupled-mechanical/joint convergence requirements. Heat remapping between cylindrical finite-volume meshes uses exact overlap in z, r² and azimuth and preserves total source power. It preserves the old piecewise-constant source, not the information that a finer optical solve might generate.

## Proposed initial acceptance policy

These are project choices, not universal numerical-analysis standards:

- Relative changes in output, absorbed pump and heat below 1%.
- Maximum crystal and plate temperature changes below 0.05 K.
- Weighted piston-removed mean OPD difference below 2 nm.
- Squared normalized vector overlap (or applicable subspace capture) above 0.999.
- Independent local optical-energy, thermal-energy and mechanical-equilibrium residuals below recorded thresholds; negligible optical-window edge power.

Each refinement family requires at least three levels and two consecutive comparisons. Whole-aperture mirror-phase sampling is reported separately. The full gate also requires window, material, mechanics, combined and tolerance refinements, mode-count and initial-field checks, and completed probe/sensitivity measurements. A green code-test job is not a green dataset qualification.

`physical_calibration_verified` and `approved_for_trusted_nn_targets` remain false: numerical refinement does not establish actual fluorescence spectra, contact conductance, bond laws, coating losses or other sample-specific inputs.

## Provenance and interruptions

Source-code SHA-256, effective case and physical configuration fingerprints and full-state SHA-256 are stored with each result. Frozen-source cases reference the exact source NPZ hash. Cached states must match the requested case/source, contain a completed status and pass content hashing. Partial/tampered states are rejected. Reports explicitly retain failures and missing required groups. Calculation budgets are limits, not promises that a difficult eigenproblem will converge.

## Sources

- NumPy FFT documentation, https://numpy.org/doc/stable/reference/routines.fft.html: normalization and azimuthal transform convention.
- SciPy interpolation documentation, https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html: spline coordinate interpolation, applied separately to complex components.
- Repository Stages 0–7 and `docs/AUDIT_FIXES_STAGE0_7.md`: physical model, corrected population/source contracts and experimental-validity limitations.

Actual execution counts and measured outcomes are recorded separately under `results/stage7v/`; no run is represented as completed merely because it appears in a manifest.
