# Stage 7V follow-up: two retained vector modes

This follow-up does not modify the Ho population equations, propagation kernels,
thermal solver, finite cooling plate, bonded interface, elasticity or Jones optics.
It changes the explicitly retained modal truncation for numerical qualification.
The original one-mode plan and its failed evidence remain available.

## Why the qualification baseline changed

The initial distributed campaign executed 33 coupled cases and 23 fixed-source
component cases. All 23 fixed-source cases completed. Nine coupled cases that
retained two, four or eight modes completed; 23 one-mode cases reached their
900-second execution limits, and one reached the outer-iteration limit.

A representative one-mode run had a field residual of 1.037 at its last saved
iteration, even though its output-power change was 1.19e-6 and temperature change
was about 0.0011 K. Its small eigensolver residual did not mean that the nonlinear
field iteration had converged. The direction/polarization of the retained vector
field kept changing. This is a failure of that numerical single-mode closure,
not proof of physical laser instability.

The completed two-, four- and eight-mode runs gave 0.790025259 W, 0.790054482 W and
0.790056644 W output, respectively, on otherwise identical grids. The two leading
branches carried essentially all output. Their dominant OAM order was zero; LG
initial guesses did not establish a stable vortex oscillator. These are finite
modal-truncation tests, not a global nonlinear stability theorem.

## Exact change and preserved controls

`examples/stage7v_two_mode_plan.py` creates a separate plan:

- Cases previously retaining one mode now retain two, with at least four candidate
  eigenpairs. All other numerical settings and tolerances are unchanged.
- Physical pump, material, crystal geometry, cooling plate, bond, support and mirror
  inputs remain identical. Physical sensitivity cases keep their original inputs.
- Completed two/four/eight-mode, candidate-spectrum and initial-start cases remain
  byte-identical in their case specifications.
- The retained-mode convergence group compares the two-mode reference to four and
  eight modes. It no longer pretends that the failed one-mode baseline is qualified.
- Original plans, failures and qualification reports are kept separately.

Reusing a previous result requires a completed status, identical source/runtime
fingerprint, identical case specification hash and matching state-file checksum.
Changed and failed cases are recalculated. No local result with a different
numerical runtime is silently combined into a remote convergence sequence.

The workflow bounds changed cases at 1800 seconds each and preserves partial
iteration histories, final states when available, failure reasons and timeouts.
The collection step evaluates the entire plan, including any missing cases.
A successful artifact upload is not a scientific acceptance pass.

## Reproduction

From the repository root:

```bash
python -m pip install -e '.[dev,plots]'
python -m pytest -q
python audit/deep_check.py

python examples/stage7v_two_mode_plan.py --output results/stage7v/two-mode-plan.json
python examples/stage7v_bounded.py \
  --plan results/stage7v/two-mode-plan.json \
  --cases reference optical_384 optical_512 \
  --seconds-per-case 1800 --output results/stage7v/two-mode
python examples/stage7_validation.py report \
  --plan results/stage7v/two-mode-plan.json \
  --output results/stage7v/two-mode --require-qualified
```

A partial selected-case execution cannot satisfy all required comparison families;
therefore `--require-qualified` must fail until the full required evidence exists.
Use `--all` on the bounded runner to execute the complete two-mode plan.

## Acceptance and scientific scope

The unchanged acceptance targets include 1% power/heat differences, 0.05 K peak
crystal/plate temperature differences, 2 nm piston-only-removed beam-weighted OPD
differences and 0.999 modal-mixture fidelity. Both terminal refinement comparisons
must pass. Every initial-start comparison must pass. Cooling/contact sensitivities
are physical changes, not mesh errors, and each altered operating point ultimately
needs its own refinement.

A numerically qualified two-mode reference would still not calibrate the real
material spectra, coatings, mounting, fluorescence energies or omitted physics.
The model remains adiabatic and uses incoherent modal photon populations, rather
than coherent beating or a full field update each round trip. Contact opening,
friction, delamination and plasticity remain outside the bilateral bond model.

`dataset_ready=false` and `experimentally_calibrated=false` remain explicit even
if every numerical family passes. Completed scientific outcomes are recorded in
the dated report under `results/stage7v/`, not inferred from CI success.

## Evidence provenance

Original distributed campaign: Actions run 35835101412, artifact 10740066509.
Archive SHA-256: `861c3e54f0addccdedda8b909062bc971aaf4446b6adf488f02dd9e28a74220a`.

Follow-up source revision: `ac1d04c82b6d28ad5fda1dc6b3aa331b5109cb0b`.
Follow-up workflow run: 35842419279. Preparation passed 269 tests and all 19
independent audit probes before the changed scientific cases were executed.
Source plan artifact: 10741592505; SHA-256
`e1711bd6c1c8ed224bf0d47cd05b80b1ccdd444e9bbc3a1a535ce8a70f3ee9a1`.

This documentation adds no claim of completed results beyond the separately
saved evidence and does not alter the physical-solver source fingerprint.
