# Stage 7W — polarization-resolved modal closure

This opt-in extension addresses the Stage 7V diagnosis of a one-mode outer loop
switching between two nearly equal-gain but distinct polarization eigenbranches.
It preserves the audited pump, four-manifold populations, modal photon equations,
heat ledger, finite copper plate, compliant bond, stress and Jones optics.
The original Stage 7 solver and its one-mode control campaign remain unchanged.

## Selection and tracking

`polarization_tracking.py` computes a raw non-Hermitian candidate bank. All
candidates, including omitted candidates, must satisfy the full-grid eigenpair
residual. An incomplete Arnoldi solve never counts as a successful selection.

Maximum-overlap rectangular assignment identifies old branch labels in the full
candidate bank before truncation. The strongest retained fields are then assigned
consistent labels. A genuinely dominant new branch is not suppressed simply to
preserve continuity: replacement resets photon warm-start seeds and must settle
again. Per-mode photon slots are never exchanged just because gain rank changes.

A retained/omitted boundary is rejected when it cuts a likely polarization
family: small complex-eigenvalue separation, nearly orthogonal vector fields and
almost identical spatial form under a constant polarization transformation. This
is a *truncation guard*, not a physical degeneracy declaration. The spatial
alignment is used only for that guard; it is never substituted for the actual
vector overlap in a convergence test. The guard thresholds are explicit numerical
parameters and require sensitivity testing if a decision is marginal.

Near-split pairs are NOT rotated or replaced by their average eigenvalue.
Only machine-near-identical complex eigenvalues (default 1e-12 relative tolerance)
may be aligned inside their common eigenspace, and every result is rechecked
against the full operator. The wider default family tolerance (1e-3) is not a
license to mix eigenvectors.

## At least two explicit branches in this model

`run_polarization_hot_cavity` defaults to two retained vector modes and requires
at least two additional candidates. Two is a conservative default for the present
polarization-unfiltered cavity, not a universal theorem that every laser emits
both polarizations. No polarizer, mode filter or artificial equal-power constraint
is added. Single-mode controls remain available through the original solver.

New field guesses are separately resolved eigenfields rather than linear blends
of distinct eigenmodes. The original heat-source under-relaxation remains. The
shared HotCavitySettings.field_relaxation value is explicitly not used by Stage 7W;
metadata records the actual field-update rule.

## Convergence cannot be hidden by a subspace rotation

Every original convergence gate remains: actual phase-aligned vector-field
change, unrelaxed heat change, crystal AND plate temperature/displacement change,
output-power change, passive loss change, periodic optical state and individual
eigenpair residual. Additional gates require modal-power change, modal gain
stationarity/change, invariant-subspace stationarity and incoherent coherency
stationarity/change.

The coherency diagnostic is

    J = sum_i P_i |E_i><E_i|.

Its relative Hilbert-Schmidt difference is calculated through small Gram matrices,
not a pixels-squared matrix. It retains polarization and spatial coherence.
Two equal-power modes can represent the same mixture under a basis rotation;
two unequal-power modes generally cannot. A stable total intensity or subspace
therefore cannot hide a changed polarization distribution. Actual nondegenerate
field convergence is still required. No tolerance is loosened to erase the stall.

## Stage 7V integration

`polarized_validation.py` produces an explicit Stage 7W plan using the Stage 7V
physics, mesh definitions, fingerprints, comparison metrics and acceptance
thresholds. The baseline is two modes. Retained-mode convergence is 2 -> 4 -> 8,
with all Gaussian/LG(+1)/LG(-1)/LG(2)/mixed starts preserved. Optical, material,
mechanical, window, candidate-count and cooling-interface sweeps are retained.
A duplicate two-mode baseline is removed, leaving 32 coupled cases.

The original one-mode plan is not overwritten. Stage 7W cases carry a distinct
closure and source fingerprint; old results cannot be silently reused as repaired
results. Missing, failed, timed-out and nonconverged cases remain unqualified.
Even a numerically converged representative case does not imply full Stage 7V
qualification or experimental/NN dataset readiness.

## Run

```bash
python -m pip install -e '.[dev,plots]'
python -m pytest -q
python audit/deep_check.py
python examples/stage7w_validation.py plan --output config/stage7w_validation.json
python examples/stage7w_validation.py run --plan config/stage7w_validation.json \
  --cases reference modes_4 modes_8 --seconds-per-case 3600 \
  --output results/stage7w/paired
python examples/stage7w_validation.py report --plan config/stage7w_validation.json \
  --output results/stage7w/paired
```

The bounded parent records timeouts separately from scientific nonconvergence.
Reports are built against the full plan, so running a small subset cannot generate
a false whole-campaign pass. `dataset_ready` remains false pending independent
physical calibration and complete qualification.

## Verification status

See the branch's Stage 7W validation workflow and its exact source revision for
executed tests and representative cases. This document does not assert that an
unexecuted case passed. Existing audit/regression workflows remain enabled.

## Sources

The iteration diagnosis is documented in
`stage7v-retry-20260923:docs/STAGE7V_RETRY_DIAGNOSIS_20260923.md`.
Underlying physical assumptions and parameter provenance are in STAGE5/6/7 and
the API 0.8 audit documentation. Mathematical/numerical references:

- SciPy eigs (non-Hermitian Arnoldi and partial-convergence contract):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html
- SciPy linear_sum_assignment (rectangular one-to-one matching):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
- Knyazev & Argentati (2002), principal subspace angles, as documented by SciPy:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.subspace_angles.html

No new measured optical or mechanical constants are claimed.
