# Stage 7V — Numerical qualification, not experimental calibration

Stage 7V adds validation orchestration to the audited API 0.8 model. The physical
rate equations, heat diffusion, finite copper plate, compliant bond, FEM and
Jones/eigenfield operators are not replaced.

## Two separate experiments

The **coupled campaign** calls the actual `run_coupled_hot_cavity` for each case,
recomputing fields, periodic Ho populations, heat, both body temperatures, stresses,
surface displacement and optical phase. The plan contains **33 cases**, including
256/384/512 optical grids, padding at fixed pixel spacing, thermal and mechanical
meshes, joint refinement, tighter integration, **1/2/4/8 retained modes**, larger
candidate eigenspectra, Gaussian/LG(+1)/LG(-1)/LG(2)/mixed initial fields, and physical
cooling/bond/support/dopant sensitivities.

The **frozen-source campaign** holds a hashed archived heat and mean-population
field fixed. It recomputes the assembly and independent weak double-pass LG probes.
It cannot predict new oscillator power or qualify the fully coupled model. The
expanded component plan contains **23 cases**, including one additional mechanical
and one additional joint refinement beyond the original 21-case study.

## Physical invariance and comparisons

ContinuousDensity uses fixed Fourier coefficients and physical wavevectors. The
analytical cylindrical mean is removed, and each numerical cell samples the SAME
physical distribution. Contrast is an amplitude bound, not a promised RMS. Archived
heat is remapped conservatively in z, r-squared and angle. Different mounting or
dopant settings are physical sensitivity cases, not grid errors.

Acceptance targets are configurable project criteria: power/heat changes <=1%;
crystal and plate peak-temperature changes <=0.05 K; beam-weighted OPD-map difference
RMS <=2 nm after **piston only** removal; complex vector mixture fidelity >=0.999.
Heat and population energy balances, mechanical residuals, field eigenpairs and
aperture sampling must also pass. Comparing a cropped region cannot conceal more
than the permitted missing optical power.

Different retained-mode counts are compared as incoherent mixtures weighted by
modal output power, not coherent sums with arbitrary phases. OAM uses both complex
polarization components around a fixed laboratory origin; unreported angular
orders remain an explicit tail. Closed phase winding includes the final contour
edge and rejects unresolved contours. Weak probes are not presented as selected
free-running vortex modes.

Numerical refinement requires **two consecutive terminal passing comparisons**.
Earlier failed coarse comparisons remain in the report, rather than being erased
or rescued by looser tolerances. Initial conditions are not ordered grid levels:
**every initial-guess comparison must pass**, not just the final two.

## Execution status is not qualification

Every case writes its specification, source/runtime fingerprint and `running`
record before execution. Partial outer-loop residuals are checkpointed to
`iterations.jsonl`. Final NPZ states carry checksums. Only identical source,
runtime and settings permit cache reuse; stale/corrupt results cannot count.

`examples/stage7v_bounded.py` puts a stated wall-time limit around each subprocess
and its workers. Timeout or resource interruption is saved as `timed_out` or
`interrupted`, never `completed`. A nonconverged optical cycle or eigenproblem is
also not a pass. The report includes all planned but unrun cases. The optional
`--require-qualified` report flag exits nonzero unless every required coupled
family is numerically qualified.

Even a fully passed numerical campaign leaves `dataset_ready=false` and
`experimentally_calibrated=false`. Real spectra, coatings, crystal/plate bonding
and omitted physics have not been measured or calibrated by a mesh study.

## Running

```bash
python -m pip install -e '.[dev,plots]'
python -m pytest -q
python audit/deep_check.py

# Planning alone does not execute anything.
python examples/stage7_validation.py plan --kind coupled \
  --output results/stage7v/coupled_plan.json

# Explicit selected cases; all intermediate and final evidence is preserved.
python examples/stage7v_bounded.py \
  --plan results/stage7v/coupled_plan.json \
  --cases reference optical_384 optical_512 \
  --seconds-per-case 900 --output results/stage7v/coupled

python examples/stage7_validation.py report \
  --plan results/stage7v/coupled_plan.json \
  --output results/stage7v/coupled --require-qualified
```

To perform component diagnosis, generate a frozen plan with `--kind frozen
--reference path/to/audited/result`. That directory must contain the real
`state.npz` and `summary.json` with repaired pump-spectrum metadata. No missing
physical arrays are invented from scalar summaries.

## GitHub Actions

The Stage 7V workflow first runs the full regression suite and independent audit.
The integration-branch run or a manual dispatch generates explicit plans, pins
numerical package versions, and runs cases as separate bounded jobs. It preserves
individual outcomes, logs, partial iteration histories, configuration hashes and
full available states as artifacts. The final report distinguishes numerical
success, failure, nonconvergence and execution limits.

A green upload/collection job means evidence was saved; it does NOT mean a
scientific acceptance threshold passed. Consult `qualification.json` and the
individual case statuses. Different numerical runtime fingerprints are not
silently merged into one grid-convergence study. Raw results remain in Actions,
not in ordinary Git history; compact reports may be committed separately.

## Current interpretation

The previously completed coarse coupled case was one grid and one retained mode.
A high-resolution eigenpair residual alone cannot certify grid or nonlinear
mode-count convergence. The finite plate and bond remain physical parts of every
new assembly solve. The existing bilateral bond does not model opening, friction,
delamination, plasticity, creep or an uncalibrated pressure/conductance law.

See `results/stage7v/` for dated executed evidence. Original pre-integration results
remain historical data and must not be relabeled as a new coupled campaign.
