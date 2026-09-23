# Ho:YAG local-agent implementation report
## Targeted numerical improvements, bounded validation, and solver-driven visualization

**Repository:** `https://github.com/aidasmik/Ho-YAG-crystal`

**Verified baseline for this handoff:** `0b18a9f1201a79c6c0efcfeeb50139f1e3e293eb`

**Purpose:** improve the existing implementation on the local PC, resolve the remaining numerical-workflow issues, and implement the live scientific visualization specification without calculations lasting days.

**Important:** do not rebuild Stages 0–7W from scratch. Reuse the existing verified physics stack.

---

# 1. Mission and non-negotiable constraints

Deliver:

1. Cross-platform bounded local execution.
2. Retained-mode family planning without weakening polarization safeguards.
3. Measured runtime improvements that do not change the physical model.
4. Replay-first scientific visualization, then a live solver worker.
5. A separate externally seeded multipass-amplifier application path.
6. Updated documentation and evidence.

Do **not** automatically launch:

- the complete 32-case Stage 7W campaign;
- multi-day coupled calculations;
- DFT, phonons, molecular dynamics, CFD, or carrier-resolved FDTD;
- large neural-network training;
- exhaustive parameter sweeps;
- remote/GitHub Actions compute without explicit user request.

The goal is a useful local development cycle that finishes in hours, not days.

---

# 2. Read first

Read before modifying code:

```text
README.md
docs/STAGE7W.md
docs/STAGE7W_RESULTS.md
docs/STAGE7V.md
docs/STAGE7.md
docs/STAGE6.md
docs/STAGE5.md
docs/AUDIT_FIXES_STAGE0_7.md
docs/AUDIT_STAGE0_7_20260922.md
results/stage7w/evidence_summary.json
results/stage7w/cases/
HoYAG_live_visualization_addendum(1).md
```

If any `AGENTS.md` exists, read and obey it before changing files in that scope. Merge this handoff with newer local rules rather than overwriting them.

---

# 3. What is already implemented

Reuse these existing modules:

| Area | Existing implementation |
|---|---|
| Passive diffraction and temporal propagation | `propagation.py`, `temporal.py` |
| Ho populations and pump handling | `populations.py`, `pump_source.py`, `spectroscopy.py` |
| Population validation | `population_state.py` |
| Spatial dopant distributions | `inhomogeneity.py`, `density_statistics.py` |
| Structured-signal amplification | `signal.py` |
| Oscillator dynamics | `resonator.py` |
| Heat and thermal coupling | `heat.py`, `thermal.py`, `thermal_resonator.py` |
| Cooling plate/interface | `cooling_plate.py` |
| Thermoelasticity | `thermomechanics.py` |
| Photoelastic/surface optics | `stress_optics.py` |
| Vector cavity | `vector_cavity.py` |
| Original coupled closure | `coupled_resonator.py` |
| Polarization-aware modal tracking | `polarization_tracking.py` |
| Stage 7W closure | `polarized_resonator.py` |
| Numerical qualification | `validation_*`, `polarized_validation.py` |

Stage 7W already includes:

- full-candidate modal matching;
- polarization-family truncation detection;
- separate photon-state labels;
- separate near-split but nondegenerate branches;
- actual vector-field convergence;
- modal-power and modal-gain checks;
- invariant-subspace checks;
- incoherent coherency checks;
- explicit rejection of retained boundaries that split a polarization family.

**Do not reimplement polarization-aware mode tracking from scratch.**

Recorded verification:

```text
299 regression tests passed
33 Stage 7W-specific tests passed
19 independent audit probes passed
```

Recorded representative Stage 7W results:

| Case | Status | Iterations | Output | Heat |
|---|---|---:|---:|---:|
| 2 modes, clamped | Converged | 7 | 0.790034 W | 0.654748 W |
| 4 modes, clamped | Rejected: polarization family split | — | Not qualified | Not qualified |
| 8 modes, clamped | Converged | 7 | 0.790115 W | 0.654756 W |
| 2 modes, free support | Converged | 7 | 0.790032 W | 0.654749 W |

The 2- versus 8-mode total-output difference is about 0.0102%, but this is not a complete mode-count convergence proof because the 4-mode intermediate case was correctly rejected.

The 4-mode rejection is a safeguard, not a defect to delete.

---

# 4. Keep oscillator and seeded amplifier separate

Maintain explicit application modes:

```text
oscillator_reference
seeded_multipass_amplifier
```

The oscillator uses the existing resonator model.

The amplifier is externally seeded. Do not use an oscillator eigensolve merely to obtain an amplifier output.

All amplifier disk encounters must access the same physical crystal state:

```text
same Ho populations
same material map
same temperature field
same stress field
same deformation field
```

Do not create independent crystal copies per encounter.

If relay distances, seed pulse duration/energy, SLM response, or pass count are unavailable:

- implement the generic encounter/relay API;
- implement a clearly labelled single-encounter fixture;
- set `seeded_amplifier_configuration_complete = false`.

Keep seed and pump parameters separate. Do not assume the seed is 10 ps because the pump is 10 ps.

---

# 5. Local checkout and environment

Existing checkout:

```bash
git status --short
git remote -v
git branch --show-current
git rev-parse HEAD
```

Do not discard local work, force-reset, force-push, or merge automatically.

New checkout:

```bash
git clone https://github.com/aidasmik/Ho-YAG-crystal.git
cd Ho-YAG-crystal
git switch -c local/bounded-validation-viewer
```

Windows environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,plots]"
```

Record CPU, RAM, GPU if present, OS, Python, NumPy, SciPy, Matplotlib, BLAS backend, and thread settings.

Do not assume GPU acceleration will help the current sparse CPU solvers.

---

# 6. Hard compute budget

Implement the budget supervisor **before expensive calculations**.

| Item | Limit |
|---|---:|
| Total automated numerical budget | **7200 s** |
| Single expensive coupled case | **900 s** |
| Total expensive coupled attempts | **4** |
| Concurrent expensive workers | **1** |
| Profiling micro-run | **180 s** |
| Regression allocation | **600 s** |
| Independent audit allocation | **300 s** |
| Viewer smoke tests | **120 s** |
| Full campaign | **disabled by default** |
| Remote compute | **disabled by default** |

Persist one global budget ledger. Restarting the CLI must not reset it.

Use:

```text
effective_case_limit = min(configured_case_limit, remaining_global_budget)
```

Add a memory ceiling and avoid sustained swapping.

At limits report explicitly:

```text
timed_out
resource_limit
budget_exhausted
```

Never call a limited run converged.

---

# 7. Priority A — cross-platform process supervision

The current bounded runner is POSIX-oriented. Create one reusable process-supervision utility.

Required:

- wall-time limit;
- global-budget limit;
- memory limit;
- cooperative cancellation;
- bounded forced termination;
- complete child-process-tree cleanup;
- streamed logs;
- atomic summary writes;
- preserved checkpoints;
- no orphan workers;
- nonzero exit code when a requested calculation fails.

POSIX: use an owned process group/session.

Windows: use a tested process-tree mechanism such as a Job Object or equivalent. Do not rely on `os.killpg`.

Add cheap tests for normal completion, timeout, exception, child spawning, cancellation, checkpoint interruption, and memory-limit handling.

---

# 8. Priority B — retained-mode family planning

Add a cheap diagnostic using one candidate eigenspectrum where possible.

Report:

```text
candidate eigenvalues
full-operator eigenpair residuals
gain ordering
branch overlaps
polarization-family relations
retained/omitted boundaries
admissible retained counts
```

Initially evaluate:

```text
2
4
6
8
```

Use a modest candidate bank, initially 10–12.

If it cannot resolve the boundary cheaply, return:

```text
candidate_bank_insufficient
```

Do not treat a finite bank as proof of global completeness.

If supported by the diagnostic, add:

```text
modes_6_complete_family_candidate
```

Six modes are a candidate, not an assumed answer. Keep the family guard active throughout the coupled run. If the boundary later becomes invalid, preserve the failure.

Preserve separate photon states for retained modes. Do not silently transfer a high photon population into a different eigenmode just because ranking changed.

---

# 9. Priority C — runtime optimization without changing physics

Profile first. Measure time in:

```text
optical pump-cycle integration
population/heat replay
thermal assembly
mechanical assembly
eigensolver
projection/interpolation
serialization
```

Optimize measured bottlenecks only.

Investigate exact sparse-factor reuse for unchanged matrices in thermal and mechanical solves. Cache keys must include every matrix-changing parameter.

Cache unchanged geometry work:

```text
tetrahedral kinematics
contact connectivity
integration weights
Cartesian/material interpolation weights
optical transfer functions
static transforms
```

Investigate one shared cycle trajectory for both heat replay and cycle-averaged populations. Keep the old path until numerical equivalence is verified.

Keep the complex non-Hermitian eigensolver. Do not replace it with a Hermitian solver merely for speed.

---

# 10. Checkpoint and resume

Each accepted checkpoint should identify:

```text
source revision
numerical-source fingerprint
configuration fingerprint
mesh fingerprint
runtime/package fingerprint
iteration or physical-time boundary
fields and branch IDs
populations
photon state
heat source
thermal state
mechanical state or reconstruction reference
convergence diagnostics
```

Distinguish:

- **Resume:** exactly compatible problem.
- **Warm start:** changed problem initialized from an explicitly transformed previous state.

A changed mesh or mode count is not a transparent resume.

---

# 11. Initial bounded campaign

At most four expensive attempts:

1. local 2-mode reference;
2. one 6-mode candidate only if boundary diagnosis supports it;
3. one 8-mode reference if needed;
4. one targeted refinement/support/cooling case chosen after profiling.

Do not automatically run the full matrix.

Keep current scientific thresholds:

```text
output/heat relative change <= 1%
peak-temperature change <= 0.05 K
beam-weighted OPD RMS <= 2 nm after piston removal
field fidelity/overlap >= 0.999 where applicable
energy and mechanical residuals must pass
```

Do not loosen them to fit the budget.

Use separate status:

```text
limited_validation_complete
full_campaign_complete
dataset_ready
```

---

# 12. Visualization implementation

The live-visualization addendum is the implementation basis.

The interface must display numerical results, not independent plausible animation.

Keep three explicit modes:

```text
live_reference
interactive_preview
replay
```

30 FPS is a renderer target, not a solver target.

Show separately:

```text
render FPS
solver updates/s
simulated time
state age
simulated-seconds/wall-second
```

Repeated rendering of one accepted state is allowed. Invented physical states are not.

## Fixed-point iteration is not physical time

Add:

```text
time_kind =
    steady_state
    outer_iteration
    physical_transient
    optical_pulse_window
```

Do not animate outer iterations as physical seconds.

---

# 13. Linked visualization views

## A. Optical assembly

For the seeded amplifier, show seed, input SLM, beam-shaping optics, signal relays, repeated encounters with the **same** disk, separate pump path, diagnostics, and the output plane.

Distinguish disk encounters from material traversals.

Only calculated fields may be used for scientific beam slices. A glowing 3D tube may be diagrammatic only.

Do not animate vortex intensity as rotating solely because phase winds.

## B. Incoming field

Display intensity or fluence, phase, pulse energy, average power where meaningful, wavelength, pixel spacing, coordinate frame, plane ID, pulse/time identifier, and normalization.

Do not call every quantity “intensity”.

Provide absolute and shape-normalized views separately.

Mask phase where amplitude is negligible.

## C. Phase modulator

Keep separate:

```text
phi_pattern
phi_correction
phi_requested
phi_applied
```

Use:

```text
phi_requested = wrap_2pi(phi_pattern + phi_correction)
A_after = a_SLM * A_before * exp(i * phi_applied)
```

For the ideal phase-only fixture, `a_SLM = 1` and local intensity immediately across the modulator must remain unchanged.

Do not fabricate latency, quantization, calibration, or response.

Retain the configured 10 Hz command schedule only as a model setting, not measured hardware response.

## D. Temperature

Show disk and plate temperature, front/rear disk surfaces, through-thickness slice, interface region, `|grad T|`, and heat flux.

Use:

```text
q = -k grad(T)
```

with correct mesh/coordinate handling. Use conservative face fluxes for energy accounting.

## E. Deformation and stress

Show front/rear displacement, thickness change, relevant stress, reference geometry, and deformed geometry.

Display-only exaggeration:

```text
x_display = x_reference + exaggeration_factor * u_physical
```

Always display the factor. Never feed exaggerated geometry back into physics.

## F. Cooling

Separate:

```text
bulk crystal heat
coating heat if modeled
crystal -> plate heat
assembly -> coolant heat
thermal storage
coolant inlet/outlet if modeled
cooling capacity if modeled
```

Do not invent cooling-capacity utilization. A fixed coolant temperature is not a finite-capacity chiller model.

## G. Output and correction

Show near field, far/focal field, phase, pulse energy, average power, gain/loss, wavefront error, and target-mode overlap.

Compare cold/reference, heated uncorrected, and heated corrected states at matched conditions.

For an incoherent multimode oscillator state, do **not** display `angle(sum(fields))` as a unique total phase.

Show per-mode phase, modal powers, total intensity, and coherency/polarization diagnostics.

---

# 14. Physics-to-viewer contract

Preserve:

```text
pump transport + populations + signal
    -> heat
    -> crystal/contact/plate temperature
    -> stress/displacement
    -> optical path and gain
    -> output
    -> measurement/controller
    -> future SLM command
```

All disk visits share one physical state.

Do not reset inversion between passes.

Do not create independent deformed disk copies.

Do not double-count surface motion and scalar optical-path corrections.

For the existing rear-coated disk geometry preserve:

```text
Delta_OPD_geom =
    2 * ((1 - n) * u_front_z + n * u_rear_z)
```

Do not blindly apply this to another geometry.

---

# 15. Snapshot contract

Solver workers publish immutable snapshots.

Recommended schema:

```text
schema_version
run_id
configuration_hash
source_hash
state_id

application_mode
fidelity_mode
solver_status

time_kind
t_sim_s
outer_iteration
pulse_id
averaging_interval
t_published_wall

mesh_ids
coordinate_frames
units

optical_state_kind
fields_at_selected_planes
field_normalization
modal_powers

pump_absorption
population_fields

phi_pattern
phi_correction
phi_requested
phi_applied

T_disk
T_plate
grad_T
heat_flux

displacement
stress
optical_path_error

energy_metrics
cooling_metrics

subsolver_timestamps
approximation_flags
sensor_flags
```

Unsupported quantities must be `null` or explicitly unavailable, never fake zero arrays.

Use a bounded display queue, e.g. two accepted snapshots. The viewer may skip stale display frames, never required physical integration events.

---

# 16. UI stack

Initial implementation:

```text
PyVista / VTK
trame
```

Keep viewer dependencies optional for headless tests.

Do not rewrite the working thermoelastic solver in FEniCSx merely for visualization.

Bind the viewer to localhost by default. Do not expose an unauthenticated control server publicly.

---

# 17. Implementation sequence

## Milestone 1 — replay viewer

Load one verified state and display:

```text
pump/input beam
output beam
Ho density
upper-manifold population
heat
disk temperature
plate temperature
front/rear deformation
OPD
mode/polarization data
```

Do not reconstruct scientific arrays from screenshots.

## Milestone 2 — live worker

Connect the same snapshot adapter to a running bounded worker.

Add pause, resume, cancel, single solver step, recording, state age, and iteration/time indicators.

## Milestone 3 — seeded amplifier

Implement generic externally seeded multipass orchestration over one shared crystal state.

If complete hardware topology is missing, keep:

```text
seeded_amplifier_configuration_complete = false
```

but finish the generic API and one verification fixture.

## Milestone 4 — bounded controls

Only expose controls with a defined physical effect.

Examples:

```text
pump power
pump position
seed energy
structured-light mask
phase correction enable/disable
supported coolant condition
contact conductance
selected plane/pass
deformation magnification
display color limits
```

Label restart-required changes.

---

# 18. Required tests

| Test | Expected result |
|---|---|
| Phase-only SLM | Local intensity preserved immediately across SLM |
| Passive propagation | Physical loss accounted correctly |
| Repeated disk encounters | Shared material/population state |
| Pump/heat removal | Solver-driven thermal relaxation |
| Crystal–plate heat | Equal/opposite interface bookkeeping |
| Crystal–plate force | Equal/opposite mechanical bookkeeping |
| Thermal balance | Heat, removal, and storage consistent |
| Uniform free heating | Expansion without invented bowl |
| Display exaggeration | No effect on physical solver |
| Viewer FPS | No change to accepted physics sequence |
| Replay | Clearly labelled, no claim of live solve |
| Incoherent modes | No fabricated total scalar phase |
| Resume | Compatible state resumes; incompatible state rejected/remapped |
| Timeout | Entire owned process tree terminates |
| Memory limit | Clean stop with evidence preserved |
| Family diagnostic | Rejects a retained boundary cutting a family |
| 6-mode candidate | Must independently pass the same guard |

---

# 19. Documentation updates

Improve existing docs rather than creating a contradictory roadmap.

Update:

```text
README.md
docs/STAGE7W.md
docs/STAGE7W_RESULTS.md
live visualization addendum
local execution guide
profiling guide
```

Preserve historical values and source revisions. Append new results rather than replacing old values.

The revised visualization addendum should explicitly cover:

```text
oscillator vs seeded amplifier
replay-first delivery
physical time vs fixed-point iteration
incoherent multimode phase restrictions
runtime and memory budgets
optional viewer dependencies
unsupported controls disabled
numerical vs experimental qualification
```

---

# 20. Suggested AGENTS.md

If no applicable `AGENTS.md` exists, create:

```markdown
# Local agent rules

Read docs/LOCAL_AGENT_REPORT.md, current stage documentation, and the
live-visualization addendum before modifying numerical code.

Reuse existing kernels. Do not weaken physical or numerical acceptance tests.

All numerical execution must use the shared local budget supervisor.
Do not automatically launch full campaigns, remote workflows, large sweeps,
or neural-network training.

Preserve population layouts, normalization, coordinate frames,
polarization conventions, and one shared physical crystal state.

Keep oscillator and seeded-amplifier modes explicit.

Never present fixed-point iterations as physical time, replay as live
calculation, or an incoherent modal mixture as a single coherent phase field.

Record tested revision, settings, runtime, memory, failures, and limits.

Do not claim dataset readiness from passing software tests.

Do not discard user changes, force-push, or merge automatically.
```

---

# 21. Final deliverables

Return:

## Code

- cross-platform bounded worker supervision;
- persistent budget ledger;
- memory limit;
- retained-family boundary diagnostic;
- optional 6-mode case;
- verified performance caches/optimizations;
- replay viewer;
- live-worker snapshot adapter;
- seeded-amplifier orchestration API or explicit incomplete status.

## Evidence

- tests;
- independent audit;
- profiling report;
- before/after runtime;
- peak memory;
- case summaries;
- checkpoints;
- hashes;
- viewer benchmark;
- failure logs.

## Separate status flags

```text
software_tests_passed
limited_numerical_validation_passed
full_campaign_complete
replay_viewer_operational
live_worker_operational
seeded_amplifier_configuration_complete
experimentally_validated
dataset_ready
```

Never collapse these into one “complete” flag.

---

# 22. Stopping rule

The task succeeds if it produces targeted verified improvements and a scientifically honest viewer within the bounded compute budget.

It does not require completing every research-level calculation.

When the budget is exhausted:

1. stop numerical jobs;
2. terminate owned child processes;
3. preserve checkpoints and logs;
4. write the current report;
5. state exactly what remains;
6. keep all scientific tolerances unchanged.

Do not extend the work into multi-day computation.

Do not restart the budget automatically.

Do not loosen scientific criteria merely to get a green result.
