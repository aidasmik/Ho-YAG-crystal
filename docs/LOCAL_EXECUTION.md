# Bounded local execution

Use Python 3.10 or newer. The reference Stage 7W path is an oscillator model;
`hoyag.seeded_amplifier` is a separate externally seeded encounter API.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[dev,viewer]'
.venv/bin/python examples/stage7w_validation.py plan \
  --output config/stage7w_validation_local.json
.venv/bin/python examples/stage7w_validation.py run \
  --plan config/stage7w_validation_local.json --cases reference \
  --seconds-per-case 900 --output results/local_stage7w --live-snapshots
```

The parent supervisor uses an owned POSIX process group or Windows Job Object,
streams `execution.log`, records `execution_status.json`, and stops the process
tree at wall time, memory, cancellation, or remaining global budget. Its shared
ledger is `.local_runtime/budget.json`, persists across CLI restarts, permits at
most four expensive coupled attempts, and charges crashed attempts when recovered.
The global automated numerical allowance is 7200 s. Coupled cases are capped at
900 s each; profile, regression, audit, and viewer runs have category limits of
180, 600, 300, and 120 s. One expensive worker is allowed at a time. A run that
hits a limit receives `timed_out`, `resource_limit`, or `budget_exhausted`; no
limit status is scientific convergence. The default memory ceiling is the
smaller of 8 GiB and 70% of installed RAM. A lower `--memory-mb` is allowed.

The standard command runs only named cases. Do not use `--all` for the full
Stage 7W campaign. The parent returns nonzero if a requested case fails or
does not converge. Check `summary.json`, `iterations.jsonl`, snapshots, and the
full-plan report. Historical result files are not overwritten.

With `--live-snapshots`, accepted outer iterations are published in
`results/local_stage7w/reference/snapshots`. Their `time_kind` is
`outer_iteration`, with no fabricated physical timestamp. Controls are applied
at the next accepted iteration boundary:

```bash
.venv/bin/python examples/live_control.py results/local_stage7w/reference pause
.venv/bin/python examples/live_control.py results/local_stage7w/reference step
.venv/bin/python examples/live_control.py results/local_stage7w/reference resume
.venv/bin/python examples/live_control.py results/local_stage7w/reference cancel
```

Cancellation can also stop the owned worker mid-iteration. Pause time counts
against the wall-time limit. A paused solver cannot be left indefinitely.

After a completed case, inspect one hot candidate bank with a separate bounded
diagnostic. A six-mode plan is written only if that finite bank supports the
boundary; the coupled family guard still applies on every iteration.

```bash
.venv/bin/python examples/stage7w_family_diagnostic.py run \
  --case-directory results/local_stage7w/reference \
  --output results/local_stage7w/family_diagnostic.json \
  --plan-output config/stage7w_six_candidate.json
```

Replay renders archived numerical arrays; an incomplete or corrupt state is
rejected. Optional PyVista/VTK and trame packages are required only for the
viewer, not for headless solver tests.

```bash
.venv/bin/python examples/replay_viewer.py results/local_stage7w/reference \
  --screenshot results/local_stage7w/replay.png
.venv/bin/python examples/live_viewer.py \
  results/local_stage7w/reference/snapshots --seconds 5 \
  --screenshot results/local_stage7w/live_benchmark.png
.venv/bin/python examples/browser_viewer.py results/local_stage7w/reference
```

Replay is not a live solve. The snapshot adapter can publish live reference
states, but it does not turn a slow fixed-point solve into physical real time.
Current controls are pause, resume, one outer step, and cancel; physical knobs
remain disabled until their solver effects and reset semantics are defined.
The browser server binds to `127.0.0.1`. Add `--live` to its command only while
a supervised Stage 7W worker is publishing snapshots. A completed-case replay
cannot accept solver controls. Numbered checkpoints are written after accepted
nonfinal outer iterations; an exact source, configuration, mesh, and runtime
match is required for automatic resume. Changed problems require an explicit
warm-start design. Numerical restart equivalence has not yet been exercised.

The local implementation does not automatically launch DFT, CFD, FDTD, long
parameter sweeps, remote compute, or training. A memory or wall-time limit does
not relax the existing numerical acceptance thresholds.

## Structured seed probes with nonuniform Ho

```bash
.venv/bin/python examples/structured_beam_gallery.py
.venv/bin/python examples/structured_beam_gallery.py \
  --density-seed 23 --cluster-count 30 --cluster-contrast 0.20 \
  --phase-mask vortex+1 \
  --output-directory results/structured_beams/seed23_vortex
```

This generates `results/structured_beams/input_output_beams.png`,
`beam_side_profiles.png`, `beam_on_ho_density.png`, `ho_density.png`,
`phase_mask.png`, `fields.npz`, and `summary.json`. Gaussian, LG(0,+1), LG(0,+2), HG(1,1), Bessel-Gaussian needle,
and order-8 super-Gaussian flattop complex inputs each have 1 W integrated
power. The
default Ho map contains 24 reproducible, randomly placed rich and poor
Gaussian clusters, with transverse radii from 0.20 to 1.25 mm, normalized to
a 1.52e26 ions/m³ active-disk mean. Change `--density-seed`, cluster count,
contrast, and radius bounds to select another declared map. `--phase-mask`
accepts `none`, `vortex+1`, `vortex-1`, `vortex+2`, `defocus`, `astigmatic`,
and `axicon`; `--phase-strength-rad` sets the non-vortex mask scale. The
requested and applied ideal phase arrays are saved separately. Phase-only
application preserves local irradiance at the SLM plane; beam reshaping occurs
after propagation. The archived Stage 7W population fractions are interpolated
to this declared density and frozen during one weak signal traversal. Output is
shown after 0.25 m of free-space propagation; phase excludes the arbitrary
global offset and masks low-intensity pixels. The plots are numerical probes,
not a new self-consistent pumped operating point or complete multipass device.
The `--solver-mode weak_probe` default uses the archived fractions. Set
`--solver-mode full_seeded_modal` to rerun the periodic four-manifold modal
pump/saturation cycle, cycle heat, bonded cooling-plate/thermoelastic solve,
and photoelastic Jones screen for the selected Ho map before propagating the
six fixed external seed fields. This mode has no cavity eigenfield update and
is subject to the bounded local supervisor budget. If the four coupled-run
slots are exhausted, the calculator offers an explicit action to archive the
old ledger and start a new bounded session.
Use `--help` for all density, phase-mask, and output-plane parameters.
Open `results/structured_beams/index.html` in a browser to switch among the
seven precomputed ideal-mask results for density seed 17. This selector changes
archived plots; it does not launch or imply a live solver run.
Use `--plots-only` for a compact plot/summary output without `fields.npz`.

To use the interactive calculator, start the local HTTP app and open its URL:

```bash
.venv/bin/python examples/structured_beam_app.py
# http://127.0.0.1:8780/results/structured_beams/index.html
```

The form accepts the phase mask, mask strength, Ho cluster seed/count/contrast,
cluster radius bounds, and propagation distance. **Calculate** creates a new
run directory with `input_output_beams.png`, `ho_density.png`,
`phase_mask.png`, and `summary.json`.
