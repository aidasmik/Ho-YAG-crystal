# Bounded local implementation evidence — 23 September 2026

Branch: `local/bounded-validation-viewer`. The unrelated `YbYAG` working tree
was left untouched. No remote computation or full 32-case campaign was started.
The numerical-source fingerprint at the end of the bounded Stage 7W work was
`a7a81953b9aab26e8857c90ec332041ec94246ed871d911ff65485d61b60eb07`;
saved numerical cases retain their own earlier fingerprints below.
The later structured-beam probe source and its evidence are recorded in
`results/structured_beams/summary.json`.

## Software and audit

- Final full suite on the final numerical source: **316 passed** in 98.36 s
  (`regression_final.log`); supervised process elapsed 98.67 s, peak
  process-tree RSS 291.6 MB. An earlier full run also passed 316 tests.
- Independent `audit/deep_check.py`: **19 checks passed**, zero failed
  (`audit.log`); supervised elapsed 9.69 s, peak RSS 129.9 MB.
- A final targeted run after strengthening checkpoint reconstruction checks:
  **35 passed** (`test_checkpoints.py`, `test_stage7w_plan.py`, `test_stage7w.py`).
- POSIX timeout, spawned descendant cleanup, memory ceiling, cancellation,
  persistent budget, and failure tests passed. Windows Job Object support is
  implemented but was not exercised on this Linux host.

## Numerical runs

| Case | Source fingerprint | State SHA-256 | Result | Supervised wall | Peak RSS |
|---|---|---|---|---:|---:|
| Two-mode reference | `9839b6642bb2d2b082c5ed47b80a3b774a26caf6adbe93e2ce62b17abcf09853` | `33a8fdecbc732ca68053e37dfbf86734405f703d8813b211cb042f31ca06b40c` | Converged, 7 iterations | 412.12 s | 950.0 MB |
| Six-mode candidate | `6b2e6f24e2b9035780ff0e94551c660890e1912b5b5b85276fb7f7bcb343d9d1` | `15f8dc5404fcd531ba7a12514982e616549ad5d53678164704866a9e559bd2dd` | Converged, 7 iterations; no split retained boundary | 482.31 s | 1253.7 MB |

Two-mode output/heat were 0.7900341761/0.6547475427 W; six-mode
output/heat were 0.7900870434/0.6547538059 W. The source fingerprints
differ because implementation work continued between launches, so these
figures do not establish strict mode-count convergence. The historical
four-mode boundary is still rejected. The local one-spectrum diagnostic found
2, 6, and 8 admissible within 10 candidates and 4 split; its maximum
full-operator eigenpair residual was `4.92e-10`. Finite-bank admissibility is
not global completeness. See both local case `summary.json` files and
`results/local_stage7w/family_diagnostic.json`.

## Profiling and visualization

The shared-cycle replay fixture was 0.07033 s median versus 0.13930 s for
separate heat/population replays, a 1.98× **component-fixture** speedup.
Heat was identical and maximum population-fraction difference was
`5.24e-12`. This is not a measured full-solver speedup. Hardware/runtime
details and each timing are in `results/local_profile/replay.json`.

Hash-verified replay rendered overview, thermal, and optics views from saved
arrays. The live worker published seven accepted outer-iteration snapshots;
the adapter/controls and localhost trame entry point were smoke-tested. The
final off-screen 960×720 nine-panel benchmark rendered 17 frames in five
seconds, **3.37 FPS**, with 578.4 MB peak process-tree RSS. The 30 FPS target
was not met. No physical-time speed is defined for steady-state outer
iterations; the benchmark had zero solver updates because it replayed a
completed case. See `viewer_benchmark.log` and the saved PNGs.

## Limits and status

The generic externally seeded multipass API and ideal phase-only fixture use
one shared crystal state. Relay/pass topology and hardware response parameters
are not supplied, so its configuration is incomplete. Snapshot replay and
local live reference transport are operational, but an interactive preview
model has not been numerically qualified. Checkpoint format, integrity, and
compatibility rejection have automated tests; full numerical restart
equivalence has not yet been run. The full retained-mode chain, mesh and
operating sweeps, experimental validation, and dataset qualification remain
outstanding. No scientific tolerance was relaxed.
