# Stage 7V evidence

The runner distinguishes numerical integration tests, fixed-source assembly
refinement and fully coupled hot-cavity qualification. Exact executed case
statuses are in the dated reports and Actions artifacts; missing cases are not
passes. Passing tests is not a declaration of experimental or dataset readiness.

## Initial local integration evidence (23 September 2026)

The exact complete source snapshot was obtained from Actions run 35832521639,
artifact 10737902453, based on main ba8c9de. The prepared Stage 7V layer initially
passed 43 integration/metric tests with no skips, and the complete repository
passed 262 tests. The subsequent extension adds 8-mode and LG2 plan coverage,
bounded execution and additional gatekeeping tests; use the final CI count for
the current tree.

A fixed-source mechanical extension used the audited heat/populations from
artifact 10718200278 (API 0.8 correction run 35780402281). OPD difference RMS:

| Pair | RMS difference | 2 nm target |
|---|---:|---|
| Original coarse to mechanical_1 | 2.578212 nm | Fail |
| mechanical_1 to mechanical_2 | 1.619859 nm | Pass |
| mechanical_2 to mechanical_3 | 0.794563 nm | Pass |

Thus the two finest consecutive mechanical comparisons pass for that fixed
archived source; the old coarse failure remains visible. This is not a fully
coupled qualification or a fracture/contact-strength prediction.

One local **two-retained-mode coupled case** completed in 321.38 seconds and eight
outer iterations on a 256-square optical grid, 24x4x12 material grid and the
configured finite plate/bond assembly. Its calculated output was 0.790025259 W,
absorbed pump 2.210988658 W, heat 0.654731779 W, peak crystal 302.3318068 K and peak
plate 293.7190146 K. Modal output powers were 0.394729703 and 0.395295556 W. These
are one operating-point result, not a mode-count or grid-convergence conclusion.
That run preceded a reporting-only correction requiring all initial-guess
comparisons; keep its source fingerprint separate from later distributed runs.

An earlier 240-second timing attempt on the one-mode stricter reference reached
three outer iterations but was interrupted before convergence. It must not be
used as a steady-state prediction.

The full distributed campaign executes the current plan separately and records
all final statuses. A resource/time-limited attempt is explicitly unqualified.
