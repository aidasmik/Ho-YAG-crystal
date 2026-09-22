# Executed Stage 4R results

`summary.csv` records actual local numerical runs, not analytical placeholders.

The four pump-sweep cases and the competing-mode run started from ground-state Ho populations. The refined 48-radial-site / 8-slice case was warm-started by interpolating the converged 36-site / 4-slice 1 mJ state. `cycles_simulated` for the refined case counts additional refinement cycles, not a second ground-state startup.

`pump100` and `pump300` have only spontaneous-background output and are below the simulated lasing threshold. `pump600` converged to a four-pump-cycle pattern; `pump1000` and the competing-mode run converged to a two-cycle pattern. Average powers are averaged over the detected period. The sole-LG1 case did not satisfy the periodic convergence criterion; its number is only a final-20-cycle transient average, not a stable vortex-laser prediction.

For the model scope and assumptions see `docs/STAGE4R_RESONATOR.md`. Raw NPZ populations, per-cycle CSVs and adaptive-time waveform CSVs are reproducible with `examples/thin_disk_resonator.py`. The conversation results bundle also contains those arrays and plots.

Validation covers cold-cavity stability, LG eigenmode reproduction including Gouy phase, passive ring-down, output coupling, two disk gain traversals per round trip, pump photon balance, weak absorption, saturation limits, stimulated optical/population energy exchange, mode degeneracy, analytic Jacobian checks, and a transient smoke test. These checks do not validate unknown coating losses or omitted thermal physics.
