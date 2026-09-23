# Structured seed probes through nonuniform Ho:YAG

[Open the phase-mask selector](index.html) for seven precomputed mask choices.

For custom calculations, start `examples/structured_beam_app.py` from the
repository root and open the same page through
`http://127.0.0.1:8780/results/structured_beams/index.html`. The form generates
a new seeded clustered Ho map and applies the selected ideal phase mask when
you press **Calculate**. Results are saved under `runs/<run-id>/`.

![Computed input and output irradiance and relative phase](input_output_beams.png)

![Imposed Ho concentration in the disk](ho_density.png)

![Ideal SLM phase and unchanged immediate irradiance](phase_mask.png)

The [vortex+1 example](vortex_plus1/input_output_beams.png) shows how a
selected mask changes the propagated output. Its [applied phase](vortex_plus1/phase_mask.png)
is saved separately.

All four inputs have 1 W integrated power at 2.0903 µm. The output plane is
one 1-mm Ho:YAG traversal followed by 0.25 m of free-space propagation.
The input/output irradiance panels use W/m²; phase is masked below 1% of
peak local irradiance and has an arbitrary global phase removed. The helical
LG modes have winding charges +1 and +2; their intensity is not animated as
rotating.

| Seed shape | Output power | Disk power gain |
|---|---:|---:|
| Gaussian TEM00 | 1.0157 W | 1.0157× |
| Helical LG(0,+1) | 1.0120 W | 1.0120× |
| Double helix LG(0,+2) | 1.0049 W | 1.0049× |
| Hermite–Gaussian HG(1,1) | 1.0048 W | 1.0048× |

The imposed Ho concentration has an active-disk mean of 1.52e26 ions/m³
and ranges from 0.865e26 to 2.600e26 ions/m³ for seed 17. It contains
24 seeded Ho-rich and Ho-poor clusters with different transverse and axial
sizes. The saved Stage 7W population fractions
were interpolated and frozen during these weak seeded probes. Pump, heat,
mechanics, saturation, and multipass hardware were not recomputed for the
new concentration. This gallery is a controlled optical illustration, not
a new self-consistent laser operating point.

Rebuild with `python examples/structured_beam_gallery.py`. Select an ideal
phase mask using `--phase-mask none|vortex+1|vortex-1|vortex+2|defocus|astigmatic|axicon`.
The mask does not change irradiance immediately across the ideal SLM.
Exact settings,
source/state hashes, and power values are in `summary.json`; complex fields
and the 3-D Ho map are in the default `fields.npz`. The `vortex+1` example
also includes its complex fields; the other precomputed masks use
`--plots-only` and retain plots and summaries. Rerun a selected mask without
that flag to save its field arrays. The full regression suite passed
**326 tests** after this addition; see `regression_final.log`.
