# Structured seed probes through nonuniform Ho:YAG

![Computed input and output irradiance and relative phase](input_output_beams.png)

![Imposed Ho concentration in the disk](ho_density.png)

All four inputs have 1 W integrated power at 2.0903 µm. The output plane is
one 1-mm Ho:YAG traversal followed by 0.25 m of free-space propagation.
The input/output irradiance panels use W/m²; phase is masked below 1% of
peak local irradiance and has an arbitrary global phase removed. The helical
LG modes have winding charges +1 and +2; their intensity is not animated as
rotating.

| Seed shape | Output power | Disk power gain |
|---|---:|---:|
| Gaussian TEM00 | 1.0137 W | 1.0137× |
| Helical LG(0,+1) | 1.0103 W | 1.0103× |
| Double helix LG(0,+2) | 1.0043 W | 1.0043× |
| Hermite–Gaussian HG(1,1) | 1.0043 W | 1.0043× |

The imposed Ho concentration has an active-disk mean of 1.52e26 ions/m³
and ranges from 1.406e26 to 1.986e26 ions/m³. It combines an axial gradient
with an off-axis dopant-rich region. The saved Stage 7W population fractions
were interpolated and frozen during these weak seeded probes. Pump, heat,
mechanics, saturation, and multipass hardware were not recomputed for the
new concentration. This gallery is a controlled optical illustration, not
a new self-consistent laser operating point.

Rebuild with `python examples/structured_beam_gallery.py`. Exact settings,
source/state hashes, and power values are in `summary.json`; complex fields
and the 3-D Ho map are in `fields.npz`. The full regression suite passed
318 tests after this addition; see `regression.log`.
