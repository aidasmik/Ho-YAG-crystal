# Figure-derived 25 at.% Yb:YAG hot spectra

Source: M. Esmaeilzadeh, H. Roohbakhsh and A. Ghaedzadeh,
"Experimental Study on Temperature Dependence of Absorption and Emission
Properties of Yb:YAG Crystal as a Disk Laser Medium" (2012), Figs. 4 and 6.
PDF copy: https://zenodo.org/records/1334966/files/13958.pdf?download=1 .

`digitize.py` traces the visually distinct **300 K and 450 K** colored curves
from a 200 dpi rendering. Run it with a locally downloaded PDF path. The
result, `endpoint_spectra.csv`, records 1 nm points and whether a matching
curve pixel was observed or an intervening gap was interpolated. The trace
and its byte-identical runtime copy under `src/ybyag/derived_spectra/` are
**figure readouts**, not author numerical records. The original PDF is not
committed here. The 25 at.% sample is 1 mm thick. Figure 6 emission was
calculated by the authors using a reciprocity relation from their absorption
measurement; it is not an independent fluorescence-spectrum measurement.

The figure labels one intermediate temperature as 390 K while the prose/table
mention 380 K, so only unambiguous endpoints were recovered. The published
941 nm absorption peak falls from about 7.5e-21 to 5.49e-21 cm2 and the
1031 nm emission peak from about 2.01e-20 to 1.22e-20 cm2 over 300-450 K.
Those anchors and the traced curves agree at figure precision. Actual
wavelength placement around the sharp 971/1031 nm peaks carries roughly
1-2 nm graphical uncertainty; minimum ordinate uncertainty is stated in
`provenance.json` and is larger at overlap/interpolation points.

`hot_25at_figure_cross_sections_m2` is an opt-in lookup with a strict 25 at.%
and 300-450 K guard. Linear interpolation between the endpoint *figures* is
only exploratory. It is not used by the 5/10/15 at.% amplifier or the NN
dataset, and cannot justify calculations near 250 C (523 K).
