# Stage 5 executed thermal example

This is a preliminary calculation, not a temperature measurement or a calibrated
prediction. It uses the 250 mm air-gap / 2% output-coupler configuration,
10 mm diameter x 1 mm Ho:YAG disk, 10 ps / 10 kHz pump at 1 mJ (10 W),
0.5 mm pump waist, and the current Stage 4R oscillator equations.

## Assumed thermal data

Rear contact h=100,000 W/(m^2 K), sink 293.15 K, adiabatic front/rim;
k=14 W/(m K), rho=4560 kg/m^3, Cp=680 J/(kg K), dn/dT=9.1e-6 /K.
Fluorescence photon energies use provisional unweighted Stark centroids.
All non-cavity fluorescence escapes; coating absorption is not included.

## Executed result

The 96 x 96 x 12 thermal grid and 36 radial x 4 axial optical quadrature converged
in nine outer iterations. The reduced feedback updates the Gaussian mode from
the weighted quadratic OPD; it is not a full multimode thermal wavefront solve.

| Quantity | Result |
|---|---:|
| Absorbed pump | 2.22019 W |
| Bulk heat | 0.63397 W |
| Escaping fluorescence | 0.52370 W |
| Output coupler power | 0.83476 W |
| Other cavity optical losses | 0.22778 W |
| Maximum temperature | 301.61184 K |
| Maximum rise over sink | 8.46184 K |
| Single-pass fitted thermal-lens power | 0.15632 m^-1 |
| Single-pass fitted focal length | 6.397 m |
| Weighted nonquadratic OPD RMS | 0.757 nm |
| Optical/thermal energy closure, relative to absorbed pump | 5.0e-7 |

The corresponding rerun with zero thermal lens gave 0.8347586 W output. The
calculated thermal-feedback change is negligible at the present tolerances;
these numbers do not establish a physically meaningful several-microwatt shift.
The older 0.808 W selected-mode reference uses a different implementation and
must not be treated as the cold member of this thermal comparison.

## Refinement

At the SAME optical heat source, thermal meshes 64x64x8, 96x96x12, 128x128x16,
and 192x192x24 gave maximum rises 8.39623, 8.46184, 8.47904 and 8.48758 K.
This checks thermal discretization, not uncertainty in the source spectroscopy,
cooling contact, pump/laser modes, or an independent optical mesh study.
The 96-to-192 transverse refinement changes peak rise by about 0.30% and fitted
lens power by about 0.93%, using fixed cold-mode fitting weights.

`summary.csv` and `thermal_refinement.json` contain compact numerical results.
Run `examples/stage5_demo.py`, then `examples/stage5_refine.py` and
`examples/stage5_plot.py` to produce raw fields, full energy budgets and plots.
The conversation results bundle includes the executed fields and figures.

The full repository CI also runs the thermal-feedback example on the real
repository sources and checks convergence and energy closure. Check its log
for that independent integration run; local focused tests alone do not
establish the entire inherited regression suite.
