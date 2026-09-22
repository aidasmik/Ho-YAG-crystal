# Stage 5 executed validation and thermal-resonator example

The new Stage 5 source and tests are in `src/hoyag/{heat,thermal,thermal_optics,thermal_resonator}.py` and `tests/test_stage5*.py`.

## Selected example

The 250 mm / 2% output-coupler cavity is used, not the 200 mm / 3% default. The disk is 10 mm diameter and 1 mm thick. Pump: 1 mJ, 10 ps, 10 kHz, Gaussian radius 0.5 mm. Rear bath 293.15 K; rear contact conductance 100000 W/(m^2 K); front and rim insulated. Bulk conductivity 14 W/(m K), dn/dT 9.1e-6 K^-1. Coating absorption heat is not inferred from mirror reflectivity.

`summary.json` is the compact 96 radial cell / 16 depth cell result; `mesh_convergence.csv` contains the three executed refinements. All are axisymmetric selected-Gaussian runs. The thermal solver itself also supports azimuthally asymmetric 3-D sources, tested separately. The 96x16 calculation was warm-started from the interpolated converged 64x12 population/photon state, not a fresh ground-state startup. Warm starts do not establish convergence; the full optical-period and thermal-feedback checks were rerun.

| Quantity | Refined model result |
|---|---:|
| Absorbed pump | 2.22025 W |
| Bulk heat | 0.634493 W |
| Net stimulated transfer from ions | 1.06129 W |
| Spontaneous radiative power from ions | 0.524465 W |
| Residual ion-storage change | 0.0000251 W |
| Output-coupler power | 0.833766 W |
| Maximum cell temperature | 301.762 K (28.612 C) |
| Maximum rise above the bath | 8.612 K |
| Single-pass OPD peak-to-valley | 62.114 nm |
| Fitted single-pass thermal focal length | 6.035 m |
| Round-trip OPD phase range | approximately 0.373 rad |

The lens fit is weighted over radius <= 0.612 mm and has 0.477 nm weighted RMS OPD residual. It is not a globally exact quadratic lens. Full supplied OPD can be used in the FFT round-trip operator; the outer oscillator iteration uses the parabolic fit only.

64x12 -> 96x16 changes output by about 0.145%, heat by 0.105%, and maximum temperature by 0.00262 K. The refined thermal heat-removal residual is about 3.1e-14 relative. The independent, volume-weighted local optical/ion/heat balance residual is 1.9e-6 of incident pump energy.

A weak LG1 field passed through the Gaussian-heated cavity retains winding +1; its one-round-trip overlap with the cold-cavity result is 0.999932. This is a probe diagnostic, not a self-consistent vortex-laser calculation.

## Reproduction

```
python -m pip install -e '.[dev,plots]'
python -m pytest tests/test_stage5_heat_thermal.py tests/test_stage5_resonator.py -q
python examples/stage5_thermal_demo.py --nr 96 --nz 16 --iterations 10 --output results/stage5/generated
python examples/stage5_plot.py --result results/stage5/generated
```

The conversation results bundle contains all three final summaries/states, the seven plots, contact-conductance sensitivity and fixed-source thermal warm-up data. No placeholder plots were substituted. The warm-up plot switches on a fixed cycle-averaged source; it is not a fully coupled laser startup/thermal trajectory.

The heat-source and cooling predictions remain conditional on unweighted legacy manifold-centroid photon energies, unknown actual rear contact, fixed room-temperature spectroscopy, and the short-pump sequential-return approximation. Thermal expansion, bulging, photoelasticity, radiation transport and full nonparabolic cavity-mode competition remain outside this stage. See `docs/STAGE5.md`.
