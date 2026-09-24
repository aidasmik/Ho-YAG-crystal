# Yb:LuAG remaining-corrections audit (2026-09-24)

This is a synthetic engineering model, not an experimentally validated amplifier prediction. The baseline was `434a6c7`. This working tree is uncommitted. The existing Ho:YAG solver was not modified.

| Requested issue | Current result | Remaining cause |
| --- | --- | --- |
| Local material fields and hot regenerative feedback | Implemented local temperature-dependent absorption/emission, periodic pump/inversion/heat/assembly outer iteration, and a scalar thermal/surface phase screen at every disk encounter. Four residuals are reported. The small smoke case converged in 3 iterations. | **Missing implementation:** polarization/photoelasticity, measured concentration-dependent bulk index, full vectorial in-disk propagation, and spatially varying lifetime/conductivity. **Missing calibration:** generic assembly properties. |
| Startup and energy balance | Regenerative recovery recomputes absorption as inversion evolves. Thermal timeline has internal steps independent of plotted times and labels the periodic source approximation. Existing cavity and population photon balances remain explicit. | **Missing implementation:** simultaneous transient population, heat, optical and controller evolution after pump/cooling changes. The thermal startup currently applies periodic heat after the fast excitation transient, so it must not be used to infer sub-millisecond startup. |
| Spatial spectral screen | Ideal relay uses incident-fluence-weighted exponential of local integrated gain, population, density, and temperature. Regenerative spectrum explicitly reports unavailable. | **Missing implementation:** wavelength-dependent regenerative cavity transport and shared-inversion spectral saturation. |
| Doping versus geometry | Radius, thickness, concentration, and chosen assembly property assumption are separate inputs. Tested 11.7/12.0/12.3 at.% at 100/150/200 µm. | **Missing calibration:** material properties at each exact concentration. |
| Optical/thermal convergence | Existing passive-cavity grid tests and new small coupled residuals run. Export requires evidence for independent optical window, pitch, axial, thermal, mechanical, and time refinements. | **Numerical convergence:** no representative multi-beam, multi-axis qualification has been completed; default 96² remains a preview. Export is therefore closed for real app results until evidence is supplied. |
| Five temperature probes | Configurable positions, material regions, footprints, response, sample rate, latency, fixed bias, readout noise, missing status, and timestamps. They sample the solved disk/plate fields. A measured-probe controller uses only delivered valid disk readings; exact-maximum feedback remains an explicitly ideal benchmark. | **Missing calibration:** controller gain and sensor placement/response are assumptions. |
| Output and dataset validity | Browser distinguishes coupled steady from lumped phase. Export keeps exact fields and noisy readings separate, saves command/target/config/version/commit/seeds, and rejects unavailable hot outputs and unresolved convergence. | **Missing calibration:** data are labeled synthetic with unvalidated material assumptions, never experimental ground truth. |

## Validation executed

- `.venv/bin/python -m pytest -q tests/test_ybluag_probes_and_export.py tests/test_ybluag_remaining_coupling.py tests/test_ybluag.py tests/test_ybluag_regenerative.py tests/test_ybluag_cold_sampling.py tests/test_ybluag_pulse_convergence.py` → **51 passed** in 39.29 s.
- `.venv/bin/python -m pytest -q tests/test_stage1.py tests/test_stage5_heat_thermal.py` → **36 passed** in 0.55 s (Ho:YAG regression).
- After the final zero-heat, local-density API, and probe/export edits, `.venv/bin/python -m pytest -q tests/test_ybluag_remaining_coupling.py tests/test_ybluag_probes_and_export.py` → **19 passed** in 0.46 s; the default app calculation plus probe/export subset → **4 passed** in 7.44 s.
- `node --check` on the extracted browser script → **passed**.
- `examples/ybluag_remaining_smoke.py` under `hoyag.local_supervisor.run_bounded`, audit category, 120 s cap → **completed** in 0.863 s, peak RSS 91.4 MB after final controller coupling. See `coupled_timeline_final.log` and `coupled_timeline_final_execution.json`. Pump 0.01 W, 32² optical grid, one optical depth cell, 4×4×1 thermal mesh, one cavity round trip, 0.1 s thermal timeline: output 7.5004128e-10 J, steady peak 20.073923 °C. Last outer residuals: ΔT 0.00758 K, heat 3.74e-5 relative, output 9.65e-7 relative, wavefront 6.97e-5 rad. Timeline used 23 internal steps and emitted 10 measurements across five probes.
- After adding measured-probe feedback, the probe, coupling, and default-app subset → **21 passed** in 9.07 s; the controller-specific missing-probe/biased-probe test → **1 passed** in 1.53 s.
- Final targeted suite, including a manufactured uniform thermo-optic OPD case and an ideal-relay spectral/main-solver comparison → **22 passed** in 1.75 s.
- Final combined Yb:LuAG plus Ho:YAG regression command (the eight test files listed above) → **92 passed** in 42.91 s.
- Manual app-path concentration smoke: 11.7/12.0/12.3 at.% all calculated at identical geometry with outputs 7.51150e-5, 7.50096e-5, and 7.49043e-5 W in a deliberately low-power cold case.
- Larger convergence campaign, hardware calibration, and NN dataset generation: **not run**.

The coupled smoke is deliberately coarse and verifies execution and residual closure only. It does not qualify accuracy of the 96² preview or the thermal assembly.
