# Executed Stage 7 coupled demonstration

Source revision: `6e4bc1aa3dc22b6bccbfe5608608f8add70a1614`.

The full project regression suite passed **162 tests in 81.42 s**, including 24 new Stage 7 tests. The separate demonstration job executes the actual repository solvers and uses `--require-converged`; a successful job requires numerical convergence, not merely successful file generation.

- Full regression run: https://github.com/aidasmik/Ho-YAG-crystal/actions/runs/35770443672
- Coupled demonstration run: https://github.com/aidasmik/Ho-YAG-crystal/actions/runs/35770443574
- Full arrays and PNG/SVG plots: https://github.com/aidasmik/Ho-YAG-crystal/actions/runs/35770443574/artifacts/10713782218
- Artifact SHA-256: `53d1cae3c5a7f2ba3d2f05c9f1f2929d3ec757571cfb366eac9c68da472daec3`.

The committed CSV and JSON values are transcribed from completed job 106890502348. The full artifact includes the fields used/predicted, mean populations, raw and relaxed heat, disk/plate temperatures, interface heat flux, Jones matrices, optical history/waveform and plots. No missing field arrays have been reconstructed from scalars.

## Conditions

10 mm diameter × 1 mm Ho:YAG disk; 20 mm diameter × 3 mm copper plate; 293.15 K coolant; h_contact=1e5 and h_coolant=1e4 W/(m² K); compliant bond with kn=1e14 and kt=1e13 Pa/m; plate underside clamped. Mirror/source/contact values remain assumptions. Cavity air gap 250 mm, output-coupler radius 500 mm, transmission 2%, rear signal reflectivity 99.95%, pump reflectivity 99.5%, other signal loss 0.5% per round trip. Pump 1 mJ, 10 kHz, 10 ps, waist radius 0.5 mm.

## Results and convergence

The one-retained-mode demonstration converged in six outer iterations on a 128×128 optical grid and a (z,r,phi) material mesh of 2×16×4 cells. Mechanical mesh: nr=4, outer_rings=2, ntheta=16, nz_disk=nz_plate=2; thermal plate depth uses 6 cells. These are coarse integration-test grids, not a precision mesh study.

Final cycle-averaged laser output: 0.7530127884 W. Absorbed pump: 2.2057198200 W. Deposited bulk heat: 0.6719956084 W. The relaxed heat driving the final assembly was 0.6715876755 W; this small residual is retained, not concealed. Peak crystal temperature: 302.3580208 K. Peak plate temperature: 293.6748733 K.

Final field fixed-point residual 1.343e-4; unrelaxed heat residual 1.652e-3; temperature change 6.681e-3 K; maximum displacement change 5.674e-11 m; output relative change 2.614e-4. Two consecutive outer iterations satisfied every configured tolerance. The last eigenpair residual was 4.395e-14. The thermal balance error was 6.18e-14 W and the local optical/population energy-ledger L1 error was 7.54e-6 relative to incident pump.

The field/rate-equation round-trip logarithmic gain mismatch diagnostic was 3.746e-4. This exposes the thin-screen/mean-field/remapping approximation. It is not silently fitted away. Whole-cycle population replay drift was 2.07e-6 and stored-ion power change 0.000346 W; periodicity is numerical, not assumed exact.

## Scope

No full grid-refinement, multimode convergence or nonlinear stability study was executed here. The selected-mode result is not a stable-vortex claim. Temperature-dependent spectroscopy, coherent mode beating, dynamic field changes within a pump cycle, contact opening/friction/plasticity and coating heating remain omitted. See `docs/STAGE7.md`.

The first validation-branch attempt correctly stopped at an incomplete three-candidate Arnoldi spectrum. The final implementation enlarged the Krylov subspace, and the quick demonstration resolves the two leading polarization candidates. It still rejects incomplete spectra rather than treating a single returned accurate eigenpair as a complete solve.
