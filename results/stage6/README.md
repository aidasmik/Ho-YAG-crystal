# Stage 6 executed assembly example

These are actual locally executed frozen-source computations using archived Stage 5 heat, not a new laser-power prediction. The archived field deposited 0.6336706994 W in the 10 mm x 1 mm disk. The previous bath-only temperature was NOT reused: heat diffusion was re-solved with a finite copper plate and two distinct thermal resistances.

Source: conversation attachment HoYAG_Stage5_results.zip, results/stage5/case_10W/thermal_state.npz. Its (z,phi,r) heat array was transposed to current (z,r,phi) order, and monotonic edges reconstructed from uniformly symmetric cell centers. This archived run is distinct from the subsequently revised default Stage 5 GitHub result. Converted NPZ SHA-256: f601ac99dc4d69eeaf46e7ff82451ac820b608e8ef755461258b6190360016df.

Configuration: 20 mm diameter x 3 mm C10100 copper plate; 293.15 K coolant; coolant h=1e4 W/(m^2 K); contact h=1e5 W/(m^2 K); kn=1e14 Pa/m and kt=1e13 Pa/m bonded interface; clamped plate underside. All mounting values are assumptions.

The refined run has 9422 nodes and 46656 linear tetrahedra. Results: peak crystal 302.139 K; peak plate 293.706 K; peak crystal von Mises stress 9.15 MPa; central-1-mm-radius front displacement PV 28.5 nm; mean reflected OPD PV 158.3 nm, including 91.8 nm thermal PV and 69.7 nm surface-geometry PV (individual PV values do not generally add). Piston-removed Gaussian-weighted mean OPD RMS is 23.27 nm. No tilt or defocus was removed. Crossed-polarizer fraction is of order 1e-5, dependent on provisional host photoelastic coefficients.

The two mesh levels in mesh_comparison.csv changed mean OPD PV by about 1.7%, peak von Mises by about 1.4%, and front-surface PV by about 10%. Polarization conversion changed by about 20%; those small values are not a converged precision prediction. This is a refinement check, not complete asymptotic mechanical convergence. Contact-edge peak stresses should not be used as bond/fracture strength predictions.

29 isolated new tests passed locally; the additional cavity-operator regression and the inherited suite are intended to run together in GitHub CI. The local Stage 5 compatibility harness is NOT included in the update. CI, not a reconstructed harness, is authoritative for repository integration.

Run the example with your current Stage 5 state using examples/stage6_assembly.py. Full NPZ fields and PNG/SVG plots accompany the conversation results bundle; the repository stores the code, summaries and provenance.
