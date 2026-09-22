# Stage 7 — Self-consistent vector hot-cavity / crystal–plate feedback

## Implemented closure and its scope

Stage 7 closes the optical-mode → local Ho populations → heat → crystal/plate
temperature → stress/displacement → Jones operator → optical-mode loop.
The field is a complex `(2, ny, nx)` vector array, not a fitted Gaussian radius.
The cavity eigenproblem is solved on the Cartesian FFT grid. Both crystal
traversals, curved output mirror, specified mirror losses, spatial gain,
thermo-refraction, both displaced crystal faces and ordered photoelastic Jones
operators are retained.

This is an **adiabatic, cycle-averaged spatial-mode closure**. The spatial
mode(s) are frozen during one fast pump-cycle population/photon calculation;
they change on the slow feedback iterations. It is not a carrier-resolved
Maxwell–Bloch solver, a coherent multimode beating simulation, or a dynamic
transverse-mode update on every 1.7 ns cavity round trip. No such claim should be
inferred from a small outer fixed-point residual.

## 7A — Vector field eigenproblem

`vector_cavity.VectorRoundTrip` caches the Stage 4R/6 transfer operator, with the
same reference plane and field convention. It exactly reproduces the existing
scalar and Jones round-trip operators in regression tests.

`solve_vector_eigenfields` uses complex non-Hermitian Arnoldi iteration to find
largest-modulus eigenpairs. Every returned mode is checked against the full-grid
operator, not a reduced Gaussian basis. There is no hidden smoothing, annular
aperture, mode-locking element, phase conjugation or vortex filter. Truly
degenerate *complex* eigenvalue subspaces can be aligned to the preceding field;
equal magnitudes alone are not enough to permit such a rotation. Failure to
converge the eigenproblem produces a nonconverged overall result.

Gaussian/LG modes are initial guesses only. `mode_count=1` is the economical
default. Additional retained modes share the same inversion and compete through
the existing photon/population equations, with incoherent modal intensities.
The chosen finite candidate/mode set is **not a global stability proof**. An LG
initial guess does not imply selection of stable vortex lasing.

## 7B — Physical normalization, populations and losses

`PlaneExchange` integrates a positive bilinear interpolation of the Cartesian
intensity over exact annular/azimuthal control-volume areas. It returns the
pre-normalization quadrature mismatch; large errors are rejected. The final
profile integrates to one, so the existing photon energy and local stimulated
rates retain their physical normalization. Both directional disk visits
contribute equally to the mean-field intensity. The 1 mm thin-disk approximation
uses this transverse profile throughout the depth, while Ho populations and heat
remain resolved at each depth cell.

The existing four-manifold ETU/cross-relaxation dynamics and sequential
picosecond pump-fluence kicks are reused. Full periodic pump cycles, including
period multiplication when detected, are solved before steady heat is used.

Passive round-trip retention is evaluated separately with **zero material gain**.
Its logarithmic loss includes the configured mirrors plus additional
field-dependent diffractive/aperture loss. The extra loss modifies the photon
lifetime, not the population equation or crystal heat source directly. Gain is
not double-counted by absorbing it into an effective negative cavity loss.
The mean-field output-coupler convention is inherited from Stage 4R.

The field's spatial gain screen uses a true cycle average of N7 and N8, not the
post-pump population. An independent replay averages over every adaptive ODE
interval. The difference between the grid operator's log gain and the
mean-field modal gain is recorded as `field_rate_log_gain_mismatch`; it exposes
the effects of remapping and thin-screen averaging rather than silently tuning
one solver to match the other.

## 7C — Heat generation

`sample_cycle_heat` from Stage 5 replays the full detected optical period,
retaining its local first-law ledger, radiation escape, stimulated energy
exchange and actual excitation-storage change. Heat is not a fixed fraction of
incident pump power. Mirror/aperture losses are not automatically heated into the
crystal or plate. No altered fluorescence energies are introduced in Stage 7.

## 7D — Finite crystal–cooling-plate mechanics

`PlateAssembly` calls the original Stage 6 solvers on every outer iteration.
The plate remains a finite solid with its own temperature and deformation, not
an isothermal or rigid boundary. Heat transfer at the bond and at the coolant
boundary remains distinct. Normal/shear interface stiffness, thermal expansion
mismatch, support constraints, stress-free temperature and front preload are all
preserved from `config/stage6_assembly.json`.

Thermal equilibrium is recomputed from the **new** optical heat field. The FEM
then uses temperatures of both bodies, and its displacement/stress fields feed
the optical Jones map. Thermal phase is not applied twice: the Stage 6 hot-disk
operator already contains the Stage 5 dn/dT contribution.

The inherited interface is a linear compliant bond, not unilateral frictional
contact or delamination. Pressure-dependent conductance, plasticity, coating
stress, temperature-dependent spectroscopy and mechanical dynamics remain outside
this implementation.

## 7E — Convergence criteria and saved states

Convergence requires consecutive accepted iterations satisfying independent
thresholds for phase-aligned vector-field change, **unrelaxed** heat-source
change, disk/plate temperature change, disk/plate displacement change, output
power change and passive modal-loss change. Both the optical periodic-state test
and full-grid eigenpair residual must also pass. Small numerical mixing factors
cannot turn a large unrelaxed heat residual into false convergence.

Outputs distinguish `fields_used` (which generated the populations and heat)
from `fields_predicted` (the updated eigenfields). Even after convergence their
remaining difference is retained. An iteration limit, unresolved periodic state
or failed Arnoldi solve is explicitly labeled, saved and never reported as a
steady hot-cavity prediction. Numerical convergence is separate from mesh
convergence and physical model validation.

## Running

From the repository root:

```bash
python -m pip install -e '.[dev,plots]'
python -m pytest -q
python examples/stage7_hot_cavity.py --quick --require-converged --output results/stage7/demo
python examples/stage7_plot.py --result results/stage7/demo

# Higher-resolution configured calculation; exit unsuccessfully unless converged:
python examples/stage7_hot_cavity.py --pump-uJ 1000 --require-converged
```

Configuration: `config/stage7_hot_cavity.json`. It selects the previously reported
**250 mm / 2%** cavity, not the separate 200 mm / 3% default. The disk remains
10 mm in diameter and 1 mm thick. The cooling plate/bond configuration remains
explicit and replaceable. `--modes 2` retains two competing vector eigenbranches.
The `--quick` flag deliberately uses coarse numerical grids; its results are
integration demonstrations, not precision thermal/mechanical predictions.

A separate GitHub workflow runs the new tests and the coupled demonstration and
saves full NPZ states, JSON history and plots as an Actions artifact. Its
`--require-converged` flag makes a failed fixed point fail the demonstration job.
The existing whole-project regression workflow remains unchanged.

## Executed demonstration

The 10 W incident pump example (1 mJ, 10 kHz, 10 ps) converged in six outer
iterations using 128×128 optical pixels and 2×16×4 material cells in (z,r,phi).
It produced 0.7530 W mean output, 0.6720 W heat, 302.358 K peak disk temperature
and 293.675 K peak plate temperature. These are **coarse-grid illustrative
results**, not mesh-refined predictions. The last field residual was 1.343e-4,
heat residual 1.652e-3, and eigenpair residual 4.395e-14. See `results/stage7/` for
source workflow IDs, numerical history and limits.

An initial three-candidate Arnoldi run stopped with an incomplete spectrum.
The checked implementation uses a larger Krylov subspace and the quick case
requests the two leading polarization candidates. It does not convert an
incomplete spectrum into a successful solve. All 162 project tests passed,
including 24 new Stage 7 tests, on the demonstration's source revision.

## Sources

- Rupp, Eichhorn and Kieleck, *Iterative 3D modeling of thermal effects in
  end-pumped continuous-wave Ho3+:YAG lasers*, Applied Physics B 129, 4 (2023),
  DOI 10.1007/s00340-022-07939-z. This supports the iterative vector-optical /
  material framework, **not experimental validation of the present pulsed disk**.
- SciPy `scipy.sparse.linalg.eigs` documentation: complex non-Hermitian Arnoldi
  eigenproblems through a `LinearOperator`; residuals are checked independently.
- Stage 5 and Stage 6 documentation retain heat, elastic, bond and photoelastic
  parameter provenance. No new measured material or coating properties are claimed.
