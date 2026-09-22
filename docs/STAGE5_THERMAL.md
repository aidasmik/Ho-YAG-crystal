# Stage 5 — Heat, 3-D temperature, and thermal optical feedback

## Scope and data provenance

Implemented 5A (heat accounting), 5B (steady/transient disk conduction), 5C
(thermo-optic OPD), and 5D (full phase insertion into the FFT cavity plus a
reduced axisymmetric Gaussian thermal-feedback loop). No previous Stage 0–4R
solver is replaced. The 10 mm diameter, 1 mm thick host remains unchanged.

**Physical uncertainty is not a numerical convergence error.** Default manifold
and fluorescence energies are the *unweighted legacy Stark centroids already
in the repository*, not measured branch-weighted emission spectra. They are
explicitly replaceable through `ManifoldEnergies`. The default heat sink contact
is an assumption. Temperature, lens strength, and output must not be presented
as calibrated predictions of an actual device.

## 5A: First-law heat source

With net absorbed pump positive and net stimulated signal extraction positive:

    Q = Ppump_abs - Psignal_ext - Pspontaneous - dU_ions/dt   [W/m^3]
    U_ions = sum_i E_i N_i                                 [J/m^3]

The storage term is essential for picosecond excitation, startup and pulse-train
transients. Absorbed pump energy held as inversion is not immediate heat.

`heat_rates` also calculates independent process-resolved terms: multiphonon,
ETU, cross-relaxation, pump/signal quantum defects and spontaneous spectral
relaxation. Their sum is compared with the first-law expression. **They are not
added on top of that expression**, which would double count heat. Signed cooling
terms are allowed; a phonon-assisted transition can absorb lattice energy.

Every spontaneous branch uses its upper manifold's own lifetime (tau5, tau6 or
tau7), not tau7 for every branch. The tiny spontaneous fraction entering modeled
cavity modes is distinct from escaping fluorescence and uses the actual signal
photon energy. All other fluorescence is assumed to escape; radiation trapping
is not modeled.

`measure_heat_cycle` reuses Stage 4R's pump fluence kick and BDF photon/population
solver. Six-point Gaussian quadrature over **every adaptive integration
interval** measures heat during narrow laser bursts. `cycle_averaged_heat`
averages over the detected one-, two-, or higher-pump-period optical cycle.
Nonperiodic runs cannot silently be labeled steady state.

The global check is:

    Ppump_abs = Qbulk + Pfluorescence_escape + Pcavity_all_losses
                + dU_ions/dt + dU_cavity_photons/dt.

Output-coupler transmission, mirror losses and relay losses are not automatically
bulk crystal heating. Coating/substrate absorption needs a separate specified
surface heat source. The first-law routines also support supplied single-pulse
state changes in J/m^3.

## 5B: Disk heat flow

`DiskHeatSolver` solves

    rho Cp dT/dt = div(k grad(T)) + Q.

The finite-volume unknowns are temperatures at Cartesian host-cell centers.
The host mask is independent of Ho density: **undoped YAG conducts heat**.
Internal face conductances use harmonic thermal resistance; opposing fluxes
cancel. The circular rim is a staircase approximation, not an exact curved FEM
boundary. Refinement checks quantify that discretization.

Coordinate convention: front face z=0, rear HR/cooled face z=1 mm. Default:

- sink 293.15 K;
- rear contact h=100,000 W/(m^2 K), an assumed value;
- front and rim adiabatic;
- k=14 W/(m K), rho=4560 kg/m^3, Cp=680 J/(kg K).

Surface h=0 means insulated; h=+infinity imposes a face Dirichlet temperature.
For finite h the center-to-face resistance dz/(2k) is in series with 1/h.
These are per-area W/(m^2 K), **not** the old rod's lumped W/K coefficients.

Steady state uses sparse preconditioned conjugate gradients. `step` uses
backward Euler with exact discrete energy bookkeeping. `deposit_energy` applies
an actual deposited-heat impulse. There is no picosecond thermal timestep.
The example uses cycle-averaged heating, not instantaneous pump absorption.

Scalar or spatially varying positive k, rho and Cp are supported. They are fixed
during each solve; no unverified Cp(T)/k(T) law or spectroscopy is silently used.

## 5C: Thermal phase and lens sign

    OPD_1pass(x,y) = sum_z [(dn/dT)(T-Tref) dz]
    phase_1pass = 2 pi OPD_1pass / lambda.

The baseline dn/dT is 9.1e-6 K^-1. It remains a Stage 0 approximation, not a newly
measured 2-um temperature curve. End-face bulging and photoelasticity are absent.
The lens fit includes piston, tip/tilt and the full quadratic Hessian. Converging
lens power is **minus** that Hessian; OPD=-r^2/(2f) gives positive 1/f.
Nonquadratic residual OPD is retained and reported separately.

## 5D: Two levels of feedback

1. `thermal_cavity_roundtrip` applies the **single-pass** OPD once on the outward
   disk traversal and once on return. A piston acquires twice its one-pass phase
   in the return field; the output at the OC has crossed only once. No mirror
   conjugation or factor-of-four phase mistake is introduced. This operator
   accepts arbitrary 2-D OPD and complex structured fields.
2. `thermal_feedback` iterates periodic oscillator -> energy-resolved heat ->
   conservative radial-to-3-D source transfer -> temperature -> weighted lens
   fit -> ABCD Gaussian mode -> re-evaluated saturated oscillator. The feedback
   is under-relaxed and requires optical/thermal convergence. Both disk lens
   traversals enter the round-trip ABCD matrix. An unstable target cavity is
   reported rather than assigned a fictitious Gaussian solution.

The reduced feedback uses the axisymmetric quadratic part only. It does not
claim coherent nonquadratic spatial-mode competition or spontaneous vortex
selection. Full phase maps can be applied by the FFT operator for such future
studies. Temperature-dependent cross sections, stress, coherent reflected-pump
pulse overlap and a self-consistent thermal pump wavefront remain excluded.

## Run

    pip install -e '.[dev,plots]'
    pytest -q
    python examples/stage5_demo.py --pump-uJ 1000
    python examples/stage5_plot.py results/stage5/generated
    python examples/stage5_refine.py results/stage5/generated

The demo defaults to the stored **250 mm / 2% OC reference geometry**. Use
`--cavity-config config/thin_disk_resonator.json` for the newer 200 mm / 3% model.
Both remain in the repository; their results are not interchangeable.

Outputs include numerical temperature/source/OPD arrays, energy budgets,
convergence histories, interpolation correction factors and all assumptions.
`stage5_refine.py` refines the thermal grid at fixed optical heat source; it is
not a claim of complete source/optical/multimode convergence.

## Validation

Tests include radiative/nonradiative and pump/signal energy limits, signed ETU,
energy-zero invariance, branch lifetimes, fluorescence bookkeeping, steady 1-D
slab solutions with ideal/finite rear contact, second-order axial convergence,
transient thermal energy conservation, cooling and steady-state approach,
off-axis 3-D heating, harmonic k transport, correct single/double-pass OPD,
astigmatic lens fitting, cold/hot ABCD–FFT equivalence, conservative source
transfer and the oscillator's whole-system energy budget.

## Sources

- Rupp, Eichhorn, Kieleck, Applied Physics B 129, 4 (2023),
  DOI 10.1007/s00340-022-07939-z: four-manifold transitions and the optical/thermal
  modeling framework, especially Eqs. 4–9. The transient storage term above is
  an explicit first-law extension to the CW heat formulation, not a claim of
  validation by that paper's CW experiment. Typesetting in Eq. 8 should not be
  copied as a universal tau7 lifetime, nor Eq. 16 as multiplication of n0 and
  dn/dT; here delta n=(dn/dT)*delta T and n=n0+delta n.
- `data/ho_yag_stark_levels.json` and `config/hoyag_stage0_parameters.json`:
  legacy energy bookkeeping and baseline material properties.
- SciPy documentation: solve_ivp/BDF and sparse.linalg.cg (rtol API requires
  SciPy >=1.12). Analytic validation and discrete conservation checks are the
  primary numerical acceptance tests, not agreement with a fabricated dataset.
