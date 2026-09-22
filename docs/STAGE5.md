# Stage 5 — Heat, circular-disk temperature and thermo-refractive cavity coupling

## Implemented scope

Stage 5A–D adds a local heat ledger, a conservative circular-disk thermal
solver, optical-path maps, and thermal phase on both resonator disk traversals.
An optional outer iteration updates the prescribed Gaussian cavity mode using a
fitted thermal lens, then recalculates population saturation and heating.

**Two distinct optical fidelity levels must not be conflated:**

1. `thermal_cavity_roundtrip` applies the full supplied two-dimensional OPD and
   gain maps to the existing FFT round-trip field operator. It retains arbitrary
   phase aberrations and supports Gaussian/HG/LG fields.
2. `run_thermal_resonator` updates a selected axisymmetric Gaussian mode through
   a weighted *parabolic fit* to the thermal OPD. This is useful reduced-order
   thermal feedback, not a self-consistent three-dimensional Fox–Li laser, nor
   proof of spontaneous vortex selection.

No existing source file or historical simulation result is overwritten.

## 5A: heat accounting

For the existing population ordering `(I5,I6,I7,I8)`, let

    U_ion = sum_i N_i E_i                       [J/m^3].

The local first law is

    Q = P_pump,net - P_signal,net - dU_ion/dt - P_spontaneous + Q_background.

All terms are in W/m^3. Optical transfers use the local net stimulated
absorption/emission, including signal reabsorption. `four_level_rhs` supplies the
full population derivative, so multiphonon relaxation, ETU and cross-relaxation
are accounted for without adding another empirical heat fraction. Endothermic
individual reactions can give negative instantaneous Q; heat is not clipped.

The spontaneous term uses each branch's **own upper-manifold lifetime**:

    P_spontaneous = sum_(i->j) N_i beta_ij <h nu_ij> / tau_i.

Output-coupler power is not subtracted again. It is only a fraction of the energy
that has already left the ions through stimulated emission. Likewise, adding a
separate quantum-defect heat source to this first-law ledger would double count
that energy.

During an instantaneous pump kick, only I7/I8 change. The map therefore gives

    deposited optical energy = (N7_after - N7_before) h nu_p,
    prompt heat = deposited optical energy - (U_after - U_before).

Between kicks, `sample_cycle_heat` replays the actual Stage 4R optical dynamics
and integrates heat, stimulated transfer and fluorescence using Gaussian
quadrature on every adaptive BDF interval. Independent lower-order quadrature
and a voxelwise first-law residual are reported. The stored-energy change is
computed from endpoints; it is not silently assumed to vanish. Subharmonic
solutions are averaged over their entire detected optical cycle.

The helper rejects a nonconverged optical solution by default. Transient heating
requires explicit `allow_transient=True` and a specified number of periods.

### Spectroscopic uncertainty

The default effective manifold energies are the **unweighted legacy Stark-level
centroids already stored in Stage 0.1**, with the ground centroid subtracted.
Default mean branch photon energies are the corresponding energy differences.
This is an energy-consistent *surrogate*, not measured line-strength-weighted
fluorescence data. `HeatSpectroscopy` accepts replacement mean branch energies.
Both values and provenance are included in output metadata.

Radiation reabsorption/transport and temperature-dependent Stark populations are
not resolved. All spontaneous photons leave the ion subsystem; the tiny fraction
seeded into the optical cavity is already radiation leaving the ions. This does
not mean that all such photons escape a real mounted crystal.

Coating losses are not automatically converted into disk heat: reflectivity
alone does not separate absorption, transmission and scattering. Measured front
or rear coating heat can be supplied explicitly as surface flux to the thermal
solver. Pump mirror loss is separately retained in the optical budget.

## 5B: disk thermal solver

`DiskThermalMesh` uses `(z,r,phi)` finite-volume cells over the actual cylinder:

- radius 5 mm, thickness 1 mm by default;
- z=0: front optical face;
- z=1 mm: rear HR-coated, cooled face;
- periodic azimuth, with `nphi=1` for axisymmetry or `nphi>=3` for 3-D asymmetry;
- zero-area radial face at the axis: automatic symmetry/no-flux condition.

Cell areas and volumes are exact annular sectors, not a square mask or a
staircased rim. Radial and axial meshes can be nonuniform. Internal face
conductances include both half-cell resistances and are symmetric. The 3-D
azimuthal operator is periodic and conservative.

    rho Cp dT/dt = div(k grad(T)) + Q.

The steady solve uses a sparse linear system. The transient solver uses backward
Euler and reports the matching discrete energy balance, including new-time
boundary flux. A fully insulated transient is supported; a fully insulated
steady problem is rejected because it has no unique absolute temperature.

Boundary conditions use `h` in **W/(m^2 K)**, not the rod's setup-specific W/K
coefficients. Each front/rear/rim condition supports insulated (`h=0`), finite
contact, or fixed-temperature (`h=inf`) behavior. The face-to-cell conductance is

    G = face_area / (half_cell_distance/k + 1/h).

The example assumptions are a 293.15 K heat sink, rear contact
`h=100000 W/(m^2 K)`, and insulated front/rim. **The rear contact is illustrative,
not a measured mounting property.** Conductivity, density and heat capacity use
the Stage 0 constant baseline, 14 W/(m K), 4560 kg/m^3 and 680 J/(kg K).
`nonlinear_steady` supports a supplied k(T) function; no unverified Debye or
high-temperature spectroscopic fit is enabled by default.

The modal-thermal adapter uses exactly the same control volumes for optical
energy deposition and thermal heat removal. Gaussian/LG annular mode intensities
are integrated analytically over each cell; there is no interpolation or global
rescaling of the heat source that could lose energy.

## 5C: thermal OPD

    Delta n = (dn/dT) (T - T_reference)
    OPD_single(r,phi) = integral Delta n dz
    phase_single = 2*pi*OPD_single/lambda.

The default coefficient 9.1e-6 K^-1 is the Rupp baseline retained in Stage 0.
The implementation is additive, `n=n0+Delta n`. Bulging, thermal expansion of the
optical path and stress/photoelasticity are **not** folded into dn/dT.

`fit_radial_thermal_lens` fits `OPD = piston - r^2/(2f)`, so a positive dioptric
power `1/f` means focusing. The fitting radius, intensity/area weights and
nonparabolic residual are returned. A fit is not a claim that the entire
thermal OPD is exactly quadratic.

## 5D: resonator coupling

The FFT wrapper applies `exp(i*k0*OPD_single)` before the outward disk traversal
and after the return traversal. Pass a **single-pass OPD**; there are already two
visits in the operator. Constant piston therefore changes round-trip phase by
`2*k0*OPD`, not `4*k0*OPD`. There is no artificial mirror complex conjugation.

For the reduced Gaussian feedback, the round-trip ABCD matrix is

    M = F P C P F,

with a single-traversal thermal lens F on each visit, the original reduced
propagation length P, and the original output-mirror curvature C. An unstable or
marginal hot cavity is reported, not forced into a fictitious stable solution.

Every outer iteration first obtains a periodic optical solution, integrates its
heat, solves the thermal steady state and updates the Gaussian radius. Optical
and thermal convergence are independent checks. Relaxed updates are a numerical
fixed-point technique, not a physical heating-time trajectory. A one-iteration
run is explicitly labelled one-way cold-mode heating. Reaching the iteration
limit is explicitly not convergence.

The complete nonparabolic OPD can be applied by the FFT wrapper, but its
aberration-induced modal loss is not silently inserted into the fixed-mode photon
solver. Temperature-dependent absorption/emission, altered coating loss and
mechanical deformation remain future extensions.

## Running

```
python -m pip install -e '.[dev,plots]'
python -m pytest tests/test_stage5_heat_thermal.py tests/test_stage5_resonator.py -q
python examples/stage5_thermal_demo.py --iterations 1
python examples/stage5_thermal_demo.py --nr 64 --nz 12 --iterations 10
python examples/stage5_plot.py --result results/stage5/generated
```

The default configuration uses the separately recorded **250 mm / 2% output
coupler** reference case. The other Stage 4R 200 mm / 3% case is not overwritten.
A new thermal run must not be compared with historical gain/output values while
quietly switching cavity parameters.

## Validation strategy

Tests cover pure-radiative, pure-multiphonon, stimulated and pump-kick energy
limits; transient first-law closure; exact disk volume; zero heat; analytical
rear-cooled one-dimensional heating with both finite and ideal contact;
radial heat flow to a cooled rim; azimuthal asymmetry and rotation;
transient heat-storage conservation; steady/transient consistency; optional
k(T); OPD units; thermal-lens sign and focal length; zero-phase regression;
two-pass phase counting; and a parabolic hot-cavity FFT eigenmode.

Passing these tests validates the numerical equations, not unknown material
parameters or a real laser assembly. Reported temperatures and output shifts
remain conditional on the assumed spectroscopy, cooling and optical model.

## Sources

Rupp, Eichhorn & Kieleck, Applied Physics B 129, 4 (2023),
DOI: 10.1007/s00340-022-07939-z. Rate equations, radiative energy subtraction,
thermal diffusion and the thermo-refractive phase approach. The published model
is CW; transient stored-energy accounting here is an explicit extension.

Existing repository Stage 0.1 Stark-level data and its provenance warnings:
`data/ho_yag_stark_levels.json`. Modern level-assignment reference:
Walsh, Grew & Barnes (2006), DOI: 10.1016/j.jpcs.2006.01.123.

SciPy `solve_ivp` (BDF/dense output) and sparse linear-algebra documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html
