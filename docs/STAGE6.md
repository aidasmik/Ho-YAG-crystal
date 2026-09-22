# Stage 6 — Crystal, cooling plate, bonded interface and vector optical phase

Stage 6 implements a finite two-body thermal/mechanical assembly and a hot-disk
Jones operator. It preserves the Stage 0–5 source and parameter files.

## Geometry and assumptions

The crystal is the requested 10 mm diameter × 1 mm Ho:YAG disk. The illustrative
cooling plate is C10100 copper, 20 mm diameter × 3 mm thick. Its underside exchanges
heat with 293.15 K coolant through h=10,000 W/(m² K). Crystal/plate thermal contact
uses h=100,000 W/(m² K). These are configurable assumptions, not mounting measurements.
The plate is a deformable solid, **not an isothermal rigid boundary**.

Coordinates: z=0 is the optical front; z=d is the crystal rear HR/bond; the plate
occupies d<z<d+t_plate. The default plate underside is mechanically clamped.
`roller` removes normal motion at the underside with minimal in-plane rigid-body
gauges; `free` applies only rigid-body gauges to the bonded assembly.

## Thermal coupling

`cooling_plate.py` combines two existing Stage 5 finite-volume heat matrices.
For each matching interface face the reciprocal conductance is

    G = A / [dz_disk/(2 k_disk) + 1/h_contact + dz_plate/(2 k_plate)].

Equal and opposite heat flows are inserted in the two bodies. Both solid
conductivities and heat capacities are retained. Steady and backward-Euler
transient solutions report a whole-assembly energy balance and actual interface
face temperatures. The contact conductance may be a spatial (r,phi) map; h=0
allows thermally insulated patches and h=inf gives perfect thermal contact.
There is no uncalibrated pressure-to-conductance law.

## Mechanical coupling

`thermomechanics.py` assembles 3-D linear tetrahedral finite elements for both
solids, solving div(sigma)=0 with

    sigma = C : [sym(grad(u)) - alpha (T-T_stress_free) I].

Different elastic moduli, Poisson ratios and expansion coefficients are assigned
to crystal and plate. An area-consistent interface spring matrix transfers
normal and shear tractions with equal/opposite force:

    traction_on_disk = -diag(kt,kt,kn) (u_disk-u_plate).

Defaults are kn=1e14 Pa/m and kt=1e13 Pa/m. This is a **compliant bonded interface**,
not contact enforced by fixing the crystal rear face. Different thermal expansion
of the two bodies therefore creates real interfacial shear and crystal stress.
A prescribed bond-quality map scales the mechanical stiffnesses independently of
thermal h. Setting kt=0 is a maintained bilateral normal-contact/sliding limit.

This implementation does NOT predict unilateral separation, Coulomb friction,
delamination, plastic solder deformation, creep, or fabrication residual stress.
For a non-bonded clamped assembly those laws and preload measurements must be
added; a tensile spring is not evidence that real contact remains closed.

Outputs include both displacement fields, strain/stress tensors, principal and
von Mises stress diagnostics, interface displacement jump, normal/shear traction,
reaction forces, virtual-work and equilibrium residuals. Linear tetrahedra and a
polygonal rim require spatial refinement; peak edge stresses are not a fracture
or bond-strength prediction.

## Optical conventions

`stress_optics.py` maps stress to **inverse relative permittivity**, B=epsilon_r^-1,
then solves the transverse eigenproblem for normal propagation. The dimensionless
cubic elasto-optic law is Delta B=p:epsilon_elastic. Only elastic strain S:sigma
is used, because the configured dn/dT is interpreted as stress-free. Adding total
strain would double-count free thermal expansion.

The default host-reference coefficients are p11=-0.029, p12=0.0091, p44=-0.0615,
with engineering-shear convention Bxy=2 p44 epsilon_elastic_xy. They are not a
validated Ho:YAG measurement at 2.09 µm. Rotation is explicit: lab z=[111],
x=[1,-1,0]/sqrt(2), y=[1,1,-2]/sqrt(6), with configurable azimuth. The original
Stage 0 stress-optic table remains unchanged; it is not silently reinterpreted as
this dimensionless cubic tensor. A hydrostatic-isotropy test guards the convention.

The reflected geometric path must include both faces and the displaced air path:

    OPD_geometry,roundtrip = 2 [(1-n) u_front,z + n u_rear,z].

The rear HR follows the **crystal rear surface**, not the cooling plate surface
when their displacements differ across a compliant bond. Rigid translation gives
2u; expansion with a fixed rear gives 2(n-1) Delta thickness. The earlier generic
approximation `2*front_bulge` is not used.

Each depth slice supplies a symmetric birefringent Jones retarder. Ordered products
are reversed on the return traversal; retarders need not commute. Reflection does
not complex-conjugate the field in this fixed laboratory polarization convention.
The operator is lossless apart from cavity gain, masks and mirror losses.

`hot_disk_cavity_roundtrip` applies the full thermal/photoelastic operator on both
disk traversals, plus the total geometric reflected OPD, to a (2,ny,nx) vector
field. It uses the existing Stage 4R cavity operator. Do not also apply Stage 5's
`thermal_cavity_roundtrip`: dn/dT is already included. The disk is represented by
collapsed screens; vector edge scattering and oblique-incidence coating physics
are outside this approximation.

## Running

From the repository root:

```bash
python -m pip install -e '.[dev,plots]'
python -m pytest -q
python examples/stage6_assembly.py \
  --stage5-state results/stage5/generated/state.npz \
  --output results/stage6/generated
python examples/stage6_plot.py --result results/stage6/generated
```

Input Stage 5 NPZ keys: `heat_W_m3` in (z,r,phi) order, `r_edges_m`, `z_edges_m`.
The demo recomputes the temperature of the finite disk+plate assembly from this
heat, rather than applying the previous bath-only disk temperature to copper.

An explicitly synthetic thermal-source test is also available:

```bash
python examples/stage6_assembly.py --illustrative-heat-W 0.634
```

Here 0.634 is total deposited HEAT in watts, not incident optical pump power.
All reports label the source and its hash. Configuration is
`config/stage6_assembly.json`. Python APIs support asymmetric 3-D sources and
mechanics; the example source is axisymmetric.

## Numerical validation and executed case

The new tests cover two-layer thermal resistance, heat/storage conservation,
nonuniform contact, plate conductivity, free expansion, CTE mismatch, support
restraint, interface action/reaction, virtual work, stress conventions, cubic
rotation, geometric path, ordered reciprocal Jones retarders, and cold-cavity
recovery. See `results/stage6/README.md` for actual executed results.

Stage 6 is a frozen-source thermo-mechanical evaluation plus an optical operator.
It does not claim a newly converged laser output power. Full hot mode/population/
heat/mechanical feedback is Stage 7; temperature-dependent spectroscopy, contact
nonlinearity and coating-layer stress remain explicit extensions.

## Sources and parameter provenance

- Rupp et al., DOI 10.1007/s00340-022-07939-z: baseline Ho:YAG mechanics and optical framework.
- Bričkus and Dement'ev (2016), DOI 10.3952/physics.v56i1.3272: cubic YAG elasto-optic conventions and thermal/free-strain distinction.
- Copper Development Association, https://alloys.copper.org/alloy/C10100: room-temperature copper properties; rounded SI conversion. E=17000 ksi, G=6400 ksi, alpha=9.4e-6 /°F, k=226 BTU/(h ft °F), cp=0.092 BTU/(lb °F).
- Plate dimensions, bond stiffness/conductance, clamping and coolant conductance are assumptions to be replaced by the actual assembly specification.
