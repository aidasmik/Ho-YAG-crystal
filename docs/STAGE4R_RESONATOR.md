# Stage 4R — Thin-disk resonator and pulsed-pump oscillator

## Optical prescription

This is a proposed simulation setup, not a reproduction of a particular experimental resonator.

```
                 1907.7 nm pump injection
                          |
                          v
heat sink | flat HR | Ho:YAG disk | dichroic ... 200 mm air gap ... concave OC -> output
          | R=99.9% | 10 mm x 1 mm |       <---- intracavity signal ---->      T=3%
                                                                            RoC=500 mm
```

The disk's rear coating is the first cavity mirror. The front surface is treated as AR-coated. The second mirror is a concave output coupler. The pump is returned once by the rear coating, giving two pump traversals. The signal also traverses the disk twice per cavity round trip. Pump-injection optics and dichroic are idealized; their signal loss is folded into the assumed 0.5% additional round-trip loss. Cooling is a mechanical placeholder only: the thermal equation is not solved here.

Both the 99.9% rear reflectivity and 3% output transmission are assumed. No coating stack, damage threshold, or alignment tolerance has been validated.

## Passive field solver

`cavity_roundtrip` is a scalar, paraxial FFT propagator with a finite circular disk aperture, real gain screens, a quadratic curved-mirror reflection phase, and output extraction. Its reference plane is the plane HR coating. Reflection is not complex conjugation. Constant carrier and reflection phases are omitted.

For the plane-concave cavity the ABCD propagation length is `L_red = L_air + d/n`. Group delay instead uses `L_air + n_group*d`.

With the supplied prescription:

- stability product g1*g2 = 0.5988883;
- mode waist at the disk HR plane = 0.4038008 mm;
- mode radius at the output coupler = 0.5217882 mm;
- round-trip time approximately 1.3465 ns;
- empty-cavity photon lifetime approximately 36.92 ns;
- uniform single-traversal intensity-gain threshold = 18.2361 m^-1.

The threshold condition is `R_HR*(1-T_OC)*(1-L_other)*exp(2*g*d)=1`.

## Dynamic laser solver

`ModalThinDiskLaser` adds stored cavity photons to the existing four-manifold Ho equations. The mode profiles are fixed by the cold-cavity calculation. Photon energies and spatially resolved populations feed back onto one another; this is no longer a weak probe of a prescribed inversion.

For a normalized mode intensity u_m and cavity energy U_m:

```
I_signal(r) = 2 * sum_m [ U_m / t_RT * u_m(r) ]
dU_m/dt = [ 2*integral_z <g>_m dz - log(1/R_RT) ] * U_m/t_RT + S_m
```

The factor of two represents the two signal directions. The output power uses the continuous-time logarithmic outcoupling rate `-log(1-T_OC)*U/t_RT`. This is the small-round-trip-gain mean-field approximation, not an exact carrier-resolved cavity map. For a 3% output coupler the logarithmic rate differs from T/t_RT by about 1.5%.

The solver reuses `four_level_rhs`; spontaneous relaxation, multiphonon relaxation, ETU, and cross-relaxation are retained. A sparse analytic Jacobian accelerates implicit BDF integration and is tested against finite differences. N8 is obtained by population conservation, not by rescaling every step. Nonphysical end states raise an error.

Pump pulses are treated as short fluence kicks, not spread into a fictitious CW pump. Each traversal uses a photon-conserving Frantz-Nodvik two-manifold map with pump absorption and stimulated emission; the other manifolds are frozen over the ps event. The returned pump then traverses the slices in reverse order. The model warns above 0.1 incident pump saturation fluence.

**Important pump approximation:** the forward and reflected 10 ps envelopes can overlap inside a 1 mm disk. Their coherent interference and simultaneous counterpropagating saturation are not resolved. The sequential fluence approximation is intended for the weakly saturating pump range used here (up to approximately 0.05 Fsat at 1 mJ). This is a physical-model uncertainty not bounded by grid refinement.

## Numerical scope

The demonstration uses 36 radial Gauss-Legendre quadrature sites over the entire 5 mm disk radius and four longitudinal slices. A radial quadrature is not a uniform 36-pixel image. The quadrature integrates the normalized cavity and pump modes without renormalizing away clipping losses. The generic modal class also accepts flattened transverse samples, but the reported runs are axisymmetric and use uniform Ho concentration.

The default runs use a 1907.7 nm Gaussian pump with 0.5 mm intensity 1/e^2 radius, 10 ps FWHM and 10 kHz repetition rate. The room-temperature effective absorption cross section for the 10 ps pulse is 1.223454786e-24 m^2, as in Stage 2P. The Ho density is 1.52e26 m^-3. The assumed spontaneous coupling into each retained mode is 1e-8; it is a seeding parameter, not a measured material constant.

The solver detects attractors repeating after one to four pump cycles, rather than incorrectly treating a two-cycle laser output as converged after one pulse. Convergence compares all local fractions, photon populations, and emitted energy. Nonconverged output is explicitly labelled finite-window transient data.

## Executed results

For the Gaussian-only dynamic mode, repeated from ground-state populations:

| Pump pulse | Average incident pump | Average output | Converged output period |
|---|---|---|---|
| 100 uJ | 1 W | 8.82e-10 W (below threshold) | 1 pump cycle |
| 300 uJ | 3 W | 8.78e-9 W (below threshold) | 1 pump cycle |
| 600 uJ | 6 W | 0.24250 W | 4 pump cycles |
| 1000 uJ | 10 W | 0.73699 W | 2 pump cycles |

The first two powers are tiny spontaneous-background outputs, not lasing output. At 1 mJ the strong bursts carry approximately 147.4 uJ, have approximately 0.85 us FWHM and 153 W peak power, and recur every 200 us. A much weaker emission occurs in the alternating pump cycle. These are model predictions of gain-switched operation, not a claim that a 10 ps pump directly creates 10 ps output.

Refining to 48 radial sites, eight longitudinal slices, and BDF rtol 5e-7 gives 0.7370208 W at 1 mJ, versus 0.7369932 W at the baseline resolution. The relative difference is approximately 0.0037%; the two-cycle pattern remains. This verifies this discretization comparison, not experimental accuracy or all possible attractors.

A competing-mode run retaining LG(0,0), LG(0,+/-1), and LG(0,+/-2) gives essentially all output in the Gaussian mode. It is NOT evidence of automatic vortex lasing. A sole-LG1 exploratory run was also made, but it had not reached the declared periodic criterion after 360 cycles and must not be labelled a steady-state vortex solution.

## Deliberate omissions

No self-consistent thermo-optic lens, stress/birefringence, dynamic deformation, Kerr lens, mode locking, Q-switch, coherent modal interference, longitudinal-mode competition, microscopic standing-wave hole burning, arbitrary 3-D wavefront evolution under saturation, or experimentally calibrated spontaneous coupling. The modal and cold-field solvers are separate fidelity levels. The existing single-pass Stage 4 field routines remain unchanged.

## Reproduction

```
pip install -e '.[dev]'
pytest -q
python examples/thin_disk_resonator.py --sweep 100 300 600 1000
python examples/thin_disk_resonator.py --pump-uJ 1000 --charges 0,1,-1,2,-2
python examples/thin_disk_resonator.py --pump-uJ 1000 --nr 48 --nz 8
```

The CLI writes NPZ arrays, per-cycle CSVs, output-waveform CSVs, and JSON configuration/convergence metadata. Read `periodic_converged` before interpreting an average as a settled operating point.

## Primary references and provenance

- M. Rupp, M. Eichhorn, C. Kieleck, Applied Physics B 129, 4 (2023), DOI 10.1007/s00340-022-07939-z: the four Ho manifolds, pump/laser rates, and resonator-model motivation. Their validated experiment is CW, not this ps-pumped resonator.
- R. A. Lorbeer et al., Optica 7, 1409 (2020): the HR-backed, front-pumped thin-disk active-mirror concept. This project does not reproduce their wedge geometry.
- L. M. Frantz and J. S. Nodvik, Journal of Applied Physics 34, 2346 (1963), DOI 10.1063/1.1702744: short-pulse saturated-amplifier fluence relation, generalized here to pump absorption/stimulated emission.
- SciPy official solve_ivp documentation: implicit BDF with a supplied sparse Jacobian.

The repository backend was read at commit 7bd59b60705657e15478b54374243ad977602dbf. Local execution used the retrieved four-level equations and default parameters. Integration with the complete existing package is additionally checked by repository CI.
