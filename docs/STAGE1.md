# Stage 1 — Passive structured-light propagation

Stage 1 implements passive scalar complex-envelope propagation in a uniform linear medium.

## Implemented

- 2-D Cartesian transverse grid
- exact angular-spectrum propagation in a homogeneous scalar medium
- Gaussian beams
- Hermite-Gaussian HG_mn modes
- Laguerre-Gaussian LG_p^l modes with vortex phase
- arbitrary phase-only masks
- relative-power normalization and diagnostics
- refractive-index parameter, so the same propagator works in air or YAG

## Physics included

The propagated field is a complex envelope E(x,y). For each spatial-frequency component,

    H(kx,ky;z) = exp(i kz z)

with

    kz = sqrt((n k0)^2 - kx^2 - ky^2).

There is no gain, resonant absorption, dopant distribution, thermal lens, stress, or nonlinear response in Stage 1.

## Numerical scope

The FFT angular-spectrum method is periodic in the transverse plane. The simulation window must therefore be chosen large enough that significant field amplitude does not reach the boundaries and wrap around.

The current `bandlimit=True` option suppresses evanescent spatial frequencies outside

    kx^2 + ky^2 <= (n k0)^2.

It is not a full propagation-distance-dependent anti-alias band-limit. The present Stage 1 validation cases have large enough windows that this distinction is negligible.

## Validation tests

The automated tests verify:

1. even and odd grids are geometrically centered consistently;
2. z=0 returns the original field;
3. forward propagation followed by the same backward propagation recovers the field;
4. passive propagation conserves integrated |E|^2;
5. a propagated Gaussian reproduces the analytical paraxial beam radius to <1%;
6. HG10 has the expected odd symmetry;
7. LG_0^2 has the expected 4-pi phase winding;
8. a phase-only mask leaves intensity unchanged at the mask plane.

## Picosecond compatibility

Stage 1 propagates a 2-D carrier-envelope field E(x,y). Stage 1P extends this to E(tau,y,x) in a retarded-time frame.

## Next stage

Stage 2P adds transient Ho:YAG populations and pump absorption. Gain/absorption should be applied as separate split-step operators rather than modifying the validated passive propagator.
