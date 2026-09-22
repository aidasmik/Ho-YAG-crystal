# Stage 1 — Passive structured-light propagation

Stage 1 implements passive scalar complex-envelope propagation in a uniform linear medium.

## Implemented

- 2-D Cartesian transverse grid
- exact angular-spectrum propagation
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

## Why angular spectrum

It keeps the phase exactly, supports arbitrary structured beams, and avoids committing the project to Gaussian/paraxial modes. This is useful later when the field is distorted by spatial gain and refractive-index maps.

## Validation tests

The automated tests verify:

1. z=0 returns the original field.
2. Passive propagation conserves integrated |E|^2.
3. A propagated Gaussian reproduces the analytical paraxial beam radius to <1% in a representative test.
4. HG10 has the expected odd symmetry.
5. LG_0^2 has the expected 4-pi phase winding.
6. A phase-only mask leaves intensity unchanged at the mask plane.

## Picosecond compatibility

Stage 1 deliberately propagates a 2-D complex envelope E(x,y), not an optical carrier. Stage 1P can extend the same API to E(x,y,tau) by applying the transverse angular-spectrum operator to each temporal-frequency slice and adding GVD.

## Next stage

Stage 2 adds the homogeneous Ho:YAG population/gain model. The passive propagator should not be modified; gain/absorption will be applied as additional split-step operators.
