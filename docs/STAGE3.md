# Stage 3 — Spatially inhomogeneous Ho concentration

Stage 3 replaces the scalar Ho density by a voxelized field

    N_Ho(z, y, x)

while preserving the validated Stage 2P single-pulse and Stage 2R pulse-train
physics.

## Implemented

- HoDensityField with explicit (nz, ny, nx) shape and crystal length
- uniform density fields
- axial linear concentration gradients
- localized 3-D Gaussian dopant-rich/dopant-poor regions
- reproducible smooth random 3-D concentration maps
- zero-doped voxels
- single-pulse pump propagation through N_Ho(z,y,x)
- repetitive-pulse accumulation through N_Ho(z,y,x)
- local population conservation at every voxel
- convergence based on the local population fraction in each active voxel

## Physics policy

Only the local total Ho density changes. The baseline spectroscopic cross
sections and split ETU/cross-relaxation coefficients remain the Rupp 1.1 at.%
values.

This is intentional for modest concentration variations, matching the Stage 0
policy. The measured total P77 concentration trend is not substituted into the
individual split coefficients because those individual concentration laws are
not uniquely known.

For large concentration excursions, concentration-dependent ETU coefficients
should be introduced as a separate calibrated extension.

## Population constraint

Each voxel conserves

    N5 + N6 + N7 + N8 = N_Ho(z,y,x).

Zero-doped voxels therefore retain exactly zero Ho population and have zero Ho
resonant absorption.

## Validation

Stage 3 tests verify:

1. uniform density-map construction;
2. reproducible/nonnegative smooth random maps;
3. exact transparency and zero populations in undoped voxels;
4. uniform Stage 3 single-pulse propagation reproduces Stage 2P;
5. a z-dependent low-fluence density field obeys
   T = exp[-sigma_a * integral N_Ho(z) dz];
6. dark relaxation preserves the local density map;
7. a uniform Stage 3 pulse train reproduces the Stage 2R history.

## Next stage

Stage 4 should add the structured 2.09 um signal field and let the stored
N7/N8 population map amplify or absorb that signal. This is where Ho-density
inhomogeneity starts producing directly observable structured-light amplitude
and phase effects.
