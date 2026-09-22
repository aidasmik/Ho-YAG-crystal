# Stage 2P — Transient Ho:YAG pump and population model

Stage 2P couples the Stage 1P picosecond pump envelope to the published four-manifold Ho:YAG rate equations.

## Implemented

- physical pulse normalization: integral |E|^2 dt dx dy = pulse energy
- local intensity in W/m^2 and photon flux in photons/(m^2 s)
- stimulated pump rates W = sigma I / (h nu)
- full I5, I6, I7, I8 transient rate equations from Rupp et al.
- ETU and cross-relaxation
- spontaneous and multiphonon relaxation
- population-dependent pump coefficient alpha = sigma_a N8 - sigma_e N7
- pump depletion and saturation along z
- split-step coupling to Stage 1P diffraction and GVD
- small-signal 2.09 um gain coefficient
- spectrum-weighted room-temperature pump absorption around 1907.7 nm

## Broadband pump treatment

The measured 295 K Ho:YAG absorption peak table is represented as Gaussian peaks using the published peak positions, peak cross sections and FWHM.

For the transient saturated solver, this is reduced to a spectrum-weighted effective absorption cross section. This captures that a 1 ps pump overlaps the 1908 nm line less efficiently than a 10 ps pump.

It does not yet model frequency-by-frequency spectral reshaping or spectral hole burning. That remains a later high-fidelity extension.

## Numerical method

At every z slice:

1. half-step passive propagation;
2. solve the four-level populations along retarded time using RK4;
3. apply the population-dependent material attenuation over dz;
4. second half-step passive propagation.

The material step uses the local midpoint intensity through the slice.

## Validation

Eight Stage 2P tests pass locally.

Key numerical checks:

- total Ho population derivative sums to zero
- ground state is stationary with no light
- physical pulse normalization returns the requested energy
- weak-pulse excitation agrees with sigma*fluence/(h nu)
- effective sigma_a:
  - 10 ps: 1.22345e-24 m^2
  - 1 ps: 8.02577e-25 m^2
- 1 mm low-fluence transmission:
  - numerical: 0.83326796669
  - Beer-Lambert: 0.83326796656
- saturation test at approximately the saturation fluence:
  - nonlinear 18 mm transmission: 0.06068
  - unsaturated Beer-Lambert value: 0.03751
  - peak I7 fraction: approximately 0.37
- low-fluence absorbed-energy versus stored-I7-energy mismatch: about 4.5e-5 relative

## Scope

This stage models a single pump pulse through homogeneous Ho:YAG.

It does not yet include:
- spatially varying Ho concentration
- pulse-to-pulse population accumulation
- exact frequency-resolved saturated absorption
- structured-signal amplification/depletion
- heat generation
- thermal feedback

The next stage is Stage 3: inhomogeneous Ho concentration N_Ho(x,y,z).
