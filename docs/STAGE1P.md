# Stage 1P — Picosecond passive propagation

Stage 1P extends the Stage 1 complex spatial field E(y,x) to a retarded-time
pulse envelope E(tau,y,x), with tau = t - z/v_g.

It remains passive and linear. Ho absorption, pump depletion, transient
populations, and gain are not included yet; those belong to Stage 2P.

## Implemented

- centered retarded-time grid
- Gaussian transform-limited pulse envelopes
- arbitrary quadratic spectral phase / GDD
- second-order GVD propagation
- structured spatiotemporal pulse construction
- combined transverse angular-spectrum diffraction and temporal GVD
- pulse-energy diagnostics
- intensity-FWHM measurement

## Model

The Stage 1 transverse operator remains

H_xy = exp(i k_z z), with k_z = sqrt((n k0)^2 - kx^2 - ky^2).

In retarded time the beta1/group-delay term is removed. The temporal operator is

H_t(Omega) = exp(i beta2 z Omega^2 / 2).

Because both operators are linear and homogeneous, they commute and can be
applied in one propagation step. Gain and nonlinear operators added later will
require z-splitting.

## Stage 0P baseline

- pump wavelength: 1907.7 nm
- YAG refractive index: 1.8018687
- beta2: -4.46068e-26 s^2/m = -44.61 fs^2/mm
- reference length: 18 mm
- GDD across 18 mm: about -803 fs^2

For a 10 ps pulse this causes negligible temporal broadening.

## Validation

Seven Stage 1P tests verify:

1. Gaussian intensity FWHM
2. transform-limited Gaussian time-bandwidth product near 0.441
3. analytical Gaussian GDD broadening
4. temporal energy conservation under GDD
5. full spatiotemporal passive-energy conservation
6. reduction to the validated Stage 1 spatial solver when beta2 = 0
7. negligible broadening of the generic 10 ps pulse through 18 mm YAG

Together with the six original Stage 1 tests, the regression suite is 13/13.

## Deliberate limitations

Stage 1P does not yet include wavelength-dependent Ho resonant absorption,
transient Ho populations, pump saturation/depletion, Kerr/SPM, self-steepening,
Raman response, higher-order dispersion, or wavelength-dependent diffraction
across the pulse spectrum.

The next physically consistent stage is Stage 2P: transient Ho:YAG rate
equations coupled to the pulsed pump and wavelength-dependent absorption.
