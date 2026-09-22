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

The measured 295 K Ho:YAG absorption peak table is represented locally around the pump as Gaussian peaks using the published peak positions, peak cross sections and FWHM.

For the transient saturated solver, this is reduced to an energy-spectrum-weighted effective absorption cross section. This correctly approximates total pump-energy attenuation. Reusing that scalar in the rate equations with the carrier photon energy introduces a negligible photon-energy weighting error for the present 1–10 ps range (below about 2e-4 relative at 1 ps).

This treatment captures that a 1 ps pump overlaps the 1908 nm absorption line less efficiently than a 10 ps pump. It does not model frequency-by-frequency spectral reshaping, spectral hole burning, or wavelength-dependent stimulated-emission cross sections.

## Numerical method

At every z slice:

1. half-step passive propagation;
2. solve the four-level populations along retarded time using RK4;
3. apply the population-dependent material attenuation over dz;
4. second half-step passive propagation.

The material step uses local midpoint intensity through the slice. The diagnostic named peak_I7_fraction now explicitly tracks the maximum over the complete pulse, not merely the end-of-window population.

## Validation

The complete project CI passes on GitHub. Stage 2P validates:

- total Ho population conservation
- ground-state stationarity
- physical pulse-energy normalization
- weak-pulse excitation against sigma*fluence/(h nu)
- picosecond bandwidth dependence of effective sigma_a
- Beer-Lambert recovery in the weak-pump limit
- saturation-induced transmission increase and positive small-signal gain
- absorbed-energy versus stored-excitation accounting
- true temporal peak-I7 tracking
- convergence with z discretization for a saturating pulse
- convergence with temporal discretization at high fluence

The published Brown et al. Gaussian-pump absorption tables provide an independent scale check: a roughly 5 nm-wide pump near 1908 nm implies an effective absorption cross section of order 8e-25 m^2 at low optical depth, consistent with the Stage 2P ~1 ps value.

## Scope and important repetition-rate limitation

This stage is a **single-pulse primitive** through homogeneous Ho:YAG.

It does not yet include:
- pulse-to-pulse population accumulation
- spatially varying Ho concentration
- exact frequency-resolved saturated absorption
- structured-signal amplification/depletion
- heat generation
- thermal feedback

The generic Stage 0P source is 10 kHz, giving 100 us between pulses. The baseline I7 spontaneous lifetime is 7.9 ms, so spontaneous decay alone would leave exp(-100 us / 7.9 ms) ~= 98.74% of an I7 population between pulses. ETU and other relaxation modify that number, but the conclusion is unchanged: the repetitive pump cannot be represented by independently resetting every pulse to the ground state.

Therefore the next required step before using the generic 10 kHz source as a physical operating point is **Stage 2R: interpulse relaxation and pulse-train accumulation**. Stage 3 (spatial Ho inhomogeneity) should build on that stateful pump model.
