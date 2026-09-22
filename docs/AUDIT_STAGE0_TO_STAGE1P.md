# Audit: Stage 0 through Stage 1P

Audit performed after Stage 1P against the repository implementation and the primary literature.

## Status

### Stage 0 / 0.1 — material database

**Status: correct for the stated baseline, with explicit approximation flags.**

The Rupp validation values in the database match Appendix A of the 2023 Ho:YAG model after unit conversion: wavelengths, 1.1 at.% Ho density, four-manifold rates, cross sections, lifetimes, branching ratios, thermal conductivity, heat capacity, density, elastic constants, dn/dT, cooling coefficients and photoelastic tensor.

The Zelmon YAG Sellmeier coefficients are also correct.

Important limitations already present in the database remain valid:
- legacy Stark levels are not treated as definitive high-resolution spectroscopy;
- unweighted manifold centroids are fallback heat-photon estimates only;
- hot 300–500 K cross-section curves still need numerical digitization before high-fidelity thermal gain feedback;
- 632.8 nm undoped-YAG dn/dT(T) is not assumed to be exact for Ho:YAG at 2.09 um;
- ETU concentration dependence is not yet known term-by-term.

### Stage 0P — generic picosecond pump

**Status: derived quantities independently rechecked and correct.**

Rechecked from the Stage 0 Sellmeier and spectroscopic values:
- n(1907.7 nm) = 1.8018687
- group index = about 1.8300395
- transit time through 18 mm = about 109.88 ps
- beta2 = about -44.62 fs^2/mm
- 18 mm GDD = about -803 fs^2
- rough pump saturation fluence = about 5.286 J/cm^2
- 10 ps transform-limited Gaussian bandwidth = about 0.535 nm

The n2 entry remains only a 1.06 um YAG reference and is correctly disabled by default.

### Stage 1 — passive spatial propagation

**Status: solver correct; one validation-test coordinate bug fixed in this audit.**

The angular-spectrum transfer function and Gaussian/HG/LG generators are correct for a scalar homogeneous medium.

The LG phase-winding test had used N/2 instead of (N-1)/2 when mapping physical coordinates back to indices on an even grid. This was a half-pixel test-only inconsistency; it did not affect the propagation solver. The audit fixes it and adds explicit even/odd centering tests.

Numerical caveat: FFT angular-spectrum propagation is periodic. The existing `bandlimit` argument suppresses evanescent components, but it is not a full anti-alias band-limiting algorithm. Simulation windows must remain large enough to avoid wrap-around.

### Stage 1P — picosecond passive propagation

**Status: correct within its documented narrowband-envelope approximation.**

The Gaussian pulse convention gives the requested intensity FWHM and the expected transform-limited time-bandwidth product. The GDD operator and Gaussian broadening formula are mutually consistent, and energy is conserved.

The retarded-time representation correctly removes first-order group delay. For Stage 2P, local population dynamics must map the moving coordinate back to local lab time as needed.

The current transverse diffraction operator uses the carrier wavelength for all temporal frequencies. This is a deliberate narrowband approximation; wavelength-dependent diffraction is not yet included. For ~1–10 ps pulses around 1.9 um the fractional pump bandwidth is small enough for this to be a reasonable first model.

## Before Stage 2P

Do not feed the current normalized Stage 1 field directly into rate equations as if |E|^2 were automatically in W/m^2. Stage 2P must define an explicit physical normalization from pulse energy/power to intensity/photon flux.

The pump absorption operator should also be spectrally resolved when the pulse bandwidth is significant relative to the Ho:YAG absorption feature.
