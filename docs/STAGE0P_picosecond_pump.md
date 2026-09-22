# Stage 0P — Generic picosecond pump extension

The exact picosecond pump laser is not yet known, so this stage deliberately uses **configurable placeholder parameters** rather than treating guessed values as experimental facts.

## Nominal simulation placeholder

The default test pulse is:

- center wavelength: 1907.7 nm
- Gaussian temporal profile
- pulse duration: 10 ps FWHM
- pulse energy: 100 uJ
- repetition rate: 10 kHz
- average power: 1 W
- D4sigma waist diameter: 1.0 mm
- M2: 1.1
- linear polarization
- zero chirp

These values are only intended to exercise the transient solver.

The initial numerical sweep ranges are 1–100 ps, 1–1000 uJ, 1–100 kHz and 0.5–1.5 mm D4sigma diameter. These ranges are **not** hardware safety or damage limits.

## Important timescale

Using the Stage 0 YAG Sellmeier model at 1907.7 nm gives:

- n = 1.80187
- group index ng ~= 1.83004
- group velocity ~= 1.638e8 m/s
- transit time through the 18 mm reference rod ~= 109.9 ps

Therefore a 1–10 ps pulse is substantially shorter than the optical transit time through the full reference crystal. The transient pump model should use a retarded-time coordinate rather than assuming the pulse is everywhere in the rod simultaneously.

## Pump spectral width

For a transform-limited Gaussian pulse:

    Delta_nu * tau = 0.441

At 1907.7 nm this gives approximately:

- 1 ps -> 5.35 nm
- 5 ps -> 1.07 nm
- 10 ps -> 0.535 nm
- 100 ps -> 0.0535 nm

The measured room-temperature Ho:YAG absorption feature around 1907.3 nm has FWHM ~= 4.5 nm.

This means a ~1 ps pump can span a substantial fraction of the absorption feature. In that regime the code should integrate the wavelength-dependent absorption cross section over the pump spectrum rather than using only sigma_a(1907.7 nm).

## Linear dispersion

Differentiating the Stage 0 Zelmon Sellmeier model at 1907.7 nm gives approximately:

- beta2 = -4.46e-26 s^2/m
- beta2 = -44.6 fs^2/mm
- GDD across 18 mm ~= -803 fs^2

This is included in the configuration. It should be modest for a many-picosecond pulse, but it is cheap to retain in the envelope propagator.

## Pump saturation scale

Using the Stage 0 pump cross sections,

    F_sat ~= h*nu / (sigma_a + sigma_e)

gives a rough value:

    F_sat ~= 5.29 J/cm^2

The nominal 100 uJ, 1 mm D4sigma Gaussian placeholder has a peak fluence of only about 0.0255 J/cm^2, or ~0.48% of this rough saturation scale.

So the default case is intentionally in a weakly saturating regime. When the real pump parameters are known, the code can automatically switch to resolving population evolution within the pulse if the fluence becomes significant relative to F_sat.

## Kerr nonlinearity

Kerr/SPM is present as an optional model term but disabled by default.

For scale only, a commonly cited YAG nonlinear index around 1064 nm is approximately:

    n2 ~= 6.2e-20 m^2/W

This value is **not assumed to be exact at 1908 nm**. With the nominal placeholder pulse it would correspond to a B-integral of only ~0.09 rad over 18 mm, so ignoring SPM is a reasonable initial simplification.

When the actual laser is known, use a wavelength-appropriate n2 if nonlinear propagation becomes relevant.

## Simulation policy

The first picosecond implementation should use:

1. complex-envelope propagation, not optical-carrier resolution;
2. retarded time t - z/vg;
3. transient Ho rate equations;
4. wavelength-resolved pump absorption automatically for sufficiently broadband pulses;
5. GVD enabled;
6. Kerr/SPM disabled until required;
7. heat accumulated per pulse and passed to the slow thermal solver.

The thermal solver must **not** run at picosecond time steps.

## Parameters to replace later

Once the actual laser is selected, replace the placeholders with measured or specified:

- pulse duration
- pulse energy
- repetition rate
- pump spectrum
- chirp
- beam waist and M2
- polarization
- pump/signal delay
- pulse-to-pulse stability

The rest of the Stage 0 material database remains unchanged.
