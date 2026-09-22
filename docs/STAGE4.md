# Stage 4 — Structured 2.09 um signal amplification and depletion

Stage 4 propagates an arbitrary structured signal through the spatial Ho
population map produced by Stages 2R/3.

## Small-signal mode

For a fixed population field,

    g(z,y,x) = sigma_e,L * N7 - sigma_a,L * N8

and the material operator is

    E -> E * exp(g dz / 2).

Therefore signal intensity obeys

    dI/dz = g I.

This mode does not change the Ho populations and is appropriate for a weak
probe or for quickly visualizing the gain map seen by a structured field.

## Saturated signal mode

A physical signal pulse is normalized so that

    integral |E_s|^2 dt dx dy = signal pulse energy.

The local stimulated rates are

    W_abs,L = sigma_a,L I_s / (h nu_L)
    W_em,L  = sigma_e,L I_s / (h nu_L).

These rates are inserted into the same four-manifold Ho equations used by the
pump model. The signal therefore depletes N7 and can increase N8 while it is
amplified. Gain saturation is not an empirical correction; it emerges from the
population dynamics.

## Propagation

The signal uses the same split-step structure:

1. half-step diffraction/GVD;
2. local gain/loss and optional population depletion;
3. half-step diffraction/GVD.

Baseline signal values from the Stage 0 YAG Sellmeier are

- wavelength: 2090.3 nm
- n: 1.799104526
- beta2: approximately -7.78457e-26 s^2/m = -77.85 fs^2/mm

The material gain operator is real, so Stage 4 itself adds amplitude gain/loss
but no Kramers-Kronig dispersive phase. Spatially varying gain followed by
diffraction can still reshape both amplitude and phase.

## Validation

Stage 4 tests verify:

1. transparency at sigma_a/(sigma_a+sigma_e);
2. uniform small-signal gain matches exp(gL);
3. a ground-state crystal matches analytical signal absorption;
4. a nonuniform gain map imprints structured amplitude while the local material
   operator itself preserves phase;
5. stimulated signal interaction conserves local total Ho density;
6. finite signal energy reduces gain relative to the small-signal limit and
   depletes N7;
7. in an isolated two-level limit, signal energy gained agrees with stored I7
   excitation energy lost.

## Scope

Stage 4 does not yet couple pump and signal simultaneously in the same
picosecond time window, nor does it include thermal refractive-index changes,
stress birefringence, or Kramers-Kronig phase associated with resonant gain.

The next useful stage is thermal loading from the pump/population cycle, because
that creates the phase distortion needed for the later correction/ML problem.
