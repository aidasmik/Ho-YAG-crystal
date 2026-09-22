# Full project audit through Stage 2P

## Result

The core architecture and equations are consistent with the intended Ho:YAG model. GitHub Actions passed 31 tests on the Stage 2P commit before this audit. This audit adds diagnostics and convergence tests.

## Stage 0 / 0.1

The baseline Rupp parameters and SI conversions are internally consistent. The Zelmon Sellmeier coefficients are correct. The 295 K Ho:YAG absorption peak table matches the published values.

The temperature-dependent and Stark-level additions remain correctly labelled as approximations where the source is not directly the final 2 um Ho:YAG quantity.

## Stage 0P

Derived group index, transit time, GVD/GDD, transform-limited bandwidth, peak fluence and rough saturation fluence are internally consistent.

The Kerr n2 value remains a reference at 1064 nm and is disabled by default, which is appropriate.

## Stage 1

The scalar angular-spectrum propagator is correct for a homogeneous isotropic medium. Gaussian/HG/LG mode definitions and power normalization are consistent.

FFT propagation is periodic, so future simulations must maintain enough transverse padding to prevent wrap-around. The existing bandlimit flag removes evanescent components; it is not a full anti-alias angular-spectrum band-limit.

## Stage 1P

The retarded-time Gaussian envelope, GDD/GVD operator and energy normalization are consistent. The carrier-wavelength diffraction approximation is a narrowband approximation and is acceptable for the current picosecond range.

For pulses approaching the shortest end of the planned range, manifold-averaged spectroscopy also assumes sufficiently rapid intramanifold thermalization. This is a model-validity assumption rather than a code defect and should be revisited if substantially sub-picosecond operation is introduced.

## Stage 2P

The four Ho-manifold rate equations match the published Rupp equations, including the factors of two in ETU/cross-relaxation. The pump attenuation coefficient and small-signal laser gain have the correct signs and units.

The physical envelope convention is deliberate: |E|^2 is optical intensity after pulse-energy scaling; E is not interpreted as electric-field amplitude in V/m.

Two audit findings:

1. The diagnostic peak_I7_fraction previously returned the end-of-pulse value. For normal picosecond Ho:YAG pumping the difference is usually tiny, but the name was formally incorrect. The audit changes it to track the true temporal maximum.
2. The spectrum-weighted sigma_a is energy-weighted, which is correct for total pump attenuation. Its reuse in the photon-rate equation with h*nu0 is an approximation. The correction is below about 0.02% at 1 ps around 1907.7 nm, so this is not a practical error in the current model.

The largest project-level limitation is repetition rate: Stage 2P is single-pulse only. With the generic 10 kHz source, the 100 us pulse spacing is far shorter than the 7.9 ms I7 lifetime. A pulse-train accumulation stage is therefore required before treating that generic source as a steady operating condition.

## Required order from here

1. Stage 2R — propagate populations between pulses and iterate to periodic steady state.
2. Stage 3 — replace scalar Ho concentration by N_Ho(x,y,z).
3. Then structured-signal amplification, thermal loading and thermo-optical feedback.

No existing Stage 0–2P result needs to be discarded; Stage 2R extends the single-pulse primitive rather than replacing it.
