# Bounded picosecond seeded Ho:YAG amplifier

The app's **Picosecond seeded amplifier** mode is a separate externally seeded
calculation. It does not use an oscillator inversion as the amplifier state.

## Inputs and provenance

`Application_Taiwan.pdf` (pp. 11-12) proposes a **Yb:YAG/Yb:LuAG** amplifier
with a 0.01 µJ seed, 10 kHz repetition, 10 signal passes, and an unspecified
picosecond pulse width. The 10 nJ, 10 kHz, and 10-traversal defaults are borrowed
as a scenario, not as Ho:YAG measurements. The 10 ps FWHM is an explicit
assumption. The local Ho reference supplies a 1 mJ, 10 ps pump, 1 mm thick
Ho:YAG disk and 2.09 µm signal wavelength. The coated-disk/bonded-copper-
heatsink assembly uses `config/stage6_assembly.json`; contact, support, cooling
and coating loss have not been measured for a particular device.

Each calculation uses one selected spatial seed, one Ho concentration map,
one shared four-manifold population field, sequential pump and signal fluence
kicks, dark recovery between 10 kHz pulses, and a local optical/population/
fluorescence heat ledger. The existing disk/heatsink thermal and mechanical
solvers generate a scalar mean thermal/photoelastic/geometric transmission
phase. That phase is applied on each amplifier traversal. The result is accepted
only after both periodic populations and the thermal phase iteration converge.
An ideal unit-magnification relay is assumed when relay distance is zero.
New app calculations archive complex input/output fields, the Ho map, population
states, heat, temperature, and hot phase in `fields.npz` for later mesh checks.
The first 96-pixel verification run predates this archive switch and retains
its plots, summary, and execution record only.

The dark-recovery ODE is independent across transverse cells. The app can
split it over 1-16 worker processes; four is the measured default on the local
22-logical-CPU host. A 20-cycle, 96-pixel profile took 20.48 s with one worker,
10.10 s with four, and 9.82 s with eight. These are kernel benchmarks, not
promises for a complete thermo-mechanical run. Pump and signal visits remain
sequential because they update the same crystal state.

For the same 96-pixel Gaussian case, the complete bounded app calculation took
416.54 s on one worker and 225.22 s on four (1.85x speedup). Peak RSS rose
from 334.8 MB to 770.1 MB. Both runs used 202 population cycles and returned
16.90753637 nJ output, 1.48851707 W heat, and 318.07138 K peak disk
temperature. The four-worker run archived 23 raw arrays in `fields.npz`.

The seed is a **short-pulse fluence approximation**. Its picosecond duration is
recorded and checked against the repetition period, but the temporal envelope,
group-velocity dispersion, Kerr effects, spectral gain reshaping, pulse overlap,
and in-pulse excited-state lens are not solved. No gain or thermal conductivity
update from local temperature is included. Scalar phase omits birefringent
Jones mixing between amplifier passes. The heat ledger uses approximate dark
fluorescence quadrature and the Stage 0.1 centroid energies. The numerical mesh
has not passed a convergence campaign. These outputs are engineering estimates,
not experimentally validated predictions.

## Refractive-index data

The repository has a YAG host index near 2.09 µm, a baseline thermo-optic
coefficient, and host-YAG photoelastic coefficients. It does **not** have a
defensible Ho:YAG-specific `dn/dN_Ho` or excited-population `dn/dN_excited`
coefficient at the signal wavelength. Those contributions default to disabled.
The optional app fields accept measured coefficients in m³/ion with a required
provenance string. The model adds their single-pass depth integral once, beyond
the existing thermal, photoelastic, and geometric terms. A populated field is
an input assumption; it does not make the coefficients experimentally valid.

For actual calibration, supply measured Ho map, index versus Ho content and
excitation, temperature-dependent 2.09 µm gain cross sections, pump/seed
profiles and spectra, coating losses, disk dimensions, contact conductance,
heatsink/coolant conditions, and relay geometry. Repeat the same map and
assembly on refined optical, thermal, and mechanical meshes before asserting
quantitative accuracy.
