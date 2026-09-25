# Yb:LuAG remaining-model implementation, 2026-09-24

Baseline: `21a36d74861da696fddd2af79246fb7b8b105179`. The Ho:YAG
oscillator and seeded-amplifier paths are unchanged. This is a synthetic
engineering model, not an experimentally validated Yb:LuAG amplifier.

The [25 September physics and desktop audit](YBLUAG_PHYSICS_DESKTOP_AUDIT_20260925.md)
supersedes the pump-recovery implementation and default pump wavelength below.
It records the common recovery kernel, independent ideal photon balance,
969 nm pulsed default, corrected references and updated native plots.

The user-facing application is the native Tkinter launcher
`examples/ybluag_desktop.py`. It invokes bounded local solver workers, shows
the beam, phase, gain and cooling results in the window, and starts no web
server. Its default architecture is ideal multipass with a 25 °C cooler
target; archived results shown at startup do not overwrite those inputs.

| Request | Status | Evidence and limit |
| --- | --- | --- |
| Coupled ideal multipass | Implemented for bounded conditions | One shared dopant, inversion, heat, temperature and OPD state; temperature-dependent pump/signal rates; saturated time-domain transport and inversion depletion on every pass; scalar disk phase at every encounter; unit-magnification ideal relay with configurable power retention (default 1). The coupled heat/temperature/mechanics fixed point checks temperature, heat, energy, OPD and coherent-field changes. No measured relay loss or phase. |
| Same-time dataset labels | Implemented gate | Only a settled coupled state with five simultaneous probe samples is eligible. The timestamp is a steady-state reference, not startup elapsed time. Transient labeled export rejects until an optical/population/thermal time integrator exists. |
| Cooling default and probes | Implemented | Default control target is 25 °C. Three default disk readings represent noncontact front-surface estimates; two plate readings are external. Custom in-volume probes remain a simulation/debug option. Sensor noise, bias, response, sampling, latency and missing flags remain explicit. |
| Static cold phase | Implemented API | Optional measured/imported disk-plane phase is separate from the SLM shaping, SLM correction and thermal OPD maps. It is imposed at every disk encounter; no concentration-index coefficient is fabricated. The browser API accepts a full-grid `static_cold_phase_rad` array. |
| Correction labels and observable separation | Implemented gate | Export requires a candidate SLM correction and a complete corrected coupled forward rerun that increases same-plane coherent overlap. Measured camera fluence, measured probes, pump, seed and SLM command are stored under `observable__`; exact fields and thermal/material maps under `truth__`. This callback must rerun the same physical configuration; a caller changing unrecorded inputs would invalidate the sample. |
| Convergence studies | Partial | Bounded 64→128 pixel-pitch comparison for four seed shapes and one low-power coupled Gaussian before/after figure are recorded in `results/ybluag_ideal_partial_validation`. Independent window, axial, thermal radial/angular/depth, mechanical mesh, tolerance and time-step refinements remain open. The export gate requires evidence for all axes, four shapes and seven metrics, matched to the configuration and current source fingerprint, so the current diagnostic cannot unlock export. |
| Material assumptions | Partial | The 12 at.% conductivity remains 7.2 W/(m K) as a stated fixed property; an undoped temperature helper is not silently substituted. The 12 at.% lifetime, host index, dn/dT, elastic constants, disk/contact/coolant conductances and phase remain assumptions or require measurement. The [temperature-dependent Yb:LuAG spectroscopy study](https://opg.optica.org/josab/abstract.cfm?uri=josab-29-9-2493) measured spectra from 20–200 °C and reports 7.2 W/(m K) for its 12 at.% example. Our spectra remain figure-guided reconstructions, not the authors' raw arrays. |
| Missing physics | Blocked by data/model scope | LuAG photoelastic tensor, ASE/parasitic lasing, B-integral, finite switch waveform, coating spectral phase/absorption, wavelength-resolved saturated gain, pulse compression and measured cooler/contact properties remain unavailable. They are not silently set to zero in a device-validity claim. |

The coupled copper assembly is supported only from 293.15 to 300 K by the
current property parameterization. This is a **model validity bound**, not a
crystal damage temperature. Higher-temperature operation requires a suitable
doped-crystal/assembly property model and calibration. The default 25 °C
controller target is within this interval; 40 °C remains selectable as an
explicit engineering scenario but cannot make an out-of-range coupled result
valid.

The ideal relay is a stated optical boundary condition. The time-domain
transport resolves saturation and pulse energy but not chirp, spectral phase,
finite Pockels switching, Kerr self-phase modulation or pulse compression.
The reported coherent field uses a spatially uniform temporal phase and
integrated fluence; it is not a reconstructed complex temporal waveform.

The four-shape diagnostic took about 3.4 s inside a 180 s bounded supervised
run with 156 MB peak process RSS. At fixed 12 mm FFT window, 64→128 coherent
overlaps were 0.9997 (Gaussian), 0.9738 (vortex), 0.9940 (flat-top) and
0.8582 (needle). The needle radius changed 11.2%. No acceptance threshold
was fitted to these outcomes. The low-power Gaussian coupled solve reached
20.078 °C and changed output energy from 9.4535039e-10 J cold to
9.4534666e-10 J coupled. These are deliberately small software checks,
not proposal operating-point predictions.

See `results/ybluag_ideal_partial_validation/cold_vs_coupled.png` for the
same-scale cold/coupled output fluence and masked phase difference. The disk
circle is the physical 5 mm radius. The plot uses the low-power bounded
fixture, so the visually small thermal effect is expected; it must not be
rescaled into a claimed high-power prediction.

## Verification

The full repository suite passed **407 tests** in 202 s with peak process RSS
338 MB under `hoyag.local_supervisor`. After the source-fingerprint export gate
and the end-to-end SLM correction/app checks were added, the focused Yb tests
passed **12 tests**. The latter include a corrected full coupled forward rerun
and JSON serialization through the browser API. These tests establish code
behavior, not material calibration or mesh convergence.
