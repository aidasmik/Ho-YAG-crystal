# Yb:YAG structured-light correction

## Physics and code map

| Stage | Main code | What is calculated |
|---|---|---|
| Material | `src/ybyag/model.py`, `src/ybyag/material_data.py` | Room-temperature Yb:YAG spectra, site density and host-property proxies. Hot gain is guarded. |
| Pump and pulse | `src/ybluag/multipass_pump.py`, `src/ybluag/gallery.py` | Pump absorption, one shared excited population, pulsed extraction and energy/photon ledgers. Optional Numba pump transport uses the same cross sections. |
| Heat and optics | `src/ybluag/assembly.py`, `src/ybluag/gallery.py` | Cycle-average heat, disk/contact/copper temperature, deformation, scalar OPD and output field. The pulsed control path uses a post-amplifier thermal phase screen. |
| Physical experiment | `src/ybyag_control/adapter.py` | Fixed crystal/SLM/camera maps, changing pump and coolant conditions, timed commands, exposures and five temperature probes. |
| Phase measurement | `src/ybyag_control/interferometry.py` | Four detector exposures, calibrated photoelectron-to-ADU response and a PSF-limited complex-field estimate. |
| Controller | `src/ybyag_control/controller.py` | Interferometric pixel map, SPGD or response matrix; only measured observations enter an update. |
| Tkinter/results | `examples/desktop_simulation.py`, `examples/ybyag_control.py`, `examples/ybyag_control_view.py` | Bounded worker, saved observations, live progress and clearly separated validation truth. |

The frozen correction target is the **uniform pumped disk's optical field before
the lumped thermal phase screen**. It has the same nominal gain and shaped
Gaussian seed as the forward solve. The actual plant then adds material and
SLM errors, external optics and its solved thermal phase. This lets the
controller correct thermal wavefront error without comparing against an
unamplified seed. It remains a model reference, not an experimentally
measured perfect beam.

The intended mask is applied at the SLM: `E_SLM = E_Gaussian ×
exp(i × (phase_target + phase_correction))`. Propagation forms the structured
disk field. After pulsed amplification, the thermal solve supplies the
deformation and OPD used in the output phase. The interpolated Yb:YAG spectra
are valid only at 293.15 K; applying a post-amplifier phase screen does not
make the gain temperature dependent. Separate [Yb:YAG material notes](YBYAG_SIMULATION.md)
identify missing data and proxy properties.

Each modeled disk encounter is one traversal of the crystal. The cold
thickness-error OPD per encounter is `(n - 1) × thickness_error`; a rear-HR
surface displacement contributes one surface-height unit of OPD per encounter
because its reflected `2 × height` OPD occurs once per two traversals. The
thermal round-trip OPD is likewise divided over the modeled traversals. The
rear-surface interpretation and symmetric split are model assumptions.

The Tkinter **Yb:YAG** tab has **Closed-loop correction…**. It opens a bounded
episode with target, correction enable, snapshot or in-situ mode, correction
mode count, finite-difference step, update/evaluation limits, diagnostic
astigmatism, SLM timing, exposure, pump, concentration and grid controls.
The dialog also exposes camera read/background noise, exposure-to-exposure
gain jitter, slow gain drift, probe noise and photodiode noise. Shot noise is
drawn from each exposure's photoelectron count. Pixel sensitivity and defects
stay fixed for one camera session; new random shot/read noise, gain jitter and
probe noise are drawn for every measurement during fitting. Gain drift follows
the episode's simulated observation time. Set **Camera and probe noise** to
**no** for a deterministic detector baseline.
The default controller is **interferometric**. It uses four phase-stepped
camera exposures to recover a complex output field, compares that measurement
with a frozen structured-field reference, and updates a pixelwise SLM phase
map through a calibrated free-space adjoint. The intended vortex or other
structured phase remains in the SLM target; the update adds only a correction.
**SPGD** and **response_matrix** remain separate selectable controllers.
The **Correction loop** tab updates after each reported physical step while
the bounded worker runs. Its camera panel also updates for each response-matrix
probe, labelled by full-solver evaluation number; accepted output fields update
only when a correction is accepted. It can replay the saved accepted steps
afterward. The **Cameras + residual phase** view shows both measured
camera arms with horizontal and vertical mean-signal profiles; dashed curves
are the frozen target-camera profiles. It also shows the piston-removed,
wrapped output phase residual against the ideal structured-field target.
For an interferometric run this tab also shows the phase residual recovered
from the four noisy detector frames, separately from the simulator-truth
phase residual. For a live probe the truth map follows that probe's simulated
field. A single camera intensity frame alone cannot measure phase. The overview camera has the same
profiles. It plots
measured camera loss and photodiode energy against iteration. A
separate **Simulation validation (truth)** tab shows phase, coherent overlap
and useful target-mode energy. Truth is never passed to the controller.
For a chronological run, select **in_situ** and **Changing coolant/pump = yes**.
The default small disturbances use a 20.2 °C coolant setpoint with 0.03 K
per-observation jitter and a slow random walk, 0.5% pump-power jitter, 0.3%
pump-radius jitter, and 5 µm pump-pointing jitter. They enter each full
physical solve. The same solve produces heating, temperature, disk deformation
and optical path change. The validation view plots true disk temperature,
measured probe readings, coolant setting, thermal OPD, and applied pump power
against simulated time. The controller still sees only measured observations.
Snapshot episodes keep fixed physical operating conditions, as required for
counterfactual finite-difference probes.

The nominal coolant setpoint now reaches both snapshot and ideal-reference
thermal solves. Noiseless probes sample the solved disk/plate fields rather
than returning a fixed room-temperature value. Probe calibration drift is a
persistent random walk with increments in K/√s; independent readout noise
remains separate. The earlier `probe_drift_K_per_s` configuration key is
accepted for old requests, but new configurations use
`probe_drift_K_per_sqrt_s`.

From the repository root on Windows, the small reproducible physical smoke is:

```powershell
& 'C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe' .\examples\run_ybyag_control.py --smoke --output .\results\control_smoke
```

Use `--mixed` to enable the fixed material maps and SLM calibration errors;
add `--in-situ` to carry the supported disk/plate temperature and five sensor
states forward. To run the configured 14-mode episode:

```powershell
& 'C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe' .\examples\run_ybyag_control.py --config .\config\ybyag_control.json --output .\results\control_14mode
```

For a more adequately sampled, phase-only correction check, add
`--grid 96 --phase-only`. These switches keep the same pulsed solver and
isolate external optical phase from fixed material and SLM errors.

### Bounded audit outcome

The latest phase-only audit used a 10 nJ seed, 0.01 W pump, 10 at.% Yb:YAG,
32² optical grid and four-step interferometric control with detector noise.
Over 40 physical evaluations, measured camera loss fell from about 0.24 to
0.12 and the measured phase-residual RMS from about 0.57 to 0.18 rad. The
final simulator-truth coherent overlap was 0.939; output energy was 9.68 nJ,
below the input seed. The grid resolves only 1.6 pixels per seed waist, so
the overlap is a coarse-grid diagnostic, not a converged performance result.
The program reports `no_measured_improvement` after the later rejected steps.
This run checks numerical coupling and command behavior; it does not validate
the proposed >10,000× amplifier gain. Camera-image agreement alone cannot
establish recovery of the complex field, particularly with material and SLM
errors present.

## Interferometric phase correction

The simulated reference arm is a stable, calibrated broad Gaussian field.
Phase shifts of 0, π/2, π and 3π/2 are applied to that arm. Each summed
intensity is converted to photoelectrons and passed through the CCD model,
including shot noise, read noise, fixed pixel response, ADC quantization and
clipping. Opposite frames are subtracted to recover the complex interference
term. A calibrated detector conversion puts the PSF-limited recovered field
in √J/m units; shot/read noise and blur remain. The controller sees only this recovered field, the ordinary camera
frames, photodiode energy and sensor observations. It never reads the
simulator's output phase or physical material map.

The nominal SLM-to-output free-space propagation is used for a phase-gradient
update. The amplifier, thermal lens and disk deformation are evaluated by the
existing coupled forward solver on the next physical observation. They are not
differentiated by the controller. Each update requires four exposure windows;
with the default 10 ms exposure, 10 ms SLM delay and 20 ms settling time, a
simulated 100 ms measurement period is feasible. The field is assumed stable
over each 40 ms phase-step sequence. This 10 Hz simulated period is not a
measured wall-clock hardware update rate.

The spatial bandwidth is constrained by the optical grid. At the default
12 mm field and 96×96 grid, the Nyquist limit is **4 mm⁻¹**. A 40 mm⁻¹
target requires at least 960 samples across that field, plus convergence
checks and an actual SLM/camera pixel-pitch calibration. The current small
grid is a method demonstration, not verification of the proposal's 40 mm⁻¹
correction bandwidth. The interferometer is idealized as co-registered;
polarization, reference-arm path drift and phase-step calibration errors are
not yet modeled.

The **Phase compensation** tab compares five maps for every correction step:
the first-order phase-only oracle calculated from simulation truth, the
controller-requested correction, the phase actually delivered by the SLM,
the wrapped requested-minus-oracle difference, and the wrapped
delivered-minus-oracle difference. A sixth panel plots both residual RMS
values through the run. The global piston is removed and each RMS is weighted
by the nominal Gaussian seed fluence, so dark edge pixels do not dominate.
All maps omit the
intentional structured-light mask, use the same ±π rad scale, and zoom to the
illuminated SLM region. The oracle is a diagnostic reference based on the
simulated thermal OPD, static disk phase and external-optics phase; it is never
given to the controller. It is not an exact inverse of spatial gain or
amplitude errors.

An interferometric candidate is now retained only when the measured combined
phase/camera score improves, camera loss stays within the configured allowance
of the best measured camera loss, and photodiode energy remains above the
minimum. The controller tries smaller phase steps when a proposal fails. If
none pass, it sends and measures the previous command again, records the
rejected trial, and stops after three consecutive unsuccessful updates.
Because coolant and detector noise advance during each trial, even the
rollback reading can differ from an earlier reading; the trace shows this
physical drift rather than forcing a monotonic plotted curve.

## Sequential hardware-style fitting

The controller selector also offers **spgd**, a stochastic parallel-gradient
descent method used in adaptive optics ([Vorontsov and Sivokon, 1998](https://opg.optica.org/josaa/abstract.cfm?uri=josaa-15-10-2745)).
Selecting this mode sets 60 requested command updates and a
400 full-solver evaluation cap. Each update perturbs all 14 correction modes
in a random signed direction, measures the plus and minus camera/photodiode
response, then applies and measures one bounded new SLM command. It keeps
regressions in the chronological trace. A measured-energy floor can trigger
a separate timed rollback command. Five consecutive updated-command readings below the
camera-loss target stop early; the run may also end at its iteration,
evaluation or supervised wall-time limit. Neither a low camera score nor a
specified update count proves complex-field recovery.

The best measured command is saved. At the end, the controller remeasures
both the current command and that saved command under the current drifting
state, then keeps the one supported by fresh camera and energy readings.
This takes additional full solves and cannot guarantee the globally best
optical field when readings are noisy. The plot labels the recheck separately
from ordinary SPGD updates.

The default in-situ measurement period is 0.1 s of **simulated** time. Three
measurements normally make one update, so 60 updates consume about 18 s of
simulated time plus initial/final observations; wall-clock solving takes much
longer. Every measurement goes through the actual pulsed forward solve, SLM
quantization/calibration, evolving pump/coolant/seed state, disk/plate thermal
state, two diagnostic camera arms, detector noise and probe response. Small
SLM calibration drift accumulates in in-situ time. Optical and thermal
sampling, material-range limits, and incomplete population continuity still
apply. The **response_matrix** method remains available for faster idealized
calibration comparisons.

The thermal operating duration in the Tkinter correction dialog warms the
in-situ disk and cooler once before fitting. Each later camera exposure
advances the carried temperature field by the configured measurement period.
Snapshot mode continues to solve each independent command from its initial
state over the specified duration. The reference camera attenuator is set
from the brighter of the two diagnostic arms at 15% of full well, leaving
headroom for command probes and small operating fluctuations. The attenuation
stays fixed throughout an episode; clipped frames remain invalid readings.
If a later command still gives three unusable captures, fitting stops with
`camera_observation_invalid` and keeps the run record. Saturated final
verification frames have no reported camera-loss score.

### Bounded speed comparison

Correction episodes now use `thermal_timeline_mode: requested_only`. The
thermal solver still takes its internal time steps, carries the same disk and
plate fields forward, and samples the same five probes. It computes the
quasistatic displacement and scalar OPD at the requested observation time.
Steady cooler sensitivity, later stabilization, and intermediate mechanical
screens remain available through `thermal_timeline_mode: full` for standalone
thermal plots. A direct comparison found identical requested-time temperature,
deformation and OPD arrays; the full optical smoke also produced identical
beam fluence, camera loss and OPD values.

An optional Numba-compiled fixed-temperature pump-transport loop preserves
the same material cross sections, ion density, alternating pump-pass direction
and Beer–Lambert cell average. It is used when the `speed` extra is installed;
the NumPy/Python implementation remains the fallback and handles spatially
varying spectral temperature. Install it with
`pip install -e ".[speed]"`. Set `YB_PUMP_BACKEND=python` to compare against
the reference implementation. Each result records `pump_transport_backend`.

For the **same four-update, 32²-grid, 0.01 W supervised smoke** on this PC,
the full thermal diagnostics and Python pump path took 23.97 s of worker
time. Requested-time thermal diagnostics with the Python pump took 8.16 s;
adding the compiled pump path took 6.81 s. All three used 35 full forward
solves. This is a measured 3.52× end-to-end reduction for that small fixture,
not a promised speedup for larger grids or the proposed high-power device.
The optional native loop and Python path matched the recorded output fluence,
camera loss, energy and peak temperature to the saved numerical precision.

The small supervised hardware smoke is:

```powershell
& 'C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe' .\examples\run_ybyag_control.py --smoke --hardware --output .\results\control_hardware_smoke
```

Remove `--smoke` for the configured 60-update episode. Use `--phase-only` to
isolate a correctable external phase disturbance while keeping chronological
noise and operating drift. Playback shows every applied command, whether its
camera loss improved or regressed, with separate truth-assisted overlap plots.

A supervised 60-update `--phase-only --grid 48` run, made before the final
saved-command recheck was added, took 540 s of wall time, 18.2 s of simulated
time, 365 full solves and 182 observations. Its measured camera loss was
0.2464 initially, reached 0.1346 at update 34, and finished at 0.1558;
34 updates improved and 26 regressed relative to the preceding reading.
The independent final camera loss was 0.1552. Truth-assisted coherent
overlap rose from 0.695 to 0.887 at the final command. This demonstrates
gradual noisy fitting, but does not validate a 60-update run with the later
saved-command recheck or establish hardware timing or laser gain.

Each command enters `hoyag.local_supervisor` and its persistent local ledger,
with a 180 s smoke or 900 s normal limit and an 8 GiB memory cap. The app's
**Stop correction** button cancels the supervised process tree. No large
sweep, dataset generation or NN training is launched.

## What the controller does

The Gaussian source receives `wrap(target_phase + basis @ coefficients)` at
the input SLM. The existing imperfect SLM response, propagation, pulsed
amplifier, optional fixed material maps, copper/contact thermal solve and
diagnostic arms are rerun for each candidate. The second arm carries a known
astigmatic diagnostic phase before propagation, separate from the corrective
SLM. A fixed reference is acquired through the same two arms and detector
geometry. The controller gets only two calibrated camera frames, an independent
photodiode energy reading, five measured probe values and validity masks,
delivered coefficients, and timestamps. It does not receive complex fields,
phase, exact maps or oracle corrections.
The current camera-loss controller logs the probes but does not use them to
choose SLM commands.

The controller builds a central finite-difference camera/energy response
matrix, solves a regularized least-squares update, clips it, and checks fresh
measurements at full, half and quarter step. Saturated frames are reacquired
up to three times. A fixed camera coordinate frame and image binning are used;
the scorer never recenters a beam. Normalized image shape is scored together
with measured energy and a hard energy-retention floor. Independent final
verification frames are saved after the accepted trajectory. The NN policy
protocol in `src/ybyag_control/controller.py` accepts the same observation and
returns a coefficient action, but no NN is trained here.

`snapshot` is counterfactual optimization: every candidate restarts the
solver from the same initial thermal state. `in_situ` carries the actual
disk/plate thermal fields and sensor response after every probe. During a
configured SLM delay the previous delivered command is solved again, then the
new command is solved through settling and exposure. Exposure timestamps and
the delivered command are logged. A rejected command already affected the
in-situ state; restoring the accepted command is another timed solve. Before
an observation starts, the worker reserves enough full-solver evaluations for
both the delayed held-command phase and the requested command, avoiding a
partly advanced thermal state when the evaluation budget is reached.

## Physical limits

The 969 nm pump and 1030 nm signal use repository Yb:YAG room-temperature
spectra and a 0.95 ms model lifetime. [Körner et al. (2012)](https://opg.optica.org/josab/abstract.cfm?uri=josab-29-9-2493)
measured temperature-dependent Yb:YAG absorption and emission. Those hot
arrays are not calibrated here for the selected 5/10/15/20 at.% samples;
states outside 293.15–300 K are rejected instead of silently using the
room-temperature gain.

Within that range, each forward solve couples pump, population, heat,
disk/contact/copper temperature, deformation and scalar OPD. The
`lumped_phase` optical gain remains at room temperature, and its computed
thermal screen is applied after amplification. It cannot model refocusing
or changed gain on each disk encounter, or photoelastic birefringence.
[Published Yb:YAG thermal-lens analysis](https://opg.optica.org/ao/abstract.cfm?uri=ao-50-32-6103)
identifies thermo-optic, expansion and photoelastic contributions.

The controller CLI fixture defaults to 0.1 W pump and two ideal-relay signal
traversals. Its 0.01 W smoke is a numerical check, not a demonstration of the
proposed ten-traversal, high-gain amplifier. Each controller observation now
records the solver's thermal-energy, population-photon and optical-energy
balance residuals. A supervised four-update smoke after these changes reached
20.413 °C peak disk temperature; maximum reported thermal relative balance
error was 5.2e-8, population-photon relative balance error 5.8e-13, and
absolute optical-energy residual 5.0e-24 J. The 32² optical grid does not
certify mesh convergence or experimental accuracy.

The present Yb:YAG solver does **not** expose a pulse-by-pulse inversion state
for reentry. Each in-situ solve recomputes its periodic population equilibrium;
thermal and sensor continuity are supported, but population continuity is not.
The room-temperature gain spectra and 293–300 K thermal material range remain
binding. Unsupported hot states are rejected. Scalar lumped thermal phase is
approximate; photoelasticity, full pulse spectra, ASE, and measured hardware
calibration remain unavailable. Camera pixels do not refine the optical grid;
the output records pixels per beam waist and flags coarse sampling. A vortex
handedness check for one two-arm configuration does not prove general phase
uniqueness. Camera-loss improvement does not guarantee improved coherent
overlap, so both are reported separately.

The saved `request.json`, `provenance.json`, `execution.json`, `execution.log`,
`progress.json` and `result.json` identify every episode. `provenance.json` records the
configuration and numerical-source hashes; `execution.json` records elapsed
time, status and peak memory. `result.json` contains command timestamps, accepted
iteration maps, per-measurement camera/noise summaries, reference observations,
independent final measurement, and a
separate truth-assisted validation section. It is a numerical demonstration,
not experimental validation or a dataset-ready claim.
