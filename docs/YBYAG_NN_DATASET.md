# Yb:YAG physical-disturbance NN dataset

Open the native Yb:YAG tab and choose **Generate NN dataset…**, or run from the repository root:

```bash
.venv/bin/python examples/ybyag_nn_dataset.py --config config/ybyag_nn_dataset.json --output results/my_dataset --workers 6
```

The command uses the repository's persistent bounded compute ledger. A production run has a three-hour per-run limit by default. If it stops at that limit, rerun the same command with `--resume`; complete setups are skipped and an interrupted setup is replayed from its start. The manifest records source revisions if code changes between segments. The Tkinter dialog also resumes when given an existing output directory. `split_counts.stress` adds a separate out-of-distribution set; the normal splits are never made by shuffling adjacent frames. `--points-per-setup 3` saves a baseline and at least two subsequent measured SLM trials per fixed setup. The JSON controls disturbance ranges, enable switches, illustrative camera parameters and camera planes. A production campaign is **not** launched automatically.

### 10 at.% overnight campaign configuration

`config/ybyag_nn_dataset_10at.json` differs from the default only in:
- `yb_at_percent_candidates: [10.0]`;
- `pump_W: 0.2`, about 298 K peak. The solver accepts at most 300 K because
  hot-gain data are missing; heating is ≈21.6 K/W, so 0.3 W is already
  rejected. This gives ≈0.6 rad peak-to-valley thermal phase over the beam;
- `modal_target_fidelity: 0.96`, so the teacher also corrects states that already
  exceed 0.9;
- 64/8/8 train/validation/test setups with 3 trials each (240 trials).

The generator now allows a single-concentration plan, and the checker takes the
required concentrations from the plan. Parallel runs process validation and
test setups first. A setup that raises (for example, a state outside 293–300 K)
is recorded under `failed_setups` in the manifest instead of ending the run.
The periodic population solver extrapolates its fixed point (per-pixel Aitken).
Convergence is still tested on an evaluated cycle at 1e-6, and
`YB_PERIODIC_ACCELERATION=0` disables it.

### Measured relative Yb distribution (default)

`ranges.yb_distribution: "measured"` replaces the seeded random Yb map with the
PL-mapped relative distribution of the matching sample: 5, 10 or 15 at.%
(`src/ybyag_dataset/data/doping_maps`, 0.1 mm grid, 1580–5533 points per sample). The
mapped quantity is R = (PL969/laser)/(PL1030/laser), divided by the sample's
interior median. It is a within-sample spectroscopic **proxy**, not a
calibrated concentration: surface/coating features and setup variation can
contribute, and the 5 at.% map shows a broad gradient that still needs an
independent check. `provenance.json` records the source scans and SHA-256 hashes.

The relative Yb scale is 1 + `yb_map_deviation_scale` × (R − 1), with the
scale set to **−1.7**. The sign is negative because 969 nm PL overlaps the
zero-phonon absorption: more Yb reabsorbs more 969 nm light and lowers R. The
sample medians confirm it (R = 0.312, 0.207 and 0.187 at 5, 10 and 15 at.%).
The magnitude is 1/(d ln R/d ln C) from the 5→10 at.% pair, −0.59. The 10→15
pair gives −3.9, but its 15% absolute ratio is not reconciled. Processing:

- the unstable outer 0.3 mm of each map's mask is trimmed (`yb_map_edge_trim_mm`);
- holes are filled from the nearest point and the map is smoothed by 0.1 mm;
- outside the measured 4–8 mm region the deviation fades to 0 over 0.5 mm;
- each setup places the map with a seeded ±0.5 mm offset and a random rotation
  (`yb_map_offset_mm`, `yb_map_rotation`), so setups still differ.

With scale −1.7 the relative Yb variation within 1.2 mm of the axis is about
3.1% (5 at.%), 2.7% (10 at.%) and 0.9% (15 at.%) RMS. At 10 at.% that is about
−0.7…+0.4 at.% under the beam, and within ±1 at.% over the whole trimmed map. The 1.5% relative cap
(`yb_concentration_max_fraction`) applies only to `yb_distribution: "random"`.
Concentrations without a measured map raise an error unless
`yb_map_nearest_sample` is true. The controller plant and the desktop Yb:YAG
pulsed calculation use the nearest measured sample by default. The desktop
offers `random_clusters` and a manual map rotation and offset.

### Doping bound and multipass propagation

The Yb doping map is now bounded by `yb_concentration_max_fraction` (1.5%):
the correlated map is scaled so its **largest** local deviation equals that
fraction, for every split including stress. Configurations without the key keep
the older `yb_concentration_rms_fraction` behaviour. With doping this uniform,
the dominant disturbance is the phase accumulated over the ten disk encounters.

The default multipass is an **image relay with 24 crystal traversals = 12
disk bounces** (`nominal.multipass.layout: "image_relay"`,
`signal_traversals: 24`), matching the YbSLAM proposal's structured-light
requirement. A bounce is two traversals (in, HR back reflection, out) at one
angle of incidence with no relay in between. The 11 relays connect successive
bounces. Every return path is a unit-magnification 4f
relay (for example a parabolic mirror with fold prisms, `relay_focal_m` = 0.5 m).
It images the disk back onto itself, so the shaped pattern arrives re-imaged
(point-inverted) at every pass instead of diffracting between passes. What still
differs per pass is:

- the angle of incidence, linearly spaced over ±`max_incidence_deg` (8°), with
  the same stretched, Snell-scaled disk phase as below;
- a lateral image shift 2εf from each fold mirror's fixed pointing error
  (`mirror_tilt_error_urad` = 5 µrad RMS, about 5 µm);
- a fixed path-length (focus) error of each relay (`relay_defocus_error_mm`
  = 0.5 mm RMS).

Both errors are saved per setup in `setup.npz` (`multipass_mirror_tilt_rad`,
`multipass_relay_defocus_m`) and reused by replays. Relay aberrations and
apertures are not modelled. A cold small-signal check with 20% Yb:YAG, 200 µm
thick and a 2 mm pump spot gave 1.6×10⁵ (200 W) to 3.2×10⁵ (300 W) gain in
24 passes, versus 180–240 in 10 passes. That check omits heating, concentration
quenching and ASE.

The alternative layout `"mirror_array"` (implemented in
`src/ybluag/multipass_geometry.py`): the seed fans between the disk and an array
of curved mirrors, so every reflection takes a slightly different path.

- Encounter k meets the disk from array height y_k (linearly spaced over
  ±`array_half_height_m` = ±0.08 m, array at `array_distance_m` = 0.5 m). The
  angle of incidence is 1–9°. The beam sees the disk phase map stretched by
  1/cos θ along the incidence plane and scaled by 1/cos θ_t for the longer
  internal path (Snell, n = 1.815).
- Path k returns through its own mirror: length 2·√(D² + y²), 1.000–1.010 m.
  The field propagates half the path to the mirror, receives its focusing and
  its pointing error, then propagates the other half back to the disk.
  Oblique mirror incidence (`array_mirror_incidence_deg` = 1.5°) makes the
  focus astigmatic: f·cos a tangential, f/cos a sagittal. With
  `mirror_focal_m: null` each mirror is mode-matched to a 0.6 mm waist on the
  disk, f = (L²/4 + z_R²)/L ≈ 1.45 m. The injected seed is not matched to that
  mode and still breathes.
- Every array mirror has a fixed pointing error for the whole setup
  (`mirror_tilt_error_urad` = 5 µrad RMS per axis, 2ε deflection). It is saved
  in `setup.npz` as `multipass_mirror_tilt_rad`, so replays rebuild the same
  geometry. The errors walk the beam on the disk: in a 192² check the output
  centroid moved by about 30–36 µm.
- After each diffraction the pixel temporal shape is replaced by the
  energy-weighted mean shape (a space-time separable approximation). The gain
  is sampled in beam coordinates, so the elongated footprint (below 1.5% at
  9°) and image inversion by the mirror sequence are not modelled.

The cold thickness/surface phase and, with any non-ideal relay, the thermal OPD
are applied at every encounter at that encounter's incidence. The solver
re-solves the optics once with half the round-trip thermal OPD on every
encounter. Heat and the thermal state come from the first solve: one
thermal-to-optical update, not a converged hot-cavity iteration. Results record
`ideal_relay`, `multipass_geometry` (angles, path lengths, focal lengths, mirror
errors) and `thermal_phase_per_encounter`. Without a `multipass` block, the
legacy uniform `inter_pass_distance_m`/`inter_pass_focal_m` path or the ideal
relay is used unchanged.

The modal teacher's passive model follows the same encounters, angles, paths
and mirror errors. In a 192² solve it reproduced the output field with overlap
1.000000. Ignoring the mirror errors gave 0.992, uniform 1 m paths at normal
incidence 0.992, and the old single lumped screen 0.960 (below the teacher's
0.99 acceptance threshold). The first-order oracle proposals of the legacy and
response teachers still assume one lumped screen; the full-solver verification
decides whether those proposals pass. Only `config/ybyag_nn_dataset.json`
enables these settings. The older configs are unchanged so their existing
datasets keep resuming.

The same mirror array is the **default everywhere a Yb:YAG ideal multipass is
solved**. The desktop Yb:YAG pulsed calculation (Amplifier & optics:
**Multipass layout**, array distance/half height, mirror incidence, mirror
pointing error and seed) and the closed-loop correction plant
(`EpisodeConfig.multipass_layout`, `mirror_tilt_error_urad`) both read their
defaults from this configuration through
`ybluag.multipass_geometry.dataset_defaults()`. Changing the `multipass` block
or `mirror_tilt_error_urad` here changes all three. Their reference solves (the
dashed desktop reference and the controller's ideal camera target) use the same
array with perfect mirrors. Select `ideal_relay` to recover the earlier unit
relay. The controller keeps its own traversal count (2 by default), so its
array has fewer encounters than the dataset's 10.

### Reduced random and thermal fluctuations

`config/ybyag_nn_dataset.json` now uses smaller trial-to-trial fluctuations so
the accumulated multipass phase dominates:

| Setting | Before | Now |
|---|---|---|
| Pump power / radius variation between trials | ±10% / ±10% | ±3% / ±3% |
| Pump pointing jitter | 1% of radius | 0.3% |
| Coolant setpoint range / thermal drift | ±0.4 K / 5 mK/s | ±0.1 K / 1 mK/s |
| Thermal-contact variation | 20% | 10% |
| Upstream residual (high-order) screen | 0.02 waves | 0.01 waves |
| SLM pixel gain noise / drift | 0.5% / 1e-4 per s | 0.25% / 5e-5 per s |
| Frame pointing jitter / alignment drift | 5 µrad / 0.02 µm/s | 2 µrad / 0.01 µm/s |
| Probe noise / drift | 0.1–0.5 K / 1 mK/√s | 0.05–0.2 K / 0.5 mK/√s |
| Camera dark / background / PRNU / DSNU | 0.1 e/s / 2 e / 1% / 0.5 e | 0.05 / 1 / 0.5% / 0.25 |
| Sensor-temperature jitter | 0.1 °C | 0.05 °C |
| Pulse energy / pointing jitter per exposure | 0.5% / 0.2 px | 0.2% / 0.1 px |

Photon shot noise and the 1.6 e read noise (ORCA-class specification) are
unchanged. The fixed per-setup errors that the controller must correct are also
unchanged: upstream Zernike aberration, thickness and surface maps, SLM
calibration and seed misalignment.

The default thickness-map RMS is 0.03% (about 30 nm for a 100 µm disk). The previous 1% setting produced about 17 rad of accumulated phase in a ten-traversal Gaussian pilot; with 0.25 m of free-space propagation from the phase-only SLM to the disk, the desired full field was unreachable even in the frozen passive model. The smaller illustrative manufacturing range keeps the same optical model and does not relax the label criteria. Unreachable sampled setups still retain their measured trials and reject correction labels.

The default `config/ybyag_nn_dataset.json` now selects `label_teacher: modal`. It uses the saved complex input field and accumulated cold/thermal phase as a fast passive proposal model, then replays selected **quantized SLM commands** through the existing full amplifier and thermal solver. The proposal fits 14 piston-free, pupil-orthogonalized Zernike modes. A 3×3 local residual stage is attempted only when the passive full-field bound permits the requested fidelity and the smooth stage has not succeeded. The two stages share `label_max_solver_evaluations`; fast optical iterations do not consume that full-solver count. The pilot has unit and manufactured-optics tests and one bounded three-trial Gaussian smoke, but no full campaign or mesh-convergence qualification yet.

The default complex-field task succeeds only if a fresh reference solve reaches `modal_target_fidelity: 0.9`, improves coherent fidelity by at least 0.01, maintains the shape guard, and retains at least 80% of baseline output energy. `improved_only` commands are saved separately from successful correction labels. The reported phase-only reachability bound assumes a frozen passive screen, fixed source amplitude and a unitary sampled propagation operator; it is a diagnostic, not a bound on every nonlinear amplifier state. When that bound falls below the requested fidelity, the teacher uses at most one full reference check and records the conditional limitation. A large target change can require a large command: the teacher optimizes continuous mode coefficients and wraps only the final SLM command. The reference solve uses the actual sampled SLM mapping and the same pre-action thermal state for each counterfactual candidate.

The bounded 5 at.% Gaussian smoke in `results/ybyag_nn_dataset/modal_teacher_final_smoke_20260928` passed file validation and fresh full-solver verification for all three trial labels. Corrected coherent fidelities were 0.9206, 0.9086 and 0.9010, with each trial meeting its beam-shape and energy guard. The last trial passed the shape guard by only about 2.2e-5, so this result does not establish robust yield across other setups, beam targets or sampling grids. The corrected search evaluates quantized SLM commands in 0.005 strength increments before spending a reference solve.

An optional `slm_phase_lut_rad` array in the configuration supplies one calibrated optical phase per SLM code; it must have `2**slm_bits` finite values. The exact reference path and fixed setup save and use that LUT. Without it, the documented illustrative linear phase response remains in use. An imperfect phase stroke can make the two ends of the command range optically different; the fast continuous optimizer is only a proposal and its exact quantized command must pass verification.

`--workers 6` (the CLI and Tkinter default) calculates up to six independent setups concurrently. Trials within each setup remain sequential to retain thermal and SLM history. Each worker writes into a private `.parallel_staging/` directory. One coordinator moves complete setup directories into the dataset and atomically updates `manifest.json`; the Tkinter progress bar advances in groups of trials. `--workers 1` retains serial execution. The shared budget supervisor still caps the complete process tree at 8 GiB and the selected wall-time limit. Six workers have not yet completed a resource-limit benchmark on this machine; lower the count if the run reaches `resource_limit`. Do not run separate generators against the same output directory. Interrupted staging directories are reused on resume. Replaced incomplete setup directories are preserved under `.parallel_staging/.../previous_final_N` for inspection.

For two machines, `--setups-per-combination 3 --shard-count 2` divides complete setup cycles without sharing a crystal between machines. Shard 1 receives one 12-combination cycle per split (108 trials at three points each); shard 0 receives two (216 trials). Each shard is independently valid and resumable, and the two manifests retain disjoint setup IDs and source hashes for later merging. The manifest records Python and numerical-library versions so results from unlike machines are traceable. Run each shard in a separate output directory with `--workers 6`; six processes may exceed available memory and the supervisor will stop the run at its configured 8 GiB tree limit.

The production setup plan contains each of 5, 10 and 15 at.% crossed with Gaussian, flat-top, vortex and needle targets **in each** of train, validation and test: 12 independent setups per split. Three measured SLM trials per setup produce 108 trial records. `--setups-per-combination 2` generates 216 trials with fresh physical setups; higher positive values have no software cap. `--points-per-setup` adds measured trials within each setup. Runtime and disk use scale with both settings; a timed-out campaign can resume. `--plan` reports the plan without running physics; `--smoke --smoke-setup-index 0..11` executes one unqualified setup from the first coverage cycle. Complete setup IDs, including the fixed crystal, SLM, optics, camera and probe session, stay in one split. The optical/SLM grid is 384² before rendering 1920×1080 camera frames. The detector resampling does not create additional physical detail, and per-target grid convergence is still required.

For a larger exploratory label set, use `config/ybyag_nn_dataset_verified_v2.json` with `--setups-per-combination 2`. It keeps the original phase-improvement and beam-shape acceptance thresholds. The label search proposes small positive and negative fractions of a simulation-truth crystal/thermal/external phase compensation, then other phase-only proposals. Each candidate is checked by a fresh full solver call; no failed candidate is exported as a valid correction. Both accepted and rejected measured SLM trials remain in the dataset. A larger number of solver-verified labels does not change `dataset_ready: false` or replace experimental calibration and numerical convergence checks.

For a bounded response-matrix pilot, use `config/ybyag_nn_dataset_response_teacher_smoke.json` with `--smoke --smoke-setup-index 0`. It probes the physical and external phase directions, then fits a regularized local complex-field response. The broader `config/ybyag_nn_dataset_response_teacher.json` probes up to eight directions, adding adaptive back-propagation, tilt, defocus and astigmatism. Both limit the proposed phase step, use the response to preview shape-safe fitted gains or two-direction combinations, verify up to three response proposals and three physics-based fallback steps with fresh full solves, and retain the **best passing** correction. The small pilot uses at most eight extra full solves per trial; the broader configuration uses at most 14. Both require at least 0.02 rad phase-RMS improvement and retain the existing shape guard. Failed probes and verification results are recorded in `correction_candidate_checks`; the measured closed-loop trial is still saved even if no label passes. These probe fields are privileged **simulated truth**, not camera measurements or a calibrated hardware response. Hardware use needs a complex-field measurement or validated phase-retrieval/calibration stage. This pilot has not established improved yields or experimental validity.

The two-mode Gaussian smoke in `results/ybyag_nn_dataset/response_teacher_2mode_smoke` completed three trials in 582 s with 1.04 GB peak process-tree RSS and passed the dataset integrity checker. All three labels passed. In the same seeded setup, two selected corrections matched the earlier oracle search; the third reduced phase RMS by 0.0233 rad, versus 0.0185 rad previously. Every selected correction came from a physics-based fallback, not the fitted response update. An earlier ten-mode pilot's first trial failed its then stricter 0.05 rad threshold and was stopped. These small checks establish bounded operation and fallback behavior, not a gain in response-fit quality or generalization.

The shape-aware and pairwise response pilots also completed three-trial Gaussian smokes (`response_teacher_shape_preview_smoke`, `response_teacher_pairwise_smoke`) in 586 s and 583 s, respectively. Both passed dataset checks and reproduced the same three selected fallbacks and phase improvements. Thus the extra response probes **did not improve the selected correction** on this setup. This is a measured negative result; do not use the extra cost for a production campaign without broader target/setup evidence or a revised phase/shape objective.

The default nominal pump is deliberately **0.1 W** so the present Yb:YAG material/thermal model can produce bounded, near-room-temperature examples with measurable thermal variation. It is not the proposal's high-power operating point. Raising pump power toward the intended amplifier regime will normally make the generator reject the state until temperature-dependent pump spectra and high-temperature assembly properties are supplied.

For sequences, the first trial evolves from a uniform coolant field for `operation_duration_s`; each following trial evolves the **previous solved disk and copper temperatures** for `sequence_step_s`. The next SLM proposal is a coordinate-search perturbation around the command retained by measured camera feedback, never by the oracle correction label. `slm_update_delay_points=1` records a delivered prior command as a delayed measured trial. Both requested and delivered commands are saved. The operating heat is a quasi-steady pulse-cycle estimate within each interval; pulse-by-pulse thermal motion is not resolved.

For each operating point, the generator draws pump power/radius and seed energy/waist, coolant and alignment changes. A fixed crystal carries smooth Yb, thickness, background-loss and surface maps plus a spatial contact map. These enter the existing Yb:YAG pulsed amplifier before the optical heat is sent to the copper/contact thermal solve. The Yb map also changes local conductivity using the measured concentration trend; at 15 at.% the relative trend from the 300 K HT family is a stated proxy because the CT family ends at 15 at.%. The same returned thermal state supplies temperature, thermomechanical displacement, scalar OPD and the five probe readings. A fixed session carries upstream Zernike/residual phase, SLM calibration maps, camera pixel patterns and alignment bias. The SLM target mask remains separate from correction and applied phase. The solver propagates the resulting field through its selected multipass geometry. The detector then propagates the complex output to focus and a known defocus plane, resamples it, counts photoelectrons and applies shot, dark, read, fixed-pattern and ADC effects. Each setup has its own `setup.npz` containing the fixed maps.

Every `point_###_measurements.npz` saves noisy camera planes, three disk-surface probe readings (and two plate probes), the assumed known relative Yb map and incoming beam shape, pump and seed settings, optical dimensions, intended target mask, and delivered/requested SLM commands. The illustrative room-temperature input equals the fixed heat-sink boundary temperature; independent ambient coupling is not solved. A separate `point_###_truth.npz` stores the actual Yb concentration map in at.%, solved temperature maps, complex input beam at the SLM, complex disk-input beam, complex output beam, their fluences, intended target, clean camera planes, disk/plate temperatures, displacement, material maps and true probe states. The JSON records the measured camera loss, accepted/rejected/delayed trial outcome, setup/source hashes and validity. A coordinate-search proposal is retained only when its **measured** phase-diverse camera shape loss improves; all measured candidates, including rejected ones, are saved.

The correction label is a **signed incremental phase update** relative to the delivered SLM command, with the complete verified command saved separately. The legacy teacher back-propagates the current and desired complex fields to propose phase-only updates, ranks several gains using a cheap optical preview, then reruns the leading candidates through the physical forward solver at the same operating state. It stops at the first verified correction. The response-matrix pilot instead fits local probe responses and checks its line-search gains, selecting the best passing gain. Either label is saved only if a fresh full run lowers phase RMS by at least `label_min_phase_improvement_rad` and preserves target beam-shape overlap above both `label_min_shape_overlap` and the baseline overlap minus `label_max_shape_drop`. Every tested candidate and its metrics are recorded. A global phase-only optimum is not established. Failed labels and solver exceptions leave the measured trial intact, with NaNs and `correction_label_valid=false`. A `point_###.png` figure displays the trial.

For the modal teacher, `truth__modal_coefficients_rad` and `modal_basis_names` identify the verified smooth or smooth-plus-residual command. An already satisfactory state gets a freshly verified zero-update `modal:hold` label. `truth__best_improved_slm_command_rad` stores a genuine verified improvement even if it misses the absolute task criterion. Metadata records coherent fidelity, energy retention, reference evaluation count, passive reachability and a status of `task_success`, `improved_only`, `control_limited_passive`, `search_incomplete` or `numerically_unresolved`. The dataset checker validates the modal fidelity, shape, energy and coefficient criteria. Old datasets retain their historical phase-RMS label criteria.

Run `python examples/check_ybyag_nn_dataset.py PATH_TO_DATASET` to check file hashes, split isolation, 5/10/15 at.% × four-target coverage in each split, complete trial sequences, separate measurements/truth, input/output complex-field energy consistency, 1080p phase-diverse stacks and valid correction masks. The report is `validation.json` and counts verified and rejected correction labels; passing it establishes file/software integrity only. Earlier v2 datasets remain readable but retain their failed first-order labels; v3 generation requires a new output directory.

### Illustrative detector and SLM anchors

The selected detector is **not** a calibrated device. The example 30,000-electron full well and 1.6-electron RMS read noise are within the [Hamamatsu ORCA-Flash4.0 V3 specifications](https://www.hamamatsu.com/us/en/product/cameras/cmos-cameras/C13440-20CU.html). That camera has 2048² native pixels; a 1920×1080 region is plausible, but the manufacturer's published spectral-response plot in its [technical note](https://www.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/sys/SCAS0134E_C13440-20CU_tec.pdf) ends at 1000 nm. Consequently the example 5% quantum efficiency at 1030 nm is **assumed**, not a quoted ORCA value. A [Hamamatsu InGaAs camera](https://www.hamamatsu.com/eu/en/product/cameras/ingaas-cameras/C12741-03-02.html) explicitly covers 950–1700 nm, but its 640×512 detector does not supply 1080p pixels. The 8-bit SLM command in this model resembles the [HOLOEYE PLUTO-2.1](https://holoeye.com/products/spatial-light-modulators/pluto-2-1-lcos-phase-only-refl/) addressing specification; its actual phase-versus-grey-level response, fill-factor transfer and 1030 nm calibration are not imported. PSF width, throughput, PRNU/DSNU and pulse jitter remain illustrative. The camera model samples pulse-to-pulse energy and pointing variation over each exposure, using their effective Gaussian blur before shot/dark/read noise and ADC conversion. No device-specific calibration or measured MTF is claimed.

## Physical limits and qualification

The generator rejects a sample if the requested Yb:YAG thermal state leaves the current 293.15–300 K parameter range. Recovered spectra now include 1.1 at.% ZPL measurements up to 300 K, 5/10/15 at.% room-temperature absorption coefficients, and figure-derived 25 at.% pump/emission endpoints at 300/450 K. These separate samples do not supply concentration-matched hot gain for the generator. Its `lumped_phase` mode uses room-temperature gain, computes heat and deformation, then applies a scalar thermal phase after amplification. This is not fully coupled hot gain. The material thickness map changes the optical ion column and cold geometric phase, while the finite-element mesh keeps nominal thickness. Background absorption uses a thin-disk first-order pump attenuation and heat approximation. Photoelasticity, a stress tensor for NN labels, camera/SLM hardware calibration, gain narrowing, ASE and nonlinear phase are unavailable. All outputs are marked `dataset_ready: false`; do not use them as experimentally validated training truth until these limits and detector ranges are resolved. The saved three disk probe readings are modeled noncontact surface measurements; they do not reveal the full internal temperature distribution to the NN.

The dataset worker suppresses the repeated `Interpolating thermal resistivity across different dopings` warning in its execution log. The approximation remains active and is disclosed in the saved physical limits; the warning is not a calculation error.

Stress samples extend alignment, contact and external-aberration ranges, raise detector exposure and increase dropout probability. They remain in a separate split. Inspect the saved saturation fraction and validity metadata before using them.
