# One-shot certified controller (V3)

Implementation: `src/ybyag_control/nn_v3.py`, `examples/train_ybyag_controller_v3.py`,
tests in `tests/test_ybyag_nn_v3.py`. Requires PyTorch (`pip install torch`);
the rest of the repository does not.

V2 regresses 14 modal coefficients from a few hundred solver-verified teacher
labels, then compares hold, full and half steps with a learned critic. Its
exploratory replay improved fidelity in 10/12 cases (mean +0.068). V3 targets a
**single applied correction per measured trial** that is either large enough to
reach the task directly or is withheld.

## What changed

| | V2 | V3 |
|---|---|---|
| Training data | 430 teacher labels (189 nonzero) | unlimited sampled passive states, no teacher labels |
| Training loss | coefficient regression + critic MSE | corrected coherent fidelity through a differentiable optical model, shape/pump penalties, state likelihood |
| Action | 14 Zernike modes | 14 modes + 12×12 smooth residual map |
| Use of measurements | CNN features only | CNN proposal, then maximum-a-posteriori phase-diversity fit to both frames |
| Uncertainty | empirical envelope of critic errors | Laplace posterior + explicit phase-retrieval twin hypotheses, per-target calibrated margin |
| Release rule | critic lower bound vs hold | worst posterior-sample gain minus calibrated margin ≥ 0.01, shape and pump-overlap guards on every sample, frames consistent with the model |

## Pipeline for one trial

1. **Inputs.** Only `input__*` arrays and nominal design: both camera frames
   (area-binned onto the 192² optical grid with shot/read/dark/PRNU variance,
   saturated and hot pixels excluded, no recentring), incoming beam shape,
   delivered SLM command, target mask, seed waist and pump radius.
2. **Network.** `OneShotNet` (FiLM-conditioned residual CNN, 2.6 M parameters)
   sees measured amplitudes, validity, the nominal zero-aberration prediction for
   the current command, the desired amplitudes and the SLM-plane field. It returns
   a full correction step and a 30-parameter state estimate: 14 SLM-plane modes,
   14 disk-plane modes, SLM global-gain error and pump-shaped disk log-gain.
3. **Refinement.** L-BFGS fits the state to both frames through the frozen
   passive model (quantized SLM rendering identical to `apply_slm`, angular-spectrum
   propagation, disk screen). The unknown camera scale is eliminated in closed
   form. Starts: every ensemble member, zero, and three **twins** of the best fit.
4. **Posterior.** Gauss–Newton covariance with the likelihood weakened by
   `max(1, χ²/dof) × covariance_inflation`; retained twin/alternative minima within
   `hypothesis_chi2` are sampled too.
5. **Plan.** The step is re-optimized to maximize mean fidelity over posterior
   samples, starting from the network step and from zero.
6. **Certificate.** Hold, network, planned and 0.75/0.5/0.25× planned steps are
   scored on every sample with exact quantization and crosstalk. A candidate is
   released only if the calibrated lower gain ≥ `min_gain`, shape ≥ max(0.95,
   hold − 0.012) and pump-overlap ratio ≥ 0.9 on **every** sample, and the
   relative amplitude residual is ≤ 0.12. Otherwise the command is unchanged.
   Uncalibrated models always hold (`hold_uncalibrated`).

The released command still needs measured verification after application:
`verify_measured_outcome` (re-exported from V2) gives the rollback decision.

## Phase-retrieval ambiguity (important)

The dataset cameras sit at the disk-exit plane and 1 m beyond it, inside the
~1.1 m Rayleigh range. From two such intensities, φ(x) and its twin −φ(−x)
are nearly indistinguishable. Zernike Zₙᵐ maps to −(−1)ⁿZₙᵐ, so **even orders
(defocus, astigmatism, spherical, i.e. the thermal lens) are sign-ambiguous**
and disk-plane phase is invisible at the disk camera. V3 therefore refines
explicit twin starts (all even modes, SLM-plane only, disk-plane only). A
state is certified only when the frames exclude the harmful twin. The unit
tests plant both cases: odd aberrations are corrected in one shot (0.80 → 1.00),
and an even-order disk screen is recovered only after the twin restart.
A third camera plane, or a known astigmatic diversity arm, would remove most
of this ambiguity in future datasets. `planes_m` in the config is read by V3.

## Commands

All expensive commands run as workers under `hoyag.local_supervisor` and the
persistent `.local_runtime/budget.json` ledger (900 s per attempt). They resume:
training from `checkpoint.pt`, evaluation/calibration from their row caches.

```bash
# label-free training (≈0.06 s/step at batch 32 on an RTX-class GPU)
python examples/train_ybyag_controller_v3.py fit MODEL --steps 8000 --device cuda
# per-target certificate margins on independent synthetic outcomes
python examples/train_ybyag_controller_v3.py calibrate MODEL --samples 160
# held-out synthetic end-to-end check with the calibrated gate
python examples/train_ybyag_controller_v3.py evaluate MODEL MODEL/eval.json --samples 64
# one measured dataset trial -> replay-ready candidate + certificate
python examples/train_ybyag_controller_v3.py predict MODEL TRIAL.json OUT/candidates/CASE.npz
python examples/replay_ybyag_nn_candidate.py TRIAL.json config/ybyag_nn_dataset.json \
  OUT/candidates/CASE.npz OUT/reports/CASE.json
# replace synthetic margins with full-solver outcomes (≥20 replays per target)
python examples/train_ybyag_controller_v3.py calibrate MODEL --replays OUT
```

`predict --proposal-only` exports the best candidate even when uncalibrated,
for offline solver replay only. Calibration is invalidated by any weight change.

## Local results (29 September 2026)

All runs used the supervisor ledger on one Windows PC with a CUDA GPU; model
files are outside the repository (`../v3_runs/model_a`).

- **Training:** 8000 steps, batch 32, one member, 488 s, 1.7 GB peak RSS. The
  network's own step on 256 fixed validation states gave mean corrected fidelity
  0.955 (mean gain +0.157).
- **Calibration:** 160 independent synthetic states, per-target 95% coverage.
  Fitted margins release Gaussian and helical corrections. For flattop and needle
  no margin separated the harmful re-planned steps, so these targets hold.
- **Held-out evaluation:** 160 further states (seed 424242, unwidened ranges),
  3.9 s per trial in single precision on the GPU:

| | Result |
|---|---|
| Corrections released | 59/160 (37%): Gaussian 25/37, helical 34/36, flattop 0/46, needle 0/41 |
| Mean true gain of released corrections | +0.111 |
| Released corrections that lowered fidelity by >0.005 | **0** |
| Released corrections that failed the true shape/pump guard | 1 |
| Certified lower bound held (true gain ≥ certified) | 96.6% |
| Trials meeting fidelity ≥ 0.9 with guards | 48.8% before → 66.9% after one application |
| Ungated network step (all 160) | mean +0.098, 3 harmful |
| Ungated re-planned step (all 160) | mean +0.083, 8 harmful |

These are synthetic passive-plant numbers. They are not comparable to V2's
full-solver replay (+0.068 mean on 12 cases). That comparison needs
`predict` + `replay_ybyag_nn_candidate.py` on dataset trials, which were not
available on this machine. The network step is the strongest candidate, while
the posterior re-plan overfits model error. Flattop and needle need either a
third camera plane or a larger calibration set restricted to network steps.

## Multipass geometry

`Spec.from_config` reads `signal_traversals` and the `multipass` block (default
24-pass image relay, or the mirror array), or the legacy uniform path. The optics apply the same per-encounter
geometry as the solver: the disk phase state is split over the encounters, each
seen at its own angle of incidence, and propagates through its own path and
array mirror. A unit test checks this against the teacher's operator to 1e-10,
including mirror pointing errors. The synthetic plant samples a fixed pointing
error for every mirror (`mirror_tilt_error_urad`). The controller's own model
uses the nominal mirrors, so beam walk from unknown mirror errors must be
absorbed by the SLM-plane state and the calibrated margins. The `model_a`
results above were trained and calibrated with the earlier single-screen
geometry. That model keeps its stored spec, so it remains self-consistent, but
it does not describe the mirror array. Retrain and recalibrate before using V3
with the current configuration.

## Bundled model (`models/ybyag_v3_model_b`, 1 October 2026)

Trained for the current default plant: 10 at.% Yb, measured Yb maps, 24-traversal
image relay with mirror pointing and relay focus errors
(`config/ybyag_nn_dataset_10at.json`). The repository holds `model.json`,
`calibration.json`, `train_state.json` and the three members' `weights.pt`
(about 10 MB each). The resumable `checkpoint.pt` files are not included.
`config/ybyag_control_nn_v3.json` and the desktop default point here.

- **Training:** 3 members, 25 000 steps, batch 16. Each member's own step on the
  fixed validation states gives mean corrected fidelity 0.959 (gain +0.076).
- **Calibration:** 160 synthetic passive states in single precision, 95% target
  coverage, separate margins for network and re-planned steps. All four targets
  are calibrated. Released fraction (harmful releases in the calibration set):
  Gaussian 39% (0), flattop 21% (0), helical 57.5% (0), needle 75% (0).
- **Not done yet:** held-out synthetic evaluation and full-solver replay on
  dataset trials.

The calibration stores SHA-256 fingerprints of the weights. Any retraining
invalidates it, and the model then holds.


- Training states come from a frozen passive model: SLM-plane aberration, SLM
  gain/quantization/crosstalk, disk phase and pump-shaped log-gain, cell-binned
  camera noise. The pulsed amplifier's nonlinear gain, the full thermal solve and
  gain narrowing are not in the training loss. Synthetic numbers are not solver
  or experimental results.
- The certificate is empirical: coverage is measured on the calibration source.
  Synthetic calibration does not certify the amplifier; full-solver replay rows
  per target are required before any hardware use, and fresh setups before any
  qualification claim.
- The pump-overlap ratio is a proxy for the dataset's 80% energy-retention guard.
- The 12×12 residual cannot represent features finer than ≈0.24 mm at the SLM.

## In-situ use in the closed-loop app

`ControllerConfig.method = "nn_v3"` runs the certified NN inside the in-situ
simulated experiment (`examples/ybyag_control.py`, desktop **Closed-loop
correction…** → Controller method `nn_v3`, preset `config/ybyag_control_nn_v3.json`).

- **Measurements.** The plant switches to `measurement_mode="dataset_camera"`:
  two phase-diverse planes (`planes_m` of the model's training config) on the
  dataset detector (1920×1080 over 10 mm, same noise model, exposure set to half
  full well at the brightest plane). The NN receives only these frames, the
  delivered SLM command, the known target mask and nominal seed/pump settings.
- **Geometry.** `episode_for_nn_model` forces grid, disk passes, seed, pump and
  thermal mesh to the model's training configuration. Target, in-situ mode,
  noise and drift settings stay user-selectable.
- **Cycle.** Measure the held command → NN proposes a full SLM command or holds
  → apply → measure. The command is kept only if the measured camera loss does
  not rise beyond `improvement_tolerance` and the photodiode energy stays above
  `minimum_energy_fraction`; otherwise the previous command is re-applied and
  measured again (physical time is not rewound). `target_confirmations`
  consecutive certified holds end the episode.
- **Safety.** An uncalibrated model always holds. `nn_allow_uncalibrated: true`
  releases proposals for offline testing only.
- **Cost.** At the 384² grid with 24 disk crossings each observation is one full
  pulsed/thermal solve (~2–3 min). The preset sets `slm_delay_s = 0` to avoid a
  second solve per observation, and `time_limit_s = 7200` for the supervised
  worker.
