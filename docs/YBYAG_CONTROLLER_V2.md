# Initial multiscale neural controller

Implementation: `src/ybyag_control/nn_v2.py` and
`examples/train_ybyag_controller_v2.py`.

This is a trainable replacement architecture, not a newly qualified checkpoint.
Existing simulator physics and the previous trained ensemble remain available.
No training campaign or full-solver campaign is launched by importing the module.

## Implemented behavior

- Physically sampled 192×192 overview plus a 128×128 camera-detail crop at native
  pixel spacing. Camera aspect ratio is preserved with validity masks; decimation
  is antialiased. Registration uses a supplied affine matrix or explicitly nominal
  geometry. Hidden camera distortions are never read from the setup truth file.
- A separate desired-target branch, shared spatial features, and four family
  adapters. Known pupil/propagation design supplies the nominal desired intensity.
- A 14-mode phase-step proposal relative to the measured current command, without
  a fixed small-step cap. It can propose the entire required modal change. This
  initial model has not been trained on target-switch transitions and is not an
  absolute target-switch controller yet.
- An action-conditioned outcome estimator predicts changes in coherent fidelity,
  shape overlap and log energy ratio. Zero action has exactly zero predicted
  change by construction. A separate head estimates current fidelity and shape.
- Selection compares hold, the full proposal, and half the proposal. It requires
  conservative empirical validation margins, supported action ranges and context.
  It holds by default if calibration is missing or insufficient. A predicted
  acceptance still requires measured verification; it is not a solver guarantee.
- A measurement-verification helper returns acceptance/rollback decisions for
  shape and energy. Applying/restoring commands and acquiring fresh measurements
  remain responsibilities of the plant adapter. This implementation does not
  automatically connect the NN to the Tkinter application or physical hardware.

Temporal GRU state, multiple alternative full proposals, 27 modes, spline residuals,
unrolled physics optimization and axial needle objectives remain later stages.
The current data do not support claims that those components have been trained.

## Data integrity and calibration

The packer opens only measurement arrays for network inputs. Truth files supply
training labels. Whole setups must remain in one split. It verifies the exact
phase command represented by each teacher label; outcomes from a higher-order
teacher action are not assigned to its truncated 14-mode approximation.

The predictor requires two phase-diverse camera frames. Missing temperature
probes are masked. Other unsupported/missing inputs raise an error rather than
silently inventing a wavefront.

Training uses train setups. Deterministically, roughly one third of validation
setups selects checkpoints; the remainder is reserved for calibration and never
used for training or early stopping. The test split is not evaluated by fitting.

Calibration requires fresh, fingerprinted solver replays of this model's own
nonzero proposals on at least 20 reserved validation setups per target, including
at least five setups with harmful proposals. Its uncertainty margins are empirical
envelopes of simultaneous baseline/outcome errors across those setups. They are
not a theorem about arbitrary new systems. Candidate selection also checks action
coordinate ranges and a six-standard-deviation context envelope; these checks are
conservative heuristics, not complete out-of-distribution detection. Insufficient
counterfactual coverage leaves the target uncalibrated and the controller holding.

Solver reports must contain trial/config/candidate fingerprints and reproduce the
original baseline. Older reports without candidate hashes cannot calibrate this
controller. Replaying a teacher command or using test results does not qualify as
validation of this model's action distribution. Negative replay examples can be
included in training when their complete setups belong to the training split.

## Commands

Run from the simulator repository, using its environment with TensorFlow installed.
The cached trial files are streamed during training, rather than keeping all
1080p measurements and truth fields in memory.

```bash
.venv/bin/python examples/train_ybyag_controller_v2.py pack \
  DATASET config/ybyag_nn_dataset.json CACHE

.venv/bin/python examples/train_ybyag_controller_v2.py fit \
  CACHE MODEL --epochs 30 --seconds 900

.venv/bin/python examples/train_ybyag_controller_v2.py predict \
  MODEL TRIAL.json candidate.npz
```

`fit` uses `hoyag.local_supervisor`, the shared persistent budget ledger, an 8 GiB
process-tree memory limit, and the configured wall-time limit. It does not reset
an exhausted ledger. Logs and a runtime record are written beside MODEL. The
training worker writes checkpoints on validation improvements; interruption leaves
the model explicitly incomplete and unavailable to the prediction command.

To collect offline verification candidates before calibration:

```bash
.venv/bin/python examples/train_ybyag_controller_v2.py predict \
  MODEL TRIAL.json REPLAYS/candidates/CASE.npz --proposal-only
```

This flag exports the full proposal for an offline solver test and explicitly
bypasses the application gate. It is not authorization to apply it to hardware.
Run the existing `examples/replay_ybyag_nn_candidate.py` through the local
supervisor to produce `REPLAYS/reports/CASE.json`. File stems must match.

Repack into a new empty cache with `--replays REPLAYS`, then run:

```bash
.venv/bin/python examples/train_ybyag_controller_v2.py calibrate NEW_CACHE MODEL
```

Repacking may add counterfactual outcomes, but source trials, setup splits and
configuration must match the fitted dataset. Calibration is invalidated by model
weight changes. Additional training data require a new fit and new calibration.

No latency, optical performance improvement, or experimental validity is claimed
until this architecture is trained and tested with independent full-solver replays.
The existing test split has been inspected during model development; use fresh
unseen setups for final qualification.

## Local training and exploratory solver replay (29 September 2026)

The first full local V2 run crashed in TensorFlow's oneDNN CPU path after packing
1,296 trials. Training was restarted from that cache with oneDNN disabled. The
bounded run completed 18 epochs with early stopping; epoch 13 had the best
validation loss, 0.0290866. The selected proposal and outcome-estimator weights
are saved locally under
`/media/aidas/Windows-SSD/FTMC/YbYAG-datasets/big_20260928/training/controller_v2_20260929_094436/model_cpu_20260929_102316`.

An offline proposal-only trial used a fresh seeded full-solver baseline and
candidate replay for one active point-2 setup in each of the 12 target/doping
combinations. All baselines reproduced the saved complex fields within relative
L2 error 2.5e-6. Fidelity improved in 10/12 cases and worsened in 2/12; mean
gain was 0.0680. The prior ensemble's mean gain on those exact cases was 0.0599.
Nine V2 cases passed the shape and energy guard, and six met every task criterion.
The vortex and needle losses were -0.0212 and -0.0158 fidelity, respectively.

The per-case report is at
`/media/aidas/Windows-SSD/FTMC/YbYAG-datasets/big_20260928/training/controller_v2_20260929_094436/model_cpu_20260929_102316/solver_tryout_12/summary.json`.
These setups had already been inspected during model development. This replay
is exploratory and does not calibrate the application gate. Validation-set
replays of this model's own commands remain necessary before automatic action;
the deployed policy currently holds.
