# Initial measured-input modal model

## Larger all-local run (29 September 2026)

The 1,296-trial local dataset contains 144 independent crystal setups per
train/validation/test split. There are 430 verified training labels; 189
require a nonzero correction. The same measurement-only input channels are
used. `examples/train_ybyag_modal_nn.py` trains the 64×64 spatial network and
regularized modal baseline. `examples/evaluate_ybyag_modal_ensemble.py`
selects a candidate using validation only; it chose a 50/50 blend. The final
SLM proposal uses 14 pupil-weighted Zernike coefficients. Only two training
labels used the additional 3×3 residual modes, so those modes are disabled.

| Split | Verified trials | Blended modal error | Zero-step error |
| --- | ---: | ---: | ---: |
| Validation | 432 | 0.1877 rad | 0.2456 rad |
| Test | 423 | 0.1841 rad | 0.2182 rad |

These errors are offline weighted modal-phase proxies. They are **not**
propagated output-beam metrics. The test split was inspected during model
development, so its result is exploratory; final qualification needs fresh
setups and fresh solver replay of predicted SLM commands. On the current test
split, the blend improves all four target groups in modal error, with the
smallest margin on needle beams. It can still harm an individual beam.

Assets are under
`/media/aidas/Windows-SSD/FTMC/YbYAG-datasets/big_20260928/training/modal_1296/`.
Generate a candidate from measured trial inputs with:

```
.venv/bin/python examples/evaluate_ybyag_modal_ensemble.py predict \
  /media/aidas/Windows-SSD/FTMC/YbYAG-datasets/big_20260928/training/modal_1296 \
  PATH/TO/point_000.json config/ybyag_nn_dataset.json PATH/TO/candidate.npz
```

## Earlier 324-trial pilot

The combined 324-trial illustrative dataset was packed on the laptop and
trained locally. The input contains two phase-diverse camera planes, the known
relative Yb map, incoming beam shape, target mask, current SLM command, pump
and seed settings, and three temperature-probe readings. Physical truth is
opened only for supervised labels. Complete crystal setups stay in their
original train, validation, and test splits.

`examples/train_ybyag_modal_nn.py` offers a convolutional prototype and a
regularized linear modal predictor. The latter is the selected checkpoint:
the convolutional model overfit 108 training trials and was worse than zero
correction on validation. Validation selected ridge penalty 10 and command
scale 0.25 without using test results.

The selected model predicts up to 23 modal coefficients. This campaign's
train split has only 14-mode successful labels, so the inference path
reconstructs the first 14 pupil-weighted Zernike modes. The 3×3 residual
stage cannot yet be learned from this train split.

| Held-out split | Verified trials | Predicted coefficient error | Zero-step error |
| --- | ---: | ---: | ---: |
| Validation | 106 | 0.2498 rad | 0.2578 rad |
| Test | 108 | 0.2514 rad | 0.2656 rad |

These are weighted modal phase-error proxies, not output-beam fidelity.
The mean test result hides target variation: the predictor improves the
Gaussian and vortex subsets, but slightly worsens flat-top and needle
subsets relative to zero correction. It must remain a proposed action with
solver verification, especially on those two targets.
The model produces a **candidate** 384×384 phase command. It has not been
accepted by a fresh amplifier/thermal solver run, so do not treat these
numbers as closed-loop correction success. The original simulator-based
teacher and command verification remain necessary until a solver replay of
model predictions is implemented and passes held-out tests.

The model, report, and preprocessing provenance are in
`results/ybyag_initial_nn/modal_324/`. Generate a command for a trial with:

```
.venv/bin/python examples/train_ybyag_modal_nn.py predict \
  results/ybyag_initial_nn/modal_324/ridge_model/ridge_model.npz \
  PATH/TO/point_000.json config/ybyag_nn_dataset.json \
  PATH/TO/candidate.npz
```
