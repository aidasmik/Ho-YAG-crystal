# NN dataset runs (metadata only)

The full datasets are too large for git: `ybyag10_night_20260930` is 8.1 GB
(240 trials). The second run, `ybyag10_night2_20261001`, is still being
generated. This folder keeps what is needed to inspect or reproduce a run
without the arrays:

- `manifest.json`: configuration, plan, seeds, software and source hashes.
- `validation.json`: output of `examples/check_ybyag_nn_dataset.py` (passed).
- `<split>/.../point_*.json`: per-trial metadata and metrics, without the `.npz`
  arrays.

## ybyag10_night_20260930

Config: `config/ybyag_nn_dataset_10at.json` (10 at.% Yb, measured Yb maps,
24-traversal image relay). Split: 64 train, 8 validation and 8 test setups,
240 trials, 0 failed setups.

| Target | Trials | Task success | Fidelity before → after | Shape after |
|---|---|---|---|---|
| Flattop super-Gaussian | 60 | 60 | 0.932 → 0.973 | 0.986 |
| Gaussian TEM00 | 60 | 60 | 0.936 → 0.976 | 0.986 |
| Helical LG(0,+1) | 60 | 57 (3 improved only) | 0.930 → 0.964 | 0.985 |
| Needle Bessel-Gaussian | 60 | 60 | 0.883 → 0.972 | 0.992 |

To regenerate the arrays (about 12 h on 5 CPU workers), run:

```bash
python examples/ybyag_nn_dataset.py --config config/ybyag_nn_dataset_10at.json --output OUT --workers 5
```

The run is deterministic for a given config and seed.
