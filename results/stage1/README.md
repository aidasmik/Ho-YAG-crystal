# Stage 1 validation results

These figures are generated from the Stage 1 passive structured-light propagator tests.

- `stage1_gaussian_radius_validation.svg`: numerical angular-spectrum propagation versus the analytical Gaussian-beam radius.
- `stage1_power_conservation.svg`: relative integrated-power error during lossless propagation.
- `stage1_hg10_symmetry.svg`: odd spatial symmetry of HG10.
- `stage1_lg_phase_winding.svg`: unwrapped phase of an LG mode with l=2 compared with 2θ.
- `stage1_phase_mask_intensity.svg`: phase-only modulation preserves intensity at the mask plane.

The pytest suite is in `tests/test_stage1.py`.
