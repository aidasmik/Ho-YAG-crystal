# Stage 7V retry: execution limit versus one-mode branch switching

Date: 23 September 2026. Production main remains `88cd102ea4c0d34eeb49a22639721ff4957f062b`. No physical solver, mesh, scientific acceptance tolerance or production configuration was changed by this investigation. Diagnostic scripts and workflows are isolated on `stage7v-retry-20260923`.

## Executed actions

1. Inspected the immutable first campaign ZIP, verifying its SHA-256 before reading every coupled summary, partial iteration history and log tail.
2. Reconstructed the actual frozen hot operators for the archived nonconverged one-mode `sensitivity_support_free` case and converged `modes_2` case. Verified numerical-core source hashes and NPZ state checksums. Solved their leading vector eigenfields and compared their polarization and spatial intensity. No new population or laser fixed point was inferred from these frozen tests.
3. Dispatched exactly one fresh **coupled-only** Stage 7V run on unchanged main, with **3,600 seconds per case** instead of 900. Original tolerances, physical inputs and the 33-case coupled plan remain unchanged. The completed 23-case fixed-source campaign is not repeated.

Extended coupled run: https://github.com/aidasmik/Ho-YAG-crystal/actions/runs/35852247652

The dispatch was confirmed by GitHub; the campaign was in progress when this note was written. This note is not a qualification result. Timeouts, iteration limits and failed field convergence remain failures to qualify, and `dataset_ready=false`.

## The stall is not only a time-budget problem

In the original one-mode reference, the last six recorded field residuals were approximately 1.009, 1.082, 1.036, 1.052, 1.060 and 1.037 against a target of 0.0005. At the final recorded iteration, relative output-power change was only 1.19e-6 and maximum temperature change was 0.00108 K. Each frozen eigenproblem itself converged, with the last residual 6.66e-15.

The 384-square case similarly reached 14 outer iterations with field residual about 1.08. The freely supported plate case reached all 24 outer iterations, but ended not_converged with field residual 1.1394. Conversely, some costly cases, including optical_512, material_2 and joint_2, hit the original wall limit before recording a complete outer iteration. Their precise bottleneck cannot be established from those empty histories alone.

## Newly measured polarization diagnosis

The nonconverged `sensitivity_support_free` state was examined without changing its populations or heat. Its used and next-predicted fields gave:

| Quantity | Measured value |
|---|---:|
| Raw complex vector overlap squared | 0.1231234433432074 |
| Phase-aligned field distance | 1.1393950122417094 |
| Polarization-summed intensity relative L2 difference | 0.011471067310262657 |
| Overlap squared after the best constant unitary polarization alignment | 0.9999129568536599 |
| Leading pair complex eigenvalue separation | 1.468078648760327e-5 |
| Leading pair logarithmic round-trip power-growth separation | 6.675511681811333e-7 |
| Current complex-eigenvalue grouping tolerance | 2e-7 |

The strongest eigenfield overlaps the current used field by only 0.1231, while the other leading polarization branch overlaps it by 0.8878. The two leading eigenfields are nearly orthogonal as vector fields but share almost the same spatial shape after a constant polarization transformation. This provides strong evidence that the persistent field changes are dominated by switching between nearly equal-gain polarization eigenbranches.

They are NOT exactly degenerate under the existing complex-eigenvalue criterion: their separation is about 1.47e-5, well above 2e-7. The globally optimized polarization alignment is a diagnostic only. It must not replace the actual vector convergence criterion or be used to claim the one-mode result converged.

For the separately converged `modes_2` state, each used-to-predicted vector overlap squared was approximately **0.9999999735425**, and the field distance was approximately **0.000162658**. Both polarization branches are retained. Their complex separation was 1.11287e-5 and logarithmic power-growth separation 2.66549e-7.

This is an iteration/truncation diagnosis within the current adiabatic model. It is not proof of physical instability of an experimental laser, and it is not justification to suppress a polarization, invent a polarizer, relax a field tolerance or merge distinct eigenvalues.

## Why still perform the extended-budget run?

The user-requested unchanged-model rerun is a control experiment: it can distinguish incomplete expensive cases from cases that continue to switch branches or reach the 24-iteration limit without convergence. The longer wall budget alone is not expected to guarantee qualification of the stalled one-mode cases. Any subsequent branch-continuation, polarization-pair or field-dynamics improvement must be separately implemented and validated, with source fingerprints kept distinct.

## Evidence and reproduction

Original campaign: https://github.com/aidasmik/Ho-YAG-crystal/actions/runs/35835101412

Original evidence artifact ID: `10740066509`.
SHA-256: `861c3e54f0addccdedda8b909062bc971aaf4446b6adf488f02dd9e28a74220a`.

History inspection run: https://github.com/aidasmik/Ho-YAG-crystal/actions/runs/35851781302
Artifact ID: `10745528485`, `stage7v-stall-diagnosis`.

Frozen eigenfield diagnosis and dispatch run: https://github.com/aidasmik/Ho-YAG-crystal/actions/runs/35852094633
Diagnostic job ID: `107151920145`; launch job ID: `107152376158`.
Diagnostic artifact ID: `10745757888`, SHA-256 `131c90f667a2548590af5df963eee508b93c9554af08eba93ce3b0df67060841`.
Launch receipt artifact ID: `10745329255`, SHA-256 `5568409fab4d9cdfdd5c206c1c7eca415293180f7be4d1c53c089a6362ef0d84`.

Diagnostic source revision: `c40e6c625c9b6b1a42ea6f33a2a5a2b8a62696d8`.

```bash
python audit/stage7v_stall_report.py \
  --archive /path/to/Stage7V_complete_campaign_evidence.zip \
  --sha256 861c3e54f0addccdedda8b909062bc971aaf4446b6adf488f02dd9e28a74220a \
  --output results/stage7v/stall_diagnosis

python audit/stage7v_eigen_diagnosis.py \
  --archive /path/to/Stage7V_complete_campaign_evidence.zip \
  --output results/stage7v/eigen_diagnosis
```

The numerical diagnostic used the archived campaign's package versions: NumPy 2.4.6 and SciPy 1.17.1. All reported measurements above were read from the completed diagnostic job, not inferred from scalar laser power or a recreated substitute state.
