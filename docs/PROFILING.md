# Local profiling and replay optimization

Recorded Stage 7W reference evidence (`results/stage7w/cases/reference.json`)
contains per-iteration clocks. Over its seven accepted outer iterations:

| Stage | Recorded total |
|---|---:|
| Optical pump-cycle integration | 104.983 s |
| Population/heat replay | 3.878 s |
| Coupled thermal/mechanical assembly | 7.905 s |
| Non-Hermitian eigensolver | 45.369 s |
| Entire iterations | 163.081 s |

These are historical GitHub Actions timings from the specified source revision,
not local before/after measurements. The optical cycle and eigensolver dominate.
Changing the eigensolver to a Hermitian solver would change the problem.

The Stage 7W shared-cycle path now accumulates heat and mean populations from
one BDF trajectory. The separate replay path remains available through
`run_polarization_hot_cavity(..., shared_cycle_replay=False)` for comparison.
The physical rate equations, quadrature, and heat ledger are unchanged. A
bounded, eight-pair local micro-benchmark is run with:

```bash
python3 examples/profile_replay.py run
```

The generated `results/local_profile/replay.json` includes individual times,
source/runtime fingerprint, CPU/RAM/GPU detection, BLAS thread configuration,
peak memory in `execution.json`, and numerical differences. The fixture is a
small marked-periodic optical state, not a converged coupled laser case. It
measures the replay component only. Do not extrapolate its speedup to the full
coupled solver without a matched local reference run.

Future optimization should target measured bottlenecks while retaining the
non-Hermitian operator, full candidate residuals, and current physical model.
Thermal/mechanical exact-factor reuse requires a cache key covering the full
assembled matrix, including mesh, material, bond, contact, support, and boundary
parameters. A cache keyed only by mesh size would be unsafe.

## Executed local measurements

On Linux with 22 logical x86_64 cores, 16.16 GB RAM, Python 3.12.3,
NumPy 1.26.4, and SciPy 1.17.1, the eight-pair micro-benchmark measured a
median 0.13930 s for separate heat/population replays and 0.07033 s for the
shared replay (1.98× for this small component fixture). Heat arrays matched
exactly and the largest population-fraction difference was `5.24e-12`.
The supervised profiling process took 2.325 s and peaked at 82.8 MB RSS.
`results/local_profile/replay.json` records the individual samples, runtime,
thread settings, and source fingerprint. BLAS backend introspection returned
an empty list; no GPU was detected by `nvidia-smi`.

The local two-mode coupled run took 412.1 s and peaked at 950 MB RSS. The
six-mode run took 482.3 s and peaked at 1254 MB RSS. They used different
numerical-source fingerprints, so their wall times do not isolate the effect
of mode count or establish a full-solver optimization speedup. The historical
Actions timing table above is a bottleneck guide, not a local before/after
control. Exact sparse-factor caching was not added without a measured safe
reuse key.
