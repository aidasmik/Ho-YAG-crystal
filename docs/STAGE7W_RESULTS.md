# Stage 7W executed results — 23 September 2026

## Status

Stage 7W is an opt-in polarization-resolved modal closure. The legacy Stage 7
solver and its extended control run are unchanged. The new source was executed
at commit `acaef27bda08c92f0e198f37813cb52340e0f864`.

- Full regression suite: **299 passed in 57.38 seconds**.
- New Stage 7W tests: **33 passed**.
- Independent physics/interface audit: **19 probes, zero failures**.
- Representative coupled runs: **three converged, one rejected**.
- Full 32-case Stage 7W/7V qualification: **not executed**.
- NN dataset readiness: **false**.

This does not turn the representative workflow green: it correctly finished
with a failure conclusion because the four-mode calculation was rejected. The
software regression result and numerical qualification result are distinct.

## Actual coupled calculations

Every case used the 10 mm diameter x 1 mm Ho:YAG disk, 250 mm cavity, 2% output
coupler, 10 ps / 1 mJ / 10 kHz pump, and the finite copper-plate/bond model.
The reference optical grid was 256 x 256 over 12 mm; material grid nr=24, nz=4,
nphi=12; mechanical mesh nr=8, ntheta=32, nz_disk=nz_plate=4. The free-support
case changed only the plate-support condition. Existing Stage 7V scientific
tolerances were not loosened.

| Case | Status | Outer iterations | Output W | Heat W | Peak crystal K | Peak plate K |
|---|---|---:|---:|---:|---:|---:|
| Two-mode reference, clamped plate | Converged | 7 | 0.7900341761 | 0.6547475427 | 302.3319194 | 293.7190278 |
| Four retained modes | Rejected: cut polarization family | 1 | Not a steady-state result | Not a steady-state result | Not qualified | Not qualified |
| Eight retained modes, clamped plate | Converged | 7 | 0.7901146696 | 0.6547561654 | 302.3320588 | 293.7190353 |
| Two modes, freely supported assembly | Converged | 7 | 0.7900318078 | 0.6547485408 | 302.3319943 | 293.7190294 |

The two-mode reference had modal outputs 0.3947342173 W and 0.3952999587 W.
The code did not impose equal powers. In the eight-mode case the first two
outputs were 0.3947744219 W and 0.3953401924 W; each other branch contributed
approximately 5e-9 to 1.2e-8 W under the configured spontaneous-seeding model.

Two versus eight retained modes changed total output by approximately **0.0102%**
and peak crystal temperature by **0.000139 K** on this grid. These are scalar
comparisons of two converged cases, not a full subspace/OPD/grid qualification.
The rejected intermediate four-mode case cannot count as a passing link in the
planned two-successive-comparison requirement.

Do not interpret the difference from the historical 0.753 W one-mode / 128-square
result as the isolated effect of Stage 7W: mode count and numerical resolution
also differ. The more comparable previously executed Stage 7V two-mode result
was already approximately 0.790025 W.

## Convergence measurements

| Case | Actual vector-field residual | Subspace residual | Raw heat residual | Final eigenpair residual |
|---|---:|---:|---:|---:|
| Two-mode reference | 3.3584e-5 | 8.4428e-6 | 1.7906e-4 | 3.3027e-15 |
| Eight-mode reference | 3.4692e-5 | See saved history | 1.7895e-4 | 8.7001e-15 |
| Free-support two-mode case | 3.2980e-5 | See saved history | 1.7909e-4 | 4.1593e-15 |

Each successful case passed all original field/heat/temperature/displacement/
power/loss gates AND the new modal-power, modal-gain, invariant-subspace and
incoherent-coherency gates for the required consecutive iterations. No global
polarization rotation was substituted for the actual field convergence test.

## Why four retained modes were refused

On its first updated hot operator, the four-mode selection would retain one
member of a higher-order polarization family while omitting its partner:

- actual vector overlap squared between the two candidates: **0.0002211211**;
- spatial overlap squared after a diagnostic constant polarization transform:
  **0.9996082815**;
- relative complex-eigenvalue gap: **1.62677e-5**;
- retained/omitted logarithmic round-trip gain difference: approximately **2.73e-7**.

Those fields are distinct eigenmodes, not exactly degenerate. The code refused
the truncation rather than mix them, silently add a mode, or claim convergence.
The record's intermediate power/temperature values belong to the initial
population/heat step and are not a converged four-mode laser prediction.

A sequence of arbitrary mode counts can cut a family even when both a smaller
and a larger basis do not. A suitable follow-up is an explicitly configured
complete-family intermediate basis (for example six, subject to the same guard),
then the full optical/material/mechanical and initial-condition sweeps. This
report does not assert that such an unexecuted six-mode case passes.

## Evidence

Full regression run:
https://github.com/aidasmik/Ho-YAG-crystal/actions/runs/35854247011
Job: `107158855750`.

Representative numerical workflow:
https://github.com/aidasmik/Ho-YAG-crystal/actions/runs/35854247190

Compact, hash-verified evidence:
`results/stage7w/evidence_summary.json` and `results/stage7w/cases/*.json`.
The collector verifies the SHA-256 of each downloaded Actions artifact before
copying its numerical summaries. No simulation values are inferred from plots.

| Artifact | ID | SHA-256 |
|---|---:|---|
| New tests + independent audit | 10747110217 | fe98e31709cb2bea329b377ad832b726816a83d29beb9ee07204256f05d6d798 |
| Two-mode reference | 10746907557 | 610788721fc5f3ef64aef25cc8aca9ee08bcdbc25b9ac64f5d1e6e9d964072c4 |
| Rejected four-mode case | 10746348640 | 9aab26fbe36e867f5ba7bf31b6adfb290a3498bb4255cb2b6ffad71d0ce41845 |
| Eight-mode case | 10747500161 | 9c2d959b15c580a3dd65aa3c523f69ccc20fe9b7ebbd106b33180240d716db14 |

The free-support artifact identifier and digest are in its saved case evidence.
Full numerical states remain in Actions artifacts; compact summaries and this
report are committed. Later documentation-only commits do not change the
numerical-source fingerprint of these results.

## Limits

The finite cooling plate and bonded interface are still included on every outer
iteration. No polarizer, equal-power constraint, changed heat fraction or new
material coefficient was introduced. The original limits remain: adiabatic
incoherent modal closure, frozen within-cycle mode shapes, prescribed bond/contact
properties, fixed spectroscopy and a finite candidate bank. Numerical convergence
does not establish global mode stability, mechanical calibration or experimental
accuracy. `dataset_ready=false` remains appropriate.

## Bounded local follow-up — 23 September 2026

These results were produced in the separate local branch
`local/bounded-validation-viewer`. They supplement the historical Actions
results above; they do not replace that evidence. The persistent local budget
ledger recorded two expensive coupled attempts, both below the 900 s per-case
limit, with a peak measured process-tree RSS below 1.3 GB.

| Local case | Outcome | Iterations | Output W | Heat W | Peak disk K | Wall s | Peak RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| Two-mode reference | Converged; all final closure gates passed | 7 | 0.7900341761 | 0.6547475427 | 302.3319194 | 412.1 | 950 MB |
| Six-mode complete-family candidate | Converged; no split boundary in seven accepted iterations | 7 | 0.7900870434 | 0.6547538059 | 302.3320148 | 482.3 | 1254 MB |

The reference state has SHA-256
`33a8fdecbc732ca68053e37dfbf86734405f703d8813b211cb042f31ca06b40c`;
the six-mode state has SHA-256
`15f8dc5404fcd531ba7a12514982e616549ad5d53678164704866a9e559bd2dd`.
Their numerical-source fingerprints differ (`9839b664…` and `6b2e6f24…`)
because local implementation changes were made between runs. Consequently,
the scalar difference between them is informative but is **not** a strict
source-matched retained-mode convergence test. No local eight-mode or grid
refinement case was run. `limited_numerical_validation_passed` means only
that these two bounded cases individually converged; the mode-count chain,
full campaign, and dataset qualification remain incomplete.

A 10-candidate diagnostic on the saved hot two-mode operator found retained
boundaries 2, 6, and 8 admissible within that finite bank, while 4 split a
polarization family. Its maximum full-operator eigenpair residual was
`4.92e-10`. The diagnostic does not prove global spectral completeness.
See `results/local_stage7w/family_diagnostic.json`, both local `summary.json`
files, and `results/local_validation/report.md` for local source hashes,
execution records, limitations, and viewer measurements.
