# Local agent rules

Read `docs/LOCAL_AGENT_REPORT.md`, current stage documentation, and
`HoYAG_live_visualization_addendum(1).md` before changing numerical code.

Reuse existing physics kernels and keep the polarization-family guard active.
All expensive local numerical runs must use `hoyag.local_supervisor` and its
persistent `.local_runtime/budget.json` ledger. Do not launch full campaigns,
remote compute, large sweeps, or neural-network training automatically.

Preserve population layout, normalization, coordinate frames, polarization
conventions, and one shared physical crystal across amplifier encounters.
Keep oscillator and externally seeded amplifier modes explicit.

Never present an outer iteration as physical time, replay as a live calculation,
or an incoherent modal mixture as one coherent phase field. Record source and
configuration fingerprints, runtime, peak memory, failures, and resource limits.
Software tests alone do not establish dataset readiness or experimental validity.

Do not discard user changes, force-push, or merge automatically.
