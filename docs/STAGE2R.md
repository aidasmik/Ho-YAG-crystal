# Stage 2R — Repetitive-pulse relaxation and population accumulation

Stage 2R extends the validated Stage 2P single-pulse primitive to a repetitive
picosecond pump.

## State cycle

For every crystal slice:

    pre-pulse populations
        -> picosecond Stage 2P pump interaction
        -> post-pulse populations
        -> dark four-manifold relaxation
        -> next pre-pulse populations

The full population field is retained as N_i(z,y,x). Identical incident pulses
are iterated until the pre-pulse state becomes periodic.

## Dark relaxation

Between pulses all optical rates are set to zero, but the complete Rupp
four-manifold dynamics remain active:

- spontaneous decay
- multiphonon relaxation
- ETU
- cross-relaxation

A conservative RK4 time step is selected from the fastest baseline Ho:YAG rate
unless an explicit max_dark_step_s is provided.

## Periodic steady state

Convergence is defined by

    max |N_i^(n+1) - N_i^(n)| / N_Ho <= tolerance

using the entire pre-pulse population field, not only one averaged inversion.

## Why this stage is required

The generic Stage 0P source uses 10 kHz, so the period is 100 us. The baseline
I7 lifetime is 7.9 ms; spontaneous decay alone leaves about 98.74% of an I7
population after 100 us. Therefore treating every pulse as if it entered a
ground-state crystal would be physically wrong.

Stage 2P remains the correct single-pulse material operator. Stage 2R adds the
memory between pulses.

## Outputs

- pulse-by-pulse transmission
- pulse-by-pulse absorbed energy
- pre-pulse peak I7 fraction
- post-pulse peak I7 fraction
- convergence residual
- converged pre-pulse population field
- last post-pulse population field
- last transmitted pulse

## Validation

Tests cover:

1. dark I7 exponential decay in the no-ETU/no-multiphonon limit;
2. recovery toward the ground state after a long dark interval;
3. clear I7 accumulation at 10 kHz;
4. increase in transmission as the repetitive pump saturates the medium;
5. fixed-point consistency after convergence;
6. strong reduction of population memory at low repetition rate.

## Scope

Stage 2R is still homogeneous in Ho concentration. Stage 3 can now replace the
scalar N_Ho by N_Ho(x,y,z) while preserving the pulse-train state machinery.
