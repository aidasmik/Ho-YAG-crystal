# Yb:LuAG active-medium path

`src/ybluag` implements a separate, executable Yb:LuAG material model. Yb³⁺ has
the ground `²F₇/₂` and excited `²F₅/₂` manifolds. The Ho:YAG four-manifold
population equations, Ho:Ho energy-transfer terms, 1907.7 nm pump, 2090.3 nm
laser and Ho:YAG results are **not** valid for Yb:LuAG.

The Yb path includes fixed-temperature CW and pulsed pump/signal transport
through a homogeneous disk. Its two-manifold population and optical equations
are

    dβ/dt = (1-β)(Wa,p + Wa,s) - β(We,p + We,s + 1/τ),
    αp = N[(1-β)σa,p - βσe,p],
    gs = N[βσe,s - (1-β)σa,s],
    dIp/dz = -αp Ip,       dIs/dz = gs Is.

Both pump stimulated emission and signal reabsorption are retained. The signal
transparency fraction is `σa,s/(σa,s+σe,s)`. A midpoint exponential axial step
preserves nonnegative intensities. Scalars and matching transverse intensity
maps are supported. The intensity maps are independent columns: diffraction,
thermal feedback, resonator mirrors, standing-wave effects and
counterpropagating pump passes are not in this CW calculation.

`propagate_structured_small_signal` takes a coherent transverse signal field
with `|field|²` in W/m², computes the pump-only CW population, then applies
Yb:LuAG gain and the existing scalar angular-spectrum diffraction kernel.
It preserves signal phase and cannot predict gain depletion by a strong signal.

`propagate_pulse` handles time-sampled pump and signal intensities through the
same two-manifold population in each depth cell. It uses exact local
constant-rate population updates and includes pump stimulated emission and
signal reabsorption. `periodic_pump_state` iterates one pulse and exact dark
decay until the population before the next pulse converges. These use a common
retarded-time frame; group-velocity walkoff, diffraction within the disk,
thermal evolution during a pulse and coherent pump/signal interference are not
resolved.

`periodic_pulse_heat` converges the shared population for repeated pump and
signal pulses, integrates the excited population over the pulse and dark part
of the period, and returns the cycle-average heat in each axial cell. It uses
the local first law: pump energy absorbed minus signal energy added minus
escaping fluorescence. The fluorescence yield parameter must be the effective
fraction of decays whose photons escape the sample; an intrinsic radiative
quantum yield alone does not account for reabsorption or trapping.

`solve_yb_assembly` accepts a Yb heat map on the existing cylindrical disk
mesh and reuses the conservative disk/plate thermal and elastic solvers.
`config/ybluag_10at_assembly.json` supplies a 10 at.% room-temperature
component example: measured 7.4 W/(m K) conductivity and 7.6×10⁻⁶ K⁻¹
expansion, with host heat capacity, density, isotropic elastic constants and
thermo-optic coefficient identified as proxies. Plate geometry, bond and
coolant settings are illustrative. The screen includes temperature-driven
scalar phase and FEM surface displacement. LuAG photoelasticity is absent;
the reported zero retardance means **not modeled**, not a prediction of no
stress birefringence. The solve rejects disk temperatures outside 293.15–300 K
because it does not have a defensible 10 at.% high-temperature material set.

## Example

Install the project and run:

```python
from ybluag import YbLuAGMaterial, propagate_cw

sample = YbLuAGMaterial(  # 10 at.%, 293.15 K, 940/1030 nm
    yb_at_percent=10.0,
)
result = propagate_cw(sample, 150e-6, 200,
                      pump_in_W_m2=1e8, signal_in_W_m2=1e5)
print(result.pump_out_W_m2, result.signal_out_W_m2)
```

The 10 at.% and 293.15 K lifetime default (0.965 ms) is Beil et al.'s pinhole
result. Other concentrations or temperatures require an explicit measured
lifetime. The Lu-site number
density is 1.42×10²⁸ m⁻³, so 10 at.% yields 1.42×10²⁷ Yb ions m⁻³.
Cross sections are converted from cm² to m². The package includes a copy of the
existing `Yb-LuAG/spectra` figure-guided reconstruction of Körner et al. so
installed builds can read it; these are **not raw author arrays**. The archived
emission reconstruction is kept for comparison, but it fails the Stark-level
reciprocity relation away from the peaks. The active Yb model derives stimulated
emission from reconstructed absorption through the McCumber relation using the
published LuAG Stark energies. Its 1030 nm, 293.15 K emission is about
3.06×10⁻²⁰ cm², roughly 2% above the upper cited sample value of
3.0×10⁻²⁰ cm²; this discrepancy reflects uncertainty in the reconstructed
absorption spectrum. It is a physics-constrained estimate, not a new
measurement. The code refuses wavelengths outside 880–1150 nm and
temperatures outside 293.15–473.15 K. The material model uses a uniform
temperature, so local heating does not update the cross sections during a run.

Optional steady-state heat per area is

    Q/A = absorbed pump - signal power increase - escaping fluorescence.

Fluorescence requires a supplied quantum yield and mean photon wavelength for
the particular sample. If either is unknown, heat is returned as `None`.
With both inputs, `heat_W_m3_by_step` gives the axial source density for a
uniform-step disk mesh; its depth integral equals `heat_W_m2`. For a mesh with
transverse cells, supply pump and signal arrays matching `(nr, nphi)` and use
`steps=mesh.nz`; the returned heat then has the `(nz, nr, nphi)` shape accepted
by `solve_yb_assembly`. The disk mesh must have uniformly spaced z cells for
this direct coupling. The same shape convention applies to
`periodic_pulse_heat.heat_W_m3_by_slice`.
This estimate does not resolve spectral fluorescence transport, reabsorption,
or nonradiative concentration quenching beyond the supplied measured lifetime
and yield. The mean fluorescence wavelength is an effective input, not simply
the chosen signal wavelength by physical necessity.

## Evidence and limits

- [Körner et al., JOSA B 29, 2493 (2012)](https://doi.org/10.1364/JOSAB.29.002493): temperature dependent absorption and emission spectra and Stark levels.
- [Beil et al., Optics Express 18, 20712 (2010)](https://doi.org/10.1364/OE.18.020712): 10 at.% sample, 940 nm absorption anchor, 1030 nm emission anchor, pinhole lifetime and site density.
- [Brenier et al., JOSA B 23, 676 (2006)](https://doi.org/10.1364/JOSAB.23.000676): concentration-dependent measured lifetimes and quantum yields; sample dependence is substantial.

The Yb optical energy ledger now connects to the existing thermal and elastic
assembly, while the Ho and Yb gain/population paths remain separate. A
quantitative Yb:LuAG polarized resonator or multipass amplifier prediction
still needs raw measured sample spectra, a LuAG photoelastic tensor, actual
pump-pass and resonator geometry, coatings, cooling contact, fluorescence
escape, and validation against the assembled device. Those inputs are not
established by the published component measurements used here.

Both material paths use `hoyag.propagation` for passive diffraction. Its bounded
private transfer-function cache avoids rebuilding the same FFT multiplier on
repeated equal steps, without changing the transfer formula or either material
model. The public transfer builder still returns a fresh array.
