# Yb:LuAG active-medium path

`src/ybluag` implements a separate, executable Yb:LuAG material model. Yb³⁺ has
the ground `²F₇/₂` and excited `²F₅/₂` manifolds. The Ho:YAG four-manifold
population equations, Ho:Ho energy-transfer terms, 1907.7 nm pump, 2090.3 nm
laser and Ho:YAG results are **not** valid for Yb:LuAG.

The current Yb path solves a **fixed-temperature, monochromatic CW, collinear**
pump and signal through a homogeneous disk. At each axial step it solves

    dβ/dt = (1-β)(Wa,p + Wa,s) - β(We,p + We,s + 1/τ),
    αp = N[(1-β)σa,p - βσe,p],
    gs = N[βσe,s - (1-β)σa,s],
    dIp/dz = -αp Ip,       dIs/dz = gs Is.

Both pump stimulated emission and signal reabsorption are retained. The signal
transparency fraction is `σa,s/(σa,s+σe,s)`. A midpoint exponential axial step
preserves nonnegative intensities. Scalars and matching transverse intensity
maps are supported. The intensity maps are independent columns: diffraction,
thermal feedback, resonator mirrors, standing-wave effects, counterpropagating
pump passes and pulse dynamics are not in this path.

`propagate_structured_small_signal` takes a coherent transverse signal field
with `|field|²` in W/m², computes the pump-only CW population, then applies
Yb:LuAG gain and the existing scalar angular-spectrum diffraction kernel.
It preserves signal phase and cannot predict gain depletion by a strong signal.

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
installed builds can read it; these are **not
raw author arrays**. The code refuses wavelengths outside 880–1150 nm and
temperatures outside 293.15–473.15 K. The material model uses a uniform
temperature, so local heating does not update the cross sections during a run.

Optional steady-state heat per area is

    Q/A = absorbed pump - signal power increase - escaping fluorescence.

Fluorescence requires a supplied quantum yield and mean photon wavelength for
the particular sample. If either is unknown, heat is returned as `None`.
This estimate does not resolve spectral fluorescence transport, reabsorption,
or nonradiative concentration quenching beyond the supplied measured lifetime
and yield. The mean fluorescence wavelength is an effective input, not simply
the chosen signal wavelength by physical necessity.

## Evidence and limits

- [Körner et al., JOSA B 29, 2493 (2012)](https://doi.org/10.1364/JOSAB.29.002493): temperature dependent absorption and emission spectra and Stark levels.
- [Beil et al., Optics Express 18, 20712 (2010)](https://doi.org/10.1364/OE.18.020712): 10 at.% sample, 940 nm absorption anchor, 1030 nm emission anchor, pinhole lifetime and site density.
- [Brenier et al., JOSA B 23, 676 (2006)](https://doi.org/10.1364/JOSAB.23.000676): concentration-dependent measured lifetimes and quantum yields; sample dependence is substantial.

The existing Ho:YAG Stage 5–7 thermal, elastic, photoelastic and polarized
closure is not connected to this Yb material path. In particular the repository
does not have a validated LuAG photoelastic tensor. A quantitative Yb:LuAG
thin-disk resonator or multipass amplifier prediction needs measured sample
spectra, concentration-dependent lifetime, actual pump-pass geometry, coatings,
cooling contact, and an independently validated coupled model.

Both material paths use `hoyag.propagation` for passive diffraction. Its bounded
private transfer-function cache avoids rebuilding the same FFT multiplier on
repeated equal steps, without changing the transfer formula or either material
model. The public transfer builder still returns a fresh array.
