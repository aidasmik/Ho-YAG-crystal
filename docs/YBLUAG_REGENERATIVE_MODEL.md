# Yb:LuAG regenerative amplifier model

The pulsed app offers two separate architectures: the existing ideal relayed
multipass amplifier and a regenerative cavity. Both use the same 12 at.%
Yb:LuAG disk, 10-pass pump transport, Gaussian seed and phase-only shaping
path, synthetic Yb distribution, and copper cooler. A regenerative round trip
visits the **same disk twice**, once toward its HR back coating and once after
reflection. The number of pump passes is independent of the number of signal
round trips.

## Solved dynamics

At each longitudinal disk slice, the signal fluence follows the short-pulse
Frantz–Nodvik map

\[F_\mathrm{out}=F_s\ln\{1+e^{g_0}[e^{F_\mathrm{in}/F_s}-1]\},\quad
F_s=h\nu/(\sigma_a+\sigma_e),\quad
g_0=N[\beta\sigma_e-(1-\beta)\sigma_a]\,\Delta z.\]

The code updates the local excited fraction by energy conservation,
\(\Delta\beta=-(F_\mathrm{out}-F_\mathrm{in})/(Nh\nu\Delta z)\).
This handles stimulated emission **and** reabsorption. The field amplitude
changes by \(\sqrt{F_\mathrm{out}/F_\mathrm{in}}\), retaining phase. Each cavity
return propagates the complex field to a concave mirror and back with an
angular-spectrum FFT, mirror curvature, a 10 mm disk aperture, HR loss, and
held-switch loss. Injection and final extraction use separate power
efficiencies. The extracted energy is reported separately from intracavity
energy.

During the intervals between round trips and between 10 kHz seeds, pump
transport and two-manifold Yb rate equations update the inversion. The
periodic pre-pulse inversion is iterated to a maximum change of \(10^{-6}\).
Pump absorption, stimulated signal transfer, and escaping fluorescence are
integrated across that period to drive the existing copper-cooler calculation.

The Frantz–Nodvik map follows [Frantz and Nodvik, *Journal of Applied
Physics* 34, 2346 (1963)](https://doi.org/10.1063/1.1702744).
A thin-disk regenerative amplifier's switch-controlled repeated disk visits
are described in the [thin-disk review](https://doi.org/10.1186/s41476-019-0108-1)
and this [experimental regenerative amplifier](https://doi.org/10.1364/OE.24.000883).
The LuAG phase and group indices used at 1030 nm, 1.8302 and 1.8488, are
estimates from the [LuAG Sellmeier measurement](https://refractiveindex.info/?shelf=main&book=Lu3Al5O12&page=Hrabovsky)
with the 7 at.% Yb:LuAG film index shift reported by
[Kurilchik et al.](https://doi.org/10.1007/s00340-019-7314-9). They are not
measurements of this 12 at.% bulk disk.

## Editable assumptions and limits

The proposal does not specify the regenerative air gap, mirror curvature,
Pockels-cell and polarizer loss, disk HR reflectivity, or injection/extraction
efficiencies. The UI exposes these as engineering assumptions. The default
values are 0.25 m, 0.5 m, 98% held power retention per round trip, 99.95% HR,
and 90% injection and extraction. A plot of energy after each round trip
makes the effect of these assumptions visible.

The short-pulse fluence model does not evolve pulse duration, chirp, spectral
gain narrowing, group-delay dispersion, self-phase modulation, amplified
spontaneous emission, or a finite Pockels switching transient. The output
temporal trace retains the input Gaussian envelope and is illustrative of
energy scaling only. The thermal OPD is calculated at the requested time but
applied as a lumped output screen; it is not fed back into the cavity after
each disk encounter. The present copper assembly also has a narrow validated
thermo-mechanical property range; extrapolated high-temperature maps are
design references, not performance predictions.

The pump-power plot recomputes the periodic optical state at five pump
powers from 20% to 100% of the selected pump. It holds the cavity and seed
fixed and does not recompute the cooler for each point. These results are
conditional predictions rather than experimentally calibrated output powers.
