# Yb:YAG missing-data search - 25 September 2026

## Results and applicability

| Missing input | Source found | Coverage / status | Consequence |
|---|---|---|---|
| Temperature-dependent 969 nm absorption, raw data | [De Vido et al. 2020](https://doi.org/10.1364/OME.386436), [STFC dataset](https://doi.org/10.5286/edata/737) | Ten original files and a quality-flagged reconstruction in `research/ybyag/devido2020`; 80-300 K, 1.1 at.% ceramic | Strong source for cold/RT ZPL validation; not evidence at 250 C or at 5/10/15/20 at.%. |
| Hot pump and signal spectra, both Yb hosts | [Korner et al. 2012](https://doi.org/10.1364/JOSAB.29.002493), [GSI record](https://repository.gsi.de/record/50281) | Measurements 20-200 C; original arrays not retrieved in this search | Best matched common source for YAG/LuAG; retain separate host spectra. The repository currently has only approximate laser-band figure readings. |
| Hot YAG pump spectra up to 300 C | [Liu et al. 2007](https://doi.org/10.1364/JOSAB.24.002081), [author-provided paper](https://www.researchgate.net/publication/239008671_Effects_of_the_temperature_dependence_of_absorption_coefficients_in_edge-pumped_YbYAG_slab_lasers) | Section 2: 4.5 at.% rod, 2.5 mm path, 0.1 nm instrument resolution, 27-300 C; Fig. 1 includes 969 nm | Figure source found, no raw arrays retrieved. Its 941 nm exponential fit must not be used at 969 nm. Abstract says 23 C while experiment/caption says 27 C; preserve this discrepancy. |
| Hot, highly doped YAG spectra | [Esmaeilzadeh et al. 2012](https://zenodo.org/records/1334966/files/13958.pdf?download=1) | 25 at.%, 1 mm disk, 300-450 K; Figs. 4 and 6 endpoint curves digitized to 1 nm in `research/ybyag/high_temp_25at` | Opt-in 25 at.% figure lookup only. Not a direct 5/10/15/20 at.% calibration or 523 K coverage. Narrow peak is near 971 nm; figure/prose intermediate-temperature legends disagree. |
| Lifetime and concentration-dependent emission | [Dong et al. 2003](https://doi.org/10.1364/JOSAB.20.001975) | Nominal 2.5, 5, 10, 20, 30 at.%; 15-300 K | Exact 20 at.% coverage exists, but no numerical lifetime arrays retrieved. This source does not close hot tau(T) above 300 K. Author-hosted indexed PDF returned 404. |
| Nonlinear index at signal wavelength | [Kabacinski et al. 2019](https://doi.org/10.1364/OE.27.011018), [author paper text](https://www.researchgate.net/publication/332238694_Nonlinear_refractive_index_measurement_by_SPM-induced_phase_regression) | Table 1: host YAG n2 = (6.13 +/- 0.07)e-20 m^2/W at 1030 nm; experiment uses 210 fs pulses | Usable as a labeled host proxy for a future nonlinear-phase estimate. Not a measured 20 at.% hot-YAG coefficient or a coating damage threshold. |
| Doped thermal conductivity vs temperature | [Cini and Mackenzie 2017](https://doi.org/10.1007/s00340-017-6848-y) | Already in repo: HT fits for 0, 5, 9.4, 22.9 at.%, 300-475 K | 20 at.% requires explicit concentration interpolation. No extension to 523 K (250 C) established by these fits. |
| Host Cp, expansion and thermo-optic response | [Sato et al. 2025](https://doi.org/10.1364/OE.540655) | Already in repo, 160-500 K host data; apparent dn/dT can include mounting stress | Do not count stress twice or call these measured 20 at.% properties. 500 K is 226.85 C. |
| Photoelasticity | [Brickus and Dementev 2016](https://doi.org/10.3952/physics.v56i1.3272) | Existing dataset contains legacy, conflicting host tensors | Data availability does not implement vector stress-optic propagation; tensor conventions and crystal orientation remain necessary. |
| Copper bond/contact, coating absorption/phase and damage, pump linewidth | Actual assembly and optical-component characterization | No universal material constant can specify this particular device | Generic values remain engineering assumptions. Require pump spectrum and coating/mount specifications for calibration. |

## What is actually obtained

The STFC files contain 7,242 rows in total, eight columns each. All values are
finite, but 20 rows contain nonpositive incident/transmitted power and 140
rows have transmitted power greater than the calibrated incident monitor.
The originals remain unchanged. SHA-256 and per-file counts are retained.
`reconstruct.py` now applies the published wavelength/monitor calibration and
Beer-Lambert relation to all 7,242 rows, retaining invalid power ratios,
isolated high spikes and the 80 K saturation region as quality flags. The
output is a sample-specific, quality-flagged numerical spectrum, not a
high-temperature lookup for the proposed disk.

Published numeric anchors from De Vido: at 300 K the ZPL peak is 969.04 nm,
peak absorption is approximately 0.8e-20 cm^2 (8e-25 m^2), and FWHM is
2.38 nm. At 80 K the reported peak exceeds 49e-20 cm^2; it is a lower
bound, not an exact fitted amplitude. Their sample is 1.1 at.%.

## Next numerical integration steps

1. **Completed for the 1.1 at.% sample:** reconstruct the ZPL records with
   invalid/censored flags and check the 300/100/80 K published anchors.
2. **Partly completed:** trace the 300 and 450 K 25 at.% absorption and
   reciprocity-derived emission curves, keeping source concentrations and
   figure uncertainty. Obtain 5/10/15 at.% measured pump and emission spectra
   for a calibrated model. Do not shift a 971 nm curve to 969 nm.
3. For a provisional common 20-200 C model, prefer the same Korner source for
   both Yb hosts, deriving pump stimulated emission with same-temperature
   detailed balance only within the measured wavelength range.
4. Convolve pump absorption/emission with the pump spectral distribution;
   wavelength alone is insufficient for a narrow ZPL.
5. Couple validated spectra to temperature-dependent material properties and
   populations, then verify conservation and sensitivity to the remaining
   lifetime, sample and mount assumptions.

The literature search has **not** established a complete calibrated
5/10/15/20 at.% Yb:YAG dataset through 250 C. The 25 at.% figure endpoints
have an explicitly opt-in helper in `ybyag.material_data`; neither they nor
the 1.1 at.% ZPL records are silently activated in the amplifier. The
existing hot-gain guard remains.

## Concentration priority: 5, 10 and 15 at.%

These concentrations now receive priority in the native input selector. The
original 20 at.% proposal point remains available and remains the launcher
default so that the proposal is not silently changed.

| Concentration | Strongest direct evidence found | Modeling status |
|---:|---|---|
| 5 at.% | Tang ceramic: 935-1040 nm RT absorption coefficient now digitized (941 nm near 6 cm^-1); RT conductivity 8.6 W m^-1 K^-1; 49% slope efficiency in its 940 nm test resonator | Direct comparison anchor, with a guarded opt-in coefficient lookup. Spectral cross sections still use the common RT table; laser efficiency is not transferred. |
| 10 at.% | Tang ceramic: RT absorption coefficient digitized (941 nm near 11 cm^-1); 41% slope efficiency. Cini has a 9.4 at.% hot conductivity fit | Opt-in comparison, not a measured hot pump or emission spectrum. |
| 15 at.% | Tang ceramic: RT absorption coefficient digitized (941 nm near 16 cm^-1). Aggarwal single-crystal k = 6.7 W m^-1 K^-1 at 298 K, rising to 16.4 W m^-1 K^-1 at 101 K | Opt-in comparison; ceramic optical sample and single-crystal thermal sample remain separate. |

Tang reports that its 1030 nm fluorescence intensity starts decreasing above
10 at.% and that its apparent lifetime rises with concentration because of
self-absorption. Those decay curves therefore cannot replace the intrinsic
0.95 ms lifetime in the rate equations. In its particular resonator, 5 at.%
outperformed 10 at.%; this does not by itself select the optimum concentration
for a 100 micrometre, ten-pump-pass regenerative amplifier.

The exact reported anchors and their qualifications are stored in
`research/ybyag/concentration_anchors.csv`. Absorption coefficients are not
treated as cross sections until sample thickness, wavelength and reflection/
scattering corrections are sufficiently specified.

The present near-RT assembly consequently evaluates to:

| Yb at.% | k (W m^-1 K^-1) | density (kg m^-3) | Cp (J kg^-1 K^-1) | Basis |
|---:|---:|---:|---:|---|
| 5 | 7.365 | 4553 | 594.7 | Cini CT resistivity interpolation; host density/Cp proxy |
| 10 | 6.899 | 4553 | 594.7 | Cini CT resistivity interpolation; host density/Cp proxy |
| 15 | 6.772 | 4850 | 571.1 | Aggarwal exact-concentration table interpolated from 298/251 K to 293.15 K; measured RT density and volumetric Cp |
| 20 | 5.803 | 4553 | 594.7 | Cini HT resistivity interpolation at 300 K; host density/Cp proxy |

Only the 15 at.% row is wholly based on an exact-concentration thermal table.
The 5 and 10 at.% Tang ceramic results remain comparison anchors because
silently mixing a particular ceramic's properties with the single-crystal fit
family would make the concentration comparison less controlled.
