# 250 mm / 2% Ho:YAG thin-disk resonator reference

This directory preserves the resonator case reported after the 10 mm diameter × 1 mm thin-disk model was extended from a single-pass amplifier to an oscillator.

## Geometry

- Ho:YAG disk: 10 mm diameter, 1 mm thick
- plane rear HR surface
- output coupler: 500 mm radius of curvature
- air gap from disk front face to output coupler: 250 mm
- output-coupler transmission: 2%
- rear signal reflectivity: 99.95%
- additional round-trip signal loss: 0.5%
- rear pump reflectivity: 99.5%

The reduced optical cavity length is about 250.556 mm. The calculated stability product is about 0.49889. The Gaussian cold-cavity radius is about 0.408 mm at the disk and 0.577 mm at the output coupler. Round-trip time is about 1.68 ns.

## Pump

The reference simulation used the generic pump:

- wavelength: 1907.7 nm
- pulse duration: 10 ps
- repetition rate: 10 kHz
- Gaussian radius: 0.5 mm
- pump energies: 100, 300, 600 and 1000 uJ per pulse
- sequential double-pass pump through the HR-backed disk

## Reference results

| Pump energy | Average incident pump | Average absorbed pump | Cycle-averaged output |
|---:|---:|---:|---:|
| 100 uJ | 1 W | 0.265 W | below threshold |
| 300 uJ | 3 W | 0.677 W | below threshold |
| 600 uJ | 6 W | 1.321 W | 0.3084 W |
| 1000 uJ | 10 W | 2.1768 W | 0.8076 W |

The numerical threshold bracket was 3.09375–3.140625 W incident average pump power, corresponding to 309.375–314.0625 uJ per pulse at 10 kHz.

At 10 W pump the selected-mode calculation produced one gain-switched optical pulse per pump period, with model-specific peak output about 38.5 W and FWHM about 1.85 us. This is **not** a 10 ps laser pulse: there is no mode-locking mechanism in the model.

A refined 48-radial-point × 8-depth-slice calculation changed the 10 W output from 0.807599 W to 0.807760 W, about 0.02%.

## Relationship to the main Stage 4R solver

The repository already contains the newer `hoyag.resonator` Stage 4R implementation. This reference case does not replace it. The configuration file is compatible with the same physical cavity parameters and preserves the exact setup/results reported for the earlier selected-mode calculation.

The main solver includes more explicit multimode/fixed-mode machinery and may not reproduce every scalar result bit-for-bit because its numerical formulation is not identical.

## Limitations

No thermal lens, disk bulging, stress birefringence, longitudinal standing-wave population grating, coherent forward/reflected 10 ps pump overlap, or mode locking is included. Mirror specifications and passive losses are assumed design values.
