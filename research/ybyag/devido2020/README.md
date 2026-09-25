# Original Yb:YAG ZPL measurements

Author: Mariastefania De Vido. Dataset: https://doi.org/10.5286/edata/737

Downloaded 2026-09-25 from STFC eData; **CC BY 4.0**:
https://creativecommons.org/licenses/by/4.0/
The ten `*_raw_data.txt` files are unchanged originals. `provenance.json`
records download URLs, SHA-256 hashes and original column descriptions.
Our added `validation.json` reports basic numerical checks. Run
`python reconstruct.py` to generate `devido_zpl_reconstructed.csv` and
`reconstruction.json`. The derived table applies the supplied calibration
and Beer-Lambert law. Nonpositive power, transmission above unity and
isolated high spikes remain present with quality flags; unusable points
have NaN usable cross sections. At 80 K, transmission below 0.1% is flagged
as a censored lower bound because source spectral impurity may dominate.

Paper: De Vido, Wojtusiak and Ertel (2020),
https://doi.org/10.1364/OME.386436 . Sample: 1.1 at.% ceramic Yb:YAG,
1.08 mm thick. This is not a measured 20 at.% dataset.

## Calibration supplied with the dataset

- Column 1: spectrometer wavelength; subtract 0.176 nm.
- Column 3: incident power monitor; multiply by 22.8.
- Column 4: transmitted power.
- Column 5: sample temperature in kelvin.
- Columns 2 and 6-8 are not specified by the repository's description;
  do not silently use column 2 as the calibrated wavelength.

The paper uses sigma_a = ln(I_incident/I_transmitted)/(N L), with
N = 1.52e20 cm^-3 and L = 0.108 cm. It discusses window/background
calibration, spectral impurity and an absorption lower bound at 80 K.
The raw files include nonpositive power readings and transmission ratios
above one; the reconstruction does not clip them into a smooth spectrum.
The usable 300 K peak is about 0.87e-20 cm2 and the 100 K peak about
27.4e-20 cm2, close to the paper's rounded 0.8e-20 and 28e-20 anchors.
The 80 K censored lower bound reaches 49.0e-20 cm2. Etalon fringes and
sample-specific baseline uncertainty remain.

Temperatures: 80, 100, 125, 150, 175, 200, 225, 250, 275 and 300 K.
**300 K is 26.85 C, not 300 C.** No extrapolation to hot operation is
supported by these files. They have not been enabled in the amplifier.
The measured sample is 1.1 at.%; the proposed 5/10/15 at.% disks require
their own concentration-resolved high-temperature characterization.
