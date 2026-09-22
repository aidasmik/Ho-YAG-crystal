# Stage 0.1 — Remaining Ho:YAG parameter research

Stage 0.1 extends the original Stage 0 database without changing the validated 2023 baseline operating point.

## What is now resolved

### Ho:YAG absorption spectrum and cryogenic/room-temperature dependence
A measured Ho:YAG absorption dataset exists from 1700–2200 nm at 83, 175, and 295 K. The database now contains the reported peak positions, peak cross sections, and FWHM values near 1908, 1928, 1933 and 1974 nm.

At 295 K the principal ~1908 nm peak is:
- peak = 1907.3 nm
- sigma_a = 1.259e-24 m^2
- FWHM = 4.5 nm

This is consistent with the Rupp validation value sigma_a,p = 1.2e-24 m^2.

### Hot Ho:YAG cross sections
Rupp's 2025 dissertation adds measured temperature-dependent Ho:YAG curves from 300–500 K:
- absorption: 1900–1920 nm
- emission: 2080–2100 nm

Both pump absorption and laser emission decrease with increasing temperature. The simulation uses linear interpolation in lambda and T.

The raw numerical curve arrays were not located in the public indexed material, so the database records the measurement ranges and method rather than inventing values. These curves only need digitization when high-fidelity thermal/gain feedback is implemented.

### Ho manifolds
The four low Ho:YAG manifolds I8, I7, I6 and I5 now have explicit Stark-energy lists in `data/ho_yag_stark_levels.json`.

The stored list is a legacy assignment reproduced in earlier Ho:YAG literature. Walsh, Grew & Barnes (2006) later remeasured the first nine manifolds and reported discrepancies with some previous assignments. Therefore the list is adequate for initial energy bookkeeping but is flagged for high-resolution refinement.

The database includes unweighted manifold-centroid wavelength estimates only as fallbacks. They are **not** treated as exact spontaneous-emission mean photon energies.

### Temperature-dependent YAG host properties
The database now includes:
- YAG Debye temperature: 760 K
- Rupp 2025 Debye formulation for Cp(T)
- Rupp 2025 Debye formulation for k(T)
- measured alpha(T) from 300–600 K
- measured dn/dT(T) from 300–600 K at 632.8 nm
- third-order dn/dT fit valid 70–600 K at 632.8 nm

The 632.8 nm dn/dT fit must not be treated as exact at 2.09 um. The original Rupp value 9.1e-6 K^-1 remains the baseline for the 2-um validation model.

## What deliberately remains configurable

### Background/parasitic absorption
This is not a universal Ho:YAG constant. It depends on crystal quality, contamination and defects. Baseline default is zero; a real crystal should be calibrated from passive transmission.

### AR-coating loss
The Rupp validation crystal was AR-coated, but no unique surface reflectivity is specified. Coating loss depends on the actual coating stack. Baseline default is ideal R=0, with measured coating reflectivity supplied when modeling a real optic.

### Concentration-dependent split ETU coefficients
Literature confirms concentration dependence and provides measured total Ho-Ho upconversion trends, but that total coefficient is not identical to each split Rupp ETU/cross-relaxation term. For modest inhomogeneity around 1.1 at.% Ho, keep the Rupp coefficients fixed initially.

## Implementation decision

Stages 1–4 should reproduce the Rupp baseline using the original constant spectroscopy.

Stage 5 should enable the temperature-dependent Debye thermal model.

Stage 5/8 high-fidelity gain feedback should use the measured Rupp 2025 sigma_a,e(lambda,T) curves after numerical digitization.

## Sources

- M. Rupp, *Multi-Physics Simulation of High-Power Solid-State Laser Systems* (2025), DOI 10.5445/IR/1000179649.
- M. Rupp, M. Eichhorn, C. Kieleck, *Applied Physics B* 129, 4 (2023), DOI 10.1007/s00340-022-07939-z.
- *Ho:YAG absorption cross sections from 1700 to 2200 nm at 83, 175, and 295 K*, *Applied Optics* 51, 8147 (2012).
- H. Furuse, R. Yasuhara, K. Hiraga, *Optical Materials Express* 4, 1794–1799 (2014), DOI 10.1364/OME.4.001794.
- B. M. Walsh, G. W. Grew, N. P. Barnes, *Journal of Physics and Chemistry of Solids* 67, 1567–1582 (2006), DOI 10.1016/j.jpcs.2006.01.123.
- N. P. Barnes, B. M. Walsh, E. D. Filer, *JOSA B* 20, 1212–1219 (2003), DOI 10.1364/JOSAB.20.001212.
