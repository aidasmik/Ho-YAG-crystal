# Stage 0 — Ho:YAG parameter database

This is the baseline dataset for reconstructing the Ho:YAG model of Rupp, Eichhorn and Kieleck.

## Baseline operating point

- Pump wavelength: 1907.7 nm
- Laser wavelength: 2090.3 nm
- Ho concentration: 1.1 at.% on Y sites
- Ho density: 1.52e26 m^-3
- Simple pump-to-laser quantum defect: 8.736 %
- Rupp constant refractive index: 1.81

For field propagation, the JSON also contains the Zelmon YAG Sellmeier equation. It gives approximately:
- n(1907.7 nm) = 1.80187
- n(2090.3 nm) = 1.79910

## Spectroscopic parameters copied from the Rupp validation model

Cross sections:
- sigma_e(2090.3 nm) = 1.2e-24 m^2
- sigma_e(1907.7 nm) = 7.7e-25 m^2
- sigma_a(2090.3 nm) = 2.1e-25 m^2
- sigma_a(1907.7 nm) = 1.2e-24 m^2

Spontaneous lifetimes:
- tau_I5 = 4.4 ms
- tau_I6 = 3.5 ms
- tau_I7 = 7.9 ms

Nonradiative/phonon rates:
- M_I5->I6 = 7.6e5 s^-1
- M_I6->I7 = 2.2e4 s^-1
- M_I7->I8 = 20.9 s^-1

ETU/cross-relaxation:
- k_I7->I5 = 3.8e-24 m^3/s
- k_I7->I6 = 4.0e-25 m^3/s
- C_I5->I7 = 1.1e-23 m^3/s
- C_I6->I7 = 2.6e-24 m^3/s

## Thermal / mechanical baseline

- T0 = 293 K
- thermal conductivity = 14 W/(m K)
- specific heat = 680 J/(kg K)
- density = 4560 kg/m^3
- Young's modulus = 310 GPa
- Poisson ratio = 0.25
- dn/dT = 9.1e-6 K^-1
- orientation = <111>

The Rupp paper uses thermal expansion alpha in the stress equations but does not list alpha in Appendix A.
The JSON therefore marks alpha = 7.0e-6 K^-1 as **provisional**, from direct YAG measurements near 300 K (Wynne et al., DOI 10.1364/AO.38.003282).

## Important separation

The following Rupp values are **not bulk material constants**:
- lateral effective thermal conductance = 0.8 W/K
- end-face effective thermal conductance = 0.05 W/K
- mechanical mounting degrees of freedom

They belong to the reference cooling/mount setup and should remain configurable.

## Inhomogeneous Ho distribution

The later model should replace the scalar Ho density with:

    N_Ho = N_Ho(x, y, z)

For the first implementation, it is reasonable to vary N_Ho spatially while keeping the Rupp spectroscopic coefficients fixed for modest concentration variations around 1.1 at.%.

For large concentration changes, ETU cannot safely be treated as constant. Barnes et al. (DOI 10.1364/JOSAB.20.001212) measured a clear concentration dependence of the total Ho:Ho upconversion parameter P77 in Ho:YAG:
- 1 at.%: 2.8e-24 m^3/s
- 2 at.%: 7.2e-24 m^3/s
- 4 at.%: 18.8e-24 m^3/s

These data are included only as auxiliary validation because P77 is not identical term-by-term to the split Rupp coefficients.

## Remaining gaps

These do **not** block Stages 1–4:
1. Full wavelength-resolved Ho:YAG absorption/emission spectra.
2. Temperature-dependent cross sections.
3. Exact manifold-average photon energies for every spontaneous heat-loss branch.
4. A validated mapping from local Ho concentration to all split ETU/cross-relaxation coefficients.

The project can therefore move directly to Stage 1 using this database.

## Main sources

- Rupp, Eichhorn, Kieleck, Applied Physics B 129, 4 (2023), DOI 10.1007/s00340-022-07939-z
- Zelmon, Small, Page, Applied Optics 37, 4933 (1998), DOI 10.1364/AO.37.004933
- Wynne, Daneu, Fan, Applied Optics 38, 3282 (1999), DOI 10.1364/AO.38.003282
- Barnes, Walsh, Filer, JOSA B 20, 1212 (2003), DOI 10.1364/JOSAB.20.001212
