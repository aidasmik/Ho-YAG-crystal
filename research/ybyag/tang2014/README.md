# 5/10/15 at.% room-temperature ceramic absorption

F. Tang et al., "Dependence of optical and thermal properties on concentration
and temperature for Yb:YAG laser ceramics," *Journal of Alloys and
Compounds* **593** (2014) 123-127, Fig. 5.
Source PDF: https://files.secure.website/wscfus/7885803/32406066/tang-yb-yag.pdf .

`digitize.py` traces the 5, 10 and 15 at.% colored absorption-coefficient
curves over 935-1040 nm at 1 nm spacing. The shorter-wavelength region is
obscured by annotation arrows and is deliberately not digitized. The figure itself labels the
roughly 941 nm peaks 6, 11 and 16 cm^-1; the recovered values reproduce
those labels to readout precision. The PDF is not committed. The derived CSV
and its byte-identical runtime copy are numerical figure readouts with an
estimated minimum uncertainty of 0.5 cm^-1. The paper does not give an
exact temperature for this room-temperature spectrum or all sample details
needed to extract an independent per-ion cross section. Fluorescence Fig. 6
is in relative units and is not an absolute emission-cross-section table.

`tang2014_rt_absorption_coefficient_m1` exposes these sample-specific room-
temperature coefficients only with explicit opt-in. The existing amplifier
uses a separate RT cross-section dataset and does not add these coefficients
as extra loss. No hot 5/10/15 at.% pump curves are supplied by this paper.
