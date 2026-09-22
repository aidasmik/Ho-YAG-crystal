# Working geometry — Ho:YAG thin disk

The active working geometry is now a circular Ho:YAG thin disk:

- diameter: 10 mm
- radius: 5 mm
- thickness: 1 mm
- face area: 78.5398 mm^2
- volume: 78.5398 mm^3

The numerical transverse domain is a square that contains the circular disk.
The Ho-density field is set to zero outside the 5 mm radius, so resonant
absorption/gain/populations are active only inside the physical disk.

The original Rupp validation geometry (4 mm diameter x 18 mm rod) is deliberately
retained in the Stage 0 parameter database and is not overwritten.

## Optical-edge limitation

The current FFT optical propagator still assumes one host refractive index across
the numerical window. Therefore the circular disk edge is represented exactly
for active-ion physics but not yet as a full air/YAG electromagnetic boundary.
For the intended centrally confined pump and structured signal beams, whose
waists are much smaller than the 5 mm radius, this approximation is negligible.
The thermal solver should use the actual circular disk boundary.
