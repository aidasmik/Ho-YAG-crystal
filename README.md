# Ho:YAG crystal simulation

Reconstruction and extension of the Ho:YAG laser/thermal model of Rupp, Eichhorn and Kieleck.

The original Rupp rod is retained as a validation case. The active working model now uses a **Ho:YAG thin disk: 10 mm diameter x 1 mm thickness**.

## Working geometry

- circular diameter: 10 mm
- thickness along propagation: 1 mm
- radius: 5 mm
- transverse simulation window: 10 mm x 10 mm by default
- Ho density outside the circular disk: zero

See `config/thin_disk_geometry.json`.

## Progress

- Stage 0 — baseline Ho:YAG material parameter database
- Stage 0.1 — extended spectroscopy and temperature-dependent material database
- Stage 0P — generic picosecond-pump parameter extension
- Stage 1 — passive structured-light propagation
- Stage 1P — picosecond passive spatiotemporal propagation
- Stage 2P — single-pulse transient Ho:YAG population, pump absorption and saturation
- Stage 2R — repetitive-pulse relaxation and periodic population accumulation
- Stage 3 — spatially inhomogeneous Ho concentration N_Ho(z,y,x)
- Stage 4 — structured 2.09 um signal amplification and gain saturation
- Stage 5 — heat deposition and temperature field (next)
