# Ho:YAG crystal simulation

Reconstruction and extension of the Ho:YAG laser/thermal model of Rupp, Eichhorn and Kieleck.

The original Rupp rod is retained as a validation case. The active working geometry is a **Ho:YAG thin disk: 10 mm diameter x 1 mm thickness**.

## Thin-disk resonator

Stage 4R adds a proposed two-mirror setup: the disk's plane rear HR coating (99.9% assumed reflectivity) and a concave 3%-transmission output coupler with 500 mm radius of curvature, separated by a 200 mm air gap. Pump and signal each traverse the disk twice on a return encounter. Additional signal round-trip loss is assumed to be 0.5%.

The cold-cavity FFT field solver and pulse-pumped fixed-mode laser solver are separate fidelity levels. The latter includes shared Ho populations, gain depletion, cavity photon storage, spontaneous seeding, and periodic pump kicks. It is not a self-consistent thermal or full coherent 3-D oscillator model. The reported competing-mode run favors the Gaussian mode, not automatic vortex lasing.

See `config/thin_disk_resonator.json`, `docs/STAGE4R_RESONATOR.md`, and `results/resonator/summary.csv`.

```
pip install -e '.[dev]'
pytest -q
python examples/thin_disk_resonator.py --sweep 100 300 600 1000
```

## Working geometry

- Circular diameter: 10 mm; thickness: 1 mm; radius: 5 mm.
- Single-pass FFT examples use a containing square window and zero Ho density outside the disk.
- The resonator validation uses a padded FFT window; the dynamic demonstration uses radial quadrature over the entire circular face.

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
- Stage 4R — HR-backed cavity, output coupling, and fixed-mode pulse-pumped oscillator
- Stage 5 — heat deposition and temperature field (not yet implemented)
