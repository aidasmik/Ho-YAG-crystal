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


## 250 mm / 2% output-coupler reference case

The resonator configuration simulated in the September 22 reference run is preserved separately in
`config/thin_disk_resonator_250mm_2pct.json`:

- 10 mm diameter × 1 mm Ho:YAG disk
- 250 mm disk-front-to-output-coupler air gap
- 500 mm output-coupler radius of curvature
- 2% output-coupler transmission
- 99.95% rear signal reflectivity
- 99.5% rear pump reflectivity
- 0.5% additional round-trip signal loss
- 10 ps, 10 kHz, 1907.7 nm pump

The corresponding reference results and threshold bracket are under
`results/resonator_250mm_2pct/`. See
`docs/RESONATOR_250MM_2PCT_REFERENCE.md` for model scope and interpretation.
