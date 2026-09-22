# Stage 1P validation results

The complete regression suite passes 13/13 tests: six original Stage 1 spatial
tests plus seven Stage 1P temporal/spatiotemporal tests.

Key results:
- 1 ps Gaussian numerical time-bandwidth product: about 0.440
- strong-GDD analytical validation: expected 1.709331 ps, measured 1.709333 ps
- generic 10 ps pulse before 18 mm YAG: 10.000026 ps
- after 18 mm YAG: 10.000026 ps
- 18 mm YAG GDD: -802.922721 fs^2

This confirms that GVD is negligible for the 10 ps placeholder. Near 1 ps,
spectral overlap with the Ho:YAG absorption band will be more important and is
handled in Stage 2P.
