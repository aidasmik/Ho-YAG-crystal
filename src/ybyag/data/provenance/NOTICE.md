# Upstream data attribution

The four files in `spectra/haseongpu_original/` are copied without numerical changes from HASEonGPU, ComputationalRadiationPhysics:

https://github.com/ComputationalRadiationPhysics/haseongpu/tree/5af022b40635b6cf22178252e243467d3d7aed90/material_library/data/legacy_yb_yag

The merged `spectra/yb_yag_293K_legacy.csv` reformats those numerical arrays. Exported RT CSV/NPZ and absorption tables are transformations of those data. Preserve this notice and the accompanying GNU GPL version 3 license when redistributing these upstream data and their transformations. The upstream project identifies its license as GPL-3.0-or-later.

Original developers retain their copyrights. The source generator identifies Copyright 2026 Tim Hanel and provides the room-temperature metadata; this package does not claim authorship of the upstream spectra or assert that Tim Hanel was their experimental author.

The other literature parameters have independent source citations in `references.json`; they are not HASEonGPU measurements. No whole research paper or paper figure is redistributed. New model helpers are independently written for this package; this notice is not a change to any license elsewhere in the parent repository.

Original Git blob SHA-1 values:

```
lambda_a.txt  8f7f44d73d83241dc0b952fd0a960072924c833f
lambda_e.txt  8f7f44d73d83241dc0b952fd0a960072924c833f
sigma_a.txt   a8f3582cf2475af0ee9323cce3a8a432a1395a55
sigma_e.txt   64c2f75ca9a613a613568a450ba35f37c0c40bf5
```

The upstream experimental publication, concentration and uncertainty are unspecified in the inspected metadata. This package does not invent them.
