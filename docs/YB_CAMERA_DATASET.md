# Exploratory Yb camera data

The native Tkinter Yb:YAG and Yb:LuAG tabs now offer **Export camera data (1080p)…** after a pulsed simulation. The exporter maps the saved per-pulse output fluence onto a 1920 × 1080 object-plane field, applies an illustrative optical throughput and silicon-camera quantum efficiency, and generates monochrome ADC frames. It adds photon shot noise, dark current, read noise, fixed pixel-response and dark-signal patterns, sparse hot/dead pixels, finite full well, quantization, a simple optical blur, and correlated **detector** temperature drift.

Each `.npz` includes `observable__camera_adu` and separate `truth__camera_plane_fluence_J_m2` / `truth__native_output_fluence_J_m2` arrays. Its `.json` records settings, seed, source request/result hashes, camera-model source hash, sensor clipping, and model limitations. The seed reproduces the same fixed pattern and temporal realization. The dialog suggests an optical throughput targeting half the full well on the first saved result; the same value applies to all selected states. Repeated frames of one solver result vary only in camera noise; they are **not** a simulation of changing crystal temperature. To include physically changing pump, disk, or cooling conditions, first solve and save separate pulsed runs, then enter those run directories in the dialog. Their ordering represents independently solved states, not a measured transient.

The command-line equivalent is:

```powershell
& 'C:\Users\Aidas\AppData\Local\Programs\Python\Python311\python.exe' .\examples\yb_camera_dataset.py --runs .\results\desktop_runs\RUN_A .\results\desktop_runs\RUN_B --output .\results\camera_datasets\example\camera --frames-per-state 2 --seed 17
```

This is an exploratory generator, not a qualified neural-network training set. The camera values are placeholders; measure quantum efficiency at the actual signal wavelength (about 1030 nm), optical throughput, gain, dark/read noise, full well, pixel pitch, point-spread function, and temperature response on the intended instrument. In particular, 1080p camera sampling does not recover detail absent from the much coarser optical solver grid. The saved Yb:YAG lumped thermal phase does not include validated hot gain; the exporter does not repair that limitation. A production training set also needs representative distributions over independently verified physical states and an external validation split.

The sensor-noise structure follows the definitions of shot noise, dark noise, photo-response nonuniformity, dark-signal nonuniformity, and saturation in the [EMVA 1288 camera characterization standard](https://www.emva.org/wp-content/uploads/EMVA1288-3.0.pdf). The numerical defaults are illustrative and are not attributed to that standard or a particular CCD.
