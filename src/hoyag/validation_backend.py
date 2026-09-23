"""Stage 7V execution adapters. Existing Stage 6/7 solvers remain unchanged.

Two explicitly different scopes:
  coupled: recompute fields, periodic Ho states, heat and disk/plate assembly;
  frozen: reuse a hashed archived mean population/heat state and vary numerical
          assembly/screen resolution or mounting parameters. No new laser power
          is inferred from this second path.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from .validation_metrics import (DiagnosticGrid, ContinuousDensity, conservative_remap,
    grid_from_axes, vector_overlap, oam_spectrum, closed_phase_winding)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def source_manifest(root):
    """Hash numerical source, not build caches; retain exact paths and bytes."""
    root = Path(root)
    source = root / "src/hoyag"
    if not source.is_dir(): raise ValueError("repository src/hoyag directory is missing")
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(source.rglob("*.py"))}
    import scipy, sys, platform
    from .validation_metrics import stable_hash
    runtime = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
               "platform": platform.platform()}
    revision = None
    if (root / ".git").exists():
        import subprocess
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                text=True, capture_output=True, check=False)
        if result.returncode == 0: revision = result.stdout.strip()
    return {"source_hash": stable_hash({"files": files, "runtime": runtime}), "files": files,
            "git_revision": revision, "fingerprint_scope": "source bytes AND numerical runtime versions", **runtime}


def assembly_config(case):
    cfg = deepcopy(case["physics"]["assembly"])
    cfg["numerics"] = {"plate_thermal_nz": case["numerics"]["plate_thermal_nz"],
                       "mechanical": deepcopy(case["numerics"]["mechanical"])}
    return cfg


def make_mesh(case):
    from .thermal import DiskThermalMesh
    g = case["physics"]["assembly"]["geometry"]
    return DiskThermalMesh.disk(radius_m=g["disk_radius_m"], thickness_m=g["disk_thickness_m"],
                                **case["numerics"]["thermal"])


def make_grid(case):
    return DiagnosticGrid.square(case["numerics"]["optical_n"], case["numerics"]["optical_window_m"])


def lg_seed(grid, waist_m, charge=0, polarization=0):
    """Input profile definition only: p=0 LG at a waist, unit integrated power."""
    if not isinstance(charge, (int, np.integer)) or not np.isfinite(waist_m) or waist_m <= 0:
        raise ValueError("integer charge and positive waist required")
    if polarization not in (0, 1): raise ValueError("polarization must be x or y")
    x, y = grid.mesh; r = np.hypot(x, y)
    field = np.zeros((2, *grid.shape), complex)
    field[polarization] = (np.sqrt(2)*r/waist_m)**abs(charge) * np.exp(-r*r/waist_m**2) * np.exp(1j*charge*np.arctan2(y, x))
    power = np.sum(abs(field)**2)*grid.dx*grid.dy
    return field/np.sqrt(power)


def initial_modes(grid, cavity, numerics):
    n = numerics["mode_count"]; guess = numerics["initial_guess"]
    if n < 1: raise ValueError("mode_count must be >=1")
    fields = []
    rng = np.random.default_rng(numerics["initial_seed"])
    for i in range(n):
        if guess == "gaussian": charge = i//2
        elif guess == "lg_plus": charge = 1+i//2
        elif guess == "lg_minus": charge = -(1+i//2)
        elif guess == "lg_two": charge = 2+i//2
        elif guess == "mixed": charge = i//2
        else: raise ValueError("unknown initial-mode family")
        f = lg_seed(grid, cavity.waist_m, charge, i%2)
        if guess == "mixed":
            f += .36*np.exp(1j*rng.uniform(-np.pi,np.pi))*lg_seed(grid, cavity.waist_m, -(i//2+1), i%2)
        f /= np.linalg.norm(f)
        fields.append(f)
    fields = np.array(fields)
    if np.linalg.matrix_rank(fields.reshape(n, -1)) < n:
        raise ValueError("initial modes are linearly dependent")
    return fields


class FrozenReference:
    """Validated Stage 7 snapshot. No regenerated populations from scalar results."""
    def __init__(self, directory, expected=None):
        self.directory = Path(directory)
        self.state_hash = sha256_file(self.directory / "state.npz")
        self.summary_hash = sha256_file(self.directory / "summary.json")
        self.summary = json.loads((self.directory / "summary.json").read_text())
        if expected:
            if self.state_hash != expected["state_sha256"] or self.summary_hash != expected["summary_sha256"]:
                raise ValueError("archived state/summary differs from the planned reference hash")
        if not self.summary.get("converged"):
            raise ValueError("frozen reference must have a converged optical/assembly state")
        # The repaired source metadata is required; older pre-audit runs are not silently reused.
        if "pump_source" not in self.summary["metadata"]:
            raise ValueError("reference lacks audited physical-pump provenance")
        with np.load(self.directory / "state.npz", allow_pickle=False) as z:
            self.data = {k: z[k].copy() for k in z.files}
        from .thermal import DiskThermalMesh
        q = self.data["raw_heat_W_m3"]
        self.mesh = DiskThermalMesh(self.data["r_edges_m"], self.data["z_edges_m"], q.shape[-1])
        if q.shape != self.mesh.shape or not np.all(np.isfinite(q)):
            raise ValueError("invalid archived heat field")
        f = self.data["mean_fractions"].reshape((4, *self.mesh.shape))
        if np.min(f) < -1e-8 or not np.allclose(f.sum(axis=0), 1, rtol=0, atol=1e-8):
            raise ValueError("invalid canonical archived mean populations")
        self.fractions = f
        self.grid = grid_from_axes(self.data["x_m"], self.data["y_m"])

    def provenance(self):
        return {"state_sha256": self.state_hash, "summary_sha256": self.summary_hash,
                "reference_density_m3": 1.52e26, "reference_density_kind": "explicit uniform audited case",
                "heat_key": "raw_heat_W_m3", "scope": "fixed archived cell-average heat and populations"}


def solve_assembly(mesh, grid, heat, cfg):
    """Use the actual Stage 6 finite disk/plate heat and FEM operators."""
    from .thermal import ThermalBoundary
    from .cooling_plate import DiskPlateHeatSolver, cooling_plate_mesh, ThermalMaterial, sample_temperature
    from .thermomechanics import DiskPlateMesh, ElasticMaterial, BondedInterface, solve_disk_plate
    from .stress_optics import build_hot_disk_screens, CubicElastoOptic, crystal_axes_111
    geom = cfg["geometry"]; th = cfg["thermal"]; mech = cfg["mechanical"]; optics = cfg["optics"]
    plate = cooling_plate_mesh(mesh, radius_m=geom["plate_radius_m"], thickness_m=geom["plate_thickness_m"],
                               nz=cfg["numerics"]["plate_thermal_nz"])
    solver = DiskPlateHeatSolver(mesh, plate,
        contact_conductance_W_m2K=th["interface_conductance_W_m2K"],
        coolant=ThermalBoundary(th["coolant_temperature_K"], th["coolant_conductance_W_m2K"]),
        disk_material=ThermalMaterial(**th["disk"]), plate_material=ThermalMaterial(**th["plate"]))
    temperature = solver.steady(heat)
    fem = DiskPlateMesh.make(radius_m=geom["disk_radius_m"], disk_thickness_m=geom["disk_thickness_m"],
        plate_radius_m=geom["plate_radius_m"], plate_thickness_m=geom["plate_thickness_m"],
        **cfg["numerics"]["mechanical"])
    td = sample_temperature(mesh, temperature.disk_temperature_K, fem.disk.centers_m)
    tp = sample_temperature(plate, temperature.plate_temperature_K, fem.plate.centers_m,
                            z_offset_m=geom["disk_thickness_m"])
    dm = ElasticMaterial(**mech["disk"])
    displacement = solve_disk_plate(fem, td, tp, disk_material=dm,
        plate_material=ElasticMaterial(**mech["plate"]), interface=BondedInterface(**mech["bond"]),
        support=mech["plate_support"], front_pressure_Pa=mech["front_pressure_Pa"])
    x, y = grid.mesh
    screens = build_hot_disk_screens(fem, displacement, mesh, temperature.disk_temperature_K, np.stack((x, y), axis=-1),
        index=optics["index"], wavelength_m=optics["wavelength_m"], dn_dT_K1=optics["dn_dT_K1"],
        reference_temperature_K=optics["reference_temperature_K"], material=dm,
        coefficients=CubicElastoOptic(**optics["cubic_elasto_optic"]),
        crystal_axes=crystal_axes_111(optics["crystal_azimuth_rad"]))
    return temperature, displacement, screens, fem


def weak_double_pass_probe(field, grid, screens, single_pass_log_gain, rear_reflectivity, mask):
    """One HR-backed, collapsed thin-disk encounter, not a resonator round trip.

    Mean populations are frozen: this is a WEAK seeded probe, not pulse extraction
    or saturated-amplifier operation. No output-coupler or cavity-air-gap operator
    is inserted. Phase and polarization follow the same Stage 6 Jones products.
    """
    from .stress_optics import apply_jones
    f = np.asarray(field, complex)
    if f.shape != (2, *grid.shape): raise ValueError("probe shape mismatch")
    if not 0 < rear_reflectivity <= 1: raise ValueError("invalid HR reflectivity")
    x = apply_jones(f.transpose(1,2,0), screens.inward_jones)
    material = np.sqrt(rear_reflectivity) * np.exp(single_pass_log_gain +
                 2j*np.pi*screens.geometry_roundtrip_opd_m/screens.wavelength_m) * mask
    x = apply_jones(x * material[..., None], screens.outward_jones)
    return x.transpose(2,0,1)


def probe_diagnostics(grid, screens, log_gain, physical):
    cfg = physical["probe"]; c = physical["cavity"]
    x, y = grid.mesh; mask = x*x+y*y <= (c["disk_diameter_m"]/2)**2
    fields, rows = [], []
    for charge in cfg["charges"]:
        seed = lg_seed(grid, cfg["waist_m"], charge)
        out = weak_double_pass_probe(seed, grid, screens, log_gain, c["disk_hr_reflectivity"], mask)
        gain = float(np.sum(abs(out)**2)/np.sum(abs(seed)**2))
        spec = oam_spectrum(out, grid, radius_m=.0015)
        fraction = spec["fractions"][spec["charges"].index(charge)]
        # Total intensity alone is not used to infer phase winding or OAM.
        winding = closed_phase_winding(out[0], grid, max(cfg["waist_m"]*np.sqrt(max(abs(charge),1)/2), grid.dx*2))
        crossed = float(np.sum(abs(out[1])**2)/np.sum(abs(out)**2))
        rows.append({"charge": charge, "power_gain": gain, "overlap_with_input": vector_overlap(seed, out),
                     "target_oam_fraction": fraction, "crossed_polarization_fraction": crossed,
                     "oam": spec, "closed_contour_winding": winding,
                     "scope": cfg["model"]})
        fields.append(out)
    return np.asarray(fields), rows


def _cavity_sampling(grid, physical):
    c = physical["cavity"]
    step = 4*np.pi*(c["disk_diameter_m"]/2)*max(grid.dx,grid.dy)/(c["wavelength_m"]*c["output_mirror_radius_m"])
    return {"max_mirror_phase_step_at_disk_edge_rad": float(step),
            "mirror_nyquist_over_disk_radius": bool(step <= np.pi),
            "qualification": "necessary aperture sampling diagnostic, not full propagation-grid convergence"}


def frozen_case(case, reference_directory):
    reference = FrozenReference(reference_directory, case["physics"]["frozen_reference"])
    density = case["physics"]["density"]
    if density["contrast_bound"] != 0 or not np.isclose(density["mean_m3"], 1.52e26, rtol=1e-12):
        raise ValueError("frozen archived populations/heat cannot represent a changed dopant distribution")
    grid, mesh, cfg = make_grid(case), make_mesh(case), assembly_config(case)
    q = conservative_remap(reference.data["raw_heat_W_m3"], reference.mesh, mesh)
    old_heat = float(np.sum(reference.data["raw_heat_W_m3"]*reference.mesh.volumes_m3))
    new_heat = float(np.sum(q*mesh.volumes_m3))
    if abs(new_heat-old_heat) > 1e-12*max(1., abs(old_heat)):
        raise RuntimeError("conservative heat transfer failed")
    temperature, displacement, screens, fem = solve_assembly(mesh, grid, q, cfg)
    from .vector_cavity import PlaneExchange
    # Gain is always sampled from the same ORIGINAL archived field. Thermal-grid
    # refinement must not also resample the physical frozen gain prescription.
    # Constants are explicit audit-baseline material inputs, not fitted outputs.
    sg = (1.2e-24*reference.fractions[2] - 2.1e-25*reference.fractions[3]) * density["mean_m3"]
    column = np.sum(sg*np.diff(reference.mesh.z_edges_m)[:,None,None], axis=0)
    log_gain = PlaneExchange(grid, reference.mesh, order=6).surface_on_grid(column)
    fields, probes = probe_diagnostics(grid, screens, log_gain, case["physics"])
    from .thermomechanics import von_mises
    metrics = {"output_W": None, "pump_absorbed_W": None, "heat_W": new_heat,
               "peak_disk_K": float(temperature.disk_temperature_K.max()),
               "peak_plate_K": float(temperature.plate_temperature_K.max()),
               "peak_disk_von_mises_Pa": float(von_mises(displacement.disk_stress_Pa).max()),
               "thermal_energy_error_relative": temperature.relative_balance_error,
               "mechanical_residual": displacement.free_residual_relative,
               "heat_remap_relative_error": abs(new_heat-old_heat)/max(abs(old_heat), 1e-20),
               "optical_population_error_relative": None, "eigen_residual": None,
               "coupled_converged": False, "component_solved": True}
    arrays = {"fields_used": fields, "x_m": grid.x, "y_m": grid.y,
              "mean_roundtrip_opd_m": screens.mean_roundtrip_opd_m,
              "thermal_single_pass_opd_m": screens.thermal_single_pass_opd_m,
              "geometry_roundtrip_opd_m": screens.geometry_roundtrip_opd_m,
              "photoelastic_mean_single_pass_opd_m": screens.photoelastic_mean_single_pass_opd_m,
              "front_uz_m": screens.front_uz_m, "rear_uz_m": screens.rear_uz_m,
              "heat_W_m3": q, "disk_temperature_K": temperature.disk_temperature_K,
              "plate_temperature_K": temperature.plate_temperature_K,
              "interface_flux_W_m2": temperature.interface_flux_W_m2,
              "r_edges_m": mesh.r_edges_m, "z_edges_m": mesh.z_edges_m,
              "inward_jones": screens.inward_jones, "outward_jones": screens.outward_jones}
    return {"status": "completed", "metrics": metrics, "probes": probes,
            "field_semantics": "independent_weak_probes", "probe_labels": [r["charge"] for r in probes],
            "source_reference": reference.provenance(), "sampling": _cavity_sampling(grid,case["physics"]),
            "mechanical_nodes": len(fem.disk.nodes_m)+len(fem.plate.nodes_m),
            "mechanical_tetrahedra": len(fem.disk.tetrahedra)+len(fem.plate.tetrahedra),
            "scope": "Frozen archived heat and mean gain; laser output NOT recomputed."}, arrays


def coupled_case(case, progress=None):
    # Lazy import: a frozen-source study need not instantiate the expensive
    # oscillator. This path requires the complete audited repository API 0.8.
    from .resonator import ThinDiskResonator
    from .coupled_resonator import HotCavitySettings, run_coupled_hot_cavity
    from .populations import HoYAGFourLevelParams
    from .vector_cavity import PlaneExchange
    grid, mesh, cfg = make_grid(case), make_mesh(case), assembly_config(case)
    c = ThinDiskResonator(**case["physics"]["cavity"])
    pump = case["physics"]["pump"]; settings = HotCavitySettings(**case["numerics"]["settings"])
    density_spec = ContinuousDensity(**case["physics"]["density"])
    density = density_spec.cell_average(mesh, case["numerics"]["density_quadrature_order"])
    p = HoYAGFourLevelParams(N_total_m3=density_spec.mean_m3)
    result = run_coupled_hot_cavity(grid, mesh, c, pump["energy_J"], cfg,
        repetition_rate_Hz=pump["repetition_rate_Hz"], pump_duration_s=pump["duration_s"],
        pump_waist_m=pump["waist_m"], params=p, density_m3=density,
        initial_fields=initial_modes(grid,c,case["numerics"]), mode_count=case["numerics"]["mode_count"],
        settings=settings, progress=progress)
    arrays = {"fields_used": result.fields_used, "x_m": grid.x, "y_m": grid.y,
              "r_edges_m": mesh.r_edges_m, "z_edges_m": mesh.z_edges_m, "density_m3": density}
    base = {"status": "completed" if result.converged else "not_converged", "solver_status": result.status,
            "field_semantics": "incoherent_cavity_modes", "history": result.history,
            "solver_metadata": result.metadata, "sampling": _cavity_sampling(grid,case["physics"]),
            "physical_density_fingerprint": density_spec.fingerprint,
            "density_volume_mean_m3": float(np.sum(density*mesh.volumes_m3)/mesh.volumes_m3.sum()),
            "scope": "Self-consistent adiabatic hot cavity with finite cooling plate and bonded mechanics."}
    if result.cycle_heat is None or result.mean_fractions is None or result.temperature is None:
        base["metrics"] = {"coupled_converged": False}
        return base, arrays
    heat = result.cycle_heat; budget = heat.budget; temp = result.temperature
    base["metrics"] = {"output_W": budget["output_W"], "pump_absorbed_W": budget["pump_absorbed_W"],
        "heat_W": budget["heat_W"], "peak_disk_K": float(temp.disk_temperature_K.max()),
        "peak_plate_K": float(temp.plate_temperature_K.max()),
        "thermal_energy_error_relative": temp.relative_balance_error,
        "mechanical_residual": result.displacement.free_residual_relative,
        "optical_population_error_relative": budget["local_ledger_L1_error_over_incident"],
        "eigen_residual": float(np.max(result.eigenfields.residuals)) if result.eigenfields else None,
        "coupled_converged": result.converged,
        "mode_power_W": list(budget["output_W_by_mode"])}
    arrays.update({"mean_roundtrip_opd_m": result.screens.mean_roundtrip_opd_m,
                   "mean_fractions": result.mean_fractions, "raw_heat_W_m3": heat.heat_W_m3.reshape(mesh.shape),
                   "assembly_heat_W_m3": result.assembly_heat_W_m3,
                   "disk_temperature_K": temp.disk_temperature_K, "plate_temperature_K": temp.plate_temperature_K,
                   "interface_flux_W_m2": temp.interface_flux_W_m2, "fields_predicted": result.fields_predicted})
    f = result.mean_fractions.reshape(4, *mesh.shape)
    gain = (p.sigma_em_laser_m2*f[2]-p.sigma_abs_laser_m2*f[3])*density
    column = np.sum(gain*np.diff(mesh.z_edges_m)[:,None,None], axis=0)
    log_gain = PlaneExchange(grid,mesh,order=settings.projection_order).surface_on_grid(column)
    _, base["probes"] = probe_diagnostics(grid,result.screens,log_gain,case["physics"])
    base["mode_oam"] = [oam_spectrum(field,grid) for field in result.fields_used]
    base["energy_budget"] = budget
    return base, arrays
