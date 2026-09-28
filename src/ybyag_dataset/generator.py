"""Physically rerun Yb:YAG before making phase-diverse synthetic observations.

The generator intentionally rejects out-of-range thermal states. It does not
turn the current room-temperature Yb:YAG model into validated hot-gain truth.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from hoyag.propagation import Grid2D, angular_spectrum_propagate
from hoyag.thermal import DiskThermalMesh
from hoyag.local_supervisor import atomic_json
from ybluag.beam_shaping import gaussian_seed_and_target_mask
from ybluag.gallery import YbGallerySettings, simulate_pulsed_seed, _assembly_configuration
from ybluag.regenerative import RegenerativeCavity
from ybluag.assembly import yb_cooler_solver
from ybluag.camera_dataset import CameraSettings, suggest_optical_throughput
from hoyag.cooling_plate import sample_temperature
from ybyag.model import YbYAGMaterial
from .distortions.common import child_seeds
from .distortions.material import sample_material
from .distortions.thermal import sample_contact, sample_operating_point
from .distortions.beam import sample_beam
from .distortions.optical import sample_optical_phase
from .distortions.slm import sample_slm_setup, apply_slm
from .distortions.camera import sample_camera_setup, capture
from .distortions.sensors import SensorSession
from .response_teacher import response_modes, response_matrix_teacher
from .field_metrics import (field_metrics, phase_residual as shared_phase_residual,
                            phase_support)
from .modal_teacher import FrozenOpticalModel, modal_basis, modal_correction_teacher


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(v) for v in value]
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _numerical_sources_sha256():
    root=Path(__file__).resolve().parents[2]
    digest=hashlib.sha256()
    for name in ("hoyag","ybluag","ybyag","ybyag_dataset"):
        folder=root/"src"/name
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    assembly=root/"config"/"ybluag_10at_assembly.json"
    digest.update(assembly.read_bytes())
    return digest.hexdigest()


def _settings(nominal, pump, beam):
    return YbGallerySettings(
        pump_power_W=pump["pump_W"], pump_radius_m=pump["radius_mm"]*1e-3,
        thickness_m=nominal["disk_thickness_um"]*1e-6,
        disk_radius_m=nominal["disk_radius_mm"]*1e-3,
        assembly_property_model="yag_rt_proxy",
        waist_m=beam["waist_mm"]*1e-3,
        post_disk_distance_m=nominal["output_distance_m"],
        slm_to_disk_distance_m=nominal["slm_to_disk_m"],
        phase_mask_name="none", cluster_contrast=0.,
        grid_n=int(nominal["grid_n"]),field_size_m=nominal["field_size_mm"]*1e-3,
        z_steps=4, thermal_nr=int(nominal["thermal_nr"]),
        thermal_nphi=int(nominal["thermal_nphi"]),thermal_nz=int(nominal["thermal_nz"]))


def _phase_target(result, desired_disk_field, grid, wavelength_m,
                  target_phase, external_phase, settings):
    """First-order phase-only oracle; retains intentional target mask."""
    timeline = result["thermal_timeline"]
    disk_phase = (np.pi*result["effective_signal_traversals"]/(wavelength_m)*
                  np.asarray(timeline["final_roundtrip_opd_m"],float))
    static = (result["effective_signal_traversals"]*
              np.asarray(result["static_cold_phase_rad"],float))
    compensation = angular_spectrum_propagate(
        desired_disk_field*np.exp(-1j*(disk_phase+static)),
        grid,wavelength_m,-settings.slm_to_disk_distance_m)
    correction = np.angle(np.exp(1j*(np.angle(compensation)-target_phase-external_phase)))
    return correction, disk_phase


def absolute_oracle_command(target_phase,correction_phase):
    """Compose the intentional structure with an absolute oracle correction."""
    return np.mod(np.asarray(target_phase)+np.asarray(correction_phase),2*np.pi)


def collocated_external_oracle(target_phase,external_phase):
    """Undo a known simulated screen at the SLM plane for oracle proposals.

    The screen is privileged simulation truth, never a measured network input.
    The proposed command still needs fresh full-solver verification.
    """
    target=np.asarray(target_phase,float)
    external=np.asarray(external_phase,float)
    if target.shape!=external.shape or not (np.all(np.isfinite(target)) and
                                            np.all(np.isfinite(external))):
        raise ValueError("target and external phase maps must be finite and aligned")
    return np.mod(target-external,2*np.pi)


REQUIRED_TARGETS = ("Gaussian TEM00", "Flattop super-Gaussian",
                    "Helical LG(0,+1)", "Needle Bessel-Gaussian")
REQUIRED_CONCENTRATIONS = (5.0, 10.0, 15.0)


def setup_plan(config, split_counts=None):
    """Assign complete, disjoint physical setups to every split."""
    counts = config["split_counts"] if split_counts is None else split_counts
    if (set(counts) != {"train", "validation", "test", "stress"} or
            any(type(v) is not int or v < 0 for v in counts.values())):
        raise ValueError("invalid setup split counts")
    concentrations = tuple(float(x) for x in config["nominal"]["yb_at_percent_candidates"])
    targets = tuple(config["nominal"]["target_candidates"])
    if (set(concentrations) != set(REQUIRED_CONCENTRATIONS) or
            set(targets) != set(REQUIRED_TARGETS)):
        raise ValueError("required 5/10/15 at.% and four structured targets")
    combinations = [(doping, target) for doping in REQUIRED_CONCENTRATIONS
                    for target in REQUIRED_TARGETS]
    if any(counts[split] < len(combinations)
           for split in ("train", "validation", "test")):
        raise ValueError("each split needs at least 12 complete setups for coverage")
    plan = {split: [] for split in counts}
    number = 0
    for split in ("train", "validation", "test", "stress"):
        for index in range(counts[split]):
            doping, target = combinations[index % len(combinations)]
            plan[split].append({"setup_id": f"{split}_crystal_{number:04d}",
                                "yb_at_percent": doping, "target": target,
                                "seed_index": number})
            number += 1
    return plan


def camera_shape_loss(measured_adu, ideal_clean, black_level_adu):
    """Amplitude-independent focus/defocus shape loss from measured cameras."""
    measured = np.maximum(np.asarray(measured_adu, float)-black_level_adu, 0)
    ideal = np.maximum(np.asarray(ideal_clean, float), 0)
    if measured.shape != ideal.shape or measured.ndim != 3:
        raise ValueError("camera and target plane stacks disagree")
    measured /= np.maximum(measured.sum(axis=(1, 2), keepdims=True), 1e-30)
    ideal /= np.maximum(ideal.sum(axis=(1, 2), keepdims=True), 1e-30)
    return float(np.mean([np.sum((a-b)**2)/max(np.sum(b*b), 1e-30)
                          for a,b in zip(measured, ideal)]))


def trial_command(retained_command, target_phase, grid, trial_index,
                  step_rad=0.25):
    """A deterministic coordinate-search proposal around the retained command."""
    if trial_index == 0:
        return np.mod(target_phase, 2*np.pi)
    x,y=grid.mesh
    radius=max(float(np.max(np.abs(grid.x))), 1e-30)
    modes=((x/radius)**2+(y/radius)**2,
           (x*x-y*y)/radius**2,
           2*x*y/radius**2,
           x/radius,
           y/radius)
    coordinate=((trial_index-1)//2) % len(modes)
    sign=1 if trial_index % 2 else -1
    return np.mod(retained_command+sign*step_rad*modes[coordinate],2*np.pi)


def _plot_validation(path, arrays, sensor):
    fig, axes = plt.subplots(4,3,figsize=(13,14),constrained_layout=True)
    panels = [
        ("Yb concentration scale",arrays["truth__yb_scale"],"viridis"),
        ("Temperature at disk surface (K)",arrays["truth__temperature_surface_K"],"inferno"),
        ("Front displacement (nm)",arrays["truth__front_displacement_m"]*1e9,"coolwarm"),
        ("Unwanted output phase (rad)",np.where(arrays["truth__phase_valid_mask"],
            arrays["truth__unwanted_phase_rad"],np.nan),"twilight"),
        ("Desired beam fluence",arrays["truth__desired_fluence_J_m2"],"viridis"),
        ("Distorted beam fluence",arrays["truth__amplitude_sqrt_J_m"]**2,"viridis"),
        ("Clean camera, focus",arrays["truth__clean_camera_fluence_J_m2"][0][::8,::8],"viridis"),
        ("Noisy camera, focus",arrays["input__camera_adu"][0][::8,::8],"viridis"),
        ("Verified SLM correction (rad)",np.where(arrays["truth__slm_valid_mask"],
            arrays["truth__verified_slm_correction_rad"],np.nan),"twilight"),
        ("Residual after correction (rad)",np.where(arrays["truth__phase_valid_mask"],
            arrays["truth__corrected_residual_phase_rad"],np.nan),"twilight"),
        ("Corrected beam fluence",arrays["truth__corrected_fluence_J_m2"],"viridis"),
        ("Thermal contact scale",arrays["truth__contact_scale"],"coolwarm"),
    ]
    for ax,(title,image,cmap) in zip(axes.flat,panels):
        ax.imshow(image,cmap=cmap,origin="lower")
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
    # Probe positions are recorded exactly in the sample; show them on T.
    ax = axes[0,1]
    disk = sensor["position_m"][:3,:2]
    n = arrays["truth__temperature_surface_K"].shape[-1]
    window = arrays["metadata__field_size_m"].item()
    ax.scatter((disk[:,0]/window+.5)*n,(disk[:,1]/window+.5)*n,
               facecolors="none",edgecolors="white",s=85,linewidths=1.5)
    fig.savefig(path,dpi=130)
    plt.close(fig)


def _stage_directory(output, split, group_id):
    return output/".parallel_staging"/split/group_id


def _complete_group_rows(output, split, group_id, points_per_setup, config_sha):
    """Find a fully written group, including one metadata file per trial."""
    folder=output/split/group_id
    setup_file=folder/"setup.npz"
    if not setup_file.is_file():
        return None
    setup_sha=_sha(setup_file)
    rows=[]
    for point in range(points_per_setup):
        metadata_file=folder/f"point_{point:03d}.json"
        if not metadata_file.is_file():
            return None
        try:
            metadata=json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (metadata.get("split")!=split or metadata.get("setup_id")!=group_id or
                metadata.get("trial_index")!=point or
                metadata.get("source_config_sha256")!=config_sha or
                metadata.get("setup_npz_sha256")!=setup_sha or
                not metadata.get("measurements_file") or
                not metadata.get("truth_file") or
                not (folder/metadata["measurements_file"]).is_file() or
                not (folder/metadata["truth_file"]).is_file()):
            return None
        rows.append(f"{split}/{group_id}/{metadata_file.name}")
    return rows


def _generate_setup_task(config_path, output, counts, points_per_setup, split, group_id):
    """One process owns one setup and all of its sequential trials."""
    stage=_stage_directory(output,split,group_id)
    generate(config_path,stage,split_counts=counts,points_per_setup=points_per_setup,
             resume=(stage/"manifest.json").is_file(),selected_setup=(split,group_id))
    return split,group_id


def _run_parallel_setups(config_path, output, counts, plan, points_per_setup,
                         manifest, manifest_path, workers):
    """Commit finished setups through one manifest writer."""
    expected={split:{row["setup_id"]:i for i,row in enumerate(rows)}
              for split,rows in plan.items()}

    def commit(split,group_id,rows):
        files=[name for name in manifest["splits"][split]
               if Path(name).parent.name!=group_id]
        files.extend(rows)
        files.sort(key=lambda name:(expected[split][Path(name).parent.name],name))
        manifest["splits"][split]=files
        atomic_json(manifest_path,manifest)

    pending=[]
    for split in ("train","validation","test","stress"):
        for spec in plan[split]:
            group_id=spec["setup_id"]
            final_rows=_complete_group_rows(output,split,group_id,points_per_setup,
                                            manifest["source_config_sha256"])
            if final_rows is not None:
                if [name for name in manifest["splits"][split]
                    if Path(name).parent.name==group_id]!=final_rows:
                    commit(split,group_id,final_rows)
            else:
                pending.append((split,group_id))
    atomic_json(manifest_path,manifest)
    if not pending:
        return manifest_path

    with ProcessPoolExecutor(max_workers=min(workers,len(pending)),
                             mp_context=get_context("spawn")) as pool:
        futures={pool.submit(_generate_setup_task,config_path,output,counts,
                             points_per_setup,split,group_id):(split,group_id)
                 for split,group_id in pending}
        for future in as_completed(futures):
            split,group_id=futures[future]
            future.result()
            stage=_stage_directory(output,split,group_id)
            rows=_complete_group_rows(stage,split,group_id,points_per_setup,
                                      manifest["source_config_sha256"])
            if rows is None:
                raise RuntimeError(f"parallel setup {group_id} has incomplete files")
            staged_group=stage/split/group_id
            final_group=output/split/group_id
            final_group.parent.mkdir(parents=True,exist_ok=True)
            if final_group.exists():
                index=1
                while (stage/f"previous_final_{index}").exists():
                    index+=1
                final_group.rename(stage/f"previous_final_{index}")
            staged_group.rename(final_group)
            commit(split,group_id,rows)
    return manifest_path


def generate(config_path, output_dir, *, split_counts=None, points_per_setup=3,
             smoke=False, resume=False, smoke_setup_index=0, workers=1,
             selected_setup=None):
    """Create disjoint setup-level splits; call only inside a bounded worker."""
    config_path = Path(config_path)
    output = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nominal, ranges = config["nominal"], config["ranges"]
    if (config["camera"]["pulses_per_exposure"] >
            nominal["repetition_rate_kHz"]*1e3*config["camera"]["exposure_s"]+1e-9):
        raise ValueError("camera exposure is too short for the pulse count and repetition rate")
    counts = config["split_counts"] if split_counts is None else split_counts
    if type(points_per_setup) is not int or points_per_setup < 3:
        raise ValueError("closed-loop sequences need at least three measured trials per setup")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    plan = setup_plan(config, counts)
    if smoke:
        if not 0<=smoke_setup_index<len(plan["train"]):
            raise ValueError("smoke setup index outside planned train setups")
        plan={"train":plan["train"][smoke_setup_index:smoke_setup_index+1],
              "validation":[],"test":[],"stress":[]}
    if selected_setup is not None:
        split,group_id=selected_setup
        if split not in plan or not any(row["setup_id"]==group_id for row in plan[split]):
            raise ValueError("selected setup is outside the planned dataset")
        plan={name:[row for row in rows if name==split and row["setup_id"]==group_id]
              for name,rows in plan.items()}
    if int(nominal["grid_n"]) < 256:
        raise ValueError("optical and SLM sampling must be at least 256² before 1080p rendering")
    calibration = config.get("camera_calibration", {"status": "illustrative"})
    if calibration.get("status") == "measured" and not calibration.get("source"):
        raise ValueError("measured camera calibration requires a source identifier")
    output.mkdir(parents=True,exist_ok=True)
    current_generator_sha=_sha(__file__)
    current_numerical_sha=_numerical_sources_sha256()
    manifest = {"schema":"ybyag_nn_closed_loop_v3", "dataset_ready":False,
        "smoke_unqualified":bool(smoke),
        "smoke_setup_index":smoke_setup_index if smoke else None,
        "source_config_sha256":_sha(config_path),
        "source_generator_sha256":current_generator_sha,
        "numerical_source_sha256":current_numerical_sha,
        "created_unix_s":time.time(), "splits":{k:[] for k in counts},
        "setup_plan":plan,"camera_calibration":calibration,
        "coverage":{split:{"setups":len(rows),"concentrations_at_percent":sorted(set(r["yb_at_percent"] for r in rows)),
                            "targets":sorted(set(r["target"] for r in rows))}
                    for split,rows in plan.items()},
        "limits":["Yb:YAG hot gain unavailable; all accepted states are room-temperature lumped-phase approximations.",
                  "Thickness uses an optical-column and static geometric-phase approximation; the FEM retains nominal thickness.",
                  "Background absorption uses first-order pre-pump attenuation and distributed heat.",
                  "Photoelasticity and stress tensor are not solved in this scalar Yb amplifier.",
                  "Phase-only labels are best passing commands from a bounded, solver-verified candidate search, not guaranteed global optima.",
                  "Response-matrix teacher, when selected, probes privileged simulated complex fields; these are not experimentally measured calibration data.",
                  "Instrument distributions are illustrative until calibrated."]}
    manifest_path=output/"manifest.json"
    if workers>1:
        for split,rows in plan.items():
            for spec in rows:
                stage=_stage_directory(output,split,spec["setup_id"])
                final_group=output/split/spec["setup_id"]
                if not final_group.exists():
                    backups=sorted(stage.glob("previous_final_*"))
                    if backups:
                        final_group.parent.mkdir(parents=True,exist_ok=True)
                        backups[-1].rename(final_group)
    if resume:
        if not manifest_path.is_file():
            raise ValueError("resume requires an existing manifest.json")
        prior=json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("schema","source_config_sha256","setup_plan","smoke_unqualified",
                    "smoke_setup_index"):
            if prior.get(key)!=manifest[key]:
                raise ValueError(f"resume refused: {key} differs from the saved run")
        for split,files in prior["splits"].items():
            if split not in manifest["splits"] or any(not (output/name).is_file() for name in files):
                raise ValueError("resume refused: saved trial metadata is missing")
        manifest=prior
        versions=manifest.setdefault("resume_source_versions",[])
        version={"generator":current_generator_sha,"numerical_sources":current_numerical_sha}
        if version not in versions:
            versions.append(version)
        if (version["generator"]!=manifest["source_generator_sha256"] or
                version["numerical_sources"]!=manifest["numerical_source_sha256"]):
            note="Resumed across source revisions; inspect per-trial fingerprints before training."
            if note not in manifest["limits"]:
                manifest["limits"].append(note)
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    elif manifest_path.exists():
        raise ValueError("output already has a manifest; use --resume or a new directory")
    if workers>1 and sum(map(len,plan.values()))>1:
        return _run_parallel_setups(config_path,output,counts,plan,points_per_setup,
                                    manifest,manifest_path,workers)
    for split in ("train","validation","test","stress"):
        for setup_spec in plan[split]:
            group_number=setup_spec["seed_index"]
            group_seed=child_seeds(int(config["seed"])+group_number,9)
            group_id=setup_spec["setup_id"]
            existing=[name for name in manifest["splits"][split]
                      if Path(name).parent.name==group_id]
            if resume and len(existing)==points_per_setup:
                rows=[json.loads((output/name).read_text(encoding="utf-8"))
                      for name in existing]
                if (sorted(row["trial_index"] for row in rows)==list(range(points_per_setup)) and
                        all((output/name).parent.joinpath(row["measurements_file"]).is_file() and
                            (output/name).parent.joinpath(row["truth_file"]).is_file()
                            for name,row in zip(existing,rows))):
                    continue
            if resume and existing:
                manifest["splits"][split]=[name for name in manifest["splits"][split]
                                           if Path(name).parent.name!=group_id]
                manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
            group_concentration=setup_spec["yb_at_percent"]
            group_target=setup_spec["target"]
            stress=config["stress_range_multiplier"] if split=="stress" else 1.
            n=int(nominal["grid_n"])
            grid=Grid2D.square(n,nominal["field_size_mm"]*1e-3)
            x,y=grid.mesh
            wavelength_m=1030e-9
            enabled=config["enabled"]
            crystal=sample_material(grid.shape,ranges,group_seed[0],
                enabled=enabled["material"],stress=stress)
            contact=sample_contact((int(nominal["thermal_nr"]),int(nominal["thermal_nphi"])),
                ranges,group_seed[1],enabled=enabled["thermal"],stress=stress)
            external,external_parameters=sample_optical_phase(x,y,5e-3,wavelength_m,
                ranges,group_seed[2],enabled=enabled["optical"],stress=stress)
            slm_setup=sample_slm_setup(grid.shape,ranges,group_seed[3],
                enabled=enabled["slm"],stress=stress)
            if "slm_phase_lut_rad" in config:
                slm_setup["phase_lut_rad"]=np.asarray(
                    config["slm_phase_lut_rad"],float)
            camera_dict={k:v for k,v in config["camera"].items() if v is not None}
            camera_settings=CameraSettings(**camera_dict)
            camera_setup=sample_camera_setup(camera_settings,ranges,group_seed[4],
                enabled=enabled["camera"],stress=stress)
            camera_temp_rng=np.random.default_rng(group_seed[4]+1)
            camera_temp_delta=0.
            folder=output/split/group_id
            folder.mkdir(parents=True,exist_ok=True)
            setup_file=folder/"setup.npz"
            np.savez_compressed(setup_file,
                yb_concentration_scale=crystal["yb_concentration_scale"],
                thickness_scale=crystal["thickness_scale"],
                background_absorption_m1=crystal["background_absorption_m1"],
                surface_figure_m=crystal["surface_figure_m"],
                contact_scale=contact,external_optics_phase_rad=external,
                slm_spatial_gain=slm_setup["spatial_gain"],
                slm_pixel_gain=slm_setup["pixel_gain"],
                slm_global_gain=np.asarray(slm_setup["global_gain"]),
                slm_bits=np.asarray(slm_setup["bits"]),
                slm_crosstalk_sigma_pixels=np.asarray(
                    slm_setup["crosstalk_sigma_pixels"]),
                slm_phase_lut_rad=np.asarray(
                    slm_setup.get("phase_lut_rad",[]),float),
                camera_prnu=camera_setup["prnu"],
                camera_dsnu_e=camera_setup["dsnu_e"],
                camera_hot=camera_setup["hot"],camera_dead=camera_setup["dead"])
            sensor_session=None
            previous_command=None
            retained_command=None
            retained_loss=float("inf")
            previous_thermal_state=None
            base_pump=sample_operating_point(nominal,ranges,group_seed[7],
                enabled=enabled["thermal"],stress=stress)
            base_beam=sample_beam(nominal,ranges,group_seed[8],
                enabled=enabled["beam"],stress=stress)
            aperture_rng=np.random.default_rng(group_seed[8]+1)
            aperture_radius_waists=aperture_rng.uniform(*ranges["seed_aperture_radius_waists"])
            aperture_decenter=aperture_rng.normal(0,
                stress*ranges["seed_aperture_decenter_waist_fraction"]*
                nominal["waist_mm"]*1e-3,2)
            for point in range(points_per_setup):
                point_seed=child_seeds(group_seed[5]+point,5)
                elapsed=(0. if point==0 else nominal["operation_duration_s"]+
                         (point-1)*nominal["sequence_step_s"])
                duration=(nominal["operation_duration_s"] if point==0 else
                          nominal["sequence_step_s"])
                pump=sample_operating_point(nominal,ranges,point_seed[0],
                    enabled=enabled["thermal"],stress=stress,elapsed_s=elapsed)
                beam=sample_beam(nominal,ranges,point_seed[1],
                    enabled=enabled["beam"],stress=stress,elapsed_s=elapsed)
                jitter_rng=np.random.default_rng(point_seed[3])
                if enabled["beam"]:
                    beam["seed_center_m"]=tuple(np.asarray(base_beam["seed_center_m"])+
                        jitter_rng.normal(0,ranges["alignment_drift_um_per_s"]*1e-6*elapsed,2))
                    beam["seed_angle_rad"]=tuple(np.asarray(base_beam["seed_angle_rad"])+
                        jitter_rng.normal(0,stress*ranges["frame_pointing_jitter_urad"]*1e-6,2))
                    beam["seed_ellipticity"]=base_beam["seed_ellipticity"]
                if enabled["thermal"]:
                    pump["pump_center_m"]=tuple(np.asarray(base_pump["pump_center_m"])+
                        jitter_rng.normal(0,stress*ranges["pump_pointing_jitter_radius_fraction"]*
                                          pump["radius_mm"]*1e-3,2))
                settings=_settings(nominal,pump,beam)
                material=YbYAGMaterial(yb_at_percent=group_concentration)
                architecture=nominal["architecture"]
                if architecture not in ("ideal_multipass","regenerative"):
                    raise ValueError("unknown amplifier architecture in dataset configuration")
                cavity=(RegenerativeCavity(
                    round_trips=int(nominal["regen_round_trips"]),
                    air_gap_m=nominal["cavity_length_m"],
                    mirror_radius_m=nominal["mirror_radius_m"],
                    disk_hr_reflectivity=nominal["disk_hr_reflectivity"],
                    held_roundtrip_retention=nominal["held_retention"],
                    injection_efficiency=nominal["injection_efficiency"],
                    extraction_efficiency=nominal["extraction_efficiency"],
                    disk_diameter_m=2*settings.disk_radius_m)
                    if architecture=="regenerative" else None)
                _,target_phase=gaussian_seed_and_target_mask(grid,settings.waist_m,1.,
                    group_target,wavelength_m,settings.slm_to_disk_distance_m)
                requested=trial_command(retained_command,target_phase,grid,point,
                    step_rad=float(config.get("trial_step_rad",.25)))
                delayed=bool(config.get("slm_update_delay_points",0) and point>0)
                prior_command=previous_command
                current_command=(prior_command if delayed and prior_command is not None
                                 else requested)
                actual_slm,_=apply_slm(requested,slm_setup,
                    drift_fraction=ranges["slm_drift_fraction_per_s"]*elapsed,
                    previous_command=previous_command,delayed=delayed)
                previous_command=requested.copy()
                physical={**crystal,"contact_scale_polar":contact,
                    "coolant_temperature_K":pump["coolant_temperature_K"],
                    "pump_center_m":pump["pump_center_m"],
                    "seed_center_m":beam["seed_center_m"],
                    "seed_angle_rad":beam["seed_angle_rad"],
                    "seed_ellipticity":beam["seed_ellipticity"],
                    "slm_actual_phase_rad":actual_slm,
                    "external_phase_rad":external}
                if enabled["beam"]:
                    physical["seed_aperture"]={
                        "radius_m":aperture_radius_waists*settings.waist_m,
                        "center_m":tuple(aperture_decenter)}
                if previous_thermal_state is not None:
                    physical["initial_disk_temperature_K"]=previous_thermal_state[0]
                    physical["initial_plate_temperature_K"]=previous_thermal_state[1]
                def counterfactual_physical(slm_phase):
                    # Every candidate starts from the same pre-action crystal
                    # and thermal state. The reference solver then evolves its
                    # own populations across all signal encounters.
                    replay={key:(value.copy() if isinstance(value,np.ndarray)
                                 else value) for key,value in physical.items()}
                    replay["slm_actual_phase_rad"]=np.asarray(slm_phase,float).copy()
                    return replay
                result=simulate_pulsed_seed(material,settings,group_target,
                    beam["seed_energy_nj"]*1e-9,nominal["seed_fwhm_ps"]*1e-12,
                    nominal["repetition_rate_kHz"]*1e3,
                    int(nominal["signal_traversals"]),
                    pump_passes=int(nominal["pump_passes"]),
                    compute_thermal=True,
                    operation_duration_s=duration,
                    cooling_mode="fixed",thermal_optical_mode="lumped_phase",
                    architecture=architecture,regenerative_cavity=cavity,
                    dataset_physical=physical)
                timeline=result["thermal_timeline"]
                if (timeline is None or not timeline["requested_material_range_valid"] or
                        not result["thermal_feedback_applied"]):
                    raise RuntimeError(f"{group_id}: thermal state outside supported Yb:YAG range; no ground truth written")
                next_thermal_state=(np.asarray(timeline["requested_disk_temperature_K"]).copy(),
                                    np.asarray(timeline["requested_plate_temperature_K"]).copy())
                field=np.asarray(result["output_complex_field_sqrt_J_m"],complex)
                x_axis=grid.x
                y_axis=grid.y
                if config["camera"]["optical_throughput"] is None and point==0:
                    example_result={"output_fluence_J_m2":abs(field)**2,
                        "x_mm":x_axis*1e3,"y_mm":y_axis*1e3,
                        "signal_wavelength_nm":1030.}
                    from dataclasses import replace
                    camera_settings=replace(camera_settings,optical_throughput=
                        min(1., suggest_optical_throughput(example_result,camera_settings)*
                            (config.get("stress_camera_throughput_multiplier",3.)
                             if split=="stress" else 1.)))
                images=[];clean_images=[];camera_diagnostics=[]
                camera_temp_delta=(camera_settings.sensor_temperature_correlation*
                    camera_temp_delta + camera_settings.sensor_temperature_jitter_C*
                    np.sqrt(1-camera_settings.sensor_temperature_correlation**2)*
                    camera_temp_rng.normal())
                for plane_index,z in enumerate(config["planes_m"]):
                    field_plane=(field if z==0 else angular_spectrum_propagate(
                        field,grid,wavelength_m,float(z)))
                    image,clean,diagnostic=capture(abs(field_plane)**2,x_axis,y_axis,
                        wavelength_m,camera_settings,camera_setup,
                        point_seed[2]+plane_index,enabled=enabled["camera"],
                        sensor_temperature_C=camera_settings.sensor_temperature_C+
                                             camera_temp_delta,
                        pulse_variation=config.get("pulse_exposure_variation"))
                    images.append(image);clean_images.append(clean)
                    camera_diagnostics.append(diagnostic)
                diversity_L1=(float(np.sum(abs(clean_images[0]-clean_images[1]))/
                    max(np.sum(clean_images[0]),1e-30)) if len(clean_images)>1 else None)
                disk_mesh=DiskThermalMesh.disk(nr=settings.thermal_nr,nphi=settings.thermal_nphi,
                    nz=settings.thermal_nz,radius_m=settings.disk_radius_m,
                    thickness_m=settings.thickness_m)
                assembly_cfg=_assembly_configuration(material,settings)
                assembly_cfg["thermal"]["interface_conductance_W_m2K"] *= contact
                assembly_cfg["thermal"]["coolant_temperature_K"]=pump["coolant_temperature_K"]
                plate=yb_cooler_solver(disk_mesh,assembly_cfg).plate
                if sensor_session is None:
                    sensor_ranges=(ranges if split!="stress" else
                        {**ranges,"probe_dropout_probability":max(.2,ranges["probe_dropout_probability"])})
                    sensor_session=SensorSession(disk_mesh,plate,sensor_ranges,group_seed[6],
                        enabled=enabled["sensors"],stress=stress,
                        force_dropout=split=="stress")
                sensor=sensor_session.sample(timeline["requested_disk_temperature_K"],
                    timeline["requested_plate_temperature_K"],elapsed+duration)
                temperature_xy=np.full(grid.shape,np.nan)
                inside=x*x+y*y <= settings.disk_radius_m**2
                temperature_xy[inside]=sample_temperature(disk_mesh,
                    timeline["requested_disk_temperature_K"],
                    np.column_stack((x[inside],y[inside],np.zeros(np.sum(inside)))))
                # Reference target is the same deliberately shaped beam without
                # external optics or thermomechanical aberration.
                source=np.exp(-(x*x+y*y)/settings.waist_m**2)
                source/=np.sqrt(np.sum(abs(source)**2)*grid.dx*grid.dy)
                desired_disk=angular_spectrum_propagate(source*np.exp(1j*target_phase),
                    grid,wavelength_m,settings.slm_to_disk_distance_m)
                desired_observed=(desired_disk if not settings.post_disk_distance_m else
                    angular_spectrum_propagate(desired_disk,grid,wavelength_m,
                                               settings.post_disk_distance_m))
                desired_fluence=abs(desired_observed)**2*result["output_energy_J"]
                target_clean=[]
                for plane_index,z in enumerate(config["planes_m"]):
                    desired_plane=(desired_observed if z==0 else
                        angular_spectrum_propagate(desired_observed,grid,wavelength_m,float(z)))
                    _, ideal, _=capture(abs(desired_plane)**2,x_axis,y_axis,
                        wavelength_m,camera_settings,camera_setup,
                        point_seed[2]+1000+plane_index,enabled=False)
                    target_clean.append(ideal)
                measured_loss=camera_shape_loss(np.stack(images),
                    np.stack(target_clean),camera_settings.black_level_adu)
                if point == 0:
                    trial_outcome="baseline_accepted"
                elif delayed and not np.array_equal(current_command,requested):
                    trial_outcome="delayed_measured"
                elif measured_loss < retained_loss-float(config.get("trial_improvement_tolerance",1e-5)):
                    trial_outcome="accepted"
                else:
                    trial_outcome="rejected"
                if trial_outcome in ("baseline_accepted", "accepted"):
                    retained_command=current_command.copy()
                    retained_loss=measured_loss
                slm_valid=np.exp(-2*(x*x+y*y)/settings.waist_m**2) >= 1e-3
                weights=abs(field)**2
                phase_valid=phase_support(field,desired_observed)
                unwanted,before_rms=shared_phase_residual(
                    field,desired_observed,weights,phase_valid)
                before_metrics=field_metrics(field,desired_observed,weights,phase_valid)
                def shape_overlap(candidate):
                    a=np.maximum(abs(candidate)**2,0)
                    b=np.maximum(desired_fluence,0)
                    return float(np.sum(np.sqrt(a*b))**2/
                        max(float(np.sum(a)*np.sum(b)),1e-30))
                before_shape=shape_overlap(field)
                phase_requirement=float(config.get("label_min_phase_improvement_rad",.01))
                shape_floor=float(config.get("label_min_shape_overlap",.95))
                shape_drop=float(config.get("label_max_shape_drop",.012))
                shape_requirement=max(shape_floor,before_shape-shape_drop)
                correction=np.full(grid.shape,np.nan)
                disk_phase=(np.pi*result["effective_signal_traversals"]/wavelength_m*
                    np.asarray(timeline["final_roundtrip_opd_m"],float))
                corrected_residual=np.full(grid.shape,np.nan)
                corrected_fluence=np.full(grid.shape,np.nan)
                corrected_command=np.full(grid.shape,np.nan)
                verified_field=None
                after_rms=None
                after_shape=None
                label_valid=False
                label_method=None
                label_rejection_reason="no candidate passed phase and shape criteria"
                candidate_checks=[]
                def phase_residual(candidate):
                    return shared_phase_residual(
                        candidate,desired_observed,weights,phase_valid)
                distance=settings.slm_to_disk_distance_m+settings.post_disk_distance_m
                back_current=angular_spectrum_propagate(field,grid,wavelength_m,-distance)
                desired_energy=np.sqrt(np.sum(abs(field)**2)/max(np.sum(abs(desired_observed)**2),1e-30))
                optical_targets=[
                    ("phase_preserving_amplitude",abs(field)*np.exp(1j*np.angle(desired_observed))),
                ]
                if before_shape<.95:
                    optical_targets.append(("full_complex_target",desired_energy*desired_observed))
                candidates=[]
                # Use privileged simulated crystal and external phase only to
                # propose a command. The full forward solve below determines
                # whether the first-order compensation actually works.
                physical_correction,_=_phase_target(result,desired_disk,grid,
                    wavelength_m,target_phase,external,settings)
                physical_command=absolute_oracle_command(target_phase,physical_correction)
                physical_delta=np.angle(np.exp(1j*(physical_command-current_command)))
                physical_candidates=[(True,0.,"first_order_crystal_oracle",gain,
                                      gain*physical_delta)
                                     for gain in (.08,-.08,.05,-.05,.12,-.12)]
                # The external screen is separately applied at the SLM plane.
                # Its exact inverse is useful when the disk phase is weak.
                nominal_oracle=collocated_external_oracle(target_phase,external)
                direct_delta=np.angle(np.exp(1j*(nominal_oracle-current_command)))
                direct_candidates=[(True,0.,"collocated_external_oracle",gain,
                                    gain*direct_delta) for gain in (1.,)]
                gains=tuple(float(g) for g in config.get("label_candidate_gains",(.05,.1,.2,.3,.4)))
                if not gains or any(not 0<g<=1 for g in gains):
                    raise ValueError("label_candidate_gains must be in (0,1]")
                max_checks=int(config.get("label_max_solver_evaluations",3))
                if max_checks<1:
                    raise ValueError("label_max_solver_evaluations must be positive")
                teacher_kind=config.get("label_teacher","legacy_candidates")
                if teacher_kind not in ("legacy_candidates","response_matrix","modal"):
                    raise ValueError("unknown label_teacher")
                modal_status=None
                passive_fidelity_limit=None
                best_improved_command=None
                best_improved_fidelity=None
                modal_names=[]
                modal_coefficients=None
                modal_reference_evaluations=0
                for method,optical_target in optical_targets:
                    back_target=angular_spectrum_propagate(optical_target,grid,wavelength_m,-distance)
                    delta=np.angle(back_target*np.conj(back_current))
                    source_weight=abs(back_current)**2*slm_valid
                    piston_delta=np.angle(np.sum(source_weight*np.exp(1j*delta)))
                    delta=np.where(slm_valid,np.angle(np.exp(1j*(delta-piston_delta))),0.)
                    for gain in gains:
                        step=gain*delta
                        preview=angular_spectrum_propagate(back_current*np.exp(1j*step),
                            grid,wavelength_m,distance)
                        _,preview_rms=phase_residual(preview)
                        preview_shape=shape_overlap(preview)
                        preview_score=(before_rms-preview_rms)-5*max(0,shape_requirement-preview_shape)
                        candidates.append((preview_shape>=shape_requirement,
                            preview_score,method,gain,step))
                candidates.sort(key=lambda item:(item[0],item[1]),reverse=True)
                selected=[]
                for candidate in physical_candidates+direct_candidates+candidates:
                    step=candidate[4]
                    if any(np.sqrt(np.sum(source_weight*np.angle(np.exp(1j*(step-other[4])))**2)/
                                   max(np.sum(source_weight),1e-30))<.02 for other in selected):
                        continue
                    selected.append(candidate)
                    if len(selected)>=max_checks:
                        break
                if teacher_kind=="response_matrix":
                    selected=[]
                    adaptive=(candidates[0][4]/candidates[0][3] if candidates else None)
                    modes=response_modes(x,y,settings.waist_m,adaptive=adaptive,
                        extra_modes=(("physical_oracle_direction",physical_delta),
                                     ("external_oracle_direction",direct_delta)),
                        max_modes=int(config.get("response_teacher_modes",10)))
                    fallback_steps=(("physical_oracle_positive_0.08",.08*physical_delta),
                                    ("physical_oracle_negative_0.08",-.08*physical_delta),
                                    ("external_oracle_positive_0.08",.08*direct_delta))
                    def response_solver(command):
                        corrected_slm,_=apply_slm(command,slm_setup,
                            drift_fraction=ranges["slm_drift_fraction_per_s"]*elapsed)
                        corrected=simulate_pulsed_seed(material,settings,group_target,
                            beam["seed_energy_nj"]*1e-9,nominal["seed_fwhm_ps"]*1e-12,
                            nominal["repetition_rate_kHz"]*1e3,
                            int(nominal["signal_traversals"]),
                            pump_passes=int(nominal["pump_passes"]),
                            compute_thermal=True,operation_duration_s=duration,
                            cooling_mode="fixed",thermal_optical_mode="lumped_phase",
                            architecture=architecture,regenerative_cavity=cavity,
                            dataset_physical=counterfactual_physical(corrected_slm))
                        valid=(corrected["thermal_feedback_applied"] and
                               corrected["thermal_timeline"]["requested_material_range_valid"])
                        return corrected["output_complex_field_sqrt_J_m"],valid
                    teacher=response_matrix_teacher(field,desired_observed,
                        current_command,modes,response_solver,
                        max_evaluations=max_checks,
                        probe_rad=float(config.get("response_probe_rad",.06)),
                        max_step_rad=float(config.get("response_max_step_rad",.45)),
                        ridge_fraction=float(config.get("response_ridge_fraction",.03)),
                        amplitude_weight=float(config.get("response_amplitude_weight",.15)),
                        min_phase_improvement_rad=phase_requirement,
                        min_shape_overlap=shape_floor,max_shape_drop=shape_drop,
                        fallback_steps=fallback_steps,adaptive_line_search=True,
                        pairwise_preview=bool(config.get("response_pairwise_preview",True)))
                    candidate_checks=teacher.checks
                    if teacher.step_rad is not None:
                        correction=np.angle(np.exp(1j*teacher.step_rad))
                        corrected_command=np.mod(current_command+teacher.step_rad,2*np.pi)
                        corrected_residual,after_rms=phase_residual(teacher.field)
                        corrected_fluence=abs(teacher.field)**2
                        verified_field=teacher.field
                        after_shape=teacher.shape_overlap
                        label_valid=True
                        label_method=f"response_matrix:{teacher.selected_candidate}"
                        label_rejection_reason=None
                elif teacher_kind=="modal":
                    selected=[]
                    try:
                        screen_phase=(result["effective_signal_traversals"]*
                            np.asarray(result["static_cold_phase_rad"],float)+disk_phase)
                        proposal_model=FrozenOpticalModel(
                            grid=grid,wavelength_m=wavelength_m,
                            slm_to_disk_m=settings.slm_to_disk_distance_m,
                            output_distance_m=settings.post_disk_distance_m,
                            input_field=result["input_complex_field_sqrt_J_m"],
                            current_command=current_command,
                            actual_slm_phase=actual_slm,
                            external_phase=external,screen_phase=screen_phase,
                            baseline_field=field,slm_setup=slm_setup,
                            drift_fraction=ranges["slm_drift_fraction_per_s"]*elapsed)
                        modal_names,basis,_=modal_basis(x,y,
                            result["input_complex_field_sqrt_J_m"],settings.waist_m,
                            radial_order=int(config.get("modal_radial_order",4)))
                        def modal_reference(command):
                            corrected_slm,_=apply_slm(command,slm_setup,
                                drift_fraction=ranges["slm_drift_fraction_per_s"]*elapsed)
                            solved=simulate_pulsed_seed(material,settings,group_target,
                                beam["seed_energy_nj"]*1e-9,
                                nominal["seed_fwhm_ps"]*1e-12,
                                nominal["repetition_rate_kHz"]*1e3,
                                int(nominal["signal_traversals"]),
                                pump_passes=int(nominal["pump_passes"]),
                                compute_thermal=True,operation_duration_s=duration,
                                cooling_mode="fixed",thermal_optical_mode="lumped_phase",
                                architecture=architecture,regenerative_cavity=cavity,
                                dataset_physical=counterfactual_physical(corrected_slm))
                            valid=(solved["thermal_feedback_applied"] and
                                   solved["thermal_timeline"]["requested_material_range_valid"])
                            return solved["output_complex_field_sqrt_J_m"],valid
                        modal_options=dict(
                            min_coherent_gain=float(config.get("modal_min_coherent_gain",.01)),
                            target_fidelity=float(config.get("modal_target_fidelity",.9)),
                            min_shape_overlap=shape_floor,max_shape_drop=shape_drop,
                            min_energy_fraction=float(config.get(
                                "modal_min_energy_fraction",.8)),
                            max_fast_iterations=int(config.get(
                                "modal_max_fast_iterations",60)))
                        residual_grid=int(config.get("modal_residual_grid",0))
                        feasible=(proposal_model.fidelity_limit(desired_observed) >=
                                  modal_options["target_fidelity"])
                        primary_budget=(max(1,max_checks//2)
                            if feasible and residual_grid and max_checks>1 else max_checks)
                        modal=modal_correction_teacher(field,desired_observed,
                            current_command,proposal_model,modal_reference,basis,
                            max_reference_evaluations=primary_budget,**modal_options)
                        modal_reference_evaluations=modal.evaluations
                        candidate_checks=[{"kind":"modal_stage","stage":"zernike"},
                                          *modal.checks]
                        if (modal.status!="task_success" and feasible and
                                residual_grid and modal.evaluations<max_checks):
                            residual_names,residual_basis,_=modal_basis(x,y,
                                result["input_complex_field_sqrt_J_m"],settings.waist_m,
                                radial_order=int(config.get("modal_radial_order",4)),
                                residual_grid=residual_grid)
                            refined=modal_correction_teacher(field,desired_observed,
                                current_command,proposal_model,modal_reference,residual_basis,
                                max_reference_evaluations=max_checks-modal.evaluations,
                                **modal_options)
                            modal_reference_evaluations+=refined.evaluations
                            candidate_checks.extend(({"kind":"modal_stage",
                                "stage":"spatial_residual"},*refined.checks))
                            if (refined.status=="task_success" or
                                    (modal.status!="task_success" and
                                     refined.best_improved_metrics is not None and
                                     (modal.best_improved_metrics is None or
                                      refined.best_improved_metrics.coherent_fidelity >
                                      modal.best_improved_metrics.coherent_fidelity))):
                                modal=refined
                                modal_names=residual_names
                        modal_status=modal.status
                        passive_fidelity_limit=modal.passive_fidelity_limit
                        best_improved_command=modal.best_improved_command_rad
                        best_improved_fidelity=(None if modal.best_improved_metrics is None
                            else modal.best_improved_metrics.coherent_fidelity)
                        modal_coefficients=modal.selected_coefficients_rad
                        if modal.step_rad is not None:
                            correction=modal.step_rad
                            corrected_command=modal.command_rad
                            corrected_residual,after_rms=phase_residual(modal.field)
                            corrected_fluence=abs(modal.field)**2
                            verified_field=modal.field
                            after_shape=modal.metrics.shape_overlap
                            label_valid=True
                            label_method=f"modal:{modal.selected_candidate}"
                            label_rejection_reason=None
                        else:
                            label_rejection_reason=modal.status
                    except Exception as exc:
                        modal_status="numerically_unresolved"
                        label_rejection_reason=(f"modal teacher failed: "
                            f"{type(exc).__name__}: {exc}")
                        candidate_checks.append({"kind":"modal_teacher_error",
                                                 "error":label_rejection_reason})
                for _,preview_score,method,gain,step in selected:
                    command=np.mod(current_command+step,2*np.pi)
                    try:
                        corrected_slm,_=apply_slm(command,slm_setup,
                            drift_fraction=ranges["slm_drift_fraction_per_s"]*elapsed)
                        corrected=simulate_pulsed_seed(material,settings,group_target,
                            beam["seed_energy_nj"]*1e-9,nominal["seed_fwhm_ps"]*1e-12,
                            nominal["repetition_rate_kHz"]*1e3,
                            int(nominal["signal_traversals"]),
                            pump_passes=int(nominal["pump_passes"]),
                            compute_thermal=True,operation_duration_s=duration,
                            cooling_mode="fixed",thermal_optical_mode="lumped_phase",
                            architecture=architecture,regenerative_cavity=cavity,
                            dataset_physical=counterfactual_physical(corrected_slm))
                        corrected_field=np.asarray(corrected["output_complex_field_sqrt_J_m"],complex)
                        residual,rms=phase_residual(corrected_field)
                        overlap=shape_overlap(corrected_field)
                        valid=(corrected["thermal_feedback_applied"] and
                               before_rms-rms>=phase_requirement and overlap>=shape_requirement)
                        candidate_checks.append({"method":method,"gain":gain,
                            "preview_score":preview_score,"phase_rms_rad":rms,
                            "shape_overlap":overlap,"passed":bool(valid)})
                        if valid:
                            correction=np.angle(np.exp(1j*(command-current_command)))
                            corrected_command=command
                            corrected_residual=residual
                            corrected_fluence=abs(corrected_field)**2
                            verified_field=corrected_field
                            after_rms=rms
                            after_shape=overlap
                            label_valid=True
                            label_method=method
                            label_rejection_reason=None
                            break
                    except Exception as exc:
                        candidate_checks.append({"method":method,"gain":gain,
                            "preview_score":preview_score,"passed":False,
                            "error":f"{type(exc).__name__}: {exc}"})
                arrays={
                    "input__camera_adu":np.stack(images),
                    "input__temperature_probes_K":sensor["measured_temperature_K"],
                    "input__temperature_probe_valid":np.isfinite(sensor["measured_temperature_K"]),
                    "input__disk_temperature_probes_K":sensor["measured_temperature_K"][:3],
                    "input__disk_temperature_probe_valid":np.isfinite(sensor["measured_temperature_K"][:3]),
                    "input__room_temperature_K":np.asarray(pump["coolant_temperature_K"]),
                    "input__pump_power_W":np.asarray(pump["pump_W"]),
                    "input__seed_energy_J":np.asarray(beam["seed_energy_nj"]*1e-9),
                    "input__pump_radius_m":np.asarray(pump["radius_mm"]*1e-3),
                    "input__seed_waist_m":np.asarray(beam["waist_mm"]*1e-3),
                    "input__disk_radius_m":np.asarray(settings.disk_radius_m),
                    "input__disk_thickness_m":np.asarray(settings.thickness_m),
                    "input__slm_to_disk_m":np.asarray(settings.slm_to_disk_distance_m),
                    "input__output_distance_m":np.asarray(settings.post_disk_distance_m),
                    "input__yb_nominal_at_percent":np.asarray(group_concentration),
                    "input__yb_relative_map":crystal["yb_concentration_scale"].astype(np.float32),
                    "input__target_phase_mask_rad":target_phase.astype(np.float32),
                    "input__incoming_beam_shape":(np.asarray(result["input_fluence_J_m2"])/
                        max(float(np.sum(result["input_fluence_J_m2"])),1e-30)).astype(np.float32),
                    "input__slm_command_rad":current_command.astype(np.float32),
                    "input__requested_slm_command_rad":requested.astype(np.float32),
                    "input__previous_slm_command_rad":(np.full(grid.shape,np.nan,np.float32)
                        if prior_command is None else prior_command.astype(np.float32)),
                    "input__previous_slm_valid":np.asarray(prior_command is not None),
                    "truth__complex_field_sqrt_J_m":field.astype(np.complex64),
                    "truth__input_complex_field_sqrt_J_m":np.asarray(
                        result["input_complex_field_sqrt_J_m"],np.complex64),
                    "truth__disk_input_complex_field_sqrt_J_m":np.asarray(
                        result["disk_input_complex_field_sqrt_J_m"],np.complex64),
                    "truth__input_fluence_J_m2":np.asarray(result["input_fluence_J_m2"],np.float32),
                    "truth__disk_input_fluence_J_m2":np.asarray(result["disk_input_fluence_J_m2"],np.float32),
                    "truth__output_fluence_J_m2":np.asarray(result["output_fluence_J_m2"],np.float32),
                    "truth__amplitude_sqrt_J_m":abs(field).astype(np.float32),
                    "truth__phase_rad":np.where(phase_valid,np.angle(field),0.).astype(np.float32),
                    "truth__phase_valid_mask":phase_valid,
                    "truth__slm_valid_mask":slm_valid,
                    "truth__unwanted_phase_rad":unwanted.astype(np.float32),
                    "truth__desired_structured_phase_rad":target_phase.astype(np.float32),
                    "truth__desired_output_phase_rad":np.angle(desired_observed).astype(np.float32),
                    "truth__verified_slm_correction_rad":(correction if label_valid else
                        np.full(grid.shape,np.nan)).astype(np.float32),
                    "truth__verified_slm_command_rad":corrected_command.astype(np.float32),
                    "truth__best_improved_slm_command_rad":(
                        np.full(grid.shape,np.nan) if best_improved_command is None else
                        best_improved_command).astype(np.float32),
                    "truth__modal_coefficients_rad":(
                        np.full(len(modal_names),np.nan) if modal_coefficients is None else
                        modal_coefficients).astype(np.float32),
                    "truth__correction_label_valid":np.asarray(label_valid),
                    "truth__corrected_residual_phase_rad":corrected_residual.astype(np.float32),
                    "truth__corrected_fluence_J_m2":corrected_fluence.astype(np.float32),
                    "truth__slm_actual_phase_rad":actual_slm.astype(np.float32),
                    "truth__external_optics_phase_rad":external.astype(np.float32),
                    "truth__thermal_phase_rad":disk_phase.astype(np.float32),
                    "truth__static_disk_phase_rad":np.asarray(result["static_cold_phase_rad"],np.float32),
                    "truth__clean_camera_fluence_J_m2":np.stack(clean_images),
                    "truth__temperature_K":np.asarray(timeline["requested_disk_temperature_K"],np.float32),
                    "truth__plate_temperature_K":np.asarray(timeline["requested_plate_temperature_K"],np.float32),
                    "truth__temperature_surface_K":temperature_xy.astype(np.float32),
                    "truth__disk_displacement_m":np.asarray(timeline["final_disk_displacement_m"],np.float32),
                    "truth__plate_displacement_m":np.asarray(timeline["final_plate_displacement_m"],np.float32),
                    "truth__front_displacement_m":np.asarray(timeline["final_front_displacement_nm"],np.float32)*1e-9,
                    "truth__rear_displacement_m":np.asarray(timeline["final_rear_displacement_nm"],np.float32)*1e-9,
                    "truth__yb_scale":crystal["yb_concentration_scale"].astype(np.float32),
                    "truth__yb_density_m3":np.asarray(result["yb_density_m3"],np.float32),
                    "truth__yb_at_percent_map":(group_concentration*np.asarray(
                        result["yb_density_m3"])/material.number_density_m3).astype(np.float32),
                    "truth__thickness_scale":crystal["thickness_scale"].astype(np.float32),
                    "truth__background_absorption_m1":crystal["background_absorption_m1"].astype(np.float32),
                    "truth__surface_figure_m":crystal["surface_figure_m"].astype(np.float32),
                    "truth__contact_scale":contact.astype(np.float32),
                    "truth__probe_temperature_K":sensor["true_temperature_K"].astype(np.float32),
                    "truth__probe_position_m":sensor["position_m"].astype(np.float32),
                    "truth__desired_fluence_J_m2":desired_fluence.astype(np.float32),
                    "metadata__disk_radius_m":np.asarray(settings.disk_radius_m),
                    "metadata__field_size_m":np.asarray(settings.field_size_m),
                    "metadata__camera_planes_m":np.asarray(config["planes_m"],float),
                }
                stem=folder/f"point_{point:03d}"
                measured_file=folder/f"point_{point:03d}_measurements.npz"
                truth_file=folder/f"point_{point:03d}_truth.npz"
                np.savez_compressed(measured_file,**{k:v for k,v in arrays.items()
                    if k.startswith("input__")})
                np.savez_compressed(truth_file,**{k:v for k,v in arrays.items()
                    if k.startswith("truth__") or k.startswith("metadata__")})
                metadata={"split":split,"crystal_id":group_id,"setup_id":group_id,
                    "sequence_id":group_id,"point_index":point,"sample_seed":point_seed,
                    "trial_index":point,"trial_outcome":trial_outcome,
                    "trial_policy":"measured_camera_coordinate_search",
                    "trial_improvement_tolerance":float(config.get("trial_improvement_tolerance",1e-5)),
                    "measured_camera_shape_loss":measured_loss,
                    "retained_camera_shape_loss":retained_loss,
                    "selected_target":group_target,
                    "sequence_time_s":elapsed+duration,
                    "thermal_interval_s":duration,
                    "thermal_initial_state_kind":timeline["initial_state_kind"],
                    "setup_seed":group_seed,"setup_npz_sha256":_sha(setup_file),
                    "physical_parameters":{"pump":pump,"beam":beam,
                        "yb_at_percent":group_concentration,
                        "contact_scale_shape":contact.shape,
                        "external_optics":external_parameters},
                    "slm_command_delayed":delayed,
                    "camera_settings":asdict(camera_settings),"camera_diagnostics":camera_diagnostics,
                    "phase_diversity_relative_L1":diversity_L1,
                    "sensor_parameters":{"bias_K":sensor["bias_K"],
                        "response_time_s":sensor["response_time_s"]},
                    "room_temperature_assumption":"Illustrative room temperature equals fixed heat-sink boundary temperature; no independent ambient coupling is solved.",
                    "source_config_sha256":manifest["source_config_sha256"],
                    "source_generator_sha256":current_generator_sha,
                    "numerical_source_sha256":current_numerical_sha,
                    "measurements_file":measured_file.name,
                    "measurements_sha256":_sha(measured_file),
                    "truth_file":truth_file.name,
                    "truth_sha256":_sha(truth_file),
                    "wavefront_rms_rad":{"uncorrected":before_rms,
                                           "corrected":after_rms},
                    "shape_overlap":{"uncorrected":before_shape,
                                     "corrected":after_shape},
                    "coherent_fidelity":{"uncorrected":before_metrics.coherent_fidelity,
                        "corrected":(None if not label_valid else field_metrics(
                            verified_field,
                            desired_observed,weights,phase_valid).coherent_fidelity)},
                    "modal_teacher_status":modal_status,
                    "passive_phase_only_fidelity_limit":passive_fidelity_limit,
                    "best_improved_coherent_fidelity":best_improved_fidelity,
                    "modal_basis_names":modal_names,
                    "modal_reference_evaluations":modal_reference_evaluations,
                    "corrected_energy_fraction":(None if not label_valid else
                        float(np.sum(corrected_fluence)/max(np.sum(abs(field)**2),1e-30))),
                    "correction_label_valid":label_valid,
                    "correction_label_method":label_method,
                    "correction_label_rejection_reason":label_rejection_reason,
                    "correction_label_thresholds":{"phase_rad":phase_requirement,
                                                   "shape_overlap":shape_requirement,
                                                   "max_shape_drop":shape_drop,
                                                   "min_shape_overlap":shape_floor,
                                                   "objective":("coherent_fidelity" if
                                                       teacher_kind=="modal" else "phase_rms"),
                                                   "min_coherent_gain":float(config.get(
                                                       "modal_min_coherent_gain",.01)),
                                                   "target_fidelity":float(config.get(
                                                       "modal_target_fidelity",.9)),
                                                   "min_energy_fraction":float(config.get(
                                                       "modal_min_energy_fraction",.8))},
                    "correction_candidate_checks":candidate_checks,
                    "correction_target_scope":(
                        "Best solver-verified local response-matrix update; probes use privileged simulated complex fields, not measured wavefronts. Bounded local fit, not a global optimum."
                        if teacher_kind=="response_matrix" else
                        "Modal teacher: passive optical proposal, continuous Zernike coefficients, exact SLM rendering and independent full-solver checks. Passive reachability is conditional; task success requires absolute coherent fidelity."
                        if teacher_kind=="modal" else
                        "Best passing candidate among ranked phase-only SLM updates; every attempted candidate is rerun with the physical solver. This is a bounded search, not a global optimum."),
                    "thermal_validity":result["thermal"].get("validity"),
                    "scope":result["dataset_physical_scope"],"dataset_ready":False}
                stem.with_suffix(".json").write_text(json.dumps(
                    _serializable(metadata),indent=2,allow_nan=False),encoding="utf-8")
                _plot_validation(stem.with_suffix(".png"),arrays,sensor)
                manifest["splits"][split].append(str(stem.relative_to(output).with_suffix(".json")))
                (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
                previous_thermal_state=next_thermal_state
    return output/"manifest.json"
