"""Physically rerun Yb:YAG before making phase-diverse synthetic observations.

The generator intentionally rejects out-of-range thermal states. It does not
turn the current room-temperature Yb:YAG model into validated hot-gain truth.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from hoyag.propagation import Grid2D, angular_spectrum_propagate
from hoyag.thermal import DiskThermalMesh
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
        ("SLM correction (rad)",np.where(arrays["truth__slm_valid_mask"],
            arrays["truth__ideal_slm_correction_rad"],np.nan),"twilight"),
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


def generate(config_path, output_dir, *, split_counts=None, points_per_setup=1,
             max_examples=8):
    """Create disjoint setup-level splits; call only inside a bounded worker."""
    config_path = Path(config_path)
    output = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    nominal, ranges = config["nominal"], config["ranges"]
    if (config["camera"]["pulses_per_exposure"] >
            nominal["repetition_rate_kHz"]*1e3*config["camera"]["exposure_s"]+1e-9):
        raise ValueError("camera exposure is too short for the pulse count and repetition rate")
    counts = config["split_counts"] if split_counts is None else split_counts
    if (not 1 <= points_per_setup <= 4 or
            sum(counts.values())*points_per_setup > max_examples or
            set(counts) != {"train","validation","test","stress"} or
            any(not isinstance(v,int) or v < 0 for v in counts.values())):
        raise ValueError("bounded group/split counts required")
    if not all(counts[k] >= 1 for k in ("train","validation","test")):
        raise ValueError("train, validation and test each need an unseen setup")
    output.mkdir(parents=True,exist_ok=True)
    manifest = {"schema":"ybyag_nn_grouped_v1", "dataset_ready":False,
        "source_config_sha256":_sha(config_path),
        "source_generator_sha256":_sha(__file__),
        "numerical_source_sha256":_numerical_sources_sha256(),
        "created_unix_s":time.time(), "splits":{k:[] for k in counts},
        "limits":["Yb:YAG hot gain unavailable; all accepted states are room-temperature lumped-phase approximations.",
                  "Thickness uses an optical-column and static geometric-phase approximation; the FEM retains nominal thickness.",
                  "Background absorption uses first-order pre-pump attenuation and distributed heat.",
                  "Photoelasticity and stress tensor are not solved in this scalar Yb amplifier.",
                  "The phase-only oracle is a first-order back-propagated correction, not a guaranteed exact optimum.",
                  "Instrument distributions are illustrative until calibrated."]}
    group_number=0
    for split in ("train","validation","test","stress"):
        for _ in range(counts[split]):
            group_seed=child_seeds(int(config["seed"])+group_number,9)
            group_id=f"{split}_crystal_{group_number:04d}"
            concentration_candidates=nominal.get("yb_at_percent_candidates",
                [nominal["yb_at_percent"]])
            if not concentration_candidates:
                raise ValueError("at least one Yb concentration candidate is required")
            group_concentration=float(concentration_candidates[group_number % len(concentration_candidates)])
            group_number+=1
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
                camera_prnu=camera_setup["prnu"],
                camera_dsnu_e=camera_setup["dsnu_e"],
                camera_hot=camera_setup["hot"],camera_dead=camera_setup["dead"])
            sensor_session=None
            previous_command=None
            previous_oracle_correction=None
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
                    nominal["selected_beam"],wavelength_m,settings.slm_to_disk_distance_m)
                requested=np.mod(target_phase+(0 if previous_oracle_correction is None
                    else previous_oracle_correction),2*np.pi)
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
                result=simulate_pulsed_seed(material,settings,nominal["selected_beam"],
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
                                             camera_temp_delta)
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
                correction,disk_phase=_phase_target(result,desired_disk,grid,wavelength_m,
                    target_phase,external,settings)
                slm_valid=np.exp(-2*(x*x+y*y)/settings.waist_m**2) >= 1e-3
                correction=np.where(slm_valid,correction,0.)
                previous_oracle_correction=correction.copy()
                # The oracle returns an absolute correction relative to the
                # intentional target. `requested` may already contain the
                # previous point's oracle, so adding it here double-counts.
                corrected_command=absolute_oracle_command(target_phase,correction)
                corrected_slm,_=apply_slm(corrected_command,slm_setup,
                    drift_fraction=ranges["slm_drift_fraction_per_s"]*elapsed)
                corrected=simulate_pulsed_seed(material,settings,nominal["selected_beam"],
                    beam["seed_energy_nj"]*1e-9,nominal["seed_fwhm_ps"]*1e-12,
                    nominal["repetition_rate_kHz"]*1e3,
                    int(nominal["signal_traversals"]),
                    pump_passes=int(nominal["pump_passes"]),
                    compute_thermal=True,
                    operation_duration_s=duration,
                    cooling_mode="fixed",thermal_optical_mode="lumped_phase",
                    architecture=architecture,regenerative_cavity=cavity,
                    dataset_physical={**physical,"slm_actual_phase_rad":corrected_slm})
                if not corrected["thermal_feedback_applied"]:
                    raise RuntimeError(f"{group_id}: corrected state outside supported thermal range")
                corrected_field=np.asarray(corrected["output_complex_field_sqrt_J_m"],complex)
                corrected_residual=np.angle(np.exp(1j*(np.angle(corrected_field)-
                    np.angle(desired_observed))))
                corrected_piston=np.angle(np.sum(abs(corrected_field)**2*
                    np.exp(1j*corrected_residual)))
                corrected_residual=np.angle(np.exp(1j*(corrected_residual-corrected_piston)))
                unwanted=np.angle(np.exp(1j*(np.angle(field)-np.angle(desired_observed))))
                weights=abs(field)**2
                phase_valid=weights >= .01*float(weights.max())
                piston=np.angle(np.sum(weights*np.exp(1j*unwanted)))
                unwanted=np.angle(np.exp(1j*(unwanted-piston)))
                unwanted=np.where(phase_valid,unwanted,0.)
                corrected_residual=np.where(phase_valid,corrected_residual,0.)
                before_rms=float(np.sqrt(np.average(unwanted[phase_valid]**2,
                    weights=weights[phase_valid])))
                after_rms=float(np.sqrt(np.average(corrected_residual[phase_valid]**2,
                    weights=weights[phase_valid])))
                arrays={
                    "input__camera_adu":np.stack(images),
                    "input__temperature_probes_K":sensor["measured_temperature_K"],
                    "input__temperature_probe_valid":np.isfinite(sensor["measured_temperature_K"]),
                    "input__pump_power_W":np.asarray(pump["pump_W"]),
                    "input__seed_energy_J":np.asarray(beam["seed_energy_nj"]*1e-9),
                    "input__pump_radius_m":np.asarray(pump["radius_mm"]*1e-3),
                    "input__seed_waist_m":np.asarray(beam["waist_mm"]*1e-3),
                    "input__slm_command_rad":current_command.astype(np.float32),
                    "input__requested_slm_command_rad":requested.astype(np.float32),
                    "input__previous_slm_command_rad":(np.full(grid.shape,np.nan,np.float32)
                        if prior_command is None else prior_command.astype(np.float32)),
                    "input__previous_slm_valid":np.asarray(prior_command is not None),
                    "truth__complex_field_sqrt_J_m":field.astype(np.complex64),
                    "truth__amplitude_sqrt_J_m":abs(field).astype(np.float32),
                    "truth__phase_rad":np.where(phase_valid,np.angle(field),0.).astype(np.float32),
                    "truth__phase_valid_mask":phase_valid,
                    "truth__slm_valid_mask":slm_valid,
                    "truth__unwanted_phase_rad":unwanted.astype(np.float32),
                    "truth__desired_structured_phase_rad":target_phase.astype(np.float32),
                    "truth__desired_output_phase_rad":np.angle(desired_observed).astype(np.float32),
                    "truth__ideal_slm_correction_rad":correction.astype(np.float32),
                    "truth__corrected_residual_phase_rad":corrected_residual.astype(np.float32),
                    "truth__corrected_fluence_J_m2":abs(corrected_field).astype(np.float32)**2,
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
                np.savez_compressed(stem.with_suffix(".npz"),**arrays)
                metadata={"split":split,"crystal_id":group_id,"setup_id":group_id,
                    "sequence_id":group_id,"point_index":point,"sample_seed":point_seed,
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
                    "source_config_sha256":manifest["source_config_sha256"],
                    "source_generator_sha256":manifest["source_generator_sha256"],
                    "numerical_source_sha256":manifest["numerical_source_sha256"],
                    "npz_sha256":_sha(stem.with_suffix(".npz")),
                    "wavefront_rms_rad":{"uncorrected":before_rms,
                                           "corrected":after_rms},
                    "correction_target_scope":"First-order phase-only inverse with frozen thermal screen; checked by a fresh physical solve, not a guaranteed global optimum.",
                    "thermal_validity":result["thermal"].get("validity"),
                    "scope":result["dataset_physical_scope"],"dataset_ready":False}
                stem.with_suffix(".json").write_text(json.dumps(
                    _serializable(metadata),indent=2,allow_nan=False),encoding="utf-8")
                _plot_validation(stem.with_suffix(".png"),arrays,sensor)
                manifest["splits"][split].append(str(stem.relative_to(output).with_suffix(".npz")))
                (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
                previous_thermal_state=next_thermal_state
    return output/"manifest.json"
