"""Small deterministic coupled Yb:LuAG smoke case; not a convergence study."""

from ybluag.gallery import YbGallerySettings, simulate_pulsed_seed
from ybluag.model import YbLuAGMaterial
from ybluag.regenerative import RegenerativeCavity


def main():
    settings = YbGallerySettings(
        pump_power_W=.01, pump_radius_m=1e-3, thickness_m=100e-6,
        assembly_property_model="proposal_12at", grid_n=32,
        field_size_m=12e-3, z_steps=1, thermal_nr=4, thermal_nphi=4,
        thermal_nz=1, cluster_contrast=0, waist_m=.6e-3,
        post_disk_distance_m=0)
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.973e-3,
                              pump_wavelength_nm=938)
    result = simulate_pulsed_seed(
        material, settings, "Gaussian TEM00", 1e-9, 10e-12, 1e4, 2,
        pump_passes=2, operation_duration_s=.1,
        architecture="regenerative",
        regenerative_cavity=RegenerativeCavity(round_trips=1),
        thermal_optical_mode="coupled_steady",
        thermal_internal_max_step_s=.01)
    timeline = result["thermal_timeline"]
    print({"status": result["coupled_steady_convergence"]["status"],
           "iterations": result["coupled_steady_convergence"]["iterations"],
           "output_energy_J": result["output_energy_J"],
           "peak_C": result["steady_coupled_temperature_max_C"],
           "last": result["coupled_steady_convergence"]["history"][-1],
           "thermal_internal_steps": timeline["internal_steps"],
           "probe_count": len(timeline["temperature_probes"]["specifications"]),
           "probe_samples": len(timeline["temperature_probes"]["samples"])})


if __name__ == "__main__":
    main()
