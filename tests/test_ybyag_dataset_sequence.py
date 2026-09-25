"""A changed pump starts from the actual previous thermal state."""
import numpy as np

from ybyag.model import YbYAGMaterial
from ybluag.gallery import YbGallerySettings, simulate_pulsed_seed


def test_sequential_thermal_initial_condition():
    material=YbYAGMaterial(yb_at_percent=10)
    base=YbGallerySettings(pump_power_W=.01,pump_radius_m=.001,
        thickness_m=100e-6,disk_radius_m=.005,
        assembly_property_model="yag_rt_proxy",grid_n=32,field_size_m=.012,
        thermal_nr=4,thermal_nphi=4,thermal_nz=2,cluster_contrast=0,
        post_disk_distance_m=0)
    first=simulate_pulsed_seed(material,base,"Gaussian TEM00",10e-9,
        10e-12,1e4,2,operation_duration_s=.5,cooling_mode="fixed",
        thermal_optical_mode="lumped_phase",
        dataset_physical={"coolant_temperature_K":293.65})
    old=first["thermal_timeline"]
    hotter=YbGallerySettings(**{**vars(base),"pump_power_W":.012})
    second=simulate_pulsed_seed(material,hotter,"Gaussian TEM00",10e-9,
        10e-12,1e4,2,operation_duration_s=.1,cooling_mode="fixed",
        thermal_optical_mode="lumped_phase",
        dataset_physical={"coolant_temperature_K":293.65,
            "initial_disk_temperature_K":old["requested_disk_temperature_K"],
            "initial_plate_temperature_K":old["requested_plate_temperature_K"]})
    new=second["thermal_timeline"]
    assert new["initial_state_kind"]=="previous_solved_state"
    assert np.isclose(new["disk_max_C"][0],old["requested_disk_max_C"])
    assert np.max(new["requested_disk_temperature_K"]) > np.max(old["requested_disk_temperature_K"])
