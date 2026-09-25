"""Four controlled Yb:YAG solver perturbations, for bounded regression use."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ybyag.model import YbYAGMaterial
from ybyag_dataset.generator import _settings
from ybluag.gallery import simulate_pulsed_seed
from hoyag.propagation import Grid2D


def main():
    cfg_path=ROOT/"config"/"ybyag_nn_dataset.json"
    cfg=json.loads(cfg_path.read_text(encoding="utf-8"))
    nominal=dict(cfg["nominal"])
    nominal.update(grid_n=32,thermal_nr=6,thermal_nphi=8,thermal_nz=3)
    material=YbYAGMaterial(yb_at_percent=nominal["yb_at_percent"])
    beam=dict(waist_mm=nominal["waist_mm"],seed_energy_nj=nominal["seed_energy_nj"])
    rows={}
    for name,pump_factor,contact_factor,yb_factor in (
            ("baseline",1.,1.,1.),("pump_up",1.2,1.,1.),
            ("contact_down",1.,.8,1.),("yb_up",1.,1.,1.05)):
        pump=dict(pump_W=nominal["pump_W"]*pump_factor,
                  radius_mm=nominal["radius_mm"])
        settings=_settings(nominal,pump,beam)
        physical=dict(contact_scale_polar=np.full((settings.thermal_nr,
            settings.thermal_nphi),contact_factor),
            yb_concentration_scale=np.full((settings.grid_n,settings.grid_n),yb_factor),
            coolant_temperature_K=nominal["coolant_temperature_C"]+273.15)
        result=simulate_pulsed_seed(material,settings,"Gaussian TEM00",10e-9,
            10e-12,1e4,2,pump_passes=10,operation_duration_s=30.,
            cooling_mode="fixed",thermal_optical_mode="lumped_phase",
            dataset_physical=physical)
        timeline=result["thermal_timeline"]
        rows[name]={
            "heat_W":float(result["cycle_average_heat_W_upper_or_assumed"]),
            "pump_absorbed_W":float(result["cycle_average_pump_absorbed_W"]),
            "disk_temperature_max_K":float(np.max(timeline["requested_disk_temperature_K"])),
            "deformation_pv_m":float(np.ptp(timeline["final_front_displacement_nm"])*1e-9),
            "opd_pv_m":float(np.ptp(timeline["final_roundtrip_opd_m"])),
            "valid":bool(timeline["requested_material_range_valid"]),
        }
    data={"schema":"controlled_ybyag_nn_coupling_v1",
          "source_config_sha256":hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
          "rows":rows}
    base=rows["baseline"]
    assert all(row["valid"] for row in rows.values())
    assert rows["pump_up"]["heat_W"] > base["heat_W"]
    assert rows["pump_up"]["disk_temperature_max_K"] > base["disk_temperature_max_K"]
    assert rows["contact_down"]["disk_temperature_max_K"] > base["disk_temperature_max_K"]
    assert rows["yb_up"]["pump_absorbed_W"] > base["pump_absorbed_W"]
    assert rows["yb_up"]["heat_W"] > base["heat_W"]
    assert rows["yb_up"]["disk_temperature_max_K"] > base["disk_temperature_max_K"]
    assert any(abs(rows[k]["deformation_pv_m"]-base["deformation_pv_m"])>1e-12
               for k in ("pump_up","contact_down","yb_up"))
    # The CT fit covers 5 at.% locally; the 15 at.% edge uses the explicit
    # relative HT concentration slope without freezing the upper half-map.
    dopings={}
    for concentration in (5.,15.):
        grid=Grid2D.square(settings.grid_n,settings.field_size_m)
        xx,_=grid.mesh
        yb_map=1+.02*xx/settings.disk_radius_m
        result=simulate_pulsed_seed(YbYAGMaterial(yb_at_percent=concentration),
            settings,"Gaussian TEM00",10e-9,10e-12,1e4,2,pump_passes=10,
            operation_duration_s=0.,cooling_mode="fixed",
            thermal_optical_mode="lumped_phase",
            dataset_physical={"yb_concentration_scale":yb_map,
                "coolant_temperature_K":nominal["coolant_temperature_C"]+273.15})
        dopings[str(int(concentration))]={"thermal_status":result["thermal"]["status"],
            "density_span_m3":float(np.ptp(result["yb_density_m3"]))}
        assert dopings[str(int(concentration))]["thermal_status"]=="computed"
        assert dopings[str(int(concentration))]["density_span_m3"]>0
    data["dopings_at_percent"]=dopings
    path=ROOT/"results"/"ybyag_nn_validation"/"controlled_coupling.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding="utf-8")
    print(json.dumps(data,indent=2))


if __name__=="__main__":
    main()
