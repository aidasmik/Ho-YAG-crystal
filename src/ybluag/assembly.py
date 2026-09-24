"""Yb:LuAG disk/plate thermoelastic assembly without invented photoelasticity."""

from dataclasses import dataclass

import numpy as np

from hoyag.cooling_plate import (DiskPlateHeatSolver, ThermalMaterial,
                                 cooling_plate_mesh, sample_temperature)
from hoyag.stress_optics import HotDiskScreens, geometric_roundtrip_opd
from hoyag.thermal import DiskThermalMesh, ThermalBoundary
from hoyag.thermomechanics import (BondedInterface, DiskPlateMesh,
                                   ElasticMaterial, solve_disk_plate)


@dataclass(frozen=True)
class YbAssemblyResult:
    temperature: object
    displacement: object
    screens: HotDiskScreens
    material_range_valid: bool = True
    scope: str = ("steady linear thermal and isotropic elastic assembly; "
                  "host thermo-optic proxy; LuAG photoelasticity omitted")


def scalar_yb_screens(fem, displacement, mesh: DiskThermalMesh,
                      temperature_K, xy_query, *, index, wavelength_m,
                      dn_dT_K1, reference_temperature_K):
    """Map temperature and actual FEM surfaces to a reciprocal scalar screen."""
    for name, value in (("index", index), ("wavelength", wavelength_m),
                        ("reference temperature", reference_temperature_K)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(dn_dT_K1):
        raise ValueError("dn/dT must be finite")
    xy = np.asarray(xy_query, dtype=float)
    if xy.shape[-1] != 2 or np.any(~np.isfinite(xy)):
        raise ValueError("xy query must be finite with last axis 2")
    shape = xy.shape[:-1]
    flat = xy.reshape(-1, 2)
    inside = np.linalg.norm(flat, axis=1) <= fem.radius_m
    selected = flat[inside]
    identity = np.broadcast_to(np.eye(2, dtype=complex), (len(flat), 2, 2)).copy()
    inward = identity.copy()
    outward = identity.copy()
    geometry = np.zeros(len(flat))
    thermal = np.zeros(len(flat))
    front = np.zeros(len(flat))
    rear = np.zeros(len(flat))
    if len(selected):
        dz = np.diff(mesh.z_edges_m)
        points = np.concatenate([np.column_stack((selected, np.full(len(selected), z)))
                                 for z in mesh.z_m])
        temp = sample_temperature(mesh, temperature_K, points).reshape(mesh.nz, -1)
        thermal_opd = np.sum(dn_dT_K1 * (temp - reference_temperature_K)
                             * dz[:, None], axis=0)
        phase = np.exp(2j * np.pi * thermal_opd / wavelength_m)
        inward[inside] *= phase[:, None, None]
        outward[inside] *= phase[:, None, None]
        surfaces = fem.disk.interpolate_layers(displacement.disk_u_m, selected,
                                               [0, fem.disk_thickness_m])
        front[inside] = surfaces[0, :, 2]
        rear[inside] = surfaces[1, :, 2]
        geometry[inside] = geometric_roundtrip_opd(front[inside], rear[inside],
                                                     index=index)
        thermal[inside] = thermal_opd
    zeros = np.zeros(len(flat))
    return HotDiskScreens(inward.reshape(shape + (2, 2)),
                          outward.reshape(shape + (2, 2)),
                          geometry.reshape(shape), thermal.reshape(shape),
                          zeros.reshape(shape), zeros.reshape(shape),
                          front.reshape(shape), rear.reshape(shape), wavelength_m)


def solve_yb_cooler_temperature(mesh: DiskThermalMesh, heat_W_m3, configuration):
    """Linear constant-property copper-cooler screen, before validity check."""
    return yb_cooler_solver(mesh, configuration).steady(heat_W_m3)


def yb_cooler_solver(mesh: DiskThermalMesh, configuration):
    """Build the same disk, bond, and finite-copper thermal operator for steady or transient use."""
    cfg = configuration
    geometry = cfg["geometry"]
    thermal_cfg = cfg["thermal"]
    numerics = cfg["numerics"]
    if (not np.isclose(mesh.r_edges_m[-1], geometry["disk_radius_m"], atol=1e-12, rtol=0)
            or not np.isclose(mesh.z_edges_m[-1], geometry["disk_thickness_m"], atol=1e-12, rtol=0)):
        raise ValueError("Yb mesh and assembly geometry disagree")
    plate = cooling_plate_mesh(mesh, radius_m=geometry["plate_radius_m"],
                               thickness_m=geometry["plate_thickness_m"],
                               nz=numerics["plate_thermal_nz"])
    solver = DiskPlateHeatSolver(
        mesh, plate,
        contact_conductance_W_m2K=thermal_cfg["interface_conductance_W_m2K"],
        coolant=ThermalBoundary(thermal_cfg["coolant_temperature_K"],
                                thermal_cfg["coolant_conductance_W_m2K"]),
        disk_material=ThermalMaterial(**thermal_cfg["disk"]),
        plate_material=ThermalMaterial(**thermal_cfg["plate"]))
    return solver


def solve_yb_assembly(mesh: DiskThermalMesh, heat_W_m3, grid, configuration, *,
                      temperature=None, allow_extrapolation=False):
    """Solve assembly; explicit extrapolation is reserved for labeled previews."""
    cfg = configuration
    geometry = cfg["geometry"]
    mechanical = cfg["mechanical"]
    optics = cfg["optics"]
    numerics = cfg["numerics"]
    plate = cooling_plate_mesh(mesh, radius_m=geometry["plate_radius_m"],
                               thickness_m=geometry["plate_thickness_m"],
                               nz=numerics["plate_thermal_nz"])
    if temperature is None:
        temperature = solve_yb_cooler_temperature(mesh, heat_W_m3, cfg)
    material_range_valid = bool(np.min(temperature.disk_temperature_K) >= 293.15 and
                                np.max(temperature.disk_temperature_K) <= 300.0)
    if not material_range_valid and not allow_extrapolation:
        raise ValueError("Yb:LuAG disk outside 293.15–300 K assembly-material range")
    fem = DiskPlateMesh.make(
        radius_m=geometry["disk_radius_m"],
        disk_thickness_m=geometry["disk_thickness_m"],
        plate_radius_m=geometry["plate_radius_m"],
        plate_thickness_m=geometry["plate_thickness_m"],
        **numerics["mechanical"])
    disk_temp = sample_temperature(mesh, temperature.disk_temperature_K,
                                   fem.disk.centers_m)
    plate_temp = sample_temperature(plate, temperature.plate_temperature_K,
                                    fem.plate.centers_m,
                                    z_offset_m=geometry["disk_thickness_m"])
    disk_material = ElasticMaterial(**mechanical["disk"])
    plate_material = ElasticMaterial(**mechanical["plate"])
    displacement = solve_disk_plate(
        fem, disk_temp, plate_temp, disk_material=disk_material,
        plate_material=plate_material,
        interface=BondedInterface(**mechanical["bond"]),
        support=mechanical["plate_support"],
        front_pressure_Pa=mechanical["front_pressure_Pa"])
    x, y = grid.mesh
    screens = scalar_yb_screens(
        fem, displacement, mesh, temperature.disk_temperature_K,
        np.stack((x, y), axis=-1), index=optics["index"],
        wavelength_m=optics["wavelength_m"],
        dn_dT_K1=optics["dn_dT_K1"],
        reference_temperature_K=optics["reference_temperature_K"])
    return YbAssemblyResult(temperature, displacement, screens,
                            material_range_valid=material_range_valid,
                            scope=("steady or transient linear assembly with constant material properties; "
                                   "outside 293.15–300 K is an unvalidated extrapolation"
                                   if not material_range_valid else
                                   "linear thermal and isotropic elastic assembly within material range"))
