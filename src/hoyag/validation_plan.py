"""Stage 7V campaign definition. Planning never executes or marks a case passed."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .validation_metrics import ContinuousDensity, stable_hash

SCHEMA = "hoyag.stage7v/1"


@dataclass(frozen=True)
class Acceptance:
    power_relative: float = 0.01
    power_absolute_floor_W: float = 1e-8
    temperature_K: float = 0.05
    opd_difference_rms_m: float = 2e-9
    field_overlap: float = 0.999
    minimum_comparison_power_fraction: float = 0.999
    maximum_comparison_power_fraction: float = 1.005
    energy_balance_relative: float = 1e-4
    mechanical_residual: float = 1e-6
    eigen_residual: float = 5e-7
    max_mode_power_fraction_change: float = 0.01

    def __post_init__(self):
        import numpy as np
        for k, v in asdict(self).items():
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{k} must be positive and finite")
        if not 0 < self.field_overlap <= 1 or not 0 < self.minimum_comparison_power_fraction <= 1:
            raise ValueError("overlap and captured-power thresholds must be <=1")


def physics_from_repository(root):
    root = Path(root)
    cavity = json.loads((root / "config/thin_disk_resonator_250mm_2pct.json").read_text())["cavity"]
    assembly = json.loads((root / "config/stage6_assembly.json").read_text())
    # Numerical choices never participate in a physical-state fingerprint.
    assembly.pop("numerics", None)
    return {"cavity": cavity, "assembly": assembly,
            "pump": {"energy_J": .001, "duration_s": 10e-12, "repetition_rate_Hz": 1e4,
                     "waist_m": .0005}, "density": asdict(ContinuousDensity()),
            "probe": {"waist_m": .000408, "charges": [0, 1, -1, 2],
                      "model": "weak seeded double-pass probe of frozen mean populations"}}


def default_numerics():
    return {"optical_n": 256, "optical_window_m": .012,
            "thermal": {"nr": 24, "nz": 4, "nphi": 12, "radial_exponent": 1.5},
            "plate_thermal_nz": 12,
            "mechanical": {"nr": 8, "outer_rings": 4, "ntheta": 32,
                           "nz_disk": 4, "nz_plate": 4, "radial_exponent": 1.6},
            "settings": {"max_outer_iterations": 24, "minimum_outer_iterations": 4,
                         "consecutive_converged": 2, "eigen_candidates": 4,
                         "optical_max_cycles": 800, "eigen_maxiter": 1000,
                         "field_tolerance": 5e-4, "heat_relative_tolerance": 5e-4,
                         "temperature_tolerance_K": .005, "displacement_tolerance_m": 5e-11,
                         "power_relative_tolerance": 5e-4, "loss_tolerance": 1e-5,
                         "optical_rtol": 1e-6, "optical_population_tolerance": 2e-7,
                         "optical_energy_tolerance": 1e-4, "projection_order": 6,
                         "max_projection_error": .01},
            "mode_count": 1, "initial_guess": "gaussian", "initial_seed": 7,
            "density_quadrature_order": 6}


def make_plan(physics, *, kind="coupled", reference=None):
    """Construct explicit one-factor, joint, initialization and sensitivity cases.

    Frozen-source campaigns use identical analysis machinery but cannot qualify
    a coupled operating point. reference is {state_sha256,summary_sha256}; paths
    are supplied at execution, not embedded as another user's local path.
    """
    if kind not in ("coupled", "frozen"):
        raise ValueError("kind must be coupled or frozen")
    if kind == "frozen" and (not reference or "state_sha256" not in reference):
        raise ValueError("frozen study requires a hashed reference state")
    physics = deepcopy(physics)
    if kind == "frozen":
        physics["frozen_reference"] = deepcopy(reference)
    base = default_numerics()
    if kind == "frozen":
        base["thermal"] = {"nr": 24, "nz": 6, "nphi": 8, "radial_exponent": 1.5}
        base["mechanical"].update(nr=4, ntheta=16, outer_rings=2, nz_disk=2, nz_plate=2)
        base["plate_thermal_nz"] = 8
    cases, groups = [], []

    def add(name, numerics=None, physical=None, purpose="convergence"):
        c = {"id": name, "kind": kind, "purpose": purpose,
             "physics": deepcopy(physics if physical is None else physical),
             "numerics": deepcopy(base if numerics is None else numerics)}
        c["physics_hash"] = stable_hash(c["physics"])
        c["spec_hash"] = stable_hash(c)
        cases.append(c)
        return name

    def group(name, ids, required=True, comparison="refinement"):
        groups.append({"name": name, "case_ids": ids, "required": required,
                       "comparison": comparison, "required_successive_pairs": min(2, max(1, len(ids)-1))})

    add("reference")
    group("optical_resolution", ["reference"] + [add(f"optical_{n}", {**deepcopy(base), "optical_n": n}) for n in (384, 512)])
    # Same spacing as 256 pixels over 12 mm. Padding and pixel spacing are not confounded.
    group("window_padding", ["reference"] + [add(f"window_{n}", {**deepcopy(base), "optical_n": n,
                                                           "optical_window_m": n*base["optical_window_m"]/base["optical_n"]}) for n in (384, 512)])
    thermal_ids = ["reference"]
    for level, values in enumerate(((36, 8, 24, 16), (48, 12, 32, 24)), start=1):
        nr, nz, nphi, pnz = values
        n = deepcopy(base); n["thermal"].update(nr=nr, nz=nz, nphi=nphi); n["plate_thermal_nz"] = pnz
        thermal_ids.append(add(f"material_{level}", n))
    group("material_resolution", thermal_ids)
    mechanical_ids = ["reference"]
    targets = ((6, 24, 3), (8, 32, 4), (12, 48, 6)) if kind == "frozen" else ((12, 48, 6), (16, 64, 8))
    for level, (nr, nt, nz) in enumerate(targets, start=1):
        n = deepcopy(base); n["mechanical"].update(nr=nr, ntheta=nt, nz_disk=nz, nz_plate=nz)
        mechanical_ids.append(add(f"mechanical_{level}", n))
    group("mechanical_resolution", mechanical_ids)
    joint_ids = ["reference"]
    joint_targets = [(384, 36, 8, 24, 12, 48, 6), (512, 48, 12, 32, 16, 64, 8)]
    if kind == "frozen":
        joint_targets.append((640, 60, 16, 40, 20, 80, 10))
    for level, (on, tr, tz, tp, mr, mt, mz) in enumerate(joint_targets, start=1):
        n = deepcopy(base); n["optical_n"] = on
        n["thermal"].update(nr=tr, nz=tz, nphi=tp)
        n["mechanical"].update(nr=mr, ntheta=mt, nz_disk=mz, nz_plate=mz)
        n["plate_thermal_nz"] = 16 if level == 1 else (24 if level == 2 else 32)
        joint_ids.append(add(f"joint_{level}", n))
    group("joint_refinement", joint_ids)
    if kind == "coupled":
        tight_ids = ["reference"]
        for index, factor in enumerate((.3, .1), start=1):
            n = deepcopy(base)
            for key in ("optical_rtol", "optical_population_tolerance", "optical_energy_tolerance",
                        "field_tolerance", "heat_relative_tolerance", "temperature_tolerance_K",
                        "displacement_tolerance_m", "power_relative_tolerance", "loss_tolerance"):
                n["settings"][key] *= factor
            n["settings"]["eigen_tolerance"] = 5e-7*factor
            tight_ids.append(add(f"tolerances_{index}", n))
        group("integration_tolerances", tight_ids)
        mode_ids = ["reference"]
        for modes in (2, 4, 8):
            n = deepcopy(base); n["mode_count"] = modes; n["settings"]["eigen_candidates"] = modes+2
            mode_ids.append(add(f"modes_{modes}", n, purpose="mode_completeness"))
        group("retained_modes", mode_ids, comparison="modal_mixture")
        candidate_ids = ["modes_2"]
        for k in (6, 8):
            n = deepcopy(base); n["mode_count"] = 2; n["settings"]["eigen_candidates"] = k
            candidate_ids.append(add(f"candidates_{k}", n, purpose="mode_completeness"))
        group("candidate_spectrum", candidate_ids, comparison="modal_mixture")
        seed_ids = ["modes_2"]
        for guess in ("lg_plus", "lg_minus", "lg_two", "mixed"):
            n = deepcopy(base); n["mode_count"] = 2; n["initial_guess"] = guess
            seed_ids.append(add(f"start_{guess}", n, purpose="initialization"))
        group("initialization", seed_ids, comparison="modal_mixture")
        # Initial conditions are not ordered refinement levels: every start matters.
        groups[-1]["required_successive_pairs"] = len(seed_ids)-1
    # Sensitivity is not numerical convergence: its cases have different physics hashes.
    sensitivity_ids = ["reference"]
    knobs = (("contact", "thermal", "interface_conductance_W_m2K"),
             ("coolant", "thermal", "coolant_conductance_W_m2K"),
             ("bond_normal", "mechanical", "normal_stiffness_Pa_m"),
             ("bond_shear", "mechanical", "tangential_stiffness_Pa_m"))
    for label, section, key in knobs:
        for factor in (.5, 2.):
            p = deepcopy(physics)
            container = p["assembly"][section] if section == "thermal" else p["assembly"][section]["bond"]
            container[key] *= factor
            sensitivity_ids.append(add(f"sensitivity_{label}_{factor:g}", physical=p, purpose="sensitivity"))
    for support in ("roller", "free"):
        p = deepcopy(physics); p["assembly"]["mechanical"]["plate_support"] = support
        sensitivity_ids.append(add(f"sensitivity_support_{support}", physical=p, purpose="sensitivity"))
    if kind == "coupled":
        p = deepcopy(physics); p["density"]["contrast_bound"] = .08
        sensitivity_ids.append(add("sensitivity_inhomogeneous", physical=p, purpose="sensitivity"))
    group("cooling_and_density_sensitivity", sensitivity_ids, required=False, comparison="sensitivity")
    return {"schema": SCHEMA, "kind": kind, "cases": cases, "groups": groups,
            "acceptance": asdict(Acceptance()),
            "comparison_domain": {"radius_m": .0015, "weight_radius_m": .000408, "nr": 96, "nphi": 256},
            "limits": ["Numerical qualification applies only to compared physical operating points.",
                       "A frozen-source study is not a coupled hot-cavity refinement.",
                       "Sensitivity results require their own grid studies before dataset use.",
                       "Material/contact calibration and omitted physics are not validated by this campaign."],
            "physics_hash": stable_hash(physics)}


def validate_plan(plan):
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported validation plan schema")
    ids = [c["id"] for c in plan["cases"]]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("case identifiers must be unique and nonempty")
    import re
    for c in plan["cases"]:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", c["id"]) or c["id"] in (".", ".."):
            raise ValueError("unsafe case id")
        if c["kind"] != plan["kind"] or c["physics_hash"] != stable_hash(c["physics"]):
            raise ValueError("physical case identity changed without rebuilding manifest")
        raw = {k: v for k, v in c.items() if k != "spec_hash"}
        if c["spec_hash"] != stable_hash(raw):
            raise ValueError("case configuration changed without rebuilding manifest")
    for g in plan["groups"]:
        if not set(g["case_ids"]).issubset(ids) or len(g["case_ids"]) < 2:
            raise ValueError("comparison group refers to missing cases")
    Acceptance(**plan["acceptance"])
    return plan
