"""Virtual point/footprint thermometers driven by the simulated disk and plate.

Measurements are synthetic. The exact thermal fields remain separate from
sensor response, fixed calibration bias, white readout noise, and latency.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from hoyag.cooling_plate import sample_temperature


@dataclass(frozen=True)
class TemperatureProbe:
    name: str
    region: str  # "disk" or "plate"
    position_m: tuple[float, float, float]  # global x,y,z; disk front is z=0
    footprint_radius_m: float = 0.0
    response_time_s: float = 0.0
    sampling_rate_Hz: float = 10.0
    latency_s: float = 0.0
    bias_K: float = 0.0
    noise_std_K: float = 0.0
    missing: bool = False

    def __post_init__(self):
        if (not self.name or self.region not in ("disk", "plate") or
                len(self.position_m) != 3 or
                not all(math.isfinite(v) for v in self.position_m) or
                not math.isfinite(self.footprint_radius_m) or self.footprint_radius_m < 0 or
                not math.isfinite(self.response_time_s) or self.response_time_s < 0 or
                not math.isfinite(self.sampling_rate_Hz) or self.sampling_rate_Hz <= 0 or
                not math.isfinite(self.latency_s) or self.latency_s < 0 or
                not math.isfinite(self.bias_K) or
                not math.isfinite(self.noise_std_K) or self.noise_std_K < 0):
            raise ValueError("invalid temperature probe specification")


def default_five_probes(disk_thickness_m: float,
                        disk_radius_m: float = 5e-3) -> tuple[TemperatureProbe, ...]:
    """Surface optical estimates and bonded-plate probes, not embedded sensors.

    The three disk readings stand for calibrated noncontact front-surface
    thermometry. They are samples of the modeled surface and do not recover
    the internal temperature field or imply that a physical sensor fits there.
    """
    d = disk_thickness_m
    edge = min(3e-3, .7*disk_radius_m)
    return (
        TemperatureProbe("disk_center", "disk", (0, 0, 0), 0.15e-3, .03),
        TemperatureProbe("disk_x", "disk", (edge, 0, 0), 0.15e-3, .03),
        TemperatureProbe("disk_y", "disk", (0, edge, 0), 0.15e-3, .03),
        TemperatureProbe("plate_center", "plate", (0, 0, d+.5e-3), .3e-3, .1),
        TemperatureProbe("plate_x", "plate", (3e-3, 0, d+.5e-3), .3e-3, .1),
    )


class ProbeArray:
    """Stateful first-order response with separately scheduled readout times."""

    def __init__(self, probes, disk_mesh, plate_mesh, *,
                 initial_temperature_K: float, seed: int = 0):
        self.probes = tuple(probes)
        if len(self.probes) != 5 or len({p.name for p in self.probes}) != 5:
            raise ValueError("exactly five uniquely named probes are required")
        self.disk_mesh, self.plate_mesh = disk_mesh, plate_mesh
        self.disk_thickness_m = float(disk_mesh.z_edges_m[-1])
        self.time_s = 0.0
        self.state_K = np.full(5, float(initial_temperature_K))
        self.local_K = np.full(5, float(initial_temperature_K))
        self.next_sample_s = np.zeros(5)
        self.rng = np.random.default_rng(seed)
        for probe in self.probes:
            mesh = disk_mesh if probe.region == "disk" else plate_mesh
            z = probe.position_m[2] - (0 if probe.region == "disk" else self.disk_thickness_m)
            radius = math.hypot(*probe.position_m[:2]) + probe.footprint_radius_m
            if radius > mesh.r_edges_m[-1] or not 0 <= z <= mesh.z_edges_m[-1]:
                raise ValueError(f"probe {probe.name} footprint is outside its material region")

    def initial_samples(self, disk_temperature_K, plate_temperature_K):
        """Sample the initial physical state at t=0, before any thermal step."""
        events = []
        for i, probe in enumerate(self.probes):
            local = self._local_average(probe, disk_temperature_K, plate_temperature_K)
            self.state_K[i] = local
            self.local_K[i] = local
            events.append(self._event(probe, 0.0, local, local))
            self.next_sample_s[i] = 1/probe.sampling_rate_Hz
        return sorted(events, key=lambda event: (event["available_time_s"], event["name"]))

    def _event(self, probe, sample_time, local, state):
        return {"name": probe.name, "region": probe.region,
                "position_m": probe.position_m,
                "footprint_radius_m": probe.footprint_radius_m,
                "sample_time_s": sample_time,
                "available_time_s": sample_time+probe.latency_s,
                "local_average_K": local,
                "noiseless_response_K": state,
                "measured_K": (None if probe.missing else
                               state+probe.bias_K+self.rng.normal(0, probe.noise_std_K)),
                "valid": not probe.missing}

    def _local_average(self, probe, disk_temperature_K, plate_temperature_K):
        x, y, z = probe.position_m
        radius = probe.footprint_radius_m
        if radius:
            angles = np.arange(8) * (2*np.pi/8)
            ring = radius/np.sqrt(2)
            points = np.vstack((np.array([[x, y, z]]),
                                np.column_stack((x+ring*np.cos(angles),
                                                 y+ring*np.sin(angles),
                                                 np.full(8, z)))))
        else:
            points = np.array([[x, y, z]])
        if probe.region == "disk":
            values = sample_temperature(self.disk_mesh, disk_temperature_K, points)
        else:
            values = sample_temperature(self.plate_mesh, plate_temperature_K, points,
                                        z_offset_m=self.disk_thickness_m)
        return float(np.mean(values))

    def advance(self, time_s, disk_temperature_K, plate_temperature_K):
        if not math.isfinite(time_s) or time_s <= self.time_s:
            raise ValueError("probe time must advance")
        events = []
        for i, probe in enumerate(self.probes):
            target = self._local_average(probe, disk_temperature_K, plate_temperature_K)
            old = float(self.state_K[i])
            initial_local = float(self.local_K[i])
            dt = time_s-self.time_s
            def response(elapsed):
                slope = (target-initial_local)/dt
                if probe.response_time_s == 0:
                    return initial_local+slope*elapsed
                tau = probe.response_time_s
                decay = math.exp(-elapsed/tau)
                return (initial_local+slope*(elapsed-tau+tau*decay)+
                        (old-initial_local)*decay)
            interval = 1/probe.sampling_rate_Hz
            while self.next_sample_s[i] <= time_s + 1e-12:
                sample_time = float(self.next_sample_s[i])
                elapsed = max(0., sample_time-self.time_s)
                state = response(elapsed)
                sampled_local = initial_local+(target-initial_local)*elapsed/dt
                events.append(self._event(probe, sample_time, sampled_local, state))
                self.next_sample_s[i] += interval
            self.state_K[i] = response(dt)
            self.local_K[i] = target
        self.time_s = float(time_s)
        return sorted(events, key=lambda event: (event["available_time_s"],
                                                  event["name"]))
