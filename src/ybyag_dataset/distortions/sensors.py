"""Five probes sampled from the same Yb disk/copper temperature arrays."""
import numpy as np
from hoyag.cooling_plate import sample_temperature
from ybluag.sensors import default_five_probes


class SensorSession:
    def __init__(self, disk_mesh, plate_mesh, ranges, seed, *, enabled=True,
                 stress=1.0, force_dropout=False):
        self.rng = np.random.default_rng(seed)
        self.probes = default_five_probes(disk_mesh.z_edges_m[-1],disk_mesh.r_edges_m[-1])
        self.disk_mesh, self.plate_mesh = disk_mesh, plate_mesh
        self.ranges, self.enabled, self.stress = ranges, enabled, stress
        self.force_dropout=force_dropout
        self.bias = self.rng.normal(0, stress*ranges["probe_bias_K"], 5) if enabled else np.zeros(5)
        self.noise_std = self.rng.uniform(*ranges["probe_noise_K"], size=5) if enabled else np.zeros(5)
        self.response_s = self.rng.uniform(*ranges["probe_response_time_s"], size=5) if enabled else np.zeros(5)
        self.position_error_m = self.rng.normal(0, stress*ranges["probe_position_uncertainty_mm"]*1e-3,(5,2)) if enabled else np.zeros((5,2))
        self.response_K = None
        self.last_time_s = None

    def sample(self, disk_temperature_K, plate_temperature_K, time_s):
        true, positions = [], []
        thickness = self.disk_mesh.z_edges_m[-1]
        for i, probe in enumerate(self.probes):
            p = np.asarray(probe.position_m, float).copy()
            p[:2] += self.position_error_m[i]
            mesh = self.disk_mesh if probe.region == "disk" else self.plate_mesh
            p[:2] *= min(1., .99*mesh.r_edges_m[-1]/max(np.hypot(*p[:2]),1e-12))
            field = disk_temperature_K if probe.region == "disk" else plate_temperature_K
            value = sample_temperature(mesh, field, p[None],
               z_offset_m=0 if probe.region == "disk" else thickness)[0]
            true.append(value)
            positions.append(p)
        true = np.asarray(true, float)
        if self.response_K is None:
            self.response_K = true.copy()
        else:
            dt = max(0, time_s-self.last_time_s)
            factor = np.where(self.response_s > 0, 1-np.exp(-dt/np.maximum(self.response_s,1e-12)), 1.)
            self.response_K += factor*(true-self.response_K)
        self.last_time_s = time_s
        measured = self.response_K.copy()
        if self.enabled:
            measured += self.bias + self.rng.normal(0,self.noise_std)
            measured += self.rng.normal(0,self.stress*self.ranges["probe_drift_K_per_s"]*time_s,5)
            dropout = self.rng.random(5) < self.stress*self.ranges["probe_dropout_probability"]
            if self.force_dropout:
                dropout[int(self.rng.integers(0,5))]=True
            measured[dropout] = np.nan
        return dict(true_temperature_K=true, measured_temperature_K=measured,
                    position_m=np.asarray(positions), names=[p.name for p in self.probes],
                    bias_K=self.bias.copy(), response_time_s=self.response_s.copy())
