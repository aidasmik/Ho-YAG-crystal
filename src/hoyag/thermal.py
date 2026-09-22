"""Stage 5B: conservative 3-D finite-volume heat flow in a circular thin disk.

Coordinates: z=0 is the optical front; z=thickness is the cooled rear HR face.
The host mask is geometry, NOT the dopant mask: undoped YAG still conducts heat.
The circular rim is voxelized (staircase); refine transverse resolution there.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg
from .propagation import Grid2D


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ThermalBoundary:
    """Contact coefficients W/(m^2 K); 0=adiabatic, +inf=Dirichlet.

    Finite h includes the center-to-face conduction resistance dx/(2k),
    in series with 1/h. These are NOT the rod's tabulated total W/K values.
    """
    sink_temperature_K: float = 293.15
    rear_h_W_m2K: float = 1e5
    front_h_W_m2K: float = 0.0
    rim_h_W_m2K: float = 0.0

    def __post_init__(self):
        _positive(self.sink_temperature_K, "sink_temperature_K")
        for v in (self.rear_h_W_m2K, self.front_h_W_m2K, self.rim_h_W_m2K):
            if np.isnan(v) or v < 0:
                raise ValueError("surface conductance must be nonnegative or +inf")


@dataclass
class ThermalResult:
    temperature_K: np.ndarray
    deposited_power_W: float
    boundary_power_W: float
    residual_W: float
    relative_linear_residual: float
    stored_energy_change_J: float = 0.0


class DiskHeatSolver:
    """Steady conduction and unconditionally stable backward-Euler stepping.

    Material properties can be scalar or voxel arrays, but remain fixed during
    a solve. Temperature-dependent constitutive laws need outer iteration; no
    unvalidated Debye polynomial is silently enabled.
    """
    def __init__(self, grid: Grid2D, nz: int = 16, thickness_m: float = 1e-3,
                 diameter_m: float = 10e-3, *, conductivity_W_mK=14.0,
                 density_kg_m3=4560.0, heat_capacity_J_kgK=680.0,
                 boundary: ThermalBoundary | None = None):
        if not isinstance(nz, (int, np.integer)) or nz < 1:
            raise ValueError("nz must be a positive integer")
        for v, name in ((thickness_m,"thickness"), (diameter_m,"diameter"),
                        (grid.dx,"dx"), (grid.dy,"dy")):
            _positive(v, name)
        if grid.nx*grid.dx < diameter_m-1e-15 or grid.ny*grid.dy < diameter_m-1e-15:
            raise ValueError("thermal grid must contain the whole disk")
        self.grid, self.nz = grid, int(nz)
        self.thickness_m, self.diameter_m = thickness_m, diameter_m
        self.dz = thickness_m / nz
        self.shape = (nz, grid.ny, grid.nx)
        x, y = grid.mesh
        face_mask = x*x+y*y <= (diameter_m/2)**2
        self.mask = np.broadcast_to(face_mask, self.shape).copy()
        if not np.any(self.mask):
            raise ValueError("thermal grid contains no disk cells")
        self.ids = np.full(self.shape, -1, int)
        self.ids[self.mask] = np.arange(np.count_nonzero(self.mask))
        self.n = np.count_nonzero(self.mask)
        self.cell_volume_m3 = grid.dx*grid.dy*self.dz
        self.boundary = boundary or ThermalBoundary()
        self.k = self._material(conductivity_W_mK, "conductivity")
        rho = self._material(density_kg_m3, "mass density")
        cp = self._material(heat_capacity_J_kgK, "heat capacity")
        self.capacity_J_K = rho*cp*self.cell_volume_m3
        self.K, self.boundary_conductance = self._assemble()

    @property
    def z_m(self):
        return (np.arange(self.nz)+0.5)*self.dz

    @property
    def material_volume_m3(self):
        return self.n*self.cell_volume_m3

    def _material(self, value, name):
        a = np.broadcast_to(np.asarray(value, float), self.shape)[self.mask].copy()
        if np.any(~np.isfinite(a)) or np.any(a <= 0):
            raise ValueError(f"{name} must be finite and positive in all host cells")
        return a

    def _assemble(self):
        diagonal = np.zeros(self.n)
        boundary_g = np.zeros(self.n)
        rows, cols, vals = [], [], []
        spacing = (self.dz, self.grid.dy, self.grid.dx)
        areas = (self.grid.dx*self.grid.dy, self.dz*self.grid.dx, self.dz*self.grid.dy)
        for axis, (d, area) in enumerate(zip(spacing, areas)):
            a_slice = [slice(None)]*3; b_slice = a_slice.copy()
            a_slice[axis] = slice(None,-1); b_slice[axis] = slice(1,None)
            a = self.ids[tuple(a_slice)].ravel(); b = self.ids[tuple(b_slice)].ravel()
            shared = (a >= 0) & (b >= 0)
            ai, bi = a[shared], b[shared]
            g = area / (d/(2*self.k[ai]) + d/(2*self.k[bi]))
            np.add.at(diagonal, ai, g); np.add.at(diagonal, bi, g)
            rows.extend((ai, bi)); cols.extend((bi, ai)); vals.extend((-g, -g))
            # Missing neighbors (including outside the square) are surfaces.
            for direction in (-1, 1):
                neighbor = np.roll(self.ids, -direction, axis=axis)
                edge = [slice(None)]*3
                edge[axis] = -1 if direction == 1 else 0
                neighbor[tuple(edge)] = -1
                surf = self.mask & (neighbor < 0)
                indices = self.ids[surf]
                h = (self.boundary.front_h_W_m2K if direction == -1 else self.boundary.rear_h_W_m2K) if axis == 0 else self.boundary.rim_h_W_m2K
                if h > 0:
                    resistance = d/(2*self.k[indices]) + (0.0 if np.isinf(h) else 1/h)
                    gs = area/resistance
                    np.add.at(diagonal, indices, gs)
                    np.add.at(boundary_g, indices, gs)
        indices = np.arange(self.n)
        rows.append(indices); cols.append(indices); vals.append(diagonal)
        K = coo_matrix((np.concatenate(vals), (np.concatenate(rows),np.concatenate(cols))),
                       shape=(self.n,self.n)).tocsr()
        return K, boundary_g

    def field(self, values, name="field"):
        a = np.broadcast_to(np.asarray(values, float), self.shape)[self.mask].copy()
        if np.any(~np.isfinite(a)):
            raise ValueError(f"{name} must be finite in host cells")
        return a

    def unpack(self, packed, outside=0.0):
        a = np.full(self.shape, outside, float)
        a[self.mask] = packed
        return a

    def _solve(self, A, rhs, rtol, initial=None):
        if not np.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError("rtol must be in (0,1)")
        if np.any(A.diagonal() <= 0):
            raise ValueError("unanchored or invalid thermal matrix")
        preconditioner = diags(1/A.diagonal())
        x, info = cg(A, rhs, x0=initial, M=preconditioner,
                     rtol=rtol, atol=0.0, maxiter=10000)
        if info != 0:
            raise RuntimeError(f"thermal CG did not converge (info={info}); refine preconditioning")
        return x, float(np.linalg.norm(A@x-rhs)/max(np.linalg.norm(rhs),1e-300))

    def steady(self, source_W_m3, *, rtol: float = 1e-10) -> ThermalResult:
        if not np.any(self.boundary_conductance > 0):
            raise ValueError("steady disk requires a heat-removing boundary")
        q = self.field(source_W_m3,"source")*self.cell_volume_m3
        delta, residual = self._solve(self.K, q, rtol)
        temperature = self.boundary.sink_temperature_K+delta
        if np.any(temperature <= 0):
            raise ValueError("computed temperature is nonphysical")
        deposited, removed = float(np.sum(q)), float(self.boundary_conductance@delta)
        return ThermalResult(self.unpack(temperature,self.boundary.sink_temperature_K),
                             deposited,removed,deposited-removed,residual)

    def step(self, temperature_K, source_W_m3, dt_s: float, *, rtol=1e-10) -> ThermalResult:
        _positive(dt_s,"dt_s")
        old = self.field(temperature_K,"temperature")-self.boundary.sink_temperature_K
        if np.any(old+self.boundary.sink_temperature_K <= 0):
            raise ValueError("temperature must be above absolute zero")
        q = self.field(source_W_m3,"source")*self.cell_volume_m3
        mass = self.capacity_J_K/dt_s
        new, residual = self._solve(self.K+diags(mass), q+mass*old, rtol, old)
        deposited, removed = float(q.sum()),float(self.boundary_conductance@new)
        stored = float(self.capacity_J_K@(new-old))
        if np.any(new+self.boundary.sink_temperature_K <= 0):
            raise ValueError("computed temperature is nonphysical")
        return ThermalResult(self.unpack(new+self.boundary.sink_temperature_K,self.boundary.sink_temperature_K),
                             deposited,removed,deposited-removed-stored/dt_s,residual,stored)

    def deposit_energy(self, temperature_K, energy_J_m3):
        """Instantaneous deposited HEAT energy, not absorbed optical energy."""
        t = self.field(temperature_K,"temperature")
        energy = self.field(energy_J_m3,"deposited energy")*self.cell_volume_m3
        new = t+energy/self.capacity_J_K
        if np.any(new <= 0):
            raise ValueError("energy impulse produced a nonphysical temperature")
        return self.unpack(new,self.boundary.sink_temperature_K)
