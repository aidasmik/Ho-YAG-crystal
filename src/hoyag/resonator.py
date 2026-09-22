"""Stage 4R: HR-backed thin-disk cavity and pulse-pumped modal oscillator.

Two deliberately distinct solvers:
* A scalar FFT round-trip operator checks the passive cavity and its LG modes.
* A spatially resolved, fixed-transverse-mode rate model evolves photons and
  the repository's four Ho manifolds between short pump kicks.

This is not a 3-D self-consistent thermal/Fox-Li laser model. The modal solver
averages standing-wave fringes, neglects coherent mode coupling, and uses a
small-round-trip-gain mean-field photon equation. Pump return encounters are
sequential fluence maps; overlapping forward/reflected 10 ps pulses are not
resolved. These approximations must remain visible in reports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre
from scipy.sparse import coo_matrix

from .pump_source import PumpSource, resolve_pump_source
from .numerical_quality import check_spectral_scalar_limit
from .populations import C0, H, I5, I6, I7, I8, HoYAGFourLevelParams, four_level_rhs


def _positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class ThinDiskResonator:
    """Plane HR disk (left), air gap, concave output coupler (right).

    The reference plane is the plane HR coating on the disk's back surface.
    Reduced ABCD length is air_gap + d/n, not air_gap + n*d. Group delay uses
    air_gap + n_group*d. Mirror and loss parameters are assumed, not measured.
    """
    disk_diameter_m: float = 0.010
    disk_thickness_m: float = 0.001
    air_gap_m: float = 0.200
    output_mirror_radius_m: float = 0.500
    output_transmission: float = 0.030
    disk_hr_reflectivity: float = 0.999
    other_roundtrip_loss: float = 0.005
    wavelength_m: float = 2.0903e-6
    host_index: float = 1.799104526293235
    host_group_index: float = 1.8315976829906033
    pump_hr_reflectivity: float = 0.999

    def __post_init__(self) -> None:
        for name in ("disk_diameter_m", "disk_thickness_m", "air_gap_m",
                     "output_mirror_radius_m", "wavelength_m", "host_index",
                     "host_group_index"):
            _positive(getattr(self, name), name)
        if not 0 < self.output_transmission < 1:
            raise ValueError("output_transmission must be between zero and one")
        for name in ("disk_hr_reflectivity", "pump_hr_reflectivity"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not np.isfinite(self.other_roundtrip_loss) or not 0 <= self.other_roundtrip_loss < 1:
            raise ValueError("other_roundtrip_loss must be in [0, 1)")

    @property
    def reduced_length_m(self) -> float:
        return self.air_gap_m + self.disk_thickness_m / self.host_index

    @property
    def stability_product(self) -> float:
        return 1.0 - self.reduced_length_m / self.output_mirror_radius_m

    @property
    def stable(self) -> bool:
        return 0.0 < self.stability_product < 1.0

    @property
    def rayleigh_range_m(self) -> float:
        if not self.stable:
            raise ValueError("unstable or marginal plane-concave cavity")
        L = self.reduced_length_m
        return math.sqrt(L * (self.output_mirror_radius_m - L))

    @property
    def waist_m(self) -> float:
        return math.sqrt(self.wavelength_m * self.rayleigh_range_m / math.pi)

    @property
    def output_mirror_spot_m(self) -> float:
        return self.waist_m * math.sqrt(1 + (self.reduced_length_m / self.rayleigh_range_m)**2)

    @property
    def roundtrip_time_s(self) -> float:
        return 2 * (self.air_gap_m + self.host_group_index * self.disk_thickness_m) / C0

    @property
    def passive_power_retention(self) -> float:
        return ((1-self.output_transmission) * self.disk_hr_reflectivity
                * (1-self.other_roundtrip_loss))

    @property
    def logarithmic_loss(self) -> float:
        return -math.log(self.passive_power_retention)

    @property
    def photon_lifetime_s(self) -> float:
        return self.roundtrip_time_s / self.logarithmic_loss

    @property
    def output_decay_rate_s1(self) -> float:
        """Continuous-time, logarithmic equivalent of output-coupler loss."""
        return -math.log1p(-self.output_transmission) / self.roundtrip_time_s

    @property
    def threshold_gain_m1(self) -> float:
        """Uniform single-traversal intensity gain for round-trip threshold."""
        return self.logarithmic_loss / (2*self.disk_thickness_m)

    def summary(self) -> dict:
        return {**asdict(self), "stable": self.stable,
                "g1_g2": self.stability_product, "reduced_length_m": self.reduced_length_m,
                "waist_m": self.waist_m, "oc_spot_m": self.output_mirror_spot_m,
                "roundtrip_time_s": self.roundtrip_time_s,
                "free_spectral_range_Hz": 1/self.roundtrip_time_s,
                "photon_lifetime_s": self.photon_lifetime_s,
                "threshold_gain_m1": self.threshold_gain_m1}


def lg0_field(x_m, y_m, waist_m: float, charge: int = 0) -> np.ndarray:
    """Unnormalized LG_0^ell field at a waist; phase in fixed lab x/y axes."""
    _positive(waist_m, "waist_m")
    if not isinstance(charge, (int, np.integer)):
        raise ValueError("charge must be an integer")
    r2 = np.asarray(x_m)**2 + np.asarray(y_m)**2
    return ((np.sqrt(2*r2)/waist_m)**abs(charge)
            * np.exp(-r2/waist_m**2)
            * np.exp(1j*charge*np.arctan2(y_m, x_m)))


def cavity_roundtrip(field, grid, cavity: ThinDiskResonator,
                     single_pass_log_gain=0.0) -> tuple[np.ndarray, np.ndarray]:
    """One unfolded scalar round trip and the field transmitted at the OC.

    Field plane: after the planar disk HR reflection, heading toward the OC.
    Half of the disk's round-trip log intensity gain is applied on each visit.
    Constant carrier/reflection phases are omitted. There is NO complex
    conjugation at a mirror. A gain screen is not the dynamic modal solver.
    """
    arr = np.asarray(field, dtype=complex)
    if arr.shape != grid.shape or np.any(~np.isfinite(arr)):
        raise ValueError("field must be finite with the transverse grid shape")
    if grid.nx*grid.dx < cavity.disk_diameter_m or grid.ny*grid.dy < cavity.disk_diameter_m:
        raise ValueError("FFT window must contain the circular disk")
    log_gain = np.broadcast_to(np.asarray(single_pass_log_gain, float), grid.shape)
    if np.any(~np.isfinite(log_gain)) or np.max(np.abs(log_gain)) > 50:
        raise ValueError("invalid single-pass log gain")
    x, y = grid.mesh
    mask = (x*x+y*y) <= (cavity.disk_diameter_m/2)**2
    fx, fy = np.meshgrid(grid.fx, grid.fy, indexing="xy")
    k0 = 2*np.pi / cavity.wavelength_m
    prop = np.exp(-0.5j * cavity.reduced_length_m / k0 * (2*np.pi)**2*(fx*fx+fy*fy))
    def travel(a):
        return np.fft.ifft2(np.fft.fft2(a)*prop)
    # Each traversal has amplitude gain exp(g*d/2), with g*d = log_gain.
    at_oc = travel(arr*mask*np.exp(log_gain/2))
    output = np.sqrt(cavity.output_transmission)*at_oc
    curved_phase = np.exp(-1j*k0*(x*x+y*y)/cavity.output_mirror_radius_m)
    back = travel(at_oc*np.sqrt(1-cavity.output_transmission)*curved_phase)
    back *= mask*np.exp(log_gain/2)*np.sqrt(cavity.disk_hr_reflectivity*(1-cavity.other_roundtrip_loss))
    return back, output


def fluence_transfer(fluence_in_J_m2, initial_log_gain, saturation_fluence_J_m2):
    """Frantz-Nodvik fluence map, including absorbing (negative gain) media.

    This is a short-pulse, two-manifold local-column map. It is independent of
    the pulse's temporal envelope but assumes fixed effective cross sections.
    """
    fs = _positive(saturation_fluence_J_m2, "saturation_fluence_J_m2")
    f, g = np.broadcast_arrays(np.asarray(fluence_in_J_m2, float), np.asarray(initial_log_gain, float))
    if np.any(~np.isfinite(f)) or np.any(f < 0) or np.any(~np.isfinite(g)):
        raise ValueError("fluence must be finite/nonnegative and log gain finite")
    u = f/fs
    # expm1/log1p prevents cancellation for the weak-pump limit.
    ans = np.zeros_like(u)
    low = u < 50
    if np.any(g[low] > 650):
        raise ValueError("unphysical gain overflows short-pulse map")
    ans[low] = np.log1p(np.exp(g[low])*np.expm1(u[low]))
    ans[~low] = np.logaddexp(0.0, g[~low] + u[~low] + np.log1p(-np.exp(-u[~low])))
    return fs*ans


@dataclass
class OscillatorResult:
    history: np.ndarray
    waveform: np.ndarray
    fractions_before_next_pump: np.ndarray
    log_photon_number: np.ndarray
    mode_labels: tuple[str, ...]
    periodic_converged: bool
    cycles_simulated: int
    period_cycles: int | None
    metadata: dict


class ModalThinDiskLaser:
    """Four-manifold, spatially saturated class-B laser with pump kicks.

    `density_m3` is (nz, ns), transverse quadrature `areas_m2` is (ns,), and
    fixed mode intensities are (nm, ns). They can come from a radial mesh or
    flattened 2D sampling. The returned radial example is not an arbitrary
    asymmetric 3-D wavefront simulation.
    """
    def __init__(self, cavity: ThinDiskResonator, areas_m2, density_m3,
                 mode_intensity_m2, pump_profile_m2, *,
                 params: HoYAGFourLevelParams | None = None,
                 pump_absorption_m2: float | None = None,
                 pump_passes: int = 2, pump_relay_efficiency: float = 1.0,
                 spontaneous_fraction_per_mode: float = 1e-8,
                 mode_labels: tuple[str, ...] | None = None,
                 pump_source: PumpSource | None = None):
        self.cavity = cavity
        if not cavity.stable:
            raise ValueError("modal model requires a stable cold cavity")
        self.params = params or HoYAGFourLevelParams()
        if not np.isclose(cavity.wavelength_m, self.params.laser_wavelength_m, rtol=1e-8, atol=0):
            raise ValueError("cavity and material signal wavelengths must match")
        self.area = np.asarray(areas_m2, float).copy()
        self.density = np.asarray(density_m3, float).copy()
        self.modes = np.asarray(mode_intensity_m2, float).copy()
        self.pump = np.asarray(pump_profile_m2, float).copy()
        if self.area.ndim != 1 or np.any(~np.isfinite(self.area)) or np.any(self.area <= 0):
            raise ValueError("areas must be finite, positive, one-dimensional")
        self.ns = len(self.area)
        if self.density.ndim != 2 or self.density.shape[1] != self.ns or self.density.shape[0] < 1:
            raise ValueError("density must be (nz, ns)")
        if np.any(~np.isfinite(self.density)) or np.any(self.density <= 0):
            raise ValueError("quadrature includes only active, positive-density voxels")
        if self.modes.ndim != 2 or self.modes.shape[1] != self.ns or len(self.modes) < 1:
            raise ValueError("modes must be (nm, ns)")
        if self.pump.shape != (self.ns,):
            raise ValueError("pump profile shape mismatch")
        for name, arr in (("modes", self.modes), ("pump", self.pump)):
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        # Do not silently renormalize an intercepted/clipped physical beam.
        if not np.allclose(self.modes @ self.area, 1.0, rtol=2e-5, atol=0):
            raise ValueError("mode quadrature must integrate to one")
        if not np.isclose(self.pump @ self.area, 1.0, rtol=2e-5, atol=0):
            raise ValueError("pump quadrature must integrate to one")
        self.nz = self.density.shape[0]
        self.nm = len(self.modes)
        self.nc = self.ns*self.nz
        self.dz = cavity.disk_thickness_m / self.nz
        self.volume = np.broadcast_to(self.dz*self.area, self.density.shape)
        if pump_source is not None and pump_absorption_m2 is not None:
            raise ValueError('specify a pump source or a scalar override, not both')
        self.pump_source = None if pump_source is None else resolve_pump_source(
            self.params.pump_wavelength_m, pump_source.duration_fwhm_s, source=pump_source)
        self.sa = (self.pump_source.effective_absorption_m2() if self.pump_source is not None
                   else (self.params.sigma_abs_pump_m2 if pump_absorption_m2 is None else float(pump_absorption_m2)))
        if not np.isfinite(self.sa) or self.sa < 0:
            raise ValueError("pump absorption must be finite and nonnegative")
        if not isinstance(pump_passes, (int, np.integer)) or pump_passes < 1:
            raise ValueError("pump_passes must be a positive integer")
        if not 0 < pump_relay_efficiency <= 1:
            raise ValueError("pump_relay_efficiency must be in (0,1]")
        if not 0 < spontaneous_fraction_per_mode < 1/self.nm:
            raise ValueError("invalid spontaneous emission coupling")
        self.pump_passes = int(pump_passes)
        self.relay = float(pump_relay_efficiency)
        self.beta_sp = float(spontaneous_fraction_per_mode)
        self.labels = mode_labels or tuple(f"mode {i}" for i in range(self.nm))
        if len(self.labels) != self.nm:
            raise ValueError("one label required per mode")
        self.ep = H*C0/self.params.pump_wavelength_m
        self.es = H*C0/self.params.laser_wavelength_m
        self.trt = cavity.roundtrip_time_s
        self.kloss = cavity.logarithmic_loss/self.trt
        self.kout = cavity.output_decay_rate_s1
        self.nvar = 3*self.nc + 2*self.nm

    def full_fractions(self, three):
        a = np.asarray(three).reshape(3, self.nz, self.ns)
        return np.concatenate((a, (1-a.sum(axis=0))[None]), axis=0)

    def check_fractions(self, three, tolerance=2e-7):
        f = self.full_fractions(three)
        if np.any(~np.isfinite(f)) or np.min(f) < -tolerance or np.max(f) > 1+tolerance:
            raise FloatingPointError("nonphysical populations: refine integration settings")

    def log_roundtrip_gain(self, fractions) -> np.ndarray:
        f = self.full_fractions(fractions) if np.size(fractions) == 3*self.nc else np.asarray(fractions)
        g = self.density*(self.params.sigma_em_laser_m2*f[I7] - self.params.sigma_abs_laser_m2*f[I8])
        return 2*self.dz * (self.modes @ (g.sum(axis=0)*self.area))

    def pump_kick(self, three, energy_J: float):
        """Sequential, photon-conserving fluence passes through actual voxels.

        Returns new first-three fractions, absorbed J, escaped J, relay/HR J.
        Each crystal voxel appears twice for the default HR-backed pump path.
        Other manifolds and decay remain frozen during the short pump event.
        """
        if not np.isfinite(energy_J) or energy_J < 0:
            raise ValueError("pump energy must be finite and nonnegative")
        self.check_fractions(three)
        f = self.full_fractions(np.asarray(three).copy())
        fsat = self.ep/(self.sa+self.params.sigma_em_pump_m2)
        fluence = energy_J*self.pump
        absorbed = 0.0
        loss = 0.0
        for traversal in range(self.pump_passes):
            indices = range(self.nz) if traversal % 2 == 0 else range(self.nz-1, -1, -1)
            for iz in indices:
                log_gain = self.dz*self.density[iz]*(self.params.sigma_em_pump_m2*f[I7,iz]-self.sa*f[I8,iz])
                fout = fluence_transfer(fluence, log_gain, fsat)
                delta = (fluence-fout)/(self.ep*self.dz*self.density[iz])
                f[I7,iz] += delta
                f[I8,iz] -= delta
                absorbed += float((fluence-fout) @ self.area)
                fluence = fout
            if traversal < self.pump_passes-1:
                eta = self.cavity.pump_hr_reflectivity if traversal % 2 == 0 else self.relay
                loss += float((1-eta)*(fluence @ self.area))
                fluence = eta*fluence
        escaped = float(fluence @ self.area)
        self.check_fractions(f[:3])
        return f[:3].copy(), absorbed, escaped, loss

    def rhs(self, _time: float, y: np.ndarray):
        three = y[:3*self.nc].reshape(3, self.nz, self.ns)
        f = self.full_fractions(three)
        log_ph = y[3*self.nc:3*self.nc+self.nm]
        # This guard avoids silent NaNs, not a saturation prescription.
        if np.any(log_ph > 200) or np.any(log_ph < -200):
            raise FloatingPointError("photon integrator left supported dynamic range")
        photons = np.exp(log_ph)
        energies = self.es*photons
        intensity = 2*(energies/self.trt) @ self.modes  # both counterpropagating waves
        rates = four_level_rhs(
            f*self.density[None], self.params,
            laser_abs_rate_s1=self.params.sigma_abs_laser_m2*intensity/self.es,
            laser_em_rate_s1=self.params.sigma_em_laser_m2*intensity/self.es,
        )/self.density[None]
        rt_gain = self.log_roundtrip_gain(f)
        spontaneous = self.beta_sp * float(np.sum(f[I7]*self.density*self.volume))/self.params.tau7_s
        # Spontaneous I7 loss is already in four_level_rhs; add only the tiny
        # selected cavity-mode fraction to the photon equations.
        log_rate = (rt_gain/self.trt - self.kloss) + spontaneous/photons
        output_power = self.kout*energies
        return np.concatenate((rates[:3].ravel(), log_rate, output_power))

    def jacobian(self, _time: float, y: np.ndarray):
        """Analytic sparse Jacobian, including global photon-population coupling."""
        f = self.full_fractions(y[:3*self.nc])
        a,b,c,d = f
        p=self.params; D=self.density
        k5=p.k75_m3_s*D; k6=p.k76_m3_s*D
        c5=p.C57_m3_s*D; c6=p.C67_m3_s*D
        l5=p.M56_s1+1/p.tau5_s; l6=p.M67_s1+1/p.tau6_s
        l7=p.M78_s1+1/p.tau7_s
        b56=p.M56_s1+p.beta56/p.tau5_s
        b67=p.M67_s1+p.beta67/p.tau6_s
        b57=p.beta57/p.tau5_s
        ph0=3*self.nc
        photons=np.exp(y[ph0:ph0+self.nm])
        flux=2/self.trt*(photons@self.modes)
        Wa=p.sigma_abs_laser_m2*flux; We=p.sigma_em_laser_m2*flux
        local=[[-c5*d-l5+c5*a, c5*a, 2*k5*c+c5*a],
               [c6*b+b56, -c6*d-l6+c6*b, 2*k6*c+c6*b],
               [-2*c6*b+2*c5*d+b57-2*c5*a-Wa,
                2*c6*d+b67-2*c6*b-2*c5*a-Wa,
                -4*(k5+k6)*c-l7-2*c6*b-2*c5*a-Wa-We]]
        rows=[];cols=[];values=[]; idx=np.arange(self.nc)
        for aa in range(3):
            for bb in range(3):
                rows.append(aa*self.nc+idx); cols.append(bb*self.nc+idx)
                values.append(local[aa][bb].ravel())
        S=self.beta_sp*float(np.sum(c*D*self.volume))/p.tau7_s
        for m in range(self.nm):
            cross=2/self.trt*photons[m]*self.modes[m]*(p.sigma_abs_laser_m2*d-p.sigma_em_laser_m2*c)
            rows.append(2*self.nc+idx);cols.append(np.full(self.nc,ph0+m));values.append(cross.ravel())
            overlap=2*self.dz/self.trt*D*self.modes[m]*self.area
            for aidx in range(3):
                grad=overlap*(p.sigma_abs_laser_m2+(p.sigma_em_laser_m2 if aidx==2 else 0))
                if aidx==2:
                    grad=grad+self.beta_sp*D*self.volume/(p.tau7_s*photons[m])
                rows.append(np.full(self.nc,ph0+m)); cols.append(aidx*self.nc+idx);values.append(grad.ravel())
            rows.append(np.array([ph0+m,ph0+self.nm+m]))
            cols.append(np.array([ph0+m,ph0+m]))
            values.append(np.array([-S/photons[m],self.kout*self.es*photons[m]]))
        return coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(self.nvar,self.nvar)).tocsc()

    def run(self, pump_energy_J: float, repetition_rate_Hz: float = 1e4, *,
            max_cycles: int = 300, min_cycles: int = 30,
            periodic_tolerance: float = 1e-5, energy_tolerance: float = 2e-3,
            rtol: float = 2e-6, atol_fraction: float = 2e-10,
            initial_fractions=None, initial_log_photons=None,
            capture_cycles: int = 4, max_step_s: float | None = None,
            max_period_cycles: int = 4,
            pump_fwhm_s: float = 10e-12) -> OscillatorResult:
        _positive(pump_energy_J, "pump_energy_J")
        _positive(repetition_rate_Hz, "repetition_rate_Hz")
        _positive(pump_fwhm_s, "pump_fwhm_s")
        if self.pump_source is not None:
            resolve_pump_source(self.params.pump_wavelength_m,pump_fwhm_s,source=self.pump_source)
            check_spectral_scalar_limit(self.pump_source.attenuation_diagnostic(
                float(np.max(np.sum(self.density,axis=0)))*self.dz*self.pump_passes))
        for v, name in ((periodic_tolerance,"periodic_tolerance"),(energy_tolerance,"energy_tolerance"),
                        (rtol,"rtol"),(atol_fraction,"atol_fraction")):
            _positive(v,name)
        if not isinstance(max_cycles,int) or not isinstance(min_cycles,int) or not 1 <= min_cycles <= max_cycles:
            raise ValueError("require 1 <= min_cycles <= max_cycles")
        if not isinstance(capture_cycles,int) or capture_cycles < 1:
            raise ValueError("capture_cycles must be a positive integer")
        period = 1/repetition_rate_Hz
        if pump_fwhm_s > 0.01*min(period, self.cavity.photon_lifetime_s):
            raise ValueError("pump pulse is too long for an instantaneous-kick model")
        fsat = self.ep/(self.sa+self.params.sigma_em_pump_m2)
        ratio = pump_energy_J*float(np.max(self.pump))/fsat
        if ratio > 0.1:
            warnings.warn("pump fluence >0.1 Fsat: return-pulse overlap approximation needs validation", RuntimeWarning)
        state = np.zeros((3,self.nz,self.ns)) if initial_fractions is None else np.asarray(initial_fractions,float).copy()
        self.check_fractions(state)
        logph = np.zeros(self.nm) if initial_log_photons is None else np.asarray(initial_log_photons,float).copy()
        if logph.shape != (self.nm,) or np.any(~np.isfinite(logph)):
            raise ValueError("initial photon state shape/values invalid")
        if not isinstance(max_period_cycles,int) or not 1 <= max_period_cycles <= 16:
            raise ValueError("max_period_cycles must be an integer from 1 to 16")
        hmax = period/12 if max_step_s is None else _positive(max_step_s,"max_step_s")
        atol = np.r_[np.full(3*self.nc,atol_fraction),np.full(self.nm,rtol*0.1),np.full(self.nm,pump_energy_J*1e-10)]
        history = []
        waveforms = []
        converged = False
        previous_energy = None
        period_cycles = None
        states_recent = []
        photons_recent = []
        energies_recent = []
        good_periods = np.zeros(max_period_cycles, dtype=int)
        periodic_residuals = None
        for cycle in range(1,max_cycles+1):
            before = state.copy()
            ph_before = logph.copy()
            state, absorbed, escaped, pump_loss = self.pump_kick(state,pump_energy_J)
            post_excess = self.log_roundtrip_gain(state)-self.cavity.logarithmic_loss
            y0=np.r_[state.ravel(),logph,np.zeros(self.nm)]
            sol=solve_ivp(self.rhs,(0,period),y0,method="BDF",rtol=rtol,atol=atol,
                          jac=self.jacobian,max_step=hmax)
            if not sol.success:
                raise RuntimeError(sol.message)
            state=sol.y[:3*self.nc,-1].reshape(3,self.nz,self.ns)
            logph=sol.y[3*self.nc:3*self.nc+self.nm,-1]
            self.check_fractions(state)
            out=sol.y[-self.nm:,-1]
            # Compare complete same-phase populations, photon state, and cycle output.
            res=float(np.max(np.abs(state-before)))
            photon_res=float(np.max(np.abs(np.exp(logph)-np.exp(ph_before))/(1+np.maximum(np.exp(logph),np.exp(ph_before)))))
            en_res=np.inf if previous_energy is None else float(np.max(np.abs(out-previous_energy)/(np.maximum(out,previous_energy)+pump_energy_J*1e-9)))
            peak=float(np.max(self.kout*self.es*np.exp(sol.y[3*self.nc:3*self.nc+self.nm]).sum(axis=0)))
            history.append([cycle,cycle*period,absorbed,escaped,pump_loss,res,photon_res,en_res,peak,
                            float(np.max(post_excess)),*out])
            previous_energy=out
            cols=np.column_stack((sol.t,self.kout*self.es*np.exp(sol.y[3*self.nc:3*self.nc+self.nm]).T))
            waveforms.append(cols)
            if len(waveforms)>capture_cycles:
                waveforms.pop(0)
            for lag in range(1,min(max_period_cycles,len(states_recent))+1):
                dpop = float(np.max(np.abs(state-states_recent[-lag])))
                pn = np.exp(logph); po = np.exp(photons_recent[-lag])
                dph = float(np.max(np.abs(pn-po)/(1+np.maximum(pn,po))))
                olde = energies_recent[-lag]
                de = float(np.max(np.abs(out-olde)/(np.maximum(out,olde)+pump_energy_J*1e-9)))
                good_periods[lag-1] = good_periods[lag-1]+1 if (dpop<=periodic_tolerance and dph<=energy_tolerance and de<=energy_tolerance) else 0
                if cycle>=min_cycles and good_periods[lag-1]>=3*lag:
                    period_cycles=lag
                    periodic_residuals={"population":dpop,"photon":dph,"cycle_energy":de}
                    converged=True
                    break
            states_recent.append(state.copy()); photons_recent.append(logph.copy()); energies_recent.append(out.copy())
            if len(states_recent)>max_period_cycles:
                states_recent.pop(0); photons_recent.pop(0); energies_recent.pop(0)
            if converged:
                break
        waveform=np.vstack([np.column_stack((w[:,0]+i*period,w[:,1:])) for i,w in enumerate(waveforms)])
        return OscillatorResult(np.asarray(history),waveform,state,logph,self.labels,converged,cycle,period_cycles,
                {"cavity":self.cavity.summary(),"pump_energy_J":pump_energy_J,
                 "repetition_rate_Hz":repetition_rate_Hz,"pump_fwhm_s":pump_fwhm_s,
                 "pump_passes":self.pump_passes,"peak_incident_fluence_over_Fsat":ratio,
                 "pump_source":self.pump_source.summary() if self.pump_source is not None else {"effective_absorption_m2":self.sa,"source":"explicit low-level cross section; no inferred spectrum"},
                 "quadrature_sites":self.ns,"z_slices":self.nz,"rtol":rtol,
                 "spontaneous_fraction_per_mode":self.beta_sp,
                 "waveform_cycles":len(waveforms),"periodic_residuals":periodic_residuals,
                 "model":"fixed-mode mean-field oscillator; short pump kicks; no thermal lens"})


def radial_laser(cavity: ThinDiskResonator | None = None, *,
                 radial_points: int = 36, z_slices: int = 4,
                 charges: tuple[int,...] = (0,), pump_waist_m: float = 0.5e-3,
                 params: HoYAGFourLevelParams | None = None,
                 radial_density_modulation: float = 0.0,
                 pump_absorption_m2: float | None = None,
                 pump_duration_fwhm_s: float = 10e-12, pump_source: PumpSource | None = None,
                 spontaneous_fraction_per_mode: float = 1e-8) -> tuple[ModalThinDiskLaser,np.ndarray]:
    """Gauss-Legendre area quadrature over the entire 10 mm disk face.

    The radial quadrature is nonuniform and not a plot raster. All p=0 LG
    modes use the geometrical cavity waist, not a freely specified signal waist.
    Charge +/- degeneracy is retained when both are included; no handedness
    selection occurs without an extra physical element.
    """
    cavity=cavity or ThinDiskResonator()
    params=params or HoYAGFourLevelParams()
    if not isinstance(radial_points,int) or radial_points<12 or not isinstance(z_slices,int) or z_slices<1:
        raise ValueError("use >=12 radial points and >=1 z slice")
    if not charges or any(not isinstance(l,(int,np.integer)) for l in charges):
        raise ValueError("charges must be nonempty integers")
    if len(set(charges)) != len(charges):
        raise ValueError("duplicate charges would double-count a mode")
    _positive(pump_waist_m,"pump_waist_m")
    if not np.isfinite(radial_density_modulation) or abs(radial_density_modulation)>0.5:
        raise ValueError("density modulation must be finite and <=50%")
    nodes,weights=roots_legendre(radial_points)
    R=cavity.disk_diameter_m/2
    r=(nodes+1)*R/2
    area=2*np.pi*r*weights*R/2
    w=cavity.waist_m
    u=np.array([2/(np.pi*w*w*math.factorial(abs(l)))*(2*r*r/(w*w))**abs(l)*np.exp(-2*r*r/(w*w)) for l in charges])
    pump=2/(np.pi*pump_waist_m**2)*np.exp(-2*r*r/pump_waist_m**2)
    N=params.N_total_m3*(1+radial_density_modulation*np.exp(-r*r/(2*(0.5e-3)**2)))
    density=np.broadcast_to(N,(z_slices,radial_points)).copy()
    source=resolve_pump_source(params.pump_wavelength_m,pump_duration_fwhm_s,
                              source=pump_source,absorption_override_m2=pump_absorption_m2)
    model=ModalThinDiskLaser(cavity,area,density,u,pump,params=params,
                  pump_source=source,
                  spontaneous_fraction_per_mode=spontaneous_fraction_per_mode,
                  mode_labels=tuple(f"LG(0,{l})" for l in charges))
    return model,r
