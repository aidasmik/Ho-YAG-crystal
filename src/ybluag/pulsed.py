"""Two-manifold Yb:LuAG pulse transport in a common retarded-time frame."""

from dataclasses import dataclass

import numpy as np

from .model import H, C, YbLuAGMaterial


@dataclass(frozen=True)
class PulseResult:
    pump_out_W_m2: np.ndarray
    signal_out_W_m2: np.ndarray
    final_excited_fraction_by_slice: np.ndarray
    absorbed_pump_fluence_J_m2: np.ndarray
    signal_fluence_change_J_m2: np.ndarray
    absorbed_pump_fluence_J_m2_by_slice: np.ndarray
    signal_fluence_change_J_m2_by_slice: np.ndarray
    excited_fraction_time_integral_s_by_slice: np.ndarray
    scope: str = ("co-propagating retarded-time pump/signal; fixed temperature; "
                  "no transverse diffraction, group-velocity walkoff or cavity")


def propagate_pulse(material: YbLuAGMaterial, time_s, pump_in_W_m2,
                    signal_in_W_m2, thickness_m: float, steps: int, *,
                    initial_excited_fraction=0.0) -> PulseResult:
    """Transport intensity samples and evolve one shared Yb population per z cell.

    Time samples include both pulse tails. During each optical pass the local
    population is frozen, then advanced to the next sample by the exact
    constant-rate solution using midpoint incident intensity. This is a
    retarded-time approximation suitable only when pump/signal walkoff and
    transverse diffraction within the disk are negligible.
    """
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or len(time) < 2 or np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be a strictly increasing finite 1D grid")
    if not np.isfinite(thickness_m) or thickness_m <= 0:
        raise ValueError("thickness_m must be finite and positive")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    pump = np.asarray(pump_in_W_m2, dtype=float)
    signal = np.asarray(signal_in_W_m2, dtype=float)
    if (pump.shape != signal.shape or pump.ndim < 1 or pump.shape[0] != len(time) or
            np.any(~np.isfinite(pump)) or np.any(~np.isfinite(signal)) or
            np.any(pump < 0) or np.any(signal < 0)):
        raise ValueError("pump/signal must match time and have finite nonnegative intensities")
    shape = pump.shape[1:]
    beta = np.broadcast_to(np.asarray(initial_excited_fraction, dtype=float),
                           (steps, *shape)).copy()
    if np.any(~np.isfinite(beta)) or np.any((beta < 0) | (beta > 1)):
        raise ValueError("initial excited fraction must be in [0, 1]")
    dz = thickness_m / steps
    pump_out = np.empty_like(pump)
    signal_out = np.empty_like(signal)

    def transport(pump_input, signal_input, current_beta, rates=False, ledger=False):
        p = np.array(pump_input, copy=True)
        s = np.array(signal_input, copy=True)
        up = np.empty_like(current_beta) if rates else None
        down = np.empty_like(current_beta) if rates else None
        pump_absorbed = np.empty_like(current_beta) if ledger else None
        signal_change = np.empty_like(current_beta) if ledger else None
        for iz in range(steps):
            alpha, gain = material.coefficients_m1(current_beta[iz])
            next_p = p * np.exp(-alpha * dz)
            next_s = s * np.exp(gain * dz)
            if ledger:
                pump_absorbed[iz] = p - next_p
                signal_change[iz] = next_s - s
            if rates:
                up[iz], down[iz] = material.rates_s1(
                    0.5 * (p + next_p), 0.5 * (s + next_s))
            p, s = next_p, next_s
        return p, s, up, down, pump_absorbed, signal_change

    absorbed_by_slice = np.zeros_like(beta)
    signal_change_by_slice = np.zeros_like(beta)
    beta_integral = np.zeros_like(beta)
    previous_absorbed = previous_signal_change = None
    for it in range(len(time)):
        (pump_out[it], signal_out[it], _, _, current_absorbed,
         current_signal_change) = transport(pump[it], signal[it], beta, ledger=True)
        if it:
            dt = time[it] - time[it - 1]
            absorbed_by_slice += 0.5 * (previous_absorbed + current_absorbed) * dt
            signal_change_by_slice += 0.5 * (previous_signal_change + current_signal_change) * dt
        previous_absorbed, previous_signal_change = current_absorbed, current_signal_change
        if it == len(time) - 1:
            break
        midpoint_pump = 0.5 * (pump[it] + pump[it + 1])
        midpoint_signal = 0.5 * (signal[it] + signal[it + 1])
        _, _, upward, downward, _, _ = transport(midpoint_pump, midpoint_signal, beta, rates=True)
        total = upward + downward + 1.0 / material.lifetime_s
        beta_steady = upward / total
        dt = time[it + 1] - time[it]
        relaxation = -np.expm1(-total * dt)
        beta_integral += beta_steady * dt + (beta - beta_steady) * relaxation / total
        beta = beta_steady + (beta - beta_steady) * (1.0 - relaxation)
        if np.any(~np.isfinite(beta)):
            raise FloatingPointError("nonfinite Yb population")
    return PulseResult(pump_out, signal_out, beta,
                       np.sum(absorbed_by_slice, axis=0),
                       np.sum(signal_change_by_slice, axis=0),
                       absorbed_by_slice, signal_change_by_slice, beta_integral)


@dataclass(frozen=True)
class PeriodicHeatResult:
    heat_W_m3_by_slice: np.ndarray
    pump_absorbed_W_m2_by_slice: np.ndarray
    signal_change_W_m2_by_slice: np.ndarray
    escaping_fluorescence_W_m2_by_slice: np.ndarray
    excited_fraction_before_pulse: np.ndarray
    cycles: int
    residual: float


def periodic_pulse_heat(material: YbLuAGMaterial, time_s, pump_in_W_m2,
                        signal_in_W_m2, thickness_m: float, steps: int,
                        repetition_rate_Hz: float, *,
                        fluorescence_quantum_yield: float,
                        mean_fluorescence_wavelength_nm: float,
                        tolerance=1e-8, max_cycles=1000) -> PeriodicHeatResult:
    """Cycle-average local lattice heat after a periodic two-manifold state.

    The fluorescence inputs represent photons escaping the sample. The result
    omits reabsorbed fluorescence and coating loss; values must be measured or
    otherwise explicitly justified for the sample.
    """
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if not np.isfinite(repetition_rate_Hz) or repetition_rate_Hz <= 0:
        raise ValueError("repetition_rate_Hz must be finite and positive")
    period = 1.0 / repetition_rate_Hz
    duration = time[-1] - time[0]
    if duration >= period:
        raise ValueError("pulse time window must be shorter than the repetition period")
    if (not np.isfinite(fluorescence_quantum_yield) or
            not 0 <= fluorescence_quantum_yield <= 1):
        raise ValueError("fluorescence_quantum_yield must be in [0, 1]")
    if not np.isfinite(mean_fluorescence_wavelength_nm) or mean_fluorescence_wavelength_nm <= 0:
        raise ValueError("mean_fluorescence_wavelength_nm must be positive")
    if (not np.isfinite(tolerance) or tolerance <= 0 or isinstance(max_cycles, bool)
            or not isinstance(max_cycles, int) or max_cycles < 1):
        raise ValueError("invalid periodic convergence settings")
    pump = np.asarray(pump_in_W_m2, dtype=float)
    if pump.ndim < 1 or pump.shape[0] != len(time):
        raise ValueError("pump shape must begin with time axis")
    beta = np.zeros((steps, *pump.shape[1:]), dtype=float)
    dark_time = period - duration
    for cycle in range(1, max_cycles + 1):
        pulse = propagate_pulse(material, time, pump, signal_in_W_m2,
                                thickness_m, steps, initial_excited_fraction=beta)
        following = pulse.final_excited_fraction_by_slice * np.exp(-dark_time / material.lifetime_s)
        error = float(np.max(np.abs(following - beta)))
        beta = following
        if error <= tolerance:
            break
    else:
        raise RuntimeError("Yb periodic pulse state did not converge")
    # Re-evaluate the final cycle from the converged initial population.
    pulse = propagate_pulse(material, time, pump, signal_in_W_m2,
                            thickness_m, steps, initial_excited_fraction=beta)
    beta_dark_integral = pulse.final_excited_fraction_by_slice * material.lifetime_s * (
        -np.expm1(-dark_time / material.lifetime_s))
    photon_energy = H * C / (mean_fluorescence_wavelength_nm * 1e-9)
    fluorescence = (material.number_density_m3 / material.lifetime_s *
                    fluorescence_quantum_yield * photon_energy *
                    (pulse.excited_fraction_time_integral_s_by_slice + beta_dark_integral) *
                    (thickness_m / steps) * repetition_rate_Hz)
    pump_power = pulse.absorbed_pump_fluence_J_m2_by_slice * repetition_rate_Hz
    signal_power = pulse.signal_fluence_change_J_m2_by_slice * repetition_rate_Hz
    heat = (pump_power - signal_power - fluorescence) / (thickness_m / steps)
    return PeriodicHeatResult(heat, pump_power, signal_power, fluorescence,
                              beta, cycle, error)


def periodic_pump_state(material: YbLuAGMaterial, time_s, pump_in_W_m2,
                        thickness_m: float, steps: int, repetition_rate_Hz: float,
                        *, tolerance=1e-8, max_cycles=1000):
    """Converge pump-only pulse and exact dark decay across the remaining period."""
    time = np.asarray(time_s, dtype=float)
    if not np.isfinite(repetition_rate_Hz) or repetition_rate_Hz <= 0:
        raise ValueError("repetition_rate_Hz must be finite and positive")
    if time.ndim != 1 or len(time) < 2 or np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must be strictly increasing")
    period = 1.0 / repetition_rate_Hz
    duration = time[-1] - time[0]
    if duration >= period:
        raise ValueError("pulse time window must be shorter than the repetition period")
    if not np.isfinite(tolerance) or tolerance <= 0 or isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles < 1:
        raise ValueError("invalid periodic convergence settings")
    pump = np.asarray(pump_in_W_m2, dtype=float)
    if pump.ndim < 1 or pump.shape[0] != len(time):
        raise ValueError("pump shape must begin with time axis")
    beta = np.zeros((steps, *pump.shape[1:]), dtype=float)
    zero_signal = np.zeros_like(pump)
    for cycle in range(1, max_cycles + 1):
        result = propagate_pulse(material, time, pump, zero_signal,
                                 thickness_m, steps, initial_excited_fraction=beta)
        following = result.final_excited_fraction_by_slice * np.exp(
            -(period - duration) / material.lifetime_s)
        error = float(np.max(abs(following - beta)))
        beta = following
        if error <= tolerance:
            return beta, cycle, error
    raise RuntimeError("Yb periodic pump state did not converge")
