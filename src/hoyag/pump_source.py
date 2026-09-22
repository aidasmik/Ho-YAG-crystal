"""Pump spectrum shared by modal, thermal and coupled-cavity models.

Physical carrier convention: A(t) exp(+i k0 z - i omega0 t).
For NumPy fft(A), optical frequency = nu0 - fftfreq. Quadratic GDD is even
in detuning and its existing sign is unchanged. The Gaussian model's duration
is the actual intensity FWHM; an explicit bandwidth is required for a chirped
pulse. All averaging weights are spectral ENERGY, not field amplitude.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss

C0 = 299_792_458.0
GAUSSIAN_TBP = 2 * np.log(2) / np.pi


def positive(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def optical_frequencies_hz(time, carrier_wavelength_m):
    positive(carrier_wavelength_m, 'carrier wavelength')
    return C0 / carrier_wavelength_m - time.frequency_hz


def normalize_spectral_weights(frequencies_hz, energy_weights):
    nu = np.asarray(frequencies_hz, float)
    w = np.asarray(energy_weights, float)
    if nu.ndim != 1 or w.shape != nu.shape or len(nu) == 0:
        raise ValueError('one-dimensional frequencies and energy weights must match')
    if np.any(~np.isfinite(nu)) or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError('finite frequencies and nonnegative energy weights required')
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError('spectrum has no finite energy')
    w = w / total
    if w[nu <= 0].sum() > 1e-12:
        raise ValueError('material spectral model requires positive optical frequency')
    valid = nu > 0
    return nu[valid].copy(), w[valid] / w[valid].sum()


@dataclass(frozen=True)
class PumpSource:
    center_wavelength_m: float = 1.9077e-6
    duration_fwhm_s: float = 10e-12
    bandwidth_fwhm_hz: float | None = None
    gdd_s2: float = 0.0
    effective_absorption_override_m2: float | None = None
    override_note: str | None = None
    spectrum_frequency_hz: tuple[float, ...] | None = None
    spectrum_energy_weights: tuple[float, ...] | None = None

    def __post_init__(self):
        positive(self.center_wavelength_m, 'pump wavelength')
        positive(self.duration_fwhm_s, 'pump intensity FWHM')
        if not np.isfinite(self.gdd_s2):
            raise ValueError('GDD must be finite')
        custom = self.spectrum_frequency_hz is not None or self.spectrum_energy_weights is not None
        if custom:
            if self.spectrum_frequency_hz is None or self.spectrum_energy_weights is None:
                raise ValueError('both custom spectrum frequencies and weights are required')
            if self.bandwidth_fwhm_hz is not None or self.gdd_s2 != 0:
                raise ValueError('custom spectrum supplies its bandwidth; do not mix Gaussian fields')
            nu, w = normalize_spectral_weights(self.spectrum_frequency_hz, self.spectrum_energy_weights)
            object.__setattr__(self, 'spectrum_frequency_hz', tuple(nu))
            object.__setattr__(self, 'spectrum_energy_weights', tuple(w))
        else:
            if self.gdd_s2 != 0 and self.bandwidth_fwhm_hz is None:
                raise ValueError('chirped duration does not uniquely determine bandwidth; provide bandwidth_fwhm_hz')
            if self.bandwidth_fwhm_hz is not None:
                bw = positive(self.bandwidth_fwhm_hz, 'spectral intensity bandwidth')
                tl = GAUSSIAN_TBP / bw
                expected = tl * np.sqrt(1 + (4*np.log(2)*self.gdd_s2/tl**2)**2)
                if not np.isclose(expected, self.duration_fwhm_s, rtol=2e-6, atol=0):
                    raise ValueError('Gaussian duration, bandwidth and GDD are inconsistent')
        if self.effective_absorption_override_m2 is not None:
            if not np.isfinite(self.effective_absorption_override_m2) or self.effective_absorption_override_m2 < 0:
                raise ValueError('cross-section override must be finite and nonnegative')
            if not isinstance(self.override_note, str) or not self.override_note.strip():
                raise ValueError('an effective cross-section override requires a provenance note')

    def energy_spectrum(self):
        if self.spectrum_frequency_hz is not None:
            return np.array(self.spectrum_frequency_hz), np.array(self.spectrum_energy_weights)
        bw = self.bandwidth_fwhm_hz or GAUSSIAN_TBP/self.duration_fwhm_s
        nodes, quad = leggauss(512)
        detuning = 5 * bw * nodes
        nu = C0/self.center_wavelength_m + detuning
        weight = quad*np.exp(-4*np.log(2)*(detuning/bw)**2)
        return normalize_spectral_weights(nu, weight)

    def effective_absorption_m2(self):
        if self.effective_absorption_override_m2 is not None:
            return float(self.effective_absorption_override_m2)
        from .spectroscopy import pump_absorption_cross_section_295K
        nu, weights = self.energy_spectrum()
        return float(weights @ pump_absorption_cross_section_295K(C0/nu))

    def attenuation_diagnostic(self, column_density_m2):
        """Compare scalar and spectral unexcited transmission, not saturation."""
        from .spectroscopy import pump_absorption_cross_section_295K
        if not np.isfinite(column_density_m2) or column_density_m2 < 0:
            raise ValueError('column density must be finite and nonnegative')
        nu, w = self.energy_spectrum()
        sigma = pump_absorption_cross_section_295K(C0/nu)
        spectral = float(w @ np.exp(-sigma*column_density_m2))
        scalar = float(np.exp(-self.effective_absorption_m2()*column_density_m2))
        return {'column_density_m2':float(column_density_m2),
                'spectral_transmission':spectral, 'scalar_transmission':scalar,
                'relative_transmission_bias':float((scalar-spectral)/max(spectral,1e-300)),
                'scope':'ground-state absorption only; does not certify saturated broadband dynamics'}

    def summary(self):
        nu, w = self.energy_spectrum()
        return {'center_wavelength_m':self.center_wavelength_m,
                'duration_fwhm_s':self.duration_fwhm_s,
                'bandwidth_fwhm_hz':self.bandwidth_fwhm_hz or (None if self.spectrum_frequency_hz is not None else GAUSSIAN_TBP/self.duration_fwhm_s),
                'gdd_s2':self.gdd_s2, 'effective_absorption_m2':self.effective_absorption_m2(),
                'spectral_model':'supplied spectral energy quadrature' if self.spectrum_frequency_hz is not None else 'Gaussian intensity spectrum',
                'mean_frequency_hz':float(w@nu),
                'effective_absorption_override_m2':self.effective_absorption_override_m2,
                'override_note':self.override_note,
                'frequency_convention':'physical exp(+ikz-iwt); nu = nu0 - numpy.fft.fftfreq'}


def resolve_pump_source(wavelength_m, duration_fwhm_s, *, source=None, absorption_override_m2=None):
    """Bind metadata and material absorption to the same physical source."""
    positive(wavelength_m, 'pump wavelength'); positive(duration_fwhm_s, 'pump duration')
    if source is not None and absorption_override_m2 is not None:
        raise ValueError('put an override in PumpSource or use absorption_override_m2, not both')
    if source is None:
        source = PumpSource(wavelength_m, duration_fwhm_s,
                effective_absorption_override_m2=absorption_override_m2,
                override_note=None if absorption_override_m2 is None else 'Explicit legacy API effective-cross-section override')
    if not isinstance(source, PumpSource):
        raise TypeError('source must be PumpSource')
    if not np.isclose(source.center_wavelength_m, wavelength_m, rtol=1e-10, atol=0):
        raise ValueError('pump-source and material carrier wavelengths differ')
    if not np.isclose(source.duration_fwhm_s, duration_fwhm_s, rtol=1e-10, atol=0):
        raise ValueError('pump-source and simulated pulse durations differ')
    return source
