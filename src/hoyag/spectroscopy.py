"""Local 295 K Ho:YAG pump-spectrum surrogate and explicit FFT convention.

The Gaussian peaks use the published positions, peak cross sections and FWHM,
not complete digitized line shapes. Scalar spectral overlap is an approximation
at finite optical depth; the weak, fixed-population spectral operator below
preserves the spectrum instead. Neither describes spectral-hole burning.
"""
from __future__ import annotations
import numpy as np
from .pump_source import optical_frequencies_hz, normalize_spectral_weights

C0 = 299_792_458.0
HOYAG_ABSORPTION_PEAKS_295K = (
    (1907.3, 1.259e-24, 4.5),
    (1928.3, 5.77e-25, 2.5),
    (1932.5, 6.88e-25, 3.0),
    (1973.5, 1.99e-25, 2.4),
)


def pump_absorption_cross_section_295K(wavelength_m):
    """Gaussian-peak cross-section surrogate, SI m², near the pump lines."""
    lam=np.asarray(wavelength_m,float)
    if np.any(~np.isfinite(lam)) or np.any(lam<=0):
        raise ValueError('wavelength must be finite and positive')
    nm=lam*1e9
    result=np.zeros_like(nm)
    for center,peak,width in HOYAG_ABSORPTION_PEAKS_295K:
        with np.errstate(under='ignore'):
            result+=peak*np.exp(-4*np.log(2)*((nm-center)/width)**2)
    return float(result) if result.ndim==0 else result


def temporal_spectral_weights(field,time):
    """Return NumPy FFT-bin frequencies and energy weights, not optical nu."""
    arr=np.asarray(field,complex)
    if arr.ndim<1 or arr.shape[0]!=time.nt or np.any(~np.isfinite(arr)):
        raise ValueError('finite field with first axis time.nt required')
    power=abs(np.fft.fft(arr,axis=0))**2
    if arr.ndim>1:
        power=power.sum(axis=tuple(range(1,arr.ndim)))
    return time.frequency_hz,power


def effective_pump_absorption_cross_section_295K(field,time,center_wavelength_m):
    """Energy-spectrum-weighted sigma with exp(+ikz-iwt) optical convention.

    Physical nu = nu0 - FFT-bin frequency. Reusing this scalar and h*nu0 in
    rate equations does NOT reproduce spectral reshaping through thick media.
    """
    _,weight=temporal_spectral_weights(field,time)
    nu,weight=normalize_spectral_weights(optical_frequencies_hz(time,center_wavelength_m),weight)
    return float(weight@pump_absorption_cross_section_295K(C0/nu))


def spectral_attenuation_diagnostic(field,time,center_wavelength_m,column_density_m2):
    """Unexcited-medium spectral vs effective-sigma transmission comparison."""
    if not np.isfinite(column_density_m2) or column_density_m2<0:
        raise ValueError('column density must be finite and nonnegative')
    _,weight=temporal_spectral_weights(field,time)
    nu,weight=normalize_spectral_weights(optical_frequencies_hz(time,center_wavelength_m),weight)
    sigma=pump_absorption_cross_section_295K(C0/nu)
    spectral=float(weight@np.exp(-column_density_m2*sigma))
    effective=float(np.exp(-column_density_m2*(weight@sigma)))
    return {'column_density_m2':float(column_density_m2),'spectral_transmission':spectral,
            'scalar_transmission':effective,'relative_transmission_bias':(effective-spectral)/max(spectral,1e-300),
            'scope':'ground-state absorption only; saturated spectral reshaping is not resolved'}


def apply_frozen_spectral_absorption(field,time,center_wavelength_m,column_density_m2):
    """Linear ground-state attenuation of each spectral component, preserving phase.

    The column density is a nonnegative scalar in m^-2. No pump saturation,
    population update, diffraction or KK index phase is included in this helper.
    """
    if not np.isfinite(column_density_m2) or column_density_m2<0:
        raise ValueError('column density must be finite and nonnegative')
    arr=np.asarray(field,complex)
    _,w=temporal_spectral_weights(arr,time)
    nu=optical_frequencies_hz(time,center_wavelength_m)
    normalize_spectral_weights(nu,w)
    valid=nu>0
    trans=np.ones(time.nt)
    trans[valid]=np.exp(-.5*column_density_m2*pump_absorption_cross_section_295K(C0/nu[valid]))
    return np.fft.ifft(np.fft.fft(arr,axis=0)*trans.reshape((time.nt,)+(1,)*(arr.ndim-1)),axis=0)
