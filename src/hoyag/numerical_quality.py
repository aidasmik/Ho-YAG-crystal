"""Explicit validity diagnostics, separate from an algebraic solver residual."""
import warnings
import numpy as np


class ModelValidityWarning(RuntimeWarning):
    pass


def check_spectral_scalar_limit(diagnostic, *, tolerance=.01, policy='warn'):
    if policy not in ('warn','error','report'):
        raise ValueError('spectral policy must be warn, error or report')
    if not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError('spectral tolerance must be positive')
    bias=abs(diagnostic['relative_transmission_bias'])
    if bias > tolerance:
        message=(f'Effective-sigma absorption has {bias:.2%} ground-state transmission '
                 f'bias over the specified column (> {tolerance:.2%}); use resolved '
                 'spectral attenuation or mark this approximation explicitly.')
        if policy=='error': raise ValueError(message)
        if policy=='warn': warnings.warn(message,ModelValidityWarning,stacklevel=2)
    return bias <= tolerance


def cavity_sampling_diagnostic(grid, cavity, *, radius_m=None):
    """Mirror quadratic-phase sampling over a stated radius, not a mode proof."""
    radius=cavity.disk_diameter_m/2 if radius_m is None else float(radius_m)
    if not np.isfinite(radius) or radius<=0:
        raise ValueError('diagnostic radius must be positive')
    step=4*np.pi*radius*max(grid.dx,grid.dy)/(cavity.wavelength_m*cavity.output_mirror_radius_m)
    return {'radius_m':radius,'worst_mirror_phase_step_rad':float(step),
            'nyquist_satisfied_over_radius':bool(step <= np.pi),
            'mesh_convergence_verified':False,
            'note':'Central modes may be resolved even when the whole aperture is not. '
                   'An eigenpair residual does not replace grid/window/mode-count refinement.'}
