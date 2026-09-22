"""One-time deterministic source migration, executed on the audit branch in CI.
This file and its publishing workflow are removed from the final source tree.
"""
from __future__ import annotations
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path):return (ROOT/path).read_text()
def write(path,text):
    if path.endswith('.py'):ast.parse(text,filename=path)
    (ROOT/path).write_text(text)
def once(text,old,new):
    count=text.count(old)
    if count!=1:raise RuntimeError(f'expected one replacement, got {count}: {old[:120]!r}')
    return text.replace(old,new,1)
def function(text,name,new,cls=None):
    scope=ast.parse(text).body
    if cls is not None:scope=next(x for x in scope if isinstance(x,ast.ClassDef) and x.name==cls).body
    node=next(x for x in scope if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name)
    lines=text.splitlines(keepends=True);indent=' '*node.col_offset
    lines[node.lineno-1:node.end_lineno]=['\n'.join(indent+line if line else '' for line in new.rstrip().splitlines())+'\n']
    return ''.join(lines)


def parameters_and_populations():
    path='src/hoyag/populations.py';s=read(path)
    s=once(s,'from .propagation import Grid2D','from .population_state import validate_populations\nfrom .propagation import Grid2D')
    s=function(s,'__post_init__','''def __post_init__(self) -> None:
    for name in ('N_total_m3','tau5_s','tau6_s','tau7_s','pump_wavelength_m','laser_wavelength_m'):
        value=getattr(self,name)
        if not np.isscalar(value) or not np.isfinite(value) or value<=0:
            raise ValueError(f'{name} must be finite and positive')
    for name in ('M56_s1','M67_s1','M78_s1','k75_m3_s','k76_m3_s','C57_m3_s','C67_m3_s',
                 'sigma_abs_pump_m2','sigma_em_pump_m2','sigma_abs_laser_m2','sigma_em_laser_m2'):
        value=getattr(self,name)
        if not np.isscalar(value) or not np.isfinite(value) or value<0:
            raise ValueError(f'{name} must be finite and nonnegative')
    for name in ('beta56','beta57','beta58','beta67','beta68','beta78'):
        value=getattr(self,name)
        if not np.isscalar(value) or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError(f'{name} must be finite and in [0,1]')
    if not np.isclose(self.beta56+self.beta57+self.beta58,1.,rtol=0,atol=1e-12):
        raise ValueError('I5 branching ratios must sum to one')
    if not np.isclose(self.beta67+self.beta68,1.,rtol=0,atol=1e-12):
        raise ValueError('I6 branching ratios must sum to one')
    if not np.isclose(self.beta78,1.,rtol=0,atol=1e-12):
        raise ValueError('this four-manifold RHS requires beta78=1')''',cls='HoYAGFourLevelParams')
    s=function(s,'stimulated_rates_from_intensity','''def stimulated_rates_from_intensity(intensity_W_m2, wavelength_m, sigma_abs_m2, sigma_em_m2):
    """Local nonnegative stimulated rates in s^-1 from physical intensity."""
    if not np.isfinite(wavelength_m) or wavelength_m<=0:
        raise ValueError('wavelength must be finite and positive')
    for sigma in (sigma_abs_m2,sigma_em_m2):
        if not np.isfinite(sigma) or sigma<0:
            raise ValueError('cross sections must be finite and nonnegative')
    intensity=np.asarray(intensity_W_m2,float)
    if np.any(~np.isfinite(intensity)) or np.any(intensity<0):
        raise ValueError('intensity must be finite and nonnegative')
    flux=intensity/(H*C0/wavelength_m)
    return sigma_abs_m2*flux,sigma_em_m2*flux''')
    s=function(s,'_physicalize_populations','''def _physicalize_populations(state, params):
    n=np.asarray(state,float)
    if n.ndim<1:
        raise FloatingPointError('population axis is missing')
    density=np.full(n.shape[1:],params.N_total_m3)
    return validate_populations(n,density,error_type=FloatingPointError)''')
    s=s.replace('if pulse_energy_J <= 0:', 'if not np.isfinite(pulse_energy_J) or pulse_energy_J <= 0:')
    s=s.replace('if np.any(intensity < 0):','if np.any(~np.isfinite(intensity)) or np.any(intensity < 0):')
    s=once(s,'wavelength_m = wavelength_m or params.pump_wavelength_m','wavelength_m = params.pump_wavelength_m if wavelength_m is None else wavelength_m')
    for marker in ['    photon_energy = H * C0 / params.pump_wavelength_m\n    out = np.empty_like(arr)',
                   '    for i in range(time.nt - 1):\n        state = _rk4_population_step(']:
        s=once(s,marker,'    state = _physicalize_populations(state, params)\n\n'+marker)
    s=once(s,'np.empty((nz, 4, grid.ny, grid.nx), dtype=float)','np.empty((4, nz, grid.ny, grid.nx), dtype=float)')
    s=once(s,'full[iz] = step.final_populations','full[:, iz] = step.final_populations')
    s=once(s,'class PumpPropagationResult:\n','class PumpPropagationResult:\n    """Population output order is (manifold,z,y,x), not the legacy z-major order."""\n    population_axes = ("manifold", "z", "y", "x")\n')
    write(path,s)


def inhomogeneity():
    path='src/hoyag/inhomogeneity.py';s=read(path)
    s=once(s,'from dataclasses import dataclass','from dataclasses import dataclass\nfrom .population_state import validate_populations\nfrom .density_statistics import bounded_mean_rms, density_statistics')
    s=function(s,'_physicalize_to_density','''def _physicalize_to_density(state,density_m3):
    return validate_populations(state,density_m3,error_type=FloatingPointError)''')
    old='''    values = mean_density_m3 * (1.0 + relative_rms * smooth)
    values = np.maximum(values, minimum_fraction * mean_density_m3)

    if mean_density_m3 > 0:
        values *= mean_density_m3 / np.mean(values)
        values = np.maximum(values, minimum_fraction * mean_density_m3)

    return HoDensityField(values, length_m)'''
    s=once(s,old,'''    values = bounded_mean_rms(smooth, mean_density_m3, relative_rms, minimum_fraction)
    return HoDensityField(values, length_m)''')
    s=s.replace('    if relative_rms < 0:', '    if not np.isfinite(relative_rms) or relative_rms < 0:')
    s=s.replace('    if mean_density_m3 < 0:', '    if not np.isfinite(mean_density_m3) or mean_density_m3 < 0:')
    s=s.replace('    if minimum_fraction < 0:', '    if not np.isfinite(minimum_fraction) or not 0 <= minimum_fraction <= 1:')
    s=once(s,'np.empty((density.nz, 4, grid.ny, grid.nx), dtype=float)','np.empty((4, density.nz, grid.ny, grid.nx), dtype=float)')
    s=once(s,'full[iz] = step.final_populations','full[:, iz] = step.final_populations')
    s=once(s,'class InhomogeneousPumpResult:\n','class InhomogeneousPumpResult:\n    """Population output order is (manifold,z,y,x)."""\n    population_axes = ("manifold", "z", "y", "x")\n')
    position=s.index('\ndef uniform_ho_density_field(')
    s=s[:position]+'\ndef ho_density_statistics(density: HoDensityField):\n    """Achieved density statistics, not merely generator input settings."""\n    return density_statistics(density.values_m3)\n\n'+s[position:]
    write(path,s)
    path='src/hoyag/pulse_train.py';s=read(path)
    s=once(s,'from dataclasses import dataclass','from dataclasses import dataclass\nfrom .population_state import validate_populations')
    old='''        total = np.sum(state, axis=0)
        if np.any(total <= 0):
            raise ValueError("initial populations contain zero total density")
        state *= (params.N_total_m3 / total)[None, ...]'''
    s=once(s,old,'''        state = validate_populations(state, np.full(expected[1:], params.N_total_m3))''')
    s=once(s,'class PulseTrainResult:\n','class PulseTrainResult:\n    population_axes = ("manifold", "z", "y", "x")\n')
    write(path,s)


def propagation():
    path='src/hoyag/propagation.py';s=read(path)
    s=function(s,'__post_init__','''def __post_init__(self) -> None:
    for name in ('nx','ny'):
        n=getattr(self,name)
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<2:
            raise ValueError(f'{name} must be an integer >=2')
    if not np.isfinite(self.dx) or not np.isfinite(self.dy) or self.dx<=0 or self.dy<=0:
        raise ValueError('grid spacings must be finite and positive')''',cls='Grid2D')
    s=once(s,'    if arr.shape != grid.shape:', '    if np.any(~np.isfinite(arr)):\n        raise ValueError("field must be finite")\n    if arr.shape != grid.shape:')
    s=function(s,'angular_spectrum_propagate','''def angular_spectrum_propagate(field, grid, wavelength_m, distance_m, *, refractive_index=1., bandlimit=True):
    """Scalar angular-spectrum propagation; bandlimit removes evanescent bins.

    This is not a distance-dependent anti-alias filter. The FFT window is periodic.
    Backward propagation of evanescent components is ill-conditioned and rejected.
    """
    arr=_check_field(field,grid)
    transfer=angular_spectrum_transfer(grid,wavelength_m,distance_m,
                                      refractive_index=refractive_index,bandlimit=bandlimit)
    if distance_m==0:
        return arr.copy()
    return np.fft.ifft2(np.fft.fft2(arr)*transfer)''')
    s+='''\n\ndef angular_spectrum_transfer(grid, wavelength_m, distance_m, *, refractive_index=1., bandlimit=True):
    """Build masked factors without evaluating any exponentially growing bin."""
    if not np.isfinite(wavelength_m) or wavelength_m<=0 or not np.isfinite(refractive_index) or refractive_index<=0:
        raise ValueError('wavelength and refractive index must be finite and positive')
    if not np.isfinite(distance_m):
        raise ValueError('distance must be finite')
    if distance_m==0:
        return np.ones(grid.shape,complex)
    fx,fy=np.meshgrid(grid.fx,grid.fy,indexing='xy')
    k=2*np.pi*refractive_index/wavelength_m
    transverse=(2*np.pi)**2*(fx*fx+fy*fy)
    propagating=transverse<=k*k
    transfer=np.zeros(grid.shape,complex)
    transfer[propagating]=np.exp(1j*np.sqrt(k*k-transverse[propagating])*distance_m)
    if not bandlimit:
        if distance_m<0 and np.any(~propagating):
            raise ValueError('backward evanescent continuation is unsupported; use bandlimit=True')
        with np.errstate(under='ignore'):
            transfer[~propagating]=np.exp(-np.sqrt(transverse[~propagating]-k*k)*distance_m)
    return transfer
'''
    s=once(s,'        if size_m <= 0:', '        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<2:\n            raise ValueError("n must be an integer >=2")\n        if not np.isfinite(size_m) or size_m <= 0:')
    write(path,s)
    path='src/hoyag/temporal.py';s=read(path)
    s=once(s,'from .propagation import Grid2D','from .propagation import Grid2D, angular_spectrum_transfer\nfrom .pump_source import optical_frequencies_hz')
    start=s.index('    fx, fy = np.meshgrid(',s.index('def propagate_spatiotemporal('))
    stop=s.index('    spectrum_xy = ',start)
    s=s[:start]+'''    hxy = angular_spectrum_transfer(grid, wavelength_m, distance_m,
                                    refractive_index=refractive_index, bandlimit=bandlimit)

'''+s[stop:]
    s=function(s,'__post_init__','''def __post_init__(self) -> None:
    if isinstance(self.nt,(bool,np.bool_)) or not isinstance(self.nt,(int,np.integer)) or self.nt<4:
        raise ValueError('nt must be an integer >=4')
    if not np.isfinite(self.dt) or self.dt<=0:
        raise ValueError('dt must be finite and positive')''',cls='TimeGrid')
    s=once(s,'        if window_s <= 0:', '        if isinstance(nt,(bool,np.bool_)) or not isinstance(nt,(int,np.integer)) or nt<4:\n            raise ValueError("nt must be an integer >=4")\n        if not np.isfinite(window_s) or window_s <= 0:')
    for name in ('_temporal','_spatiotemporal'):
        node=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==name)
        part=''.join(s.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
        new=part.replace('    if arr.shape','    if np.any(~np.isfinite(arr)):\n        raise ValueError("field must be finite")\n    if arr.shape',1)
        s=s.replace(part,new,1)
    write(path,s)


def signal():
    path='src/hoyag/signal.py';s=read(path)
    s=once(s,'from dataclasses import dataclass','from dataclasses import dataclass\nfrom .population_state import PopulationField, validate_populations')
    s=function(s,'validate_population_field','''def validate_population_field(populations_by_slice, density, grid):
    """Canonical manifold-first field; reject wrong local totals, never guess axes."""
    density.validate_grid(grid)
    if isinstance(populations_by_slice, PopulationField):
        if not np.array_equal(populations_by_slice.density_m3,density.values_m3):
            raise ValueError('PopulationField and geometry densities disagree')
        populations_by_slice=populations_by_slice.values_m3
    return validate_populations(populations_by_slice,density.values_m3)''')
    s=function(s,'_physicalize_signal_populations','''def _physicalize_signal_populations(state,density_m3):
    return validate_populations(state,density_m3,error_type=FloatingPointError)''')
    a=s.index('    # Reference gain from the same initial populations, without depletion.')
    b=s.index('    energy_changes = ',a)
    s=s[:a]+'''    # Use the SAME complex pulse, phase and GVD, freezing populations only.
    # Taking sqrt(fluence) would erase vortex phase and space-time correlations.
    small_ref = propagate_structured_signal_frozen(
        signal, grid, time, pulse_energy_J, state, density, params,
        refractive_index=refractive_index, beta2_s2_per_m=beta2_s2_per_m,
        include_passive_propagation=include_passive_propagation,
    )

'''+s[b:]
    s=once(s,'small_signal_power_gain_reference=float(small_ref.power_gain)', 'small_signal_power_gain_reference=float(small_ref.energy_gain)')
    s+='''\n\n@dataclass
class FrozenSignalResult:
    field_out: np.ndarray
    input_energy_J: float
    output_energy_J: float
    energy_gain: float


def propagate_structured_signal_frozen(field, grid, time, pulse_energy_J,
        populations_by_slice, density, params=None, *,
        refractive_index=DEFAULT_SIGNAL_REFRACTIVE_INDEX,
        beta2_s2_per_m=DEFAULT_SIGNAL_BETA2_S2_PER_M, include_passive_propagation=True):
    """Full complex E(t,y,x) weak-signal reference with frozen populations.

    Preserves arbitrary spatial phase, temporal phase and space-time correlations.
    Identical diffraction, GVD, slice geometry and normalization to the saturated
    path. Only the stimulated population update is disabled.
    """
    params=params or HoYAGFourLevelParams()
    state=validate_population_field(populations_by_slice,density,grid)
    signal=scale_pulse_to_energy(field,grid,time,pulse_energy_J)
    ein=spatiotemporal_energy(signal,grid,time)
    for iz in range(density.nz):
        if include_passive_propagation:
            signal=propagate_spatiotemporal(signal,grid,time,params.laser_wavelength_m,density.dz_m/2,
                         refractive_index=refractive_index,beta2_s2_per_m=beta2_s2_per_m)
        gain=small_signal_gain_coefficient_m1(state[:,iz],params)
        signal*=np.exp(.5*gain[None]*density.dz_m)
        if include_passive_propagation:
            signal=propagate_spatiotemporal(signal,grid,time,params.laser_wavelength_m,density.dz_m/2,
                         refractive_index=refractive_index,beta2_s2_per_m=beta2_s2_per_m)
    eout=spatiotemporal_energy(signal,grid,time)
    return FrozenSignalResult(signal,float(ein),float(eout),float(eout/ein))
'''
    write(path,s)


def pump_binding():
    path='src/hoyag/resonator.py';s=read(path)
    s=once(s,'from .populations import ', 'from .pump_source import PumpSource, resolve_pump_source\nfrom .numerical_quality import check_spectral_scalar_limit\nfrom .populations import ')
    s=once(s,'mode_labels: tuple[str, ...] | None = None):','mode_labels: tuple[str, ...] | None = None,\n                 pump_source: PumpSource | None = None):')
    old='        self.sa = self.params.sigma_abs_pump_m2 if pump_absorption_m2 is None else float(pump_absorption_m2)'
    s=once(s,old,'''        if pump_source is not None and pump_absorption_m2 is not None:
            raise ValueError('specify a pump source or a scalar override, not both')
        self.pump_source = None if pump_source is None else resolve_pump_source(
            self.params.pump_wavelength_m, pump_source.duration_fwhm_s, source=pump_source)
        self.sa = (self.pump_source.effective_absorption_m2() if self.pump_source is not None
                   else (self.params.sigma_abs_pump_m2 if pump_absorption_m2 is None else float(pump_absorption_m2)))''')
    s=once(s,'        _positive(self.sa, "pump_absorption_m2")','        if not np.isfinite(self.sa) or self.sa < 0:\n            raise ValueError("pump absorption must be finite and nonnegative")')
    marker='        _positive(pump_fwhm_s, "pump_fwhm_s")'
    s=once(s,marker,marker+'''
        if self.pump_source is not None:
            resolve_pump_source(self.params.pump_wavelength_m,pump_fwhm_s,source=self.pump_source)
            check_spectral_scalar_limit(self.pump_source.attenuation_diagnostic(
                float(np.max(np.sum(self.density,axis=0)))*self.dz*self.pump_passes))''')
    old='"pump_passes":self.pump_passes,"peak_incident_fluence_over_Fsat":ratio,'
    s=once(s,old,old+'\n                 "pump_source":self.pump_source.summary() if self.pump_source is not None else {"effective_absorption_m2":self.sa,"source":"explicit low-level cross section; no inferred spectrum"},')
    s=once(s,'pump_absorption_m2: float = 1.223454786e-24,','pump_absorption_m2: float | None = None,\n                 pump_duration_fwhm_s: float = 10e-12, pump_source: PumpSource | None = None,')
    old='    model=ModalThinDiskLaser(cavity,area,density,u,pump,params=params,\n                  pump_absorption_m2=pump_absorption_m2,'
    s=once(s,old,'''    source=resolve_pump_source(params.pump_wavelength_m,pump_duration_fwhm_s,
                              source=pump_source,absorption_override_m2=pump_absorption_m2)
    model=ModalThinDiskLaser(cavity,area,density,u,pump,params=params,
                  pump_source=source,''')
    write(path,s)
    path='src/hoyag/thermal_resonator.py';s=read(path)
    s=once(s,'from .populations import ', 'from .pump_source import resolve_pump_source\nfrom .populations import ')
    s=once(s,'params=None, density_m3=None, pump_absorption_m2=1.223454786e-24,','params=None, density_m3=None, pump_absorption_m2=None,\n                                pump_duration_fwhm_s=10e-12, pump_source=None,')
    old='    return ModalThinDiskLaser(cavity,mesh.face_areas_m2.ravel(),density.reshape(mesh.nz,-1),\n        modes,pump,params=p,pump_absorption_m2=pump_absorption_m2,'
    s=once(s,old,'''    source=resolve_pump_source(p.pump_wavelength_m,pump_duration_fwhm_s,
                              source=pump_source,absorption_override_m2=pump_absorption_m2)
    return ModalThinDiskLaser(cavity,mesh.face_areas_m2.ravel(),density.reshape(mesh.nz,-1),
        modes,pump,params=p,pump_source=source,''')
    s=once(s,'initial_fractions=None, initial_log_photons=None, progress=None):','initial_fractions=None, initial_log_photons=None, progress=None,\n                          pump_source=None, pump_absorption_m2=None):')
    old='model=modal_laser_on_thermal_mesh(mesh,cavity,waist_m=w,params=params,pump_waist_m=pump_waist_m)'
    s=once(s,old,'''model=modal_laser_on_thermal_mesh(mesh,cavity,waist_m=w,params=params,pump_waist_m=pump_waist_m,
                 pump_duration_fwhm_s=pump_duration_fwhm_s,pump_source=pump_source,
                 pump_absorption_m2=pump_absorption_m2)''')
    write(path,s)
    path='src/hoyag/coupled_resonator.py';s=read(path)
    s=once(s,'from .populations import ', 'from .pump_source import resolve_pump_source\nfrom .numerical_quality import cavity_sampling_diagnostic\nfrom .populations import ')
    s=once(s,'spectroscopy=None,progress=None):','spectroscopy=None,progress=None,pump_source=None,pump_absorption_m2=None):')
    marker='    settings=settings or HotCavitySettings();p=params or HoYAGFourLevelParams()'
    s=once(s,marker,marker+'''
    source=resolve_pump_source(p.pump_wavelength_m,pump_duration_s,
                              source=pump_source,absorption_override_m2=pump_absorption_m2)''')
    s=once(s,'np.asarray(profiles),pump,params=p,pump_absorption_m2=1.223454786e-24,', 'np.asarray(profiles),pump,params=p,pump_source=source,')
    marker="'mode_count':mode_count,'pump_energy_J':pump_energy_J,'repetition_rate_Hz':repetition_rate_Hz,"
    s=once(s,marker,marker+'''
          'pump_source':source.summary(),'optical_sampling':cavity_sampling_diagnostic(grid,cavity),
          'mesh_convergence_verified':False,'validated_for_dataset':False,''')
    write(path,s)


def update_consumers_and_audit():
    path='tests/test_stage2p.py';s=read(path)
    s=s.replace('for s in result.final_populations_by_slice','for s in np.moveaxis(result.final_populations_by_slice, 0, 1)')
    s=s.replace('for populations in result.final_populations_by_slice','for populations in np.moveaxis(result.final_populations_by_slice, 0, 1)')
    write(path,s)
    path='tests/test_stage4.py';s=read(path)
    s=s.replace('result.final_populations_by_slice[:, I7] if False else result.final_populations_by_slice[I7]', 'result.final_populations_by_slice[I7]')
    write(path,s)
    path='audit/deep_check.py';s=read(path)
    old='raw=result.final_populations_by_slice;reference=np.moveaxis(raw,1,0)'
    new='''raw=result.final_populations_by_slice
    if getattr(result,'population_axes',None)==('manifold','z','y','x'):
        reference=raw.copy()
    else:
        reference=np.moveaxis(raw,1,0)'''
    s=once(s,old,new)
    s=once(s,"'audited_base':'b55a1faf462f6433d4e9401a000682426de46467',", "'original_audited_base':'b55a1faf462f6433d4e9401a000682426de46467',")
    marker="    print('AUDIT_FINAL '+json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)"
    s=once(s,marker,marker+"\n    if report['failed_checks']:\n        raise SystemExit('Independent audit failures: '+', '.join(report['failed_checks']))")
    write(path,s)
    path='pyproject.toml';s=read(path);s=s.replace('version = "0.7.0"','version = "0.8.0"');write(path,s)
    path='src/hoyag/__init__.py';s=read(path)
    s+='''\nfrom .population_state import PopulationField, convert_population_layout, POPULATION_AXES
from .pump_source import PumpSource
from .signal import propagate_structured_signal_frozen
__all__ += ["PopulationField", "convert_population_layout", "POPULATION_AXES", "PumpSource", "propagate_structured_signal_frozen"]
'''
    write(path,s)
    path='README.md';s=read(path)
    title='# Ho:YAG crystal simulation\n'
    notice='''
## Audit corrections — API 0.8

The Stage 0–7 audit defects are corrected. All spatial population results now use
**(manifold,z,y,x)**; legacy single-pulse archives require explicit axis conversion.
The weak reference preserves the original complex pulse, and Stage 5/7 absorption
is bound to a shared, documented pump spectrum. See `docs/AUDIT_FIXES_STAGE0_7.md`
and `results/audit_fixes/verification.json` for migration and executed checks.
Fixed-point convergence is still not a claim of mesh or experimental validation.

'''
    s=once(s,title,title+notice);write(path,s)


def main():
    parameters_and_populations();inhomogeneity();propagation();signal();pump_binding();update_consumers_and_audit()
    print('Deterministic source migration complete; running independent regressions next.',flush=True)

if __name__=='__main__':main()
