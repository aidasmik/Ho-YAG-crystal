"""PyVista replay of saved solver fields. All panels use archived arrays."""
from __future__ import annotations

import time
import numpy as np

from .snapshots import ScientificSnapshot


def _image(pv, data, x, y, name):
    data = np.asarray(data, float)
    if data.ndim != 2:
        raise ValueError(f'{name} must be a 2D solver array')
    ny, nx = data.shape
    if x is None or y is None:
        x = np.arange(nx, dtype=float)
        y = np.arange(ny, dtype=float)
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != nx or len(y) != ny:
        raise ValueError(f'{name} coordinate shape mismatch')
    dx = float(x[1]-x[0]) if nx>1 else 1.
    dy = float(y[1]-y[0]) if ny>1 else 1.
    if np.allclose(np.diff(x),dx,rtol=1e-9,atol=1e-12) and np.allclose(np.diff(y),dy,rtol=1e-9,atol=1e-12):
        grid = pv.ImageData(dimensions=(nx,ny,1), spacing=(dx,dy,1),
                            origin=(float(x[0]),float(y[0]),0))
        grid.point_data[name] = data.ravel(order='C')
    else:
        xx,yy=np.meshgrid(x,y,indexing='xy')
        grid=pv.StructuredGrid(xx,yy,np.zeros_like(xx))
        grid.point_data[name]=data.ravel(order='F')
    return grid


def scientific_maps(snapshot: ScientificSnapshot, *, mode=0, polarization=0) -> dict:
    """Return mapped numerical quantities with exact units and coordinate axes."""
    a = snapshot.arrays
    maps = {}
    x = a.get('x_m')*1e3 if 'x_m' in a else None
    y = a.get('y_m')*1e3 if 'y_m' in a else None
    material_r = ((a['r_edges_m'][:-1]+a['r_edges_m'][1:])/2
                  *1e3 if 'r_edges_m' in a else None)
    plate_r = ((a['plate_r_edges_m'][:-1]+a['plate_r_edges_m'][1:])/2
               *1e3 if 'plate_r_edges_m' in a else None)
    def material_axes(data, radial):
        phi=2*np.pi*(np.arange(data.shape[-2])+.5)/data.shape[-2]
        return np.arange(data.shape[-1]) if radial is None else radial,phi
    fields = a.get('output_coupler_fields')
    powers = snapshot.metadata.get('modal_powers_W')
    if fields is not None and powers is not None:
        if len(fields) != len(powers):
            raise ValueError('modal powers and output fields disagree')
        dx = float(x[1]-x[0])*1e-3; dy = float(y[1]-y[0])*1e-3
        total = np.zeros(fields.shape[-2:], float)
        for field, power in zip(fields, powers):
            raw = np.sum(abs(field)**2,axis=0)
            norm = float(raw.sum()*dx*dy)
            if norm <= 0 or power < 0:
                raise ValueError('invalid output field or modal power')
            total += power*raw/norm
        maps['output_average_irradiance_W_m2']=(total,x,y)
        wavelength=snapshot.metadata.get('wavelength_m')
        if wavelength is not None:
            angular=np.zeros_like(total)
            for field,power in zip(fields,powers):
                far=np.fft.fftshift(np.fft.fft2(field,axes=(-2,-1)),axes=(-2,-1))
                raw=np.sum(abs(far)**2,axis=0)
                angular+=power*raw/raw.sum()
            fx=np.fft.fftshift(np.fft.fftfreq(total.shape[1],d=dx))
            fy=np.fft.fftshift(np.fft.fftfreq(total.shape[0],d=dy))
            maps['farfield_power_per_bin_W']=(angular,wavelength*fx*1e3,wavelength*fy*1e3)
        component = fields[mode,polarization]
        phase = np.angle(component)
        phase[abs(component) < 1e-3*abs(component).max()] = np.nan
        maps[f'mode_{mode}_polarization_{polarization}_phase_rad']=(phase,x,y)
    if 'density_m3' in a:
        data=a['density_m3'][0].T
        maps['Ho_density_front_m3']=(data,*material_axes(data,material_r))
    if 'pump_input_average_irradiance_W_m2' in a:
        data=a['pump_input_average_irradiance_W_m2'].T
        maps['pump_input_average_irradiance_W_m2']=(data,*material_axes(data,material_r))
    if 'mean_fractions' in a:
        fractions=a['mean_fractions']
        if fractions.ndim==3 and 'raw_heat_W_m3' in a:
            fractions=fractions.reshape(4,*a['raw_heat_W_m3'].shape)
        data=fractions[2,0].T
        maps['upper_manifold_fraction_front']=(data,*material_axes(data,material_r))
    if 'raw_heat_W_m3' in a:
        data=a['raw_heat_W_m3'][0].T
        maps['disk_heat_front_W_m3']=(data,*material_axes(data,material_r))
    if 'disk_temperature_K' in a:
        front=a['disk_temperature_K'][0].T;rear=a['disk_temperature_K'][-1].T
        maps['disk_front_temperature_K']=(front,*material_axes(front,material_r))
        maps['disk_rear_temperature_K']=(rear,*material_axes(rear,material_r))
        if 'z_edges_m' in a and material_r is not None:
            zcenter=(a['z_edges_m'][:-1]+a['z_edges_m'][1:])*500
            maps['disk_temperature_zr_phi0_K']=(a['disk_temperature_K'][:,:,0].T,
                                                  zcenter,material_r)
        conductivity=snapshot.metadata.get('disk_conductivity_W_mK')
        if conductivity is not None and 'r_edges_m' in a and 'z_edges_m' in a:
            from .thermal_views import cylindrical_temperature_derivatives
            derived=cylindrical_temperature_derivatives(a['disk_temperature_K'],
                a['r_edges_m'],a['z_edges_m'],conductivity)
            grad=derived['grad_T_magnitude_K_m'][0].T/1e3
            radial=derived['heat_flux_W_m2'][0,:,:,0].T
            maps['disk_front_grad_T_K_mm']=(grad,*material_axes(grad,material_r))
            maps['disk_front_radial_heat_flux_W_m2']=(radial,*material_axes(radial,material_r))
    if 'plate_temperature_K' in a:
        data=a['plate_temperature_K'][0].T
        maps['plate_interface_temperature_K']=(data,*material_axes(data,plate_r))
    if 'interface_flux_W_m2' in a:
        data=a['interface_flux_W_m2'].T
        maps['crystal_plate_face_flux_W_m2']=(data,*material_axes(data,material_r))
    if 'mean_roundtrip_opd_m' in a:
        maps['roundtrip_OPD_nm']=(a['mean_roundtrip_opd_m']*1e9,x,y)
    return maps


def _populate(plotter, pv, snapshot, mode, polarization, deformation_exaggeration,
              view='overview'):
    if not np.isfinite(deformation_exaggeration) or deformation_exaggeration < 0:
        raise ValueError('invalid display-only deformation factor')
    maps = scientific_maps(snapshot, mode=mode, polarization=polarization)
    if view=='overview':
        keys=['output_average_irradiance_W_m2',f'mode_{mode}_polarization_{polarization}_phase_rad',
              'Ho_density_front_m3','upper_manifold_fraction_front','disk_heat_front_W_m3',
              'disk_front_temperature_K','plate_interface_temperature_K','roundtrip_OPD_nm']
        titles=['Output average irradiance (W/m2) | x/y mm',
                f'Mode {mode} pol {polarization} phase (rad) | x/y mm',
                'Ho density (m^-3) | r mm, phi rad',
                'Upper manifold fraction | r mm, phi rad',
                'Disk heat (W/m3) | r mm, phi rad',
                'Disk front T (K) | r mm, phi rad',
                'Plate interface T (K) | r mm, phi rad',
                'Roundtrip OPD (nm) | x/y mm']
    elif view=='thermal':
        keys=['disk_front_temperature_K','disk_rear_temperature_K',
              'disk_temperature_zr_phi0_K','disk_front_grad_T_K_mm',
              'disk_front_radial_heat_flux_W_m2','crystal_plate_face_flux_W_m2',
              'plate_interface_temperature_K','disk_heat_front_W_m3']
        titles=['Disk front T (K) | r mm, phi rad','Disk rear T (K) | r mm, phi rad',
                'Disk T at phi=0 (K) | z/r mm','Front |grad T| (K/mm)',
                'Front radial heat flux (W/m2)','Crystal to plate face flux (W/m2)',
                'Plate interface T (K)','Disk heat (W/m3)']
    elif view=='optics':
        keys=['pump_input_average_irradiance_W_m2','output_average_irradiance_W_m2',
              'farfield_power_per_bin_W',f'mode_{mode}_polarization_{polarization}_phase_rad',
              'roundtrip_OPD_nm','upper_manifold_fraction_front',
              'Ho_density_front_m3','disk_heat_front_W_m3']
        titles=['Prescribed pump average irradiance (W/m2)',
                'Output average irradiance (W/m2)',
                'Far-field power/angular bin (W) | angle mrad',
                f'Mode {mode} pol {polarization} phase (rad)',
                'Roundtrip OPD (nm)','Upper manifold fraction',
                'Ho density (m^-3)','Disk heat (W/m3)']
    else:
        raise ValueError('view must be overview, thermal, or optics')
    for slot,key in enumerate(keys):
        plotter.subplot(slot//3,slot%3)
        if key not in maps:
            plotter.add_text(f'{titles[slot]}\nunavailable',font_size=7)
            continue
        data,x,y = maps[key]
        mesh = _image(pv,data,x,y,key)
        plotter.add_mesh(mesh,scalars=key,cmap='twilight' if 'phase' in key else 'viridis',
                         nan_color='gray',show_scalar_bar=False)
        plotter.add_text(titles[slot],font_size=7,position='upper_left')
        finite=data[np.isfinite(data)]
        if finite.size:
            plotter.add_text(f'min {finite.min():.4g} | max {finite.max():.4g}',
                             font_size=7,position='lower_left')
        plotter.view_xy()
    plotter.subplot(2,2)
    a=snapshot.arrays
    if 'disk_nodes_m' in a and 'disk_displacement_m' in a:
        points=a['disk_nodes_m']+deformation_exaggeration*a['disk_displacement_m']
        cloud=pv.PolyData(points)
        cloud.point_data['physical_u_z_nm']=a['disk_displacement_m'][:,2]*1e9
        plotter.add_mesh(cloud,scalars='physical_u_z_nm',render_points_as_spheres=True,
                         point_size=3,cmap='coolwarm')
        plotter.add_text(f'disk physical u_z (nm); display deformation x{deformation_exaggeration:g}',
                         font_size=7)
        plotter.view_xy()
    else:
        plotter.add_text('Deformation unavailable in this snapshot',font_size=7)


def render_replay(snapshot: ScientificSnapshot, *, off_screen=False,
                  screenshot=None, mode=0, polarization=0,
                  deformation_exaggeration=1., view='overview'):
    """Create a 3x3 scientific replay panel; missing arrays stay unavailable."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError('install the optional viewer dependencies: pip install -e ".[viewer]"') from exc
    plotter = pv.Plotter(shape=(3,3), off_screen=off_screen, window_size=(960,720))
    _populate(plotter,pv,snapshot,mode,polarization,deformation_exaggeration,view)
    started=time.perf_counter()
    if screenshot is not None:
        plotter.show(screenshot=str(screenshot),auto_close=True)
    elif off_screen:
        plotter.show(auto_close=True)
    else:
        plotter.show()
    return {'render_seconds':time.perf_counter()-started,
            'render_FPS_one_frame':1/max(time.perf_counter()-started,1e-9),
            'solver_updates_per_s':None,'simulated_seconds_per_wall_second':None,
            'state_age_s':max(0,time.time()-snapshot.metadata['t_published_wall'])}


def monitor_live(directory, *, target_fps=30., seconds=None, off_screen=False,
                 screenshot=None, mode=0, polarization=0, view='overview'):
    """Render the latest accepted snapshot repeatedly without advancing physics."""
    from .snapshots import newest_snapshot
    if target_fps<=0 or target_fps>120:
        raise ValueError('target render FPS must be in (0,120]')
    if off_screen and seconds is None:
        raise ValueError('headless live benchmark needs a finite duration')
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError('install the optional viewer dependencies') from exc
    snap=newest_snapshot(directory)
    if snap is None:
        raise ValueError('no accepted live snapshot has been published')
    plotter=pv.Plotter(shape=(3,3),off_screen=off_screen,window_size=(960,720))
    _populate(plotter,pv,snap,mode,polarization,1.,view)
    plotter.show(interactive_update=True,auto_close=False,interactive=not off_screen)
    started=time.perf_counter();last_frame=started
    last_id=snap.metadata['state_id'];updates=0;frames=0
    try:
        while seconds is None or time.perf_counter()-started<seconds:
            frame_begin=time.perf_counter()
            fresh=newest_snapshot(directory)
            if fresh is not None and fresh.metadata['state_id']!=last_id:
                snap=fresh;last_id=fresh.metadata['state_id'];updates+=1
                plotter.clear()
                _populate(plotter,pv,snap,mode,polarization,1.,view)
            elapsed=max(time.perf_counter()-started,1e-9)
            age=max(0.,time.time()-snap.metadata['t_published_wall'])
            plotter.subplot(2,2)
            plotter.add_text(
                f'{snap.metadata["fidelity_mode"]} | state {snap.metadata["state_id"]}\n'
                f'render {frames/elapsed:.1f} FPS | solver {updates/elapsed:.3f} updates/s\n'
                f'state age {age:.1f} s | simulated time unavailable ({snap.metadata["time_kind"]})\n'
                'simulated seconds/wall second unavailable',
                name='live_metrics',position='lower_left',font_size=8)
            try:
                plotter.update(stime=1,force_redraw=True)
            except (RuntimeError,AttributeError):
                break
            frames+=1
            wait=max(0.,1/target_fps-(time.perf_counter()-frame_begin))
            if wait:time.sleep(wait)
            last_frame=time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        if screenshot is not None:
            plotter.screenshot(str(screenshot))
        plotter.close()
    elapsed=max(time.perf_counter()-started,1e-9)
    return {'rendered_frames':frames,'render_FPS':frames/elapsed,
            'solver_updates':updates,'solver_updates_per_s':updates/elapsed,
            'state_age_s':max(0.,time.time()-snap.metadata['t_published_wall']),
            'simulated_seconds_per_wall_second':None,
            'time_kind':snap.metadata['time_kind']}
