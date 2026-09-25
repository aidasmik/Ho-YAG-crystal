"""Fixed CCD-like response and plane-specific photon-to-ADU captures."""
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from ybluag.camera_dataset import CameraSettings, H, C
from .common import child_seeds


def sample_camera_setup(settings: CameraSettings, ranges, seed, *, enabled=True,
                        stress=1.0):
    shape = (settings.height, settings.width)
    s = child_seeds(seed, 3)
    rng = np.random.default_rng(s[0])
    if not enabled:
        return dict(prnu=np.ones(shape), dsnu_e=np.zeros(shape), hot=np.zeros(shape, bool),
                    dead=np.zeros(shape, bool), shift_pixels=(0., 0.),
                    rotation_rad=0., scale=1.)
    defects = np.random.default_rng(s[1]).random(shape)
    return dict(
        prnu=np.maximum(0, 1+rng.normal(0, settings.prnu_rms, shape)).astype(np.float32),
        dsnu_e=rng.normal(0, settings.dsnu_rms_e, shape).astype(np.float32),
        dead=defects < settings.dead_pixel_fraction,
        hot=(defects >= settings.dead_pixel_fraction) &
            (defects < settings.dead_pixel_fraction+settings.hot_pixel_fraction),
        shift_pixels=tuple(np.random.default_rng(s[2]).normal(
            0, stress*ranges["camera_shift_pixels"], 2)),
        rotation_rad=float(rng.normal(0, stress*ranges["camera_rotation_deg"]*np.pi/180)),
        scale=float(1+rng.normal(0, stress*ranges["camera_scale_fraction"])))


def capture(fluence_J_m2, x_m, y_m, wavelength_m, settings, setup, seed,
            *, enabled=True, sensor_temperature_C=20.):
    """Resample physical object-plane fluence, then count photoelectrons."""
    rng = np.random.default_rng(seed)
    h, w = settings.height, settings.width
    pixel_m = settings.object_fov_width_mm*1e-3/w
    xx = np.arange(w)+.5-w/2
    yy = np.arange(h)+.5-h/2
    yy, xx = np.meshgrid(yy, xx, indexing="ij")
    shift_y, shift_x = setup["shift_pixels"]
    theta = setup["rotation_rad"]
    scale = setup["scale"]
    object_x = ((xx-shift_x)*np.cos(theta)+(yy-shift_y)*np.sin(theta))*pixel_m/scale
    object_y = (-(xx-shift_x)*np.sin(theta)+(yy-shift_y)*np.cos(theta))*pixel_m/scale
    x = np.asarray(x_m, float)
    y = np.asarray(y_m, float)
    if (object_x.min() < x[0] or object_x.max() > x[-1] or
            object_y.min() < y[0] or object_y.max() > y[-1]):
        raise ValueError("camera field or misregistration exceeds optical grid")
    xi = np.interp(object_x, x, np.arange(len(x)))
    yi = np.interp(object_y, y, np.arange(len(y)))
    clean = map_coordinates(np.asarray(fluence_J_m2, float), [yi, xi],
                            order=1, mode="nearest")
    clean = np.maximum(0, gaussian_filter(clean, settings.psf_sigma_pixels))
    expected = (clean*pixel_m**2*settings.pulses_per_exposure*
                settings.optical_throughput*settings.qe_at_signal*wavelength_m/(H*C))
    expected *= setup["prnu"]
    dark = settings.dark_current_e_s*settings.exposure_s*2**(
        (sensor_temperature_C-20)/settings.dark_current_doubling_C)
    if enabled:
        electrons = rng.poisson(np.clip(expected+dark+settings.background_e,
                                        0, settings.full_well_e*50)).astype(np.float32)
        electrons += setup["dsnu_e"] + rng.normal(0, settings.read_noise_e, (h,w))
        electrons[setup["dead"]] = 0
        electrons[setup["hot"]] += settings.full_well_e*.5
    else:
        electrons = expected
    saturated = float(np.mean(electrons >= settings.full_well_e))
    clipped = np.clip(electrons, 0, settings.full_well_e)
    max_adu = 2**settings.adc_bits-1
    adu = np.rint(settings.black_level_adu+clipped/settings.full_well_e*
                  (max_adu-settings.black_level_adu)).clip(0,max_adu).astype(np.uint16)
    return adu, clean.astype(np.float32), dict(saturated_fraction=saturated,
         expected_electron_peak=float(expected.max()), sensor_temperature_C=sensor_temperature_C)
