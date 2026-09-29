"""Measurement-only multiscale modal controller; no implicit solver/training runs.

V2 initial stage: 14-mode full-step proposal, target adapters and learned outcome
comparison. Recurrent state and dense residual correction require further data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from hoyag.propagation import Grid2D, angular_spectrum_propagate
from ybyag_dataset.modal_teacher import modal_basis

TARGETS = ('Gaussian TEM00', 'Flattop super-Gaussian', 'Helical LG(0,+1)',
           'Needle Bessel-Gaussian')
MODES = 14
INPUTS = ('overview', 'detail', 'target', 'context', 'family')
SCHEMA = 'ybyag_controller_v2_initial_1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pupil(measured, config):
    incoming = np.asarray(measured['input__incoming_beam_shape'], float)
    if incoming.ndim != 2 or incoming.shape[0] != incoming.shape[1]:
        raise ValueError('square optical grid required')
    if not np.all(np.isfinite(incoming)) or np.any(incoming < 0) or incoming.max() <= 0:
        raise ValueError('invalid measured input beam')
    grid = Grid2D.square(len(incoming), float(config['nominal']['field_size_mm']) * 1e-3)
    names, basis, weight = modal_basis(*grid.mesh, np.sqrt(incoming),
        float(measured['input__seed_waist_m']), radial_order=4, residual_grid=0)
    if len(names) != MODES:
        raise ValueError('unexpected modal basis')
    return grid, names, basis, weight


def features(measured, metadata, config, *, overview_size=192, detail_size=128):
    """Use only declared measurements and nominal design; never setup/truth arrays.

    Default registration is nominal, not the hidden sampled camera distortion.
    Optional calibrated object-to-pixel affine matrix maps x/y metres to col/row.
    Missing registration remains a declared domain uncertainty.
    """
    cam = np.asarray(measured['input__camera_adu'], float)
    if cam.ndim != 3 or cam.shape[0] != 2 or not np.all(np.isfinite(cam)):
        raise ValueError('two finite phase-diverse camera frames required')
    cs = metadata['camera_settings']
    h, w = cam.shape[1:]
    pixel = float(cs['object_fov_width_mm']) * 1e-3 / w
    if pixel <= 0:
        raise ValueError('invalid pixel calibration')
    nominal_affine = np.array([[1/pixel, 0, w/2-.5], [0, 1/pixel, h/2-.5]])
    affine = np.asarray(cs.get('object_to_pixel_affine', nominal_affine), float)
    if affine.shape != (2, 3) or not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:, :2])) < 1e-12:
        raise ValueError('invalid camera registration')
    signal = np.maximum(cam-float(cs['black_level_adu']), 0)
    saturated = cam >= 2**int(cs['adc_bits'])-1
    totals = signal.sum(axis=(1, 2))
    if np.any(totals <= 0):
        raise ValueError('dark camera frame')
    scale = np.maximum(signal.max(axis=(1, 2)), 1)
    images = signal/scale[:, None, None]
    side = pixel*w
    axis = (np.arange(overview_size)+.5-overview_size/2)*side/overview_size
    yy, xx = np.meshgrid(axis, axis, indexing='ij')

    def camera_sample(x, y, spacing):
        coords = affine @ np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
        cols, rows = coords.reshape(2, *x.shape)
        # Antialias before decimation; preserve native resolution in the crop.
        down = max(np.linalg.svd(affine[:, :2], compute_uv=False))*spacing
        blur = max(0., .5*np.sqrt(max(down**2-1, 0)))
        valid = (cols >= 0) & (cols <= w-1) & (rows >= 0) & (rows <= h-1)
        channels = [map_coordinates(gaussian_filter(im, blur), [rows, cols],
                    order=1, mode='constant', cval=0) for im in images]
        for bad in saturated:
            valid &= map_coordinates(bad.astype(float), [rows, cols], order=0,
                                     mode='constant', cval=1) < .5
        return channels, valid.astype(float)

    camera, mask = camera_sample(xx, yy, side/overview_size)
    grid, _, _, _ = pupil(measured, config)
    gx, gy = grid.mesh
    rows = (yy-gy[0, 0])/grid.dy
    cols = (xx-gx[0, 0])/grid.dx
    def optical(a):
        a = np.asarray(a, float)
        if a.shape != grid.shape or not np.all(np.isfinite(a)):
            raise ValueError('invalid measured optical map')
        down = side/overview_size/grid.dx
        smooth = gaussian_filter(a, .5*np.sqrt(max(down**2-1, 0)))
        return map_coordinates(smooth, [rows, cols], order=1, mode='constant', cval=0)
    current = np.asarray(measured['input__slm_command_rad'], float)
    incoming = np.asarray(measured['input__incoming_beam_shape'], float)
    overview = np.stack([*camera, mask,
        optical(measured['input__yb_relative_map'])-1,
        optical(incoming/max(incoming.max(), 1e-30)),
        optical(np.sin(current)), optical(np.cos(current))], -1)
    # Native-pixel crop around the nominal optical axis, not a recentered beam.
    crop_axis = (np.arange(detail_size)+.5-detail_size/2)*pixel
    cy, cx = np.meshgrid(crop_axis, crop_axis, indexing='ij')
    crop, cmask = camera_sample(cx, cy, pixel)
    detail = np.stack([*crop, cmask], -1)
    phase = np.asarray(measured['input__target_phase_mask_rad'], float)
    waist = float(measured['input__seed_waist_m'])
    source = np.exp(-(gx*gx+gy*gy)/waist**2)
    distance = float(measured['input__slm_to_disk_m'])+float(measured['input__output_distance_m'])
    desired = angular_spectrum_propagate(source*np.exp(1j*phase), grid, 1030e-9, distance)
    intensity = abs(desired)**2
    target = np.stack([optical(intensity/max(intensity.max(), 1e-30)),
                      optical(np.sin(phase)), optical(np.cos(phase))], -1)
    room = float(measured['input__room_temperature_K'])
    probes = np.asarray(measured['input__disk_temperature_probes_K'], float)
    valid = np.asarray(measured['input__disk_temperature_probe_valid'], bool) & np.isfinite(probes)
    if probes.shape != (3,) or valid.shape != (3,):
        raise ValueError('expected three temperature probes with validity')
    scalars = ['pump_power_W', 'seed_energy_J', 'pump_radius_m', 'seed_waist_m',
               'disk_radius_m', 'disk_thickness_m', 'slm_to_disk_m',
               'output_distance_m', 'yb_nominal_at_percent']
    context = np.r_[np.where(valid, probes-room, 0), valid.astype(float), room,
                    [float(measured['input__'+k]) for k in scalars],
                    np.log1p(totals), np.log1p(scale), pixel,
                    float(cs.get('exposure_s', 0)), float(cs['black_level_adu']),
                    saturated.mean(), float('object_to_pixel_affine' in cs)]
    family = np.eye(4)[TARGETS.index(metadata['selected_target'])]
    result = dict(overview=overview, detail=detail, target=target, context=context, family=family)
    if not all(np.all(np.isfinite(a)) for a in result.values()):
        raise ValueError('nonfinite measured features')
    return {k: np.asarray(v, np.float32) for k, v in result.items()}


def build_models(context_width, *, overview_size=192, detail_size=128):
    """Lazy TensorFlow dependency. Returns proposal+baseline and action critic."""
    import tensorflow as tf
    L = tf.keras.layers
    def encoder():
        ins = [L.Input((overview_size, overview_size, 7), name='overview'),
               L.Input((detail_size, detail_size, 3), name='detail'),
               L.Input((overview_size, overview_size, 3), name='target'),
               L.Input((context_width,), name='context'), L.Input((4,), name='family')]
        def branch(x, channels):
            for c in channels:
                skip = L.Conv2D(c, 1, strides=2, padding='same')(x)
                x = L.Conv2D(c, 3, strides=2, padding='same', activation='swish')(x)
                x = L.Conv2D(c, 3, padding='same')(x)
                x = L.Activation('swish')(L.Add()([x, skip]))
            return L.Dense(64, activation='swish')(L.Flatten()(x))
        branches = [branch(ins[0], (16, 24, 32, 48)),
                    branch(ins[1], (12, 16, 24, 32)),
                    branch(ins[2], (8, 12, 16, 24))]
        context = L.Dense(64, activation='swish')(L.Concatenate()(ins[3:]))
        h = L.Dense(128, activation='swish')(L.Concatenate()([*branches, context]))
        return ins, h
    ins, h = encoder()
    adapters = L.Reshape((4, MODES))(L.Dense(4*MODES, kernel_initializer='zeros')(h))
    # Family adapters share the spatial/context representation.
    coeff = L.Dot(axes=(1, 1), name='coefficients')([adapters, ins[4]])
    baseline = L.Dense(2, activation='sigmoid', name='baseline')(h)
    proposal = tf.keras.Model(ins, [coeff, baseline], name='modal_v2_proposal')
    ci, ch = encoder()
    action = L.Input((MODES,), name='action')
    layer1 = L.Dense(64, activation='swish')
    layer2 = L.Dense(3)
    zero = L.Rescaling(0.)(action)
    predicted = layer2(layer1(L.Concatenate()([ch, action])))
    held = layer2(layer1(L.Concatenate()([ch, zero])))
    # Hold is exactly zero change, even before training.
    delta = L.Subtract(name='outcome_delta')([predicted, held])
    critic = tf.keras.Model([*ci, action], delta, name='modal_v2_critic')
    return proposal, critic


@dataclass(frozen=True)
class GateLimits:
    min_gain: float = .01
    min_shape: float = .95
    max_shape_drop: float = .012
    min_energy_fraction: float = .8


def choose(actions, deltas, baseline, bounds, *, calibrated, supported=None, limits=GateLimits()):
    """Compare conservative outcome bounds with hold; predictions are not proof."""
    actions, deltas = np.asarray(actions), np.asarray(deltas)
    baseline, bounds = np.asarray(baseline), np.asarray(bounds)
    if (actions.ndim != 2 or actions.shape[1] != MODES or
        deltas.shape != (len(actions), 3) or baseline.shape != (2,) or bounds.shape != (5,) or
        not all(np.all(np.isfinite(a)) for a in (actions, deltas, baseline, bounds)) or
        np.any(bounds < 0) or not np.all(actions[0] == 0)):
        raise ValueError('invalid candidate or calibration arrays')
    if not calibrated:
        return 0, 'hold_uncalibrated'
    # bounds: baseline fidelity/shape errors; delta fidelity/shape/log-energy errors.
    low = deltas-bounds[2:]
    shape_low = baseline[1]-bounds[1]+low[:, 1]
    accepted = (low[:, 0] >= limits.min_gain) & (shape_low >= limits.min_shape)
    accepted &= (low[:, 1] >= -limits.max_shape_drop)
    accepted &= low[:, 2] >= np.log(limits.min_energy_fraction)
    if supported is not None:
        supported = np.asarray(supported, bool)
        if supported.shape != (len(actions),):
            raise ValueError('candidate support mask has wrong shape')
        accepted &= supported
    accepted[0] = False
    score = np.where(accepted, low[:, 0], -np.inf)
    if not np.any(accepted):
        return 0, 'hold_no_supported_improvement'
    return int(np.argmax(score)), 'proposed_requires_measurement_verification'


def command(measured, config, coefficients):
    _, _, basis, _ = pupil(measured, config)
    c = np.asarray(coefficients, float)
    if c.shape != (MODES,) or not np.all(np.isfinite(c)):
        raise ValueError('invalid modal coefficients')
    current = np.asarray(measured['input__slm_command_rad'], float)
    if current.shape != basis.shape[1:] or not np.all(np.isfinite(current)):
        raise ValueError('invalid current SLM command')
    step = np.einsum('k,kij->ij', c, basis)
    return np.mod(current+step, 2*np.pi), step


def verify_measured_outcome(previous, observed, *, limits=GateLimits()):
    """Return a rollback decision from calibrated measured shape and energy.

    Callers must acquire a new observation after rollback: physical time is not
    rewound. Phase fidelity is optional and must be measured/reconstructed, not truth.
    """
    for reading in (previous, observed):
        if not reading.get('valid', False):
            return False, 'rollback_invalid_measurement'
        if not all(np.isfinite(reading[k]) for k in ('shape', 'energy_J')) or reading['energy_J'] <= 0:
            return False, 'rollback_invalid_measurement'
    if (observed['shape'] < max(limits.min_shape, previous['shape']-limits.max_shape_drop)
        or observed['energy_J']/previous['energy_J'] < limits.min_energy_fraction):
        return False, 'rollback_shape_or_energy'
    return True, 'measured_shape_energy_pass'
