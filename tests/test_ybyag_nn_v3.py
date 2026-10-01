import numpy as np
import pytest

torch = pytest.importorskip('torch')

from ybyag_control import nn_v3 as v3
from ybyag_dataset.distortions.slm import apply_slm
from ybyag_dataset.distortions.camera import capture
from ybluag.camera_dataset import CameraSettings


def small_spec(**kw):
    values = dict(slm_n=128, field_m=6e-3, factor=2, crop=32, planes_m=(0., 1.))
    values.update(kw)
    return v3.Spec(**values)


def test_render_matches_dataset_slm_rendering():
    spec = small_spec()
    optics = v3.Optics(spec, 'cpu', 'double')
    command = np.random.default_rng(3).uniform(-4, 10, (128, 128))
    for gain in (0., .02):
        setup = dict(global_gain=1+gain, spatial_gain=np.ones((128, 128)),
                     pixel_gain=np.ones((128, 128)), phase_offset=np.zeros((128, 128)),
                     bits=8, crosstalk_sigma_pixels=.35)
        reference, _ = apply_slm(command, setup)
        phasor = optics.render(optics.tensor(command)[None], optics.tensor([gain]))[0].numpy()
        assert np.max(abs(phasor-np.exp(1j*reference))) < 1e-10


def test_step_map_is_piston_free_and_zero_at_hold():
    optics = v3.Optics(small_spec(), 'cpu', 'double')
    zero = optics.step_map(torch.zeros(1, v3.SLM_MODES, dtype=torch.float64),
                           torch.zeros(1, v3.RESIDUAL, v3.RESIDUAL, dtype=torch.float64))
    assert float(zero.abs().max()) == 0
    step = optics.step_map(torch.zeros(1, v3.SLM_MODES, dtype=torch.float64),
                           torch.ones(1, v3.RESIDUAL, v3.RESIDUAL, dtype=torch.float64))
    assert abs(float((step[0]*optics.slm_weight).sum())) < 1e-10


def test_camera_binning_matches_area_average():
    spec = v3.Spec()
    grid = spec.slm_grid()
    x, y = grid.mesh
    fluence = np.exp(-2*((x-3e-4)**2+(y+2e-4)**2)/(.8e-3)**2)
    # About half full well at the peak pixel.
    settings = CameraSettings(optical_throughput=1e-6, qe_at_signal=1.)
    setup = dict(prnu=np.ones((1080, 1920)), dsnu_e=np.zeros((1080, 1920)),
                 hot=np.zeros((1080, 1920), bool), dead=np.zeros((1080, 1920), bool),
                 shift_pixels=(0., 0.), rotation_rad=0., scale=1.)
    peak = fluence.max()
    frames = []
    for scale in (1., .5):
        adu, clean, _ = capture(scale*fluence, grid.x, grid.y, 1030e-9, settings, setup, 1,
                                enabled=False)
        frames.append(adu)
    cs = dict(vars(settings))
    meas, var, valid, saturated = v3.bin_camera(np.stack(frames), cs, spec)
    expected = v3.block_mean(fluence, spec.factor)
    sel = valid[0] & (expected > .05*peak)
    assert sel.sum() > 100
    ratio = meas[0][sel]/expected[sel]
    assert np.std(ratio)/np.mean(ratio) < .02
    assert np.all(var[valid] > 0) and saturated.shape == (2,)


def test_network_starts_at_hold():
    net = v3.OneShotNet(crop=32)
    out = net(torch.zeros(2, v3.CHANNELS, 32, 32), torch.zeros(2, v3.CONTEXT))
    assert float(out['coefficients'].abs().max()) == 0
    assert float(out['residual'].abs().max()) == 0
    assert out['state_mean'].shape == (2, v3.STATE)


def planted_trial(spec, theta_true):
    optics = v3.Optics(spec, 'cpu', 'double')
    xs, ys = spec.slm_grid().mesh
    waist = spec.waist_m
    amp = np.exp(-(xs*xs+ys*ys)/waist**2)
    command = np.zeros((spec.slm_n, spec.slm_n))
    pump = optics.pump_profile(optics.tensor([spec.pump_radius_m]))
    out = optics.model(optics.tensor(theta_true)[None], optics.tensor(amp)[None],
                       optics.tensor(command)[None], pump)
    intensity = (out['planes'].abs()**2)[0].numpy()
    meas = 2e4*intensity/intensity.max()
    var = (meas+3.)/144
    valid = np.ones_like(meas, bool)
    context = v3.context_vector('Gaussian TEM00', waist, spec.pump_radius_m, spec,
                                meas.sum((1, 2))*144, [0., 0.])
    trial = v3.TrialInputs('Gaussian TEM00', waist, spec.pump_radius_m, amp, command,
                           np.zeros_like(command), meas.astype(np.float32),
                           var.astype(np.float32), valid, context)
    return optics, trial


def true_fidelity(optics, trial, theta, command):
    t = optics.tensor
    pump = optics.pump_profile(t([trial.pump_radius_m]))
    out = optics.model(t(theta)[None], t(trial.amp)[None], t(command)[None], pump, planes=False)
    desired = optics.desired(t([trial.waist_m]), t(trial.target)[None])['output']
    f, _, _ = v3.overlaps(out['output'], desired, out['disk'], pump)
    return float(f)


def run_planted(theta, samples=12):
    spec = small_spec()
    optics, trial = planted_trial(spec, theta)
    limits = v3.CertificateLimits(calibrated=True, samples=samples, refine_iterations=80,
                                  plan_iterations=40, plan_samples=4)
    controller = v3.OneShotController(spec, [v3.OneShotNet(crop=spec.crop)], 'cpu', limits)
    result = controller.propose(trial)
    before = true_fidelity(optics, trial, theta, trial.command)
    after = true_fidelity(optics, trial, theta, result['command'])
    return result['certificate'], before, after


def test_measured_refinement_certifies_a_one_shot_correction():
    # Odd upstream aberrations (tilt, coma, trefoil) are unchanged by the
    # phase-retrieval twin phi(x) -> -phi(-x), so two planes determine them.
    theta = np.zeros(v3.STATE)
    theta[[0, 1, 5, 6, 7]] = (.35, -.25, -.2, .4, .3)
    cert, before, after = run_planted(theta)
    assert cert['model_consistent']
    assert cert['decision'] != 'hold'
    assert after > before+.2
    # The released lower bound held for the true state.
    assert after-before >= cert['candidates'][cert['decision']]['gain_certified']


def test_twin_search_resolves_even_aberrations_safely():
    # Even aberrations have a sign twin at these planes and disk-plane phase is
    # invisible at the disk camera. The explicit twin restarts must reach the
    # measured state, and whatever is released must obey its certified bound.
    theta = np.zeros(v3.STATE)
    theta[:5] = (.35, -.25, .5, .3, -.2)
    theta[v3.SLM_MODES+2] = .3
    cert, before, after = run_planted(theta)
    assert cert['reduced_chi2'] < 1e-3
    assert np.allclose(cert['state_estimate'][:v3.SLM_MODES+v3.DISK_MODES], theta[:v3.SLM_MODES+v3.DISK_MODES], atol=.02)
    released = cert['candidates'][cert['decision']]
    assert after-before >= released['gain_certified']-1e-9
    assert cert['decision'] != 'hold' and after > before+.2


def test_uncalibrated_controller_holds_but_reports_candidates():
    spec = small_spec()
    theta = np.zeros(v3.STATE)
    theta[2] = .6
    _, trial = planted_trial(spec, theta)
    limits = v3.CertificateLimits(samples=6, refine_iterations=30, plan_iterations=15,
                                  plan_samples=2)
    controller = v3.OneShotController(spec, [v3.OneShotNet(crop=spec.crop)], 'cpu', limits)
    result = controller.propose(trial)
    assert result['certificate']['decision'] == 'hold'
    assert result['certificate']['status'] == 'hold_uncalibrated'
    assert np.array_equal(result['command'], np.mod(trial.command, 2*np.pi))
    assert 'planned' in result['candidate_steps']


def test_margin_fit_covers_and_blocks_harm():
    rng = np.random.default_rng(0)
    rows = []
    for family in v3.TARGETS:
        for _ in range(60):
            true = rng.uniform(-.02, .2)
            slack = rng.uniform(-.02, .05)
            true_slack = slack-rng.uniform(0, .01)
            rows.append(dict(family=family, gain_lower=true+rng.normal(.01, .02),
                             gain_mean=true+.01, true_gain=true, shape_slack=slack,
                             true_shape_slack=true_slack, pump_ok=True,
                             guard_ok=true_slack >= 0))
    base, slope, shape, report = v3.fit_margins(rows, target_coverage=.95)
    for i, family in enumerate(v3.TARGETS):
        sel = [r for r in rows if r['family'] == family]
        certified = np.array([r['gain_lower']-base[i]-slope[i]*max(r['gain_mean'], 0) for r in sel])
        true = np.array([r['true_gain'] for r in sel])
        assert np.mean(certified <= true) >= .95
        slack = np.array([r['shape_slack'] for r in sel])
        true_slack = np.array([r['true_shape_slack'] for r in sel])
        released = (certified >= .01) & (slack >= shape[i])
        assert not np.any(released & ((true < -.005) | (true_slack < 0)))
        assert released.any()
        assert report[family]['calibrated']
    few = v3.fit_margins(rows[:5])[3]
    assert not few['Gaussian TEM00']['calibrated']
