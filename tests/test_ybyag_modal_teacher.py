"""Known optical plants for the continuous modal correction teacher."""
import numpy as np

from hoyag.propagation import Grid2D, angular_spectrum_propagate
from ybyag_dataset.distortions.slm import apply_slm
from ybyag_dataset.field_metrics import field_metrics, phase_residual, phase_support
from ybyag_dataset.modal_teacher import (
    FrozenOpticalModel, _loss_gradient, modal_basis, modal_correction_teacher)


def _fixture(n=64):
    grid = Grid2D.square(n, .006)
    x, y = grid.mesh
    waist = .0007
    source = np.exp(-(x*x+y*y)/waist**2).astype(complex)
    command = np.zeros(grid.shape)
    slm = {"global_gain":1.,"spatial_gain":np.ones(grid.shape),
           "pixel_gain":np.ones(grid.shape),"phase_offset":command,
           "bits":0,"crosstalk_sigma_pixels":0.}
    return grid, x, y, waist, source, command, slm


def _model(grid, source, command, screen, baseline, slm):
    return FrozenOpticalModel(
        grid=grid,wavelength_m=1.03e-6,slm_to_disk_m=0.,
        output_distance_m=0.,input_field=source,current_command=command,
        actual_slm_phase=command,external_phase=command,
        screen_phase=screen,baseline_field=baseline,slm_setup=slm)


def test_phase_metric_uses_one_illuminated_support_and_piston():
    grid, x, y, _, source, _, _ = _fixture(32)
    desired = source.copy()
    baseline = source*np.exp(1j*(.3*x / .0007 + .4))
    weights = abs(baseline)**2
    support = phase_support(baseline, desired)
    residual, rms = phase_residual(baseline, desired, weights, support)
    shifted, shifted_rms = phase_residual(baseline*np.exp(1.7j),
                                           desired, weights, support)
    assert np.allclose(residual, shifted, atol=1e-12)
    assert np.isclose(rms, shifted_rms)
    assert np.isclose(field_metrics(baseline, desired, weights, support).
                      coherent_fidelity,
                      field_metrics(baseline*np.exp(1.7j), desired, weights,
                                    support).coherent_fidelity)


def test_modal_adjoint_gradient_matches_central_difference():
    grid, x, y, waist, source, command, slm = _fixture(48)
    screen = .5*x/waist + .3*(x*x-y*y)/waist**2
    baseline = source*np.exp(1j*screen)
    model = _model(grid, source, command, screen, baseline, slm)
    _, basis, _ = modal_basis(x, y, source, waist)
    coefficients = np.zeros(len(basis))
    coefficients[:3] = (.03,-.02,.04)
    _, gradient, _, _ = _loss_gradient(coefficients,basis,model,source,.95,30.,1e-5)
    for index in range(3):
        direction = np.zeros_like(coefficients)
        direction[index] = 1e-5
        plus = _loss_gradient(coefficients+direction,basis,model,source,.95,30.,1e-5)[0]
        minus = _loss_gradient(coefficients-direction,basis,model,source,.95,30.,1e-5)[0]
        assert np.isclose(gradient[index],(plus-minus)/2e-5,
                          rtol=1e-5,atol=1e-6)


def test_modal_adjoint_gradient_with_propagation_and_shape_penalty():
    grid, x, y, waist, source, command, slm = _fixture(48)
    screen = .6*x/waist
    propagated = angular_spectrum_propagate(source,grid,1.03e-6,.08)
    baseline = angular_spectrum_propagate(
        propagated*np.exp(1j*screen),grid,1.03e-6,.05)
    model = FrozenOpticalModel(
        grid=grid,wavelength_m=1.03e-6,slm_to_disk_m=.08,
        output_distance_m=.05,input_field=source,current_command=command,
        actual_slm_phase=command,external_phase=command,
        screen_phase=screen,baseline_field=baseline,slm_setup=slm)
    _, basis, _ = modal_basis(x,y,source,waist)
    coefficients=np.zeros(len(basis))
    coefficients[:2]=(.1,-.05)
    _, gradient, _, shape = _loss_gradient(
        coefficients,basis,model,source,.9999,30.,1e-5)
    assert shape < .9999
    for index in range(2):
        direction=np.zeros_like(coefficients)
        direction[index]=1e-5
        plus=_loss_gradient(coefficients+direction,basis,model,source,
                            .9999,30.,1e-5)[0]
        minus=_loss_gradient(coefficients-direction,basis,model,source,
                             .9999,30.,1e-5)[0]
        assert np.isclose(gradient[index],(plus-minus)/2e-5,
                          rtol=1e-5,atol=1e-6)


def test_multwrap_smooth_aberration_gets_verified_full_correction():
    grid, x, y, waist, source, command, slm = _fixture()
    screen = 5*(x*x+y*y)/waist**2 + .5*x/waist
    baseline = source*np.exp(1j*screen)
    model = _model(grid, source, command, screen, baseline, slm)
    names, basis, weights = modal_basis(x, y, source, waist)
    assert len(names) == 14
    assert np.max(abs(np.einsum('ijk,jk->i',basis,weights))) < 1e-9
    calls = []
    def verify(trial):
        calls.append(trial.copy())
        return source*np.exp(1j*(screen+trial)), True
    result = modal_correction_teacher(
        baseline,source,command,model,verify,basis,
        max_reference_evaluations=4,target_fidelity=.99,
        min_coherent_gain=.01)
    assert result.status == "task_success"
    assert result.metrics.coherent_fidelity > .99
    assert result.metrics.phase_rms_rad < 1e-3
    assert result.evaluations == len(calls) <= 4
    assert result.passive_fidelity_limit > .999
    assert any(row["kind"] == "backprop_initialization" and row["used"]
               for row in result.checks)


def test_shape_guard_finds_strength_between_coarse_steps():
    grid = Grid2D.square(96, .006)
    x, y = grid.mesh
    waist = .00065
    source = np.exp(-(x*x+y*y)/waist**2).astype(complex)
    desired = angular_spectrum_propagate(source, grid, 1.03e-6, .2)
    screen = 2*(x*x+y*y)/waist**2
    baseline = desired*np.exp(1j*screen)
    command = np.zeros(grid.shape)
    slm = {"global_gain":1., "spatial_gain":np.ones(grid.shape),
           "pixel_gain":np.ones(grid.shape), "phase_offset":command,
           "bits":8, "crosstalk_sigma_pixels":.35}
    model = FrozenOpticalModel(
        grid=grid, wavelength_m=1.03e-6, slm_to_disk_m=.2,
        output_distance_m=0., input_field=source, current_command=command,
        actual_slm_phase=command, external_phase=command,
        screen_phase=screen, baseline_field=baseline, slm_setup=slm)
    _, basis, _ = modal_basis(x, y, source, waist)
    result = modal_correction_teacher(
        baseline, desired, command, model,
        lambda trial: (model.forward_command(trial), True), basis,
        max_reference_evaluations=1, target_fidelity=.82,
        min_shape_overlap=.973, max_shape_drop=.1)
    assert result.status == "task_success"
    selected_gain = float(result.selected_candidate.removeprefix("modal_"))
    assert .5 < selected_gain < .75
    assert result.metrics.coherent_fidelity >= .82
    assert result.metrics.shape_overlap >= .973
    # The previous 0.75 and 0.5 gain grid cannot pass both criteria.
    full_coefficients = result.selected_coefficients_rad / selected_gain
    support = phase_support(baseline, desired)
    for gain in (.75, .5):
        trial = np.einsum("i,ijk->jk", gain*full_coefficients, basis)
        metrics = field_metrics(model.forward_command(np.mod(trial, 2*np.pi)),
                                desired, abs(baseline)**2, support)
        assert metrics.coherent_fidelity < .82 or metrics.shape_overlap < .973


def test_fixed_amplitude_bound_flags_unreachable_full_field():
    grid, x, y, waist, source, command, slm = _fixture(48)
    target = np.exp(-((x-.001)**2+y*y)/waist**2)
    model = _model(grid,source,command,command,source,slm)
    expected = float(np.abs(np.sum(source*target))**2 /
                     (np.sum(np.abs(source)**2)*np.sum(target**2)))
    assert np.isclose(model.fidelity_limit(target), expected)
    _, basis, _ = modal_basis(x,y,source,waist)
    def verify(trial):
        return source*np.exp(1j*trial),True
    result = modal_correction_teacher(source,target,command,model,verify,basis,
        max_reference_evaluations=2,target_fidelity=.9)
    assert result.passive_fidelity_limit < .9
    assert result.status == "control_limited_passive"
    assert result.step_rad is None


def test_already_corrected_state_gets_freshly_verified_hold_label():
    grid, x, y, waist, source, command, slm = _fixture(48)
    model = _model(grid,source,command,command,source,slm)
    _, basis, _ = modal_basis(x,y,source,waist)
    calls=[]
    def verify(trial):
        calls.append(trial.copy())
        return source.copy(),True
    result = modal_correction_teacher(source,source,command,model,verify,basis,
        max_reference_evaluations=3,target_fidelity=.9)
    assert result.status == "task_success"
    assert result.selected_candidate == "hold"
    assert result.evaluations == len(calls) == 1
    assert np.array_equal(result.step_rad,command)
    assert np.allclose(result.selected_coefficients_rad,0.)


def test_calibrated_slm_lut_keeps_real_endpoint_mismatch():
    grid, _, _, _, _, command, slm = _fixture(8)
    slm["bits"] = 8
    slm["phase_lut_rad"] = np.linspace(0,1.8*np.pi,256)
    phase_zero,_ = apply_slm(command,slm)
    phase_negative,_ = apply_slm(command-1e-3,slm)
    assert np.all(phase_zero == 0.)
    assert np.allclose(phase_negative,1.8*np.pi)
