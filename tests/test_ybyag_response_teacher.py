"""A small known optical plant verifies the response teacher's control logic."""
import numpy as np

from ybyag_dataset.response_teacher import response_matrix_teacher, response_modes


def test_response_teacher_finds_and_verifies_phase_correction():
    axis=np.linspace(-1,1,33)
    x,y=np.meshgrid(axis,axis)
    amplitude=np.exp(-2*(x*x+y*y))
    aberration=.24*x+.16*(x*x-y*y)
    desired=amplitude.astype(complex)
    baseline=amplitude*np.exp(1j*aberration)
    command=np.zeros_like(x)
    calls=[]
    def plant(trial):
        calls.append(trial.copy())
        return amplitude*np.exp(1j*(aberration+trial)),True
    modes=response_modes(x,y,1.,max_modes=5)
    result=response_matrix_teacher(baseline,desired,command,modes,plant,
        max_evaluations=8,min_phase_improvement_rad=.05)
    assert result.step_rad is not None
    assert result.phase_rms_rad<.03
    assert result.shape_overlap>.999
    assert result.evaluations==len(calls)<=8
    assert sum(row["kind"]=="verification" for row in result.checks)==3
    assert all(row["kind"]!="verification" or "phase_rms_rad" in row
               for row in result.checks)


def test_response_teacher_rejects_changes_that_fail_shape_guard():
    axis=np.linspace(-1,1,25)
    x,y=np.meshgrid(axis,axis)
    amplitude=np.exp(-2*(x*x+y*y))
    desired=amplitude.astype(complex)
    baseline=amplitude*np.exp(.15j*x)
    def plant(trial):
        return amplitude*(1+3*abs(trial))*np.exp(1j*(.15*x+trial)),True
    modes=response_modes(x,y,1.,max_modes=2)
    result=response_matrix_teacher(baseline,desired,np.zeros_like(x),modes,plant,
        max_evaluations=5,min_phase_improvement_rad=.05,
        min_shape_overlap=.99999,max_shape_drop=0.)
    assert result.step_rad is None
    assert result.evaluations<=5
    assert any(row["kind"]=="verification" and not row["passed"]
               for row in result.checks)


def test_response_teacher_respects_solver_budget_and_invalid_probes():
    axis=np.linspace(-1,1,13)
    x,y=np.meshgrid(axis,axis)
    amplitude=np.exp(-x*x-y*y)
    baseline=amplitude*np.exp(.2j*x)
    calls=0
    def plant(trial):
        nonlocal calls
        calls+=1
        return amplitude*np.exp(1j*(.2*x+trial)),calls!=1
    modes=response_modes(x,y,1.,max_modes=6)
    result=response_matrix_teacher(baseline,amplitude.astype(complex),
        np.zeros_like(x),modes,plant,max_evaluations=4,
        min_phase_improvement_rad=.01)
    assert calls==result.evaluations<=4
    assert result.step_rad is None
    assert result.checks[0]["kind"]=="probe"
    assert not result.checks[0]["passed"]


def test_response_teacher_uses_verified_fallback_when_fit_is_weak():
    axis=np.linspace(-1,1,25)
    x,y=np.meshgrid(axis,axis)
    amplitude=np.exp(-x*x-y*y)
    aberration=.2*x
    baseline=amplitude*np.exp(1j*aberration)
    def plant(trial):
        return amplitude*np.exp(1j*(aberration+trial)),True
    # A defocus-only response cannot remove the tilt, but a separately
    # proposed physical direction can, after a fresh plant evaluation.
    modes=response_modes(x,y,1.,max_modes=3)[2:3]
    result=response_matrix_teacher(baseline,amplitude.astype(complex),
        np.zeros_like(x),modes,plant,max_evaluations=6,
        min_phase_improvement_rad=.05,
        fallback_steps=(("physical_tilt",-aberration),))
    assert result.step_rad is not None
    assert result.phase_rms_rad<.01
    assert result.selected_candidate=="physical_tilt"
    assert any(row.get("candidate")=="physical_tilt" and row["passed"]
               for row in result.checks)


def test_response_teacher_can_verify_fallback_after_failed_probe():
    axis=np.linspace(-1,1,17)
    x,y=np.meshgrid(axis,axis)
    amplitude=np.exp(-x*x-y*y)
    aberration=.2*x
    baseline=amplitude*np.exp(1j*aberration)
    calls=0
    def plant(trial):
        nonlocal calls
        calls+=1
        return amplitude*np.exp(1j*(aberration+trial)),calls>1
    modes=response_modes(x,y,1.,max_modes=1)
    result=response_matrix_teacher(baseline,amplitude.astype(complex),
        np.zeros_like(x),modes,plant,max_evaluations=3,
        min_phase_improvement_rad=.05,
        fallback_steps=(("trusted_fallback",-aberration),))
    assert result.selected_candidate=="trusted_fallback"
    assert result.evaluations==calls==2


def test_adaptive_line_search_uses_measured_shape_response():
    axis=np.linspace(-1,1,33)
    x,y=np.meshgrid(axis,axis)
    amplitude=np.exp(-x*x-y*y)
    aberration=.3*x
    def plant(command):
        phase_command=np.angle(np.exp(1j*command))
        return amplitude*(1+1j*(aberration+phase_command)+2*phase_command),True
    baseline,_=plant(np.zeros_like(x))
    modes=response_modes(x,y,1.,max_modes=1)
    result=response_matrix_teacher(baseline,amplitude.astype(complex),
        np.zeros_like(x),modes,plant,max_evaluations=4,
        min_phase_improvement_rad=.01,min_shape_overlap=.98,
        adaptive_line_search=True)
    fit=next(row for row in result.checks if row["kind"]=="fit")
    assert max(fit["preview_selected_gains"])<1.
    assert result.selected_candidate.startswith("response_fit_")
    assert result.shape_overlap>=.98


def test_pairwise_preview_proposes_verified_two_mode_update():
    axis=np.linspace(-1,1,29)
    x,y=np.meshgrid(axis,axis)
    amplitude=np.exp(-x*x-y*y)
    aberration=.12*x+.08*(x*x-y*y)
    def plant(command):
        return amplitude*np.exp(1j*(aberration+command)),True
    baseline,_=plant(np.zeros_like(x))
    all_modes=response_modes(x,y,1.,max_modes=5)
    modes=[all_modes[0],all_modes[3]]
    result=response_matrix_teacher(baseline,amplitude.astype(complex),
        np.zeros_like(x),modes,plant,max_evaluations=5,
        min_phase_improvement_rad=.02,pairwise_preview=True)
    assert result.evaluations<=5
    assert result.selected_candidate.startswith("response_pair_")
    assert result.phase_rms_rad<.1
    assert any(row["kind"]=="pairwise_preview" for row in result.checks)
