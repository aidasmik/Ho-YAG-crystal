"""Stage 7W regressions use exact matrix spectra and the audited physical engine."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, ArpackNoConvergence
from hoyag.polarization_tracking import (TrackingSettings, normalized_modes, match_branches,
    mixture_distance, invariant_subspace_distance, polarization_family,
    select_tracked_fields, solve_tracked_eigenfields)
from hoyag.polarized_resonator import (convergence_metrics, accepted_closure,
                                     run_polarization_hot_cavity)
from hoyag.coupled_resonator import HotCavitySettings


class MatrixCavity:
    shape=(2,2,2)
    def __init__(self,matrix):self.matrix=np.asarray(matrix,complex)
    def propagate(self,field):
        return (self.matrix@np.asarray(field).reshape(-1)).reshape(self.shape),np.zeros(self.shape,complex)
    def linear_operator(self):return LinearOperator((8,8),matvec=lambda x:self.matrix@x,dtype=complex)


def basis():
    return np.eye(8,dtype=complex).reshape(8,2,2,2)


def spectrum(split=1e-5):
    # Indices 0 and 4: same scalar spatial mode, orthogonal x/y polarization.
    diagonal=np.array([1.02+0j,.8,.6,.4,(1.02-2e-7)*np.exp(1j*split),.7,.5,.3],complex)
    indices=np.array([0,4,1,5])
    return MatrixCavity(np.diag(diagonal)),diagonal[indices],basis()[indices]


def test_full_bank_match_is_rectangular_and_keeps_labels():
    _,_,bank=spectrum()
    prev=bank[[1,0]]
    labels,overlap=match_branches(prev,bank)
    assert labels.tolist()==[1,0]
    assert np.allclose(overlap,1,rtol=0,atol=1e-14)


def test_gain_rank_crossing_preserves_both_photon_slots():
    op,val,bank=spectrum()
    prev=bank[[1,0]]
    result=select_tracked_fields(op,prev,val,bank)
    assert result.converged
    assert np.allclose(abs(np.einsum('ijkl,ijkl->i',prev.conj(),result.fields)),1)
    assert np.allclose(result.eigenvalues,val[[1,0]],rtol=1e-12,atol=0)
    assert not result.diagnostics['photon_seed_reset_branches']


def test_one_mode_cut_through_polarization_pair_is_not_accepted():
    op,val,bank=spectrum()
    result=select_tracked_fields(op,bank[:1],val,bank)
    assert not result.converged
    assert 'insufficient_mode_capacity' in result.status
    assert len(result.diagnostics['split_boundary_pairs'])==1


def test_split_complex_eigenvalues_are_never_rotated_to_force_tracking():
    op,val,bank=spectrum(split=1.47e-5)
    prev=np.stack([(bank[0]+bank[1])/np.sqrt(2),(bank[0]-bank[1])/np.sqrt(2)])
    result=select_tracked_fields(op,prev,val,bank,tolerance=1e-10)
    assert result.converged
    assert result.diagnostics['exact_degenerate_groups']==[]
    assert max(result.residuals)<1e-12
    overlaps=abs(result.fields.reshape(2,-1).conj()@bank[:2].reshape(2,-1).T)**2
    assert np.allclose(overlaps.max(axis=1),1.,atol=1e-12)
    assert not np.allclose(result.fields,prev)


def test_exact_degeneracy_aligns_only_within_true_eigenspace():
    diagonal=np.array([1.02,.8,.6,.4,1.02,.7,.5,.3])
    op=MatrixCavity(np.diag(diagonal));indices=[0,4,1,5]
    bank=basis()[indices];val=diagonal[indices]
    prev=np.stack([(bank[0]+1j*bank[1])/np.sqrt(2),(bank[0]-1j*bank[1])/np.sqrt(2)])
    result=select_tracked_fields(op,prev,val,bank,tolerance=1e-10)
    assert result.converged
    assert np.allclose(result.fields,prev,rtol=0,atol=1e-12)
    assert max(result.residuals)<1e-12


def test_omitted_candidate_failure_is_not_ignored():
    op,val,bank=spectrum();bad=val.copy();bad[-1]+=.01
    result=select_tracked_fields(op,bank[:2],bad,bank)
    assert not result.converged
    assert 'candidate eigenpair' in result.status


def test_genuinely_dominant_new_mode_is_not_suppressed_for_continuity():
    op,val,bank=spectrum()
    previous=bank[2:]
    result=select_tracked_fields(op,previous,val,bank)
    assert result.converged
    assert set(result.diagnostics['photon_seed_reset_branches'])=={0,1}
    assert np.allclose(np.sort(abs(result.eigenvalues)),np.sort(abs(val[:2])))


def test_large_retained_boundary_that_splits_second_pair_is_rejected():
    diagonal=np.array([1.04,1.03,.6,.4,1.04-1e-7,1.03-1e-7,.5,.3])
    op=MatrixCavity(np.diag(diagonal));indices=[0,4,1,5,2,6]
    bank=basis()[indices];val=diagonal[indices]
    result=select_tracked_fields(op,bank[:3],val,bank)
    assert not result.converged and result.diagnostics['split_boundary_pairs']


def test_non_partner_same_gain_does_not_trip_polarization_guard():
    op,val,bank=spectrum()
    # Separate x-polarized spatial pixels have no constant-polarization overlap.
    paired,diag=polarization_family(bank[0],bank[2],1.02,1.02+1e-8,TrackingSettings())
    assert not paired
    assert diag['best_constant_polarization_overlap_squared']==0


def test_orthogonal_polarizations_are_partners_even_when_split():
    _,val,bank=spectrum()
    paired,diag=polarization_family(bank[0],bank[1],val[0],val[1],TrackingSettings())
    assert paired
    assert diag['complex_relative_gap']>TrackingSettings().exact_eigenvalue_tolerance


def test_subspace_invariance_does_not_erase_unequal_power_rotation():
    f=basis()[[0,4]]
    rotated=np.stack([(f[0]+f[1])/np.sqrt(2),(f[0]-f[1])/np.sqrt(2)])
    assert invariant_subspace_distance(f,rotated)<1e-12
    assert mixture_distance(f,[.5,.5],rotated,[.5,.5])<3e-8
    assert mixture_distance(f,[.9,.1],rotated,[.9,.1])>.5


def test_incoherent_mixture_preserves_polarization_not_only_total_intensity():
    f=basis()[[0,4]]
    assert np.allclose(np.sum(abs(f[0])**2,axis=0),np.sum(abs(f[1])**2,axis=0))
    assert mixture_distance(f,[1.,0.],f,[0.,1.])>1.


def test_low_rank_mixture_metric_matches_explicit_dense_coherency():
    rng=np.random.default_rng(11)
    a=normalized_modes(rng.normal(size=(2,2,2,2))+1j*rng.normal(size=(2,2,2,2)))
    b=normalized_modes(rng.normal(size=(3,2,2,2))+1j*rng.normal(size=(3,2,2,2)))
    p=np.array([.3,.7]);q=np.array([.2,.1,.8])
    x=a.reshape(2,-1).T;y=b.reshape(3,-1).T
    ja=(x*p)@x.conj().T;jb=(y*q)@y.conj().T
    exact=np.linalg.norm(ja-jb)/max(np.linalg.norm(ja),np.linalg.norm(jb))
    assert np.isclose(mixture_distance(a,p,b,q),exact,rtol=1e-12)


def test_global_phase_and_paired_permutation_leave_mixture_unchanged():
    f=basis()[[0,4]];p=np.array([.2,.8])
    assert mixture_distance(f,p,f[::-1]*np.exp(1.3j),p[::-1])<3e-8


def test_new_modal_power_gate_catches_constant_total_output():
    f=basis()[[0,4]];gain=np.array([.03,.03])
    metrics=convergence_metrics(f,f,[.8,.2],gain,gain,
                  previous_fields=f,previous_powers=[.2,.8],previous_gain=gain)
    assert metrics['field_residual']<1e-12
    assert metrics['modal_power_relative_change']>1.
    assert metrics['mixture_relative_change']>.5


def test_convergence_requires_old_and_new_independent_gates():
    s=HotCavitySettings()
    keys=['field_residual','subspace_residual','mixture_stationarity','mixture_relative_change',
          'heat_residual','temperature_change_K','displacement_change_m','output_relative_change',
          'modal_power_relative_change','loss_residual','gain_stationarity','modal_gain_change','eigen_residual']
    row={k:0. for k in keys}
    row.update(eigen_converged=True,optical_periodic_converged=True)
    assert all(accepted_closure(row,s).values())
    for key in keys:
        broken={**row,key:1.}
        assert not all(accepted_closure(broken,s).values()),key
        broken={**row,key:None}
        assert not all(accepted_closure(broken,s).values()),key
    assert not all(accepted_closure({**row,'photon_seed_reset_branches':[1]},s).values())
    assert not all(accepted_closure({**row,'split_boundary_pairs':[{}]},s).values())


@pytest.mark.parametrize('bad',[np.zeros((2,2,2,2)),np.ones((2,2,2,2)),np.ones((2,3,2,2))])
def test_rank_deficient_or_wrong_polarization_fields_rejected(bad):
    with pytest.raises(ValueError):normalized_modes(bad)


@pytest.mark.parametrize('bad',[[1.,-1.],[np.nan,1.],[1.],[np.inf,0.]])
def test_invalid_modal_weights_rejected(bad):
    f=basis()[[0,4]]
    with pytest.raises(ValueError):mixture_distance(f,bad,f,[.5,.5])


def test_exact_degeneracy_setting_cannot_be_used_to_merge_split_pair():
    with pytest.raises(ValueError):TrackingSettings(exact_eigenvalue_tolerance=1e-4)


def test_stage7w_does_not_silently_promote_single_mode_request():
    with pytest.raises(ValueError,match='two explicit'):
        run_polarization_hot_cavity(None,None,None,1e-3,None,mode_count=1)


def test_insufficient_candidate_budget_rejected_before_solve():
    with pytest.raises(ValueError,match='additional candidate'):
        run_polarization_hot_cavity(None,None,None,1e-3,None,mode_count=4,
                                   settings=HotCavitySettings(eigen_candidates=4))


def test_real_arnoldi_solver_uses_nonhermitian_vector_operator():
    op,_,bank=spectrum()
    result=solve_tracked_eigenfields(op,bank[:2],candidates=4,tolerance=1e-8,maxiter=100)
    assert result.converged,result.status
    assert max(result.residuals)<1e-8
    assert result.operator_calls>0


def test_incomplete_arnoldi_bank_cannot_pass_using_good_partial_pair(monkeypatch):
    import hoyag.polarization_tracking as m
    op,values,bank=spectrum()
    def failed(*args,**kwargs):
        raise ArpackNoConvergence('forced partial result',values[:2],bank[:2].reshape(2,-1).T)
    monkeypatch.setattr(m,'eigs',failed)
    result=solve_tracked_eigenfields(op,bank[:2],candidates=4)
    assert not result.converged
    assert 'complete' in result.status


def test_all_fields_and_physics_components_are_reused_without_global_patching():
    import inspect
    import hoyag.polarized_resonator as m
    import hoyag.coupled_resonator as old
    assert m.PlateAssembly is old.PlateAssembly
    assert m.FieldCoupledLaser is old.FieldCoupledLaser
    source=inspect.getsource(m.run_polarization_hot_cavity)
    assert 'pump_source=source' in source
    assert 'fields=predicted.copy()' in source
    assert 's.field_relaxation)*' not in source
