import numpy as np

from hoyag.family_planning import diagnose_candidate_bank


class DiagonalOperator:
    shape = (2, 1, 6)
    def __init__(self, values):
        self.values = np.asarray(values)
    def propagate(self, field):
        return (self.values.reshape(self.shape)*field, None)


def test_family_boundary_and_finite_bank_are_explicit():
    # Same spatial pixel, orthogonal polarizations, nearly split eigenvalues.
    values = np.array([1-.01*i for i in range(12)], complex)
    values[9] = .96999
    fields = np.eye(12).reshape(12,2,1,6)
    # Indices 3 and 9 are distinct polarization states of the same pixel.
    result = diagnose_candidate_bank(DiagonalOperator(values), values, fields,
                                     counts=(2,4,6,8,12))
    assert result['boundaries']['12']['status']=='candidate_bank_insufficient'
    assert result['boundaries']['4']['status']=='split_polarization_family'
    assert result['modes_6_complete_family_candidate'] is True
    assert result['global_completeness_verified'] is False
    assert max(result['full_operator_residuals']) < 1e-14


def test_bad_eigenpair_never_admissible():
    values=np.array([1-.01*i for i in range(12)],complex)
    fields=np.eye(12).reshape(12,2,1,6)
    result=diagnose_candidate_bank(DiagonalOperator(values),values+.1,fields)
    assert result['status']=='candidate_eigenpair_failure'
    assert result['admissible_retained_counts']==[]
