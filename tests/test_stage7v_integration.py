"""Integration boundary test against a COMPLETE audited repository installation.

This intentionally nonconverged one-cycle run tests that orchestration calls the
real Stage 7 API and retains nonconvergence rather than calling it a pass. It is
not a coupled-grid-refinement study. A partial offline kernel snapshot skips it.
"""
from copy import deepcopy
from pathlib import Path
import pytest

pytest.importorskip('hoyag.coupled_resonator', reason='requires complete audited Stage 7 source')
from hoyag.validation_plan import physics_from_repository, make_plan
from hoyag.validation_backend import coupled_case


def test_real_coupled_api_failure_to_converge_remains_visible():
    root=Path(__file__).resolve().parents[1]
    case=deepcopy(make_plan(physics_from_repository(root))['cases'][0])
    n=case['numerics'];n['optical_n']=128
    n['thermal'].update(nr=6,nz=2,nphi=4)
    n['mechanical'].update(nr=2,ntheta=8,outer_rings=1,nz_disk=2,nz_plate=2)
    n['plate_thermal_nz']=2
    n['settings'].update(max_outer_iterations=1,minimum_outer_iterations=1,
                         optical_min_cycles=1,optical_max_cycles=1)
    record,arrays=coupled_case(case)
    assert record['status']=='not_converged'
    assert record['metrics']['coupled_converged'] is False
    assert arrays['fields_used'].shape==(1,2,128,128)
