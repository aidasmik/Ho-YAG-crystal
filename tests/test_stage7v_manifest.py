"""A byte-valid cached state cannot qualify a different requested case."""
from copy import deepcopy
import numpy as np
from hoyag.validation_metrics import fingerprint
from hoyag.validation_campaign import make_plan,write_json,file_sha256,verify_saved,compile_report


def test_report_rejects_same_directory_with_different_manifest(tmp_path):
    plan=make_plan('pilot');case=plan['cases'][0]
    folder=tmp_path/'base256';folder.mkdir()
    x=np.linspace(-.003,.003,32)
    field=np.zeros((1,2,32,32),complex);field[0,0]=1
    np.savez(folder/'state.npz',x_m=x,y_m=x,fields=field,mean_opd_m=np.zeros((32,32)))
    summary={'case':case,'status':'COMPLETED',
      'identity':{'case_fingerprint':fingerprint(case),'physics_fingerprint':'same',
                  'solver_source_sha256':'code','source_state_sha256':None},
      'scalars':{'output_W':1.,'heat_W':.6,'absorbed_W':2.,'disk_peak_K':302.,'plate_peak_K':294.,
                 'thermal_balance_relative':0.,'mechanical_residual':0.,'energy_ledger_relative':0.},
      'state_sha256':file_sha256(folder/'state.npz'),'elapsed_s':1.,'edge_power_fraction':0.,
      'mirror_sampling':{'nyquist_satisfied_over_radius':True},'probes':[]}
    write_json(folder/'summary.json',summary)
    assert verify_saved(folder) is not None
    correct=compile_report(plan,tmp_path)
    assert correct['cases'][0]['status']=='COMPLETED'
    changed=deepcopy(plan);changed['cases'][0]['grid']['optical_n']=512
    wrong=compile_report(changed,tmp_path)
    assert wrong['cases'][0]['status']=='INVALID_CONFIG'
    assert 'base256:INVALID_CONFIG' in wrong['case_quality_issues']
    assert 'seeded_probes' in wrong['uncompleted_measurements']
    assert not wrong['numerical_qualification_passed']
    # Altering only the stored descriptive input is also detected.
    summary['case']=changed['cases'][0]
    write_json(folder/'summary.json',summary)
    wrong=compile_report(plan,tmp_path)
    assert wrong['cases'][0]['status']=='INVALID_CONFIG'
