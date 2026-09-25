"""Export ordinary SI CSV and NPZ arrays; no network access is required."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import warnings

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import yb_yag as m


def export(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    wl = np.arange(905.0, 1096.0)
    sa, se = m.cross_sections_m2(wl)
    rows = np.column_stack([wl, sa, se])
    np.savetxt(output_dir / 'rt_cross_sections_SI.csv', rows, delimiter=',',
               header='wavelength_nm,sigma_abs_m2,sigma_em_m2', comments='', fmt='%.10g')
    np.savez_compressed(output_dir / 'rt_cross_sections.npz', wavelength_nm=wl,
                        temperature_K=np.array(293.15), sigma_abs_m2=sa, sigma_em_m2=se,
                        quality=np.array('legacy_repository_array'))
    laser_wl = np.arange(1020.0, 1061.0)  # Resampling does not add information.
    temps = np.array([293.15, 353.15, 413.15, 473.15])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', m.ApproximationWarning)
        aa, ee = m.cross_sections_m2(laser_wl[None,:], temps[:,None],
                                    dataset='laser_band_figure', allow_approximate=True)
    np.savez_compressed(output_dir / 'laser_band_figure_APPROXIMATE.npz',
                        wavelength_nm=laser_wl, temperature_K=temps,
                        sigma_abs_m2=aa, sigma_em_m2=ee,
                        quality=np.array('manual_emission_readings_and_McCumber_derived_absorption'))
    with (output_dir / 'laser_band_figure_APPROXIMATE.csv').open('w', newline='') as f:
        w=csv.writer(f);w.writerow(['temperature_K','wavelength_nm','sigma_abs_m2','sigma_em_m2','quality'])
        for i, T in enumerate(temps):
            for j, x in enumerate(laser_wl):
                w.writerow([T,x,aa[i,j],ee[i,j],'manual_emission_and_derived_absorption'])
    with (output_dir / 'RT_absorption_vs_doping.csv').open('w', newline='') as f:
        w=csv.writer(f);w.writerow(['Yb_at_percent','wavelength_nm','alpha_m-1','quality'])
        for c in (2,5,10,12,15):
            for x, a in zip(wl, m.absorption_coefficient_m1(wl,c)):
                w.writerow([c,x,a,'derived_N_times_legacy_sigma;not_doping_resolved_measurement'])
    optical_wl=np.arange(400.0,5001.0,5.0)
    np.savetxt(output_dir/'host_refractive_index.csv',np.column_stack([optical_wl,m.n_yag(optical_wl)]),
               delimiter=',',header='wavelength_nm,n_host_RT',comments='',fmt='%.10g')
    T=np.arange(160.0,501.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',m.ApproximationWarning)
        dn=m.dn_dT_per_K(T,dataset='sato2025',allow_apparent=True)
    out=np.column_stack([T,m.thermal_conductivity_host(T),m.heat_capacity_host_J_kgK(T),
                         m.thermal_expansion_per_K(T,dataset='sato2025'),dn,m.density_host_kg_m3(T)])
    np.savetxt(output_dir/'host_thermal_Sato2025.csv',out,delimiter=',',comments='',fmt='%.10g',
        header='temperature_K,k_W_mK,Cp_J_kgK,alpha_per_K,apparent_dn_dT_per_K,rho_kg_m3_derived')
    (output_dir/'WARNINGS.txt').write_text(
        'RT spectra are exact legacy software arrays, not verified spectrometer raw data.\n'
        'Temperature-dependent laser-band spectra are coarse manual figure readings.\n'
        'Their absorption is derived, not measured independently. No pump-band high-T spectra.\n'
        'Host properties are not Yb-doping calibrations. Sato dn/dT may include mounting stress.\n'
        'Dense resampling is not improved experimental resolution. See source package provenance.\n')
    return sorted(p.name for p in output_dir.iterdir() if p.is_file())


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'generated')
    args=ap.parse_args()
    print(json.dumps(export(args.output),indent=2))
