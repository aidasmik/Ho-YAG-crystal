"""Small deterministic audit; launch under hoyag.local_supervisor."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
from ybluag.model import YbLuAGMaterial, H, C
from ybluag.diagnostics import gain_feasibility
from ybluag.multipass_pump import transport_multipass_pump, recover_pumped_population


def main():
    material = YbLuAGMaterial(yb_at_percent=12, lifetime_s=.000973,
                              pump_wavelength_nm=969)
    beta = np.array([[0.0, .2]])
    scale = np.ones_like(beta)
    mean, absorbed, _ = transport_multipass_pump(material, np.full(2, 1e8),
                                                scale, .001, 3, beta)
    up, down = material.rates_s1(mean, 0)
    predicted = material.number_density_m3*.001*(up*(1-beta)-down*beta)
    photons = absorbed/(H*C/969e-9)
    recovery = []
    initial = np.full((2, 1), .01)
    for count in (4, 8, 16):
        final, pump_integral, excited_integral = recover_pumped_population(
            material, np.full(1, 1e8), np.ones_like(initial), 100e-6, 10,
            initial, 1e-4, count)
        pump_photons = pump_integral/(H*C/969e-9)
        used = material.number_density_m3*50e-6*(
            final-initial+excited_integral/material.lifetime_s)
        recovery.append({'substeps':count,
                         'photon_relative_L1':float(np.sum(abs(pump_photons-used))/np.sum(pump_photons))})
    files = sorted((ROOT/'src/ybluag').glob('*.py')) + [ROOT/'config/ybslam_proposal_luag.json']
    fingerprint = hashlib.sha256()
    for path in files:
        fingerprint.update(path.relative_to(ROOT).as_posix().encode())
        fingerprint.update(path.read_bytes())
    report = {
        'numerical_source_sha256':fingerprint.hexdigest(),
        'configuration': {'pump_nm':969, 'signal_nm':1030, 'yb_at_percent':12,
                          'lifetime_s_assumed':.000973, 'thickness_m':100e-6,
                          'pump_passes':10, 'signal_traversals':10},
        'thick_cell_photon_relative_max':float(np.max(abs(predicted/photons-1))),
        'depleted_recovery_refinement':recovery,
        'gain_bound':gain_feasibility(material, thickness_m=100e-6, pump_passes=10,
                                     signal_traversals=10, input_energy_J=10e-9,
                                     requested_output_energy_J=100e-6),
        'seed_fwhm_duty_fraction':10e-12*1e4,
        'scope':'Cold homogeneous analytical bound and small recovery fixtures; not a device validation.',
    }
    output = ROOT/'results/ybluag_desktop_audit/physics_checks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
