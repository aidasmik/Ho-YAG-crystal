"""Run the Stage 4R thin-disk oscillator, saving actual histories and metadata.

From the repository root:
    pip install -e '.[dev]'
    python examples/thin_disk_resonator.py --sweep 100 300 600 1000
    python examples/thin_disk_resonator.py --pump-uJ 1000 --charges 0,1,-1,2,-2

The single-charge LG1 option is a constrained-mode calculation, not evidence
that an ordinary two-mirror resonator spontaneously generates a vortex.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from hoyag.resonator import ThinDiskResonator, radial_laser

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/thin_disk_resonator.json')
    parser.add_argument('--pump-uJ', type=float, default=1000.0)
    parser.add_argument('--sweep', type=float, nargs='+')
    parser.add_argument('--charges', default='0')
    parser.add_argument('--nr', type=int)
    parser.add_argument('--nz', type=int)
    parser.add_argument('--max-cycles', type=int)
    parser.add_argument('--output', type=Path, default=ROOT/'results/resonator/generated')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cavity = ThinDiskResonator(**cfg['cavity'])
    charges = tuple(int(x) for x in args.charges.split(','))
    num = cfg['numerics']; pump = cfg['pump']
    model, radius = radial_laser(
        cavity, charges=charges,
        radial_points=args.nr or num['radial_quadrature_points'],
        z_slices=args.nz or num['z_slices'], pump_waist_m=pump['waist_radius_m'],
        pump_absorption_m2=pump['effective_absorption_cross_section_m2'],
        spontaneous_fraction_per_mode=num['spontaneous_fraction_per_mode'],
    )
    if not np.isclose(pump['wavelength_m'], model.params.pump_wavelength_m, rtol=1e-9, atol=0):
        raise ValueError('Change material pump cross sections/wavelength together, not the configuration label alone.')
    if pump['traversals'] != model.pump_passes:
        raise ValueError('This example uses a two-traversal HR-backed pump; configure other paths explicitly.')
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(cavity.summary(), indent=2))
    summary = []
    for uJ in args.sweep or [args.pump_uJ]:
        result = model.run(
            uJ*1e-6, pump['repetition_rate_Hz'],
            max_cycles=args.max_cycles or num['max_cycles'],
            rtol=num['relative_tolerance'],
            periodic_tolerance=num['population_periodic_tolerance'],
            energy_tolerance=num['photon_and_output_periodic_tolerance'],
            max_period_cycles=num['max_period_cycles'],
            pump_fwhm_s=pump['duration_fwhm_s'],
        )
        n_average = result.period_cycles or min(20, result.cycles_simulated)
        average_output = result.history[-n_average:,10:].mean(axis=0)*pump['repetition_rate_Hz']
        label = f"pump_{uJ:g}uJ_l" + '_'.join(map(str, charges))
        meta = {**result.metadata,
                'periodic_converged': result.periodic_converged,
                'period_cycles': result.period_cycles,
                'cycles_simulated': result.cycles_simulated,
                'mode_labels': result.mode_labels,
                'averaging_cycles': n_average,
                'average_output_W_by_mode': average_output.tolist(),
                'status': 'periodic' if result.periodic_converged else 'finite-window transient only'}
        (args.output/(label+'.json')).write_text(json.dumps(meta, indent=2))
        np.savez_compressed(args.output/(label+'.npz'),
            history=result.history, waveform=result.waveform,
            fractions=result.fractions_before_next_pump,
            log_photons=result.log_photon_number, r=radius, area=model.area,
            density=model.density, modes=model.modes, pump=model.pump)
        columns = ['pump_cycle','time_s','absorbed_pump_J','escaped_pump_J','pump_optics_loss_J',
                   'population_change_one_cycle','photon_change_one_cycle','output_change_one_cycle',
                   'peak_output_W','max_postpump_log_gain_excess']
        columns += [f'output_mode_{i}_J' for i in range(model.nm)]
        np.savetxt(args.output/(label+'_history.csv'), result.history, delimiter=',',
                   header=','.join(columns), comments='')
        np.savetxt(args.output/(label+'_waveform.csv'),result.waveform,delimiter=',',
                   header='time_s,'+','.join(f'output_mode_{i}_W' for i in range(model.nm)),comments='')
        row = {'pump_uJ':uJ,'pump_average_W':uJ*1e-6*pump['repetition_rate_Hz'],
               'output_average_W':float(average_output.sum()),
               'periodic_converged':result.periodic_converged,
               'period_pump_cycles':result.period_cycles,'cycles_simulated':result.cycles_simulated}
        summary.append(row)
        print(row, flush=True)
    with (args.output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)


if __name__ == '__main__':
    main()
