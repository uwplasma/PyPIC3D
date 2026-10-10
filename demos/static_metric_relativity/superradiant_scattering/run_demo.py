"""Run the source-free Cartesian Kerr PEC experiment from the repository root."""
import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import jax

from .analysis import analyze
from .diagnostics import measure, snapshot
from .evolution import VacuumEvolution
from .initial_data import seed_fields
from .parameters import Parameters, load_parameters, build_runtime


ARTIFACTS = ('energy.csv', 'metadata.json', 'growth.json', 'energy_growth.png')
advance = jax.jit(lambda solver, state, count: solver.advance(state, count))
initialize = jax.jit(lambda solver, D, B: solver.initialize(D, B))
diagnose = jax.jit(snapshot)
measure_pair = jax.jit(lambda solver, D, B: measure(D, B, solver))


def host_diagnostics(values):
    result = {key: float(value) for key, value in jax.device_get(values).items()}
    if not all(math.isfinite(value) for value in result.values()) or not result['finite_fields']:
        raise FloatingPointError('Nonfinite field or diagnostic; run stopped')
    if min(result['div_D_cells'], result['div_B_cells']) <= 0:
        raise ValueError('Empty exterior constraint diagnostic region')
    return result


def run(p=Parameters(), *, overwrite=False):
    p.validate()
    output = Path(p.output_directory)
    existing = [output/name for name in ARTIFACTS if (output/name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f'Demo artifacts already exist in {output}; pass --overwrite')
    started = time.monotonic()
    print(f'Building {p.cells}^3 metric and vacuum state (first JIT compilation can take a few minutes)...', flush=True)
    static, dynamic, metric, steps, cfl_dt = build_runtime(p)
    solver = VacuumEvolution(p, static, dynamic, metric)
    D, B = seed_fields(p, dynamic)
    state = initialize(solver, D, B)
    initial = host_diagnostics(diagnose(solver, state))
    if initial['fido_energy_exterior'] <= 0 or initial['radial_fido_flux_moment'] >= 0:
        raise ValueError('Seed must have positive exterior energy and inward FIDO flux')
    if max(initial['div_D_relative'], initial['div_B_relative']) > 1.e-10:
        raise ValueError('Seed failed vacuum divergence acceptance')
    scale = math.sqrt(p.initial_energy/initial['fido_energy_exterior'])
    state = jax.tree.map(lambda a: a*scale, state)
    initial = host_diagnostics(diagnose(solver, state))
    dt = float(dynamic.dt)
    interval_steps = max(1, round(p.output_interval/dt))
    metadata = dict(parameters=asdict(p), dt=dt, cfl_dt=cfl_dt, total_steps=steps,
                    actual_output_interval=interval_steps*dt, horizon=p.horizon,
                    omega_h=p.omega_h, carrier_in_superradiant_band=bool(
                        not p.flat and 0 < p.omega < p.omega_h),
                    backend=jax.default_backend(), jax_version=jax.__version__,
                    units='Gaussian G=c=1; spin=a/M; lengths and times in code units',
                    boundaries='Existing FIDO D/B PEC on all six faces',
                    horizon_flux='Positive into BH; coordinate staircase at r=r_plus',
                    status='running', completed_steps=0)
    output.mkdir(parents=True, exist_ok=True)
    # Delete only demo-owned files after initialization has succeeded.
    for path in existing:
        path.unlink()
    metadata_path = output/'metadata.json'

    def save_metadata():
        metadata['elapsed_seconds'] = time.monotonic()-started
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')

    save_metadata()
    print(f'{p.cells}^3 vacuum cells; dt={dt:.6g}; steps={steps}; backend={jax.default_backend()}', flush=True)
    cumulative_flux = 0.
    previous = None
    try:
        with (output/'energy.csv').open('w', newline='') as stream:
            writer = None
            done = 0
            while True:
                values = initial if done == 0 else host_diagnostics(diagnose(solver, state))
                now = done*dt
                flux = values['killing_flux_horizon_in']+values['killing_flux_wall_out']
                if previous is not None:
                    cumulative_flux += (now-previous[0])*(flux+previous[1])/2
                previous = now, flux
                row = dict(step=done, time=now, **values,
                           energy_amplification=values['fido_energy_exterior']/initial['fido_energy_exterior'],
                           equivalent_amplitude=math.sqrt(values['fido_energy_exterior']/initial['fido_energy_exterior']),
                           cumulative_killing_flux_out=cumulative_flux,
                           killing_budget_residual=values['killing_energy_exterior']-
                               initial['killing_energy_exterior']+cumulative_flux)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                metadata['completed_steps'] = done
                save_metadata()
                print(f't={now:.6g}: U_ext={values["fido_energy_exterior"]:.8g}, '
                      f'gain={row["energy_amplification"]:.6g}', flush=True)
                if done == steps:
                    break
                count = min(interval_steps, steps-done)
                state = advance(solver, state, count)
                jax.block_until_ready(state)
                done += count
        metadata['status'] = 'complete'
        save_metadata()
        summary = analyze(output)
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        save_metadata()
        raise
    print(f'Wrote {output}/energy.csv and energy_growth.png; fit: {summary["status"]}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('kerr.toml'))
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    p = load_parameters(args.config)
    if args.output_dir is not None:
        p = replace(p, output_directory=args.output_dir)
    run(p, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
