"""Run a fresh BZ monopole using settings in simulation_parameters.py.

Run from this directory with:
    python run_bz_monopole.py

The fixed-step explicit simulation runs from zero to SimulationParameters.end_time.
Diagnostic snapshots and final particle/field arrays are saved as NumPy files.
Runtime metric, finite-state, displacement, capacity, and exterior field-constraint
checks remain enabled. Interruptions leave completed snapshots in place.
"""
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from PyPIC3D.boundary_conditions.staggered import refresh_fields as refresh_vector
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.relativity.field_state import physical_vector
from PyPIC3D.solvers.GR_yee.static_metric import (
    compute_covariant_E, compute_covariant_H, update_B, update_D)
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
if __package__:
    from .simulation_parameters import SimulationParameters, build_runtime, shard_array
    from .current_filter import filter_current
    from .magnetization import measure_magnetization
    from .plasma_injector import empty_particles, inject_pairs
    from .diagnostics import (
        constraint_residuals, validate_constraint_settings,
        check_constraints, diagnostics,
    )
    from .output import (
        RadialBoundaryBudget, plot_diagnostics, output_metadata,
        save_final_state, prepare_output_directory,
    )
else:
    from simulation_parameters import SimulationParameters, build_runtime, shard_array
    from current_filter import filter_current
    from magnetization import measure_magnetization
    from plasma_injector import empty_particles, inject_pairs
    from diagnostics import (
        constraint_residuals, validate_constraint_settings,
        check_constraints, diagnostics,
    )
    from output import (
        RadialBoundaryBudget, plot_diagnostics, output_metadata,
        save_final_state, prepare_output_directory,
    )

def monopole_field(p, metric, dynamic):
    """Continuum monopole density sqrt(gamma) B^r on the magnetic Yee locations."""
    theta = dynamic.grids.tiled_vertex_grid[1][..., None, :, None]
    Br = p.B0*jnp.sin(theta)*jnp.ones_like(metric.B[0].sqrt_gamma)
    return (Br, jnp.zeros_like(Br), jnp.zeros_like(Br))


def refresh_fields(vector, locations, static, metric):
    """Refresh D or B density halos, including the frozen horizon layers."""
    return refresh_vector(vector, static, locations, 'B' if locations == B_FIELD_LOCATIONS else 'D', metric)


def initialize_fields(p, static, dynamic, metric):
    """Densitized state at t=0 with Taylor-started D^{-1} and B^{-1/2}, B^{-3/2}."""
    B0 = tuple(shard_array(x, static) for x in monopole_field(p, metric, dynamic))
    B0=refresh_fields(B0, B_FIELD_LOCATIONS, static, metric)
    zero = jnp.zeros_like(B0[0])
    D0 = (zero,)*3
    def rhs(D, B):
        E = compute_covariant_E(D, B, metric)
        H = compute_covariant_H(D, B, metric)
        db = update_B(E, (zero,)*3, metric, static, dynamic, 1.)
        dd = update_D((zero,)*3, H, (zero,)*3, metric, static, dynamic, 1.)
        return (refresh_fields(dd, D_FIELD_LOCATIONS, static, metric),
                refresh_fields(db, B_FIELD_LOCATIONS, static, metric))
    dD, dB = rhs(D0, B0)
    ddD, ddB = rhs(dD, dB)
    def at(initial, first, second, t):
        return tuple(x+t*v+0.5*t*t*a for x, v, a in zip(initial, first, second))
    dt = dynamic.dt
    Bhalf = at(B0, dB, ddB, -0.5*dt)
    previous = (at(D0, dD, ddD, -dt), at(B0, dB, ddB, -1.5*dt))
    # add_external_fields expects a pair of vectors (D_external, B_external).
    external = (D0, D0)
    return (D0, Bhalf, D0, zero, zero, external, metric, previous, jnp.asarray(False)), B0


def apply_sponge(fields, background, p, static, dynamic):
    """Damp deviations from the monopole; report constraint changes separately.

    This is an absorbing numerical layer, not a charge-conserving physical source.
    Constraint diagnostics therefore exclude it and its adjacent stencil cells.
    The damping is pointwise and linear, so it acts on the densities directly.
    """
    D, B = fields[0], fields[1]
    def damp(vector, baseline, locations):
        result = []
        for i, loc in enumerate(locations):
            grid = dynamic.grids.tiled_center_grid if loc[0] == "C" else dynamic.grids.tiled_vertex_grid
            r = grid[0][..., :, None, None]
            ramp = jnp.clip((r-p.sponge_start)/(p.r_max-p.sponge_start), 0., 1.)**4
            factor = jnp.exp(-p.sponge_rate*dynamic.dt*ramp)
            result.append(baseline[i]+factor*(vector[i]-baseline[i]))
        return refresh_fields(tuple(result), locations, static, fields[6])
    zero = tuple(jnp.zeros_like(x) for x in D)
    return (damp(D, zero, D_FIELD_LOCATIONS), damp(B, background, B_FIELD_LOCATIONS))+fields[2:]


def make_step(p, species, static, dynamic, background, *,
              current_filter_passes=0):
    """Check deterministic kernels, leaving rejection sampling outside checkify.

    Errors cross the host boundary before any failed state can be consumed.
    """
    injection_steps = max(1, math.ceil(p.injection_interval/float(dynamic.dt)))
    @jax.jit
    def inject(particles, fields, key, index):
        metric = fields[6]
        # magnetization reads the physical field; the leapfrog seed reads densities
        mag = measure_magnetization(particles, species, physical_vector(fields[1], metric.B),
                                    metric, static, dynamic)
        return inject_pairs(particles, species, mag, fields[0], fields[1], metric,
                            static, dynamic, p, key, index)
    def evolve(particles, fields):
        transform = None
        if current_filter_passes:
            transform = lambda current: filter_current(current, static, current_filter_passes)
        errors, (particles, fields, boundary) = time_loop_static_metric(
            particles, species, fields, static, dynamic, return_diagnostics=True,
            return_errors=True, current_transform=transform)
        fields = apply_sponge(fields, background, p, static, dynamic)
        finite = finite_state(particles, fields)
        return errors, (particles, fields, boundary), finite
    checked = jax.jit(evolve)
    def execute(particles, fields, key, index):
        if int(index) % injection_steps == 0:
            # Key by injection event, not timestep: matched physical injection
            # schedules must draw the same candidates during dt refinement.
            event = index // injection_steps
            particles, key, report = inject(particles, fields, key, event)
            for error in report.errors:
                error.throw()
            requested, inserted, rejected = report.requested, report.inserted, report.rejected
            if bool(jnp.any(rejected)):
                raise RuntimeError(f"Injection capacity failure at step {index}: "
                                   f"requested={requested}, inserted={inserted}, rejected={rejected}")
        else:
            requested = inserted = rejected = jnp.zeros(p.devices, jnp.int64)
        errors, (particles, fields, boundary), finite = checked(particles, fields)
        errors.throw()
        if not bool(finite):
            raise FloatingPointError(f'Nonfinite particle or field state at step {index}')
        if bool(boundary.invalid_push):
            raise RuntimeError(f'Unsupported particle displacement at step {index}')
        if bool(fields[-1]):
            raise RuntimeError(f'Particle migration/capacity failure at step {index}')
        check_sharding(particles, fields, static)
        return particles, fields, key, (requested, inserted, rejected, boundary)
    return execute


def finite_state(particles, fields):
    valid = jnp.all(~particles.active | (jnp.isfinite(particles.x).all(-1)
                                        & jnp.isfinite(particles.u).all(-1)))
    for value in (*fields[0], *fields[1], *fields[2], fields[3], fields[4],
                  *fields[7][0], *fields[7][1]):
        valid &= jnp.isfinite(value).all()
    return valid


def check_sharding(particles, fields, static):
    from jax.sharding import NamedSharding, PartitionSpec
    layout = NamedSharding(static.field_mesh, PartitionSpec('tile_x', 'tile_y', 'tile_z'))
    for value in (*jax.tree.leaves(particles), *fields[0], *fields[1], *fields[2],
                  *fields[7][0], *fields[7][1]):
        if not value.sharding.is_equivalent_to(layout, value.ndim):
            raise RuntimeError('Particle or field state lost radial device sharding')


def evolve(particles, species, fields, key, p, static, dynamic, background, output):
    """Evolve a fresh initial state to p.end_time, saving checked snapshots."""
    validate_constraint_settings(p.gauss_tolerance, p.magnetic_divergence_tolerance,
                                 p.constraint_check_interval)
    output = prepare_output_directory(output)
    dt = float(dynamic.dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    target = math.ceil(p.end_time/dt)
    metadata = output_metadata(p, static, dynamic)
    execute = make_step(p, species, static, dynamic, background,
                        current_filter_passes=p.current_filter_passes)
    measure = jax.jit(lambda pts, fs: constraint_residuals(
        pts, species, fs, static, dynamic, p, current_filter_passes=p.current_filter_passes))
    budget = RadialBoundaryBudget()
    next_snapshot = p.output_interval

    def check_state(pts, fs, step):
        try:
            check_constraints(measure(pts, fs),
                              gauss_tolerance=p.gauss_tolerance,
                              magnetic_divergence_tolerance=p.magnetic_divergence_tolerance)
        except (RuntimeError, FloatingPointError) as error:
            raise type(error)(f'Step {step}, t={step*dt:g}: {error}') from error

    def snapshot(step):
        data = diagnostics(particles, species, fields, p, static, dynamic,
                           current_filter_passes=p.current_filter_passes)
        data.update(metadata, step=np.asarray(step), time=np.asarray(step*dt),
                    boundary_budget=budget.as_array())
        np.savez(output/f'snapshot_{step:012d}.npz', **data)
        return data

    check_sharding(particles, fields, static)
    if not bool(finite_state(particles, fields)):
        raise FloatingPointError('Nonfinite initial state')
    check_state(particles, fields, 0)
    snapshot(0)
    last_refresh = time.perf_counter()
    with tqdm(total=target, desc='BZ monopole', unit='step', mininterval=1.,
              dynamic_ncols=True) as progress:
        progress.set_postfix(time='0 M', active=int(particles.active.sum()), refresh=False)
        for step in range(1, target+1):
            new, newfields, newkey, report = execute(particles, fields, key, step-1)
            if step % p.constraint_check_interval == 0 or step == target:
                check_state(new, newfields, step)
            particles, fields, key = new, newfields, newkey
            _, _, _, boundary = report
            budget.accumulate(boundary, dt)
            now = time.perf_counter()
            if now-last_refresh >= 1. or step == target:
                progress.set_postfix(time=f'{step*dt:g} M', active=int(particles.active.sum()),
                                     refresh=False)
                last_refresh = now
            progress.update(1)
            if step*dt >= next_snapshot or step == target:
                data = snapshot(step)
                next_snapshot = (math.floor(step*dt/p.output_interval)+1)*p.output_interval
    np.savez(output/'diagnostics.npz', **data)
    save_final_state(output/'final_state.npz', particles, species, fields, target,
                     metadata, dynamic, budget.as_array())
    plot_diagnostics(data, output/'figure6_diagnostics.png')
    return particles, fields, key, target


def run(parameters=None):
    p = parameters if parameters is not None else SimulationParameters()
    validate_constraint_settings(p.gauss_tolerance, p.magnetic_divergence_tolerance,
                                 p.constraint_check_interval)
    output = prepare_output_directory(p.output_directory)
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_platforms', 'cpu' if p.backend == 'cpu' else 'cuda')
    if jax.default_backend() not in (('cpu',) if p.backend == 'cpu' else ('gpu', 'cuda')):
        raise RuntimeError(f'The initialized JAX backend differs from backend={p.backend!r}')
    if len(jax.devices()) < p.devices:
        raise RuntimeError(f'Requested {p.devices} devices, but only {len(jax.devices())} are visible')
    if p.current_filter_passes and p.r_min+(p.current_filter_passes+2)*p.dr >= p.horizon:
        raise ValueError('Source filter and absorption stencil must fit inside the horizon; increase nr')
    static, dynamic, metric, _ = build_runtime(p)
    particles, species = empty_particles(p, static)
    fields, background = initialize_fields(p, static, dynamic, metric)
    return evolve(particles, species, fields, jax.random.PRNGKey(p.seed), p,
                  static, dynamic, background, output)


def main():
    import sys
    if len(sys.argv) != 1:
        raise SystemExit('This runner takes no arguments; edit simulation_parameters.py.')
    run(SimulationParameters())


if __name__ == '__main__':
    main()
