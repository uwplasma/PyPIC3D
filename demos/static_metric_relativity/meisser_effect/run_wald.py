#!/usr/bin/env python3
"""Run the FPIC Schwarzschild-to-Kerr vacuum Wald demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm import tqdm

import wald_solution as wald

from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.diagnostics.async_writer import (
    create_async_tiled_openpmd_field_writer,
    enqueue_openpmd_field_output,
)
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.relativity.field_state import physical_vector
from PyPIC3D.solvers.GR_yee.static_metric import (
    compute_covariant_E,
    compute_covariant_H,
    update_B,
    update_D,
)
from PyPIC3D.utilities.simulation_helpers import setup_pmd_files


jax.config.update("jax_enable_x64", True)


def apply_target_absorber(
    field,
    target,
    field_kind,
    metric,
    static_parameters,
    dynamic_parameters,
    absorber_parameters,
    step_dt,
):
    """Damp density deviations toward the analytical Kerr field at the outer wall."""

    deviation = tuple(
        component - target_component
        for component, target_component in zip(field, target)
    )
    deviation = apply_tiled_supergaussian_absorber(
        deviation,
        absorber_parameters,
        dynamic_parameters,
        step_dt,
    )
    matched = tuple(
        target_component + deviation_component
        for target_component, deviation_component in zip(target, deviation)
    )
    locations = D_FIELD_LOCATIONS if field_kind == 'D' else B_FIELD_LOCATIONS
    return refresh_fields(matched, static_parameters, locations, field_kind, metric)


def step_vacuum_wald(
    state,
    target,
    metric,
    static_parameters,
    dynamic_parameters,
    absorber_parameters,
):
    """Advance one production static-metric leapfrog step of D/B densities with zero current."""

    D_n, B_n_minushalf, D_n_minusone, B_n_minusthreehalves = state
    D_target, B_target = target
    dt = dynamic_parameters.dt

    D_n_minushalf = tuple(
        0.5 * (D_n[i] + D_n_minusone[i]) for i in range(3)
    )
    B_n_minusone = tuple(
        0.5 * (B_n_minushalf[i] + B_n_minusthreehalves[i])
        for i in range(3)
    )

    E_n_minusonehalf = compute_covariant_E(
        D_n_minushalf,
        B_n_minushalf,
        metric,
    )
    B_n = update_B(
        E_n_minusonehalf,
        B_n_minusone,
        metric,
        static_parameters,
        dynamic_parameters,
        dt,
    )
    B_n = apply_target_absorber(
        B_n,
        B_target,
        'B',
        metric,
        static_parameters,
        dynamic_parameters,
        absorber_parameters,
        dt,
    )

    E_n = compute_covariant_E(D_n, B_n, metric)
    H_n = compute_covariant_H(D_n, B_n, metric)

    B_n_plushalf = update_B(
        E_n,
        B_n_minushalf,
        metric,
        static_parameters,
        dynamic_parameters,
        dt,
    )
    B_n_plushalf = apply_target_absorber(
        B_n_plushalf,
        B_target,
        'B',
        metric,
        static_parameters,
        dynamic_parameters,
        absorber_parameters,
        dt,
    )

    zero_current = tuple(jnp.zeros_like(component) for component in D_n)
    D_n_plushalf = update_D(
        D_n_minushalf,
        H_n,
        zero_current,
        metric,
        static_parameters,
        dynamic_parameters,
        dt,
    )
    D_n_plushalf = apply_target_absorber(
        D_n_plushalf,
        D_target,
        'D',
        metric,
        static_parameters,
        dynamic_parameters,
        absorber_parameters,
        dt,
    )

    H_n_plushalf = compute_covariant_H(D_n_plushalf, B_n_plushalf, metric)
    D_n_plusone = update_D(
        D_n,
        H_n_plushalf,
        zero_current,
        metric,
        static_parameters,
        dynamic_parameters,
        dt,
    )
    D_n_plusone = apply_target_absorber(
        D_n_plusone,
        D_target,
        'D',
        metric,
        static_parameters,
        dynamic_parameters,
        absorber_parameters,
        dt,
    )

    return D_n_plusone, B_n_plushalf, D_n, B_n_minushalf


def advance_steps(
    state,
    steps,
    target,
    metric,
    static_parameters,
    dynamic_parameters,
    absorber_parameters,
):
    """Advance a chunk without placing Python control flow in the hot loop."""

    def advance_one(_, current_state):
        return step_vacuum_wald(
            current_state,
            target,
            metric,
            static_parameters,
            dynamic_parameters,
            absorber_parameters,
        )

    return jax.lax.fori_loop(0, steps, advance_one, state)


def field_map(state, metric, B0):
    """Select the two requested PyPIC3D mesh diagnostics; files store physical B."""

    D, B = state[:2]
    return {
        "E_parallel": wald.parallel_electric_field(D, B, metric, B0),
        "B": physical_vector(B, metric.B),
    }


def output_paths(config):
    """Return the files owned by this demo run and its analysis."""

    output_dir = Path(config["output"]["directory"])
    name = config["output"]["field_filename"]
    extension = config["output"]["field_extension"]
    analysis = config["analysis"]
    return (
        output_dir / f"{name}{extension}",
        output_dir / f"{name}.pmd",
        output_dir / analysis["comparison_filename"],
        output_dir / analysis["movie_filename"],
    )


def prepare_output(config, overwrite):
    """Create the output directory without silently mixing two runs."""

    paths = output_paths(config)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Demo output already exists: {names}. Pass --overwrite to replace it."
        )

    paths[0].parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def run(config, overwrite=False):
    """Initialize, evolve, and dump the configured Wald relaxation."""

    prepare_output(config, overwrite)
    static_parameters, dynamic_parameters, absorber_parameters = (
        wald.build_pypic_parameters(config)
    )
    metric = wald.initialize_metric(static_parameters, dynamic_parameters)
    D_initial, B_initial, _, _ = wald.initialize_schwarzschild_seed(
        config,
        static_parameters,
        dynamic_parameters,
        metric,
    )
    D_target, B_target, _, _ = wald.initialize_kerr_target(
        config,
        static_parameters,
        dynamic_parameters,
        metric,
    )
    state = (D_initial, B_initial, D_initial, B_initial)

    output = config["output"]
    output_dir = Path(output["directory"])
    setup_pmd_files(
        str(output_dir),
        output["field_filename"],
        output["field_extension"],
    )
    writer = create_async_tiled_openpmd_field_writer(
        static_parameters,
        dynamic_parameters,
        str(output_dir),
        filename=output["field_filename"],
        file_extension=output["field_extension"],
        queue_size=int(output["queue_size"]),
    )

    target = (D_target, B_target)
    advance_chunk = jax.jit(
        lambda current_state, steps: advance_steps(
            current_state,
            steps,
            target,
            metric,
            static_parameters,
            dynamic_parameters,
            absorber_parameters,
        )
    )

    total_steps = int(static_parameters.Nt)
    dump_interval = int(output["dump_interval"])
    B0 = float(config["physics"]["B0"])
    completed_steps = 0
    dump_index = 0
    run_error = None

    try:
        enqueue_openpmd_field_output(
            writer,
            field_map(state, metric, B0),
            dynamic_parameters,
            dump_index,
            completed_steps,
        )

        progress = tqdm(total=total_steps, unit="step", desc="Wald evolution")
        while completed_steps < total_steps:
            steps = min(dump_interval, total_steps - completed_steps)
            state = advance_chunk(state, steps)
            jax.block_until_ready(state)

            completed_steps += steps
            dump_index += 1
            progress.update(steps)
            enqueue_openpmd_field_output(
                writer,
                field_map(state, metric, B0),
                dynamic_parameters,
                dump_index,
                completed_steps,
            )
        progress.close()
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        writer.close(raise_errors=run_error is None)

    print(f"Wrote {dump_index + 1} snapshots to {output_paths(config)[0]}")
    return state


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("wald_demo.toml"),
        help="Wald demo TOML file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the configured demo data and analysis artifacts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = wald.load_configuration(args.config)
    run(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
