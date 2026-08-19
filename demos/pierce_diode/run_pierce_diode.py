#!/usr/bin/env python3
"""Run the warm one-dimensional Pierce diode with cathode injection."""

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import toml
from tqdm import tqdm

from PyPIC3D.diagnostics.async_writer import (
    create_async_tiled_openpmd_field_writer,
    create_async_tiled_openpmd_particle_writer,
    enqueue_openpmd_field_output,
    enqueue_openpmd_particle_output,
)
from PyPIC3D.diagnostics.output_adapters import build_field_output_map
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.particles.particle_tile_communication import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.pusher.particle_push import particle_push
from PyPIC3D.solvers.electrostatic.electrostatic_yee import (
    calculate_electrostatic_fields,
)
from PyPIC3D.solvers.electrostatic.time_loop import time_loop_electrostatic
from PyPIC3D.utilities.field_helpers import add_external_fields
from PyPIC3D.utilities.grids import grid_domain_bounds
from PyPIC3D.utilities.simulation_helpers import setup_pmd_files


CONFIG_FILE = Path(__file__).with_name("pierce_diode.toml")


def load_configuration(config_path):
    """Load a Pierce TOML and keep its output beside the input file."""

    config_path = Path(config_path).expanduser().resolve()
    config = toml.load(config_path)
    config.setdefault("simulation_parameters", {}).setdefault(
        "output_dir",
        str(config_path.parent),
    )
    return config


def species_index(species_names, requested_name):
    """Return the configured species index used by the tiled arrays."""

    try:
        return tuple(species_names).index(requested_name)
    except ValueError as exc:
        raise ValueError(
            f"Pierce species '{requested_name}' was not initialized. "
            f"Available species: {tuple(species_names)}"
        ) from exc


def match_neutralizing_ions(particles, electron_index, ion_index):
    """Collocate stationary ions with electrons for an exactly neutral start."""

    electron_x = particles.x[:, :, :, electron_index, :, :]
    electron_active = particles.active[:, :, :, electron_index, :]

    x = particles.x.at[:, :, :, ion_index, :, :].set(electron_x)
    u = particles.u.at[:, :, :, ion_index, :, :].set(0.0)
    active = particles.active.at[:, :, :, ion_index, :].set(electron_active)

    return TiledParticles(x=x, u=u, active=active)


def injected_macroparticle_count(step, injection_rate):
    """Return the deterministic source count for one timestep."""

    step = jnp.asarray(step)
    injection_rate = jnp.asarray(injection_rate)
    injected_before = jnp.floor(step * injection_rate).astype(jnp.int32)
    injected_after = jnp.floor((step + 1) * injection_rate).astype(jnp.int32)
    return injected_after - injected_before, injected_before


def inject_electrons(
    particles,
    step,
    random_key,
    dynamic_parameters,
    *,
    electron_index,
    injection_rate,
    beam_velocity,
    thermal_velocity,
    max_injected_per_step,
):
    """Fill inactive cathode-tile slots with the particles born this step."""

    n_injected, injected_before = injected_macroparticle_count(
        step,
        injection_rate,
    )
    lanes = jnp.arange(max_injected_per_step, dtype=jnp.int32)
    random_key, velocity_key, y_key, z_key = jax.random.split(random_key, 4)
    velocity = beam_velocity * jnp.array((1.0, 0.0, 0.0))
    velocity = velocity + thermal_velocity * jax.random.normal(
        velocity_key,
        shape=(max_injected_per_step, 3),
    )

    (x_bounds, y_bounds, z_bounds) = grid_domain_bounds(dynamic_parameters)
    global_injection_index = injected_before + lanes + 1
    safe_injection_rate = jnp.where(injection_rate > 0.0, injection_rate, 1.0)
    birth_step = global_injection_index / safe_injection_rate
    particle_age = (step + 1 - birth_step) * dynamic_parameters.dt

    x = x_bounds[0] + velocity[:, 0] * particle_age
    y = jax.random.uniform(
        y_key,
        shape=(max_injected_per_step,),
        minval=y_bounds[0],
        maxval=y_bounds[1],
    )
    z = jax.random.uniform(
        z_key,
        shape=(max_injected_per_step,),
        minval=z_bounds[0],
        maxval=z_bounds[1],
    )
    injected_x = jnp.stack((x, y, z), axis=-1)

    cathode_x = particles.x[0, 0, 0, electron_index]
    cathode_u = particles.u[0, 0, 0, electron_index]
    cathode_active = particles.active[0, 0, 0, electron_index]
    free = ~cathode_active
    free_rank = jnp.cumsum(free.astype(jnp.int32)) - 1
    candidate_index = jnp.clip(free_rank, 0, max_injected_per_step - 1)
    fill = free & (free_rank < n_injected)

    cathode_x = jnp.where(
        fill[:, jnp.newaxis],
        injected_x[candidate_index],
        cathode_x,
    )
    cathode_u = jnp.where(
        fill[:, jnp.newaxis],
        velocity[candidate_index],
        cathode_u,
    )
    cathode_active = cathode_active | fill

    x_all = particles.x.at[0, 0, 0, electron_index].set(cathode_x)
    u_all = particles.u.at[0, 0, 0, electron_index].set(cathode_u)
    active_all = particles.active.at[0, 0, 0, electron_index].set(
        cathode_active
    )
    overflow = n_injected > jnp.sum(free.astype(jnp.int32))

    # Unused candidates are deliberately sampled so the JAX shape and random
    # key progression do not depend on the fractional source count.
    return (
        TiledParticles(x=x_all, u=u_all, active=active_all),
        random_key,
        n_injected,
        overflow,
    )


def pierce_diode_step(
    particles,
    species_config,
    fields,
    dynamic_parameters,
    step,
    random_key,
    *,
    static_parameters,
    electron_index,
    injection_rate,
    beam_velocity,
    thermal_velocity,
    max_injected_per_step,
):
    """Advance one electrostatic step with cathode injection before deposition."""

    (
        E_tiles,
        B_tiles,
        J_tiles,
        rho_tiles,
        phi_tiles,
        external_fields,
        pml_state,
        overflow_previous,
    ) = fields

    push_E_tiles, push_B_tiles = add_external_fields(
        E_tiles,
        B_tiles,
        external_fields,
    )
    particles = particle_push(
        particles,
        species_config,
        push_E_tiles,
        push_B_tiles,
        static_parameters,
        dynamic_parameters,
    )
    particles = update_tiled_particle_positions(
        particles,
        species_config,
        dynamic_parameters.dt,
    )

    particles, random_key, n_injected, injection_overflow = inject_electrons(
        particles,
        step,
        random_key,
        dynamic_parameters,
        electron_index=electron_index,
        injection_rate=injection_rate,
        beam_velocity=beam_velocity,
        thermal_velocity=thermal_velocity,
        max_injected_per_step=max_injected_per_step,
    )
    particles, retile_overflow = refresh_tiled_particle_tiles(
        particles,
        static_parameters,
        dynamic_parameters,
    )

    E_tiles, phi_tiles, rho_tiles = calculate_electrostatic_fields(
        static_parameters,
        dynamic_parameters,
        particles,
        species_config,
        rho_tiles,
        phi_tiles,
    )

    overflow = overflow_previous | injection_overflow | retile_overflow
    fields = (
        E_tiles,
        B_tiles,
        J_tiles,
        rho_tiles,
        phi_tiles,
        external_fields,
        pml_state,
        overflow,
    )
    return particles, fields, random_key, n_injected


def make_jitted_pierce_step(static_parameters, source_parameters, electron_index):
    """Close static solver and source choices over the hot timestep."""

    injection_rate = float(source_parameters["macro_injection_per_timestep"])
    max_injected_per_step = max(1, math.ceil(injection_rate))
    beam_velocity = float(source_parameters["beam_velocity"])
    thermal_velocity = float(source_parameters["thermal_velocity"])

    def step_function(
        particles,
        species_config,
        fields,
        dynamic_parameters,
        step,
        random_key,
    ):
        return pierce_diode_step(
            particles,
            species_config,
            fields,
            dynamic_parameters,
            step,
            random_key,
            static_parameters=static_parameters,
            electron_index=electron_index,
            injection_rate=injection_rate,
            beam_velocity=beam_velocity,
            thermal_velocity=thermal_velocity,
            max_injected_per_step=max_injected_per_step,
        )

    return jax.jit(step_function)


def _raise_if_overflowed(fields):
    if bool(jax.device_get(fields[-1])):
        raise RuntimeError(
            "Pierce particle injection or tiled particle communication "
            "exceeded its fixed capacity"
        )


def _open_writers(
    plotting_parameters,
    static_parameters,
    dynamic_parameters,
):
    output_dir = Path(static_parameters.output_dir) / "data"
    field_writer = None
    particle_writer = None

    if plotting_parameters["plot_openpmd_fields"]:
        setup_pmd_files(str(output_dir), "fields", ".h5")
        field_writer = create_async_tiled_openpmd_field_writer(
            static_parameters,
            dynamic_parameters,
            str(output_dir),
            filename="fields",
            file_extension=".h5",
            queue_size=int(plotting_parameters.get("openpmd_field_queue_size", 2)),
        )

    if plotting_parameters["plot_openpmd_particles"]:
        setup_pmd_files(str(output_dir), "particles", ".h5")
        particle_writer = create_async_tiled_openpmd_particle_writer(
            static_parameters,
            dynamic_parameters,
            str(output_dir),
            filename="particles",
            file_extension=".h5",
            queue_size=int(
                plotting_parameters.get("openpmd_particle_queue_size", 2)
            ),
        )

    return field_writer, particle_writer


def _write_output(
    field_writer,
    particle_writer,
    particles,
    fields,
    species_config,
    species_names,
    static_parameters,
    dynamic_parameters,
    plotting_parameters,
    plot_number,
    step,
):
    if particle_writer is not None:
        enqueue_openpmd_particle_output(
            particle_writer,
            particles,
            dynamic_parameters,
            plot_number,
            step,
            species_config=species_config,
            species_names=species_names,
        )

    if field_writer is not None:
        field_map = build_field_output_map(
            fields,
            particles,
            species_config,
            static_parameters,
            dynamic_parameters,
            include_fluid_velocity=bool(plotting_parameters["plotvelocities"]),
            include_charge_density=bool(plotting_parameters["plotchargedensity"]),
        )
        enqueue_openpmd_field_output(
            field_writer,
            field_map,
            dynamic_parameters,
            plot_number,
            step,
        )


def run_pierce_diode(config):
    """Initialize and run the configured open Pierce diode system."""

    source_parameters = config["pierce_diode"]
    np.random.seed(int(source_parameters["random_seed"]))

    (
        initialized_loop,
        particles,
        fields,
        static_parameters,
        dynamic_parameters,
        plotting_parameters,
        plasma_parameters,
        species_config,
    ) = initialize_simulation(config)
    if initialized_loop is not time_loop_electrostatic:
        raise ValueError("The Pierce diode driver requires solver='electrostatic'.")

    species_names = plotting_parameters["particle_species_names"]
    electron_index = species_index(
        species_names,
        source_parameters["electron_species"],
    )
    ion_index = species_index(
        species_names,
        source_parameters["ion_species"],
    )
    particles = match_neutralizing_ions(
        particles,
        electron_index,
        ion_index,
    )

    jitted_step = make_jitted_pierce_step(
        static_parameters,
        source_parameters,
        electron_index,
    )
    random_key = jax.random.key(int(source_parameters["random_seed"]))
    field_writer, particle_writer = _open_writers(
        plotting_parameters,
        static_parameters,
        dynamic_parameters,
    )

    total_injected = 0
    loop_error = None
    try:
        for step in tqdm(range(static_parameters.Nt)):
            if step % plotting_parameters["plotting_interval"] == 0:
                plot_number = step // plotting_parameters["plotting_interval"]
                _write_output(
                    field_writer,
                    particle_writer,
                    particles,
                    fields,
                    species_config,
                    species_names,
                    static_parameters,
                    dynamic_parameters,
                    plotting_parameters,
                    plot_number,
                    step,
                )

            particles, fields, random_key, n_injected = jitted_step(
                particles,
                species_config,
                fields,
                dynamic_parameters,
                jnp.asarray(step, dtype=jnp.int32),
                random_key,
            )
            _raise_if_overflowed(fields)
            total_injected += int(jax.device_get(n_injected))

    except BaseException as exc:
        loop_error = exc
        raise
    finally:
        writer_error = None
        for writer in (particle_writer, field_writer):
            if writer is None:
                continue
            try:
                writer.close(raise_errors=loop_error is None)
            except BaseException as exc:
                if writer_error is None:
                    writer_error = exc
        if loop_error is None and writer_error is not None:
            raise writer_error

    print(f"Injected macroparticles: {total_injected}")
    return (
        particles,
        fields,
        static_parameters,
        dynamic_parameters,
        plotting_parameters,
        plasma_parameters,
        species_config,
        total_injected,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILE,
        help="Pierce-diode TOML configuration",
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    run_pierce_diode(load_configuration(args.config))


if __name__ == "__main__":
    main()
