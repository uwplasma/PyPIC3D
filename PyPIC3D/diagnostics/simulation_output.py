"""Simulation snapshots, scalar diagnostics, and output writer lifetime."""

from contextlib import contextmanager
import os

import jax

from PyPIC3D.diagnostics.async_writer import (
    create_async_tiled_openpmd_field_writer,
    create_async_tiled_openpmd_particle_writer,
    enqueue_openpmd_field_output,
    enqueue_openpmd_particle_output,
)
from PyPIC3D.diagnostics.output_adapters import build_field_output_map
from PyPIC3D.diagnostics.plotting import write_data as _write_data
from PyPIC3D.diagnostics.diagnostic_quantities import compute_energy_components, compute_total_momentum
from PyPIC3D.relativity.interpolate_metric import particle_lorentz_factor
from PyPIC3D.utilities.simulation_helpers import setup_pmd_files


def simulation_energy_components(particles, species, fields, static, dynamic):
    """Read ordinary and dark fields from the runtime tuple for flat-space diagnostics."""
    E, B, J, rho, phi, external_fields, *rest = fields
    dark_fields = dark_pml = None
    if static.solver == "dark_matter_yee":
        dark_fields = fields[7]
        dark_pml = None if fields[6] is None else fields[6][1]
    return compute_energy_components(particles, E, B, external_fields, static, dynamic,
                                     species, dark_fields=dark_fields, dark_pml=dark_pml)


def print_energy(label, energies, *, dark_matter=False):
    """Report energies already evaluated on every process."""
    if jax.process_index() != 0:
        return
    electric, magnetic, kinetic, dark = energies
    print(f"{label} Electric Field Energy: {electric:.6e} J")
    print(f"{label} Magnetic Field Energy: {magnetic:.6e} J")
    print(f"{label} Kinetic Energy: {kinetic:.6e} J")
    if dark_matter:
        print(f"{label} Dark Field Energy: {dark:.6e} J")
    print(f"Total {label} Energy: {electric + magnetic + kinetic + dark:.6e} J\n")


def write_data(filename, time, value):
    """Collective diagnostics run everywhere; only rank zero appends scalar files."""
    if jax.process_index() == 0:
        _write_data(filename, time, value)


def write_work_energy_tracker(tracker, output_dir):
    """Write one synchronized work-energy sample."""
    from PyPIC3D.diagnostics.gr_work_energy_theorem import tracker_values

    for name, value in tracker_values(tracker).items():
        write_data(f"{output_dir}/data/work_energy_{name}.txt", tracker.latest.time, value)


def write_scalar_diagnostics(particles, species, fields, static, dynamic, step, initial_energy):
    """Write the active solver's scalar diagnostics at the driver's output cadence."""
    output_dir = static.output_dir
    if static.solver == "dynamic_metric":
        from PyPIC3D.diagnostics.static_metric import bssn_constraint_norms

        norms = bssn_constraint_norms(particles, species, fields, static, dynamic)
        for name, value in norms.items():
            write_data(f"{output_dir}/data/{name}.txt", dynamic.bssn_parameters.time, value)
        if static.gr_work_energy_theorem and 0 < step < static.Nt:
            write_work_energy_tracker(dynamic.work_energy_tracker, output_dir)
    elif static.solver != "static_metric":
        electric, magnetic, kinetic, dark = simulation_energy_components(
            particles, species, fields, static, dynamic)
        total = electric + magnetic + kinetic + dark
        time = step * dynamic.dt
        if static.solver == "dark_matter_yee":
            write_data(f"{output_dir}/data/dark_field_energy.txt", time, dark)
        write_data(f"{output_dir}/data/total_energy.txt", time, total)
        write_data(f"{output_dir}/data/energy_error.txt", time,
                   abs(initial_energy - total) / max(initial_energy, 1e-10))
        write_data(f"{output_dir}/data/electric_field_energy.txt", time, electric)
        write_data(f"{output_dir}/data/magnetic_field_energy.txt", time, magnetic)
        write_data(f"{output_dir}/data/kinetic_energy.txt", time, kinetic)
        momentum = compute_total_momentum(particles, species_config=species)
        write_data(f"{output_dir}/data/total_momentum.txt", time, momentum)


@contextmanager
def open_simulation_output(static_parameters, dynamic_parameters, plotting_parameters):
    """Close both writers on failure.

    Every process queues snapshots into its own series, so nothing here is
    collective. A .pmd pointer holds a single name, so a multiprocess run leaves
    it to the merge tool rather than writing one that dangles until then.
    """
    output_dir = static_parameters.output_dir
    field_writer = particle_writer = None
    loop_error = None
    shared = jax.process_count() == 1
    try:
        if plotting_parameters["plot_openpmd_fields"]:
            if shared:
                setup_pmd_files(os.path.join(output_dir, "data"), "fields", ".h5")
            field_writer = create_async_tiled_openpmd_field_writer(
                static_parameters,
                dynamic_parameters,
                os.path.join(output_dir, "data"),
                filename="fields",
                file_extension=".h5",
                queue_size=int(plotting_parameters.get("openpmd_field_queue_size", 2)),
            )
        if plotting_parameters["plot_openpmd_particles"]:
            if shared:
                setup_pmd_files(os.path.join(output_dir, "data"), "particles", ".h5")
            particle_writer = create_async_tiled_openpmd_particle_writer(
                static_parameters,
                dynamic_parameters,
                os.path.join(output_dir, "data"),
                filename="particles",
                file_extension=".h5",
                queue_size=int(plotting_parameters.get("openpmd_particle_queue_size", 2)),
            )
        yield field_writer, particle_writer
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


def write_simulation_output(particles, species_config, fields, static_parameters,
                            dynamic_parameters, plotting_parameters, t, writers):
    """Submit particle and field snapshots without gathering global arrays."""
    field_writer, particle_writer = writers
    particle_species_names = plotting_parameters.get("particle_species_names")
    dynamic_metric = static_parameters.solver == "dynamic_metric"
    gr_metric = dynamic_metric or static_parameters.solver == "static_metric"

    if plotting_parameters["plot_openpmd_particles"]:
        gamma = None
        output_particles = particles
        if dynamic_metric:
            from PyPIC3D.relativity.dynamical_relativity import synchronize_particles
            output_particles = synchronize_particles(
                particles, species_config, fields, static_parameters, dynamic_parameters)
        if gr_metric:
            metric = fields[6]
            gamma = particle_lorentz_factor(output_particles, metric, static_parameters, dynamic_parameters)
            # particle u holds covariant u_i, so Gamma needs the metric at each particle
        enqueue_openpmd_particle_output(
            particle_writer, output_particles, dynamic_parameters, t, t,
            species_config=species_config, species_names=particle_species_names,
            gamma=gamma,
        )

    if plotting_parameters["plot_openpmd_fields"]:
        field_map = build_field_output_map(
            fields,
            particles,
            species_config,
            static_parameters,
            dynamic_parameters,
            include_fluid_velocity=bool(plotting_parameters["plotvelocities"]),
            include_charge_density=bool(plotting_parameters["plotchargedensity"]),
        )
        plotting_parameters["field_map"] = field_map
        enqueue_openpmd_field_output(field_writer, field_map, dynamic_parameters, t, t)
