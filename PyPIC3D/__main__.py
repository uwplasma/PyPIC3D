# Christopher Woolford July 2024
# 3D PIC code in Python using the JAX library

import os
import time
import traceback

import jax
from jax import block_until_ready
from tqdm import tqdm

from PyPIC3D.diagnostics.plotting import write_data as _write_data
from PyPIC3D.diagnostics.async_writer import (
    create_async_tiled_openpmd_field_writer,
    create_async_tiled_openpmd_particle_writer,
    enqueue_openpmd_field_output,
    enqueue_openpmd_particle_output,
    make_tiled_field_snapshot,
    make_tiled_particle_snapshot,
)
from PyPIC3D.diagnostics.openPMD import (
    TiledMeshLayout,
    write_tiled_field_snapshot_openpmd,
    write_tiled_particle_snapshot_openpmd,
)
from PyPIC3D.diagnostics.output_adapters import build_field_output_map
from PyPIC3D.diagnostics.diagnostic_quantities import (
    compute_energy,
    compute_dark_energy,
    compute_total_momentum,
)
from PyPIC3D.utilities.field_helpers import add_external_fields
from PyPIC3D.utilities.simulation_helpers import setup_pmd_files
from PyPIC3D.utilities.toml_helpers import (
    dump_parameters_to_toml,
    load_config_file,
)
from PyPIC3D.initialization import initialize_simulation
from PyPIC3D.relativity.interpolate_metric import particle_lorentz_factor


def write_data(*args, **kwargs):
    """Collective diagnostics are evaluated everywhere; rank zero writes scalars."""
    if jax.process_index() == 0:
        _write_data(*args, **kwargs)


def _raise_if_tiled_particles_overflowed(fields):
    if len(fields) < 8:
        return

    overflow = fields[-1]
    if bool(jax.device_get(overflow)):
        raise RuntimeError("tiled particle tile capacity overflowed during periodic retile")


def write_work_energy_tracker(tracker, output_dir):
    """Write one synchronized sample through the scalar time-series output."""
    from PyPIC3D.diagnostics.gr_work_energy_theorem import tracker_values

    for name, value in tracker_values(tracker).items():
        write_data(f"{output_dir}/data/work_energy_{name}.txt", tracker.latest.time, value)


def run_PyPIC3D(config_file, *, pair_injector=None, injection_key=None,
                injection_interval_steps=1, bssn_state=None, bssn_evolution_parameters=None,
                field_initializers=None):
    """Run a simulation, optionally injecting neutral pairs in the static solver.

    The injector follows time_loop_static_metric's callable contract. Supply an
    explicit RNG key and reserve enough particle slots in the initial configuration.
    """
    from PyPIC3D.distributed import mpi_communicator
    comm = mpi_communicator()
    try:
        (
            loop,
            particles,
            fields,
            static_parameters,
            dynamic_parameters,
            plotting_parameters,
            plasma_parameters,
            species_config,
        ) = initialize_simulation(config_file, bssn_state=bssn_state,
                                  bssn_evolution_parameters=bssn_evolution_parameters,
                                  field_initializers=field_initializers)
    except BaseException:
        if comm is not None:
            traceback.print_exc()
            comm.Abort(1)
        raise

    dt = dynamic_parameters.dt
    Nt = static_parameters.Nt
    output_dir = static_parameters.output_dir
    particle_species_names = plotting_parameters.get("particle_species_names")
    static_metric = static_parameters.solver == "static_metric"
    dynamic_metric = static_parameters.solver == "dynamic_metric"
    gr_metric = static_metric or dynamic_metric
    work_energy_theorem = dynamic_metric and static_parameters.gr_work_energy_theorem

    if pair_injector is not None:
        if not static_metric:
            raise ValueError("Pair injection requires solver='static_metric'")
        if injection_key is None or injection_interval_steps < 1:
            raise ValueError("Pair injection requires an RNG key and a positive step interval")

    def loop_with_static_parameters(
        particles,
        species_config,
        fields,
        dynamic_parameters,
        key,
        step,
    ):
        injection = {} if pair_injector is None else dict(
            pair_injector=pair_injector, injection_key=key, step=step,
            injection_interval_steps=injection_interval_steps)
        return loop(
            particles,
            species_config,
            fields,
            static_parameters,
            dynamic_parameters,
            **injection,
        )

    jit_loop = jax.jit(loop_with_static_parameters)

    initial_energy = None
    if not gr_metric:
        E, B, J, rho, phi, external_fields, *rest = fields
        total_E, total_B = add_external_fields(E, B, external_fields)
        e_energy, b_energy, kinetic_energy = compute_energy(
            particles,
            total_E,
            total_B,
            static_parameters,
            dynamic_parameters,
            species_config=species_config,
        )
        dark_energy = (compute_dark_energy(fields[7], static_parameters, dynamic_parameters,
                                              None if fields[6] is None else fields[6][1])
                       if static_parameters.solver == "dark_matter_yee" else 0.0)
        initial_energy = e_energy + b_energy + kinetic_energy + dark_energy

    field_writer = None
    particle_writer = None
    if comm is not None:
        layout = TiledMeshLayout(
            global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
            tile_shape=tuple(int(width) for width in static_parameters.tile_shape),
            guard_cells=int(static_parameters.guard_cells),
        )
    if comm is None and plotting_parameters["plot_openpmd_fields"]:
        setup_pmd_files(os.path.join(output_dir, "data"), "fields", ".h5")
        field_writer = create_async_tiled_openpmd_field_writer(
            static_parameters,
            dynamic_parameters,
            os.path.join(output_dir, "data"),
            filename="fields",
            file_extension=".h5",
            queue_size=int(plotting_parameters.get("openpmd_field_queue_size", 2)),
        )
    if comm is None and plotting_parameters["plot_openpmd_particles"]:
        setup_pmd_files(os.path.join(output_dir, "data"), "particles", ".h5")
        particle_writer = create_async_tiled_openpmd_particle_writer(
            static_parameters,
            dynamic_parameters,
            os.path.join(output_dir, "data"),
            filename="particles",
            file_extension=".h5",
            queue_size=int(plotting_parameters.get("openpmd_particle_queue_size", 2)),
        )

    matched_gr = gr_metric and static_parameters.current_filter != "none"
    # A common-time final snapshot is needed when the work-energy budget writes its
    # endpoint at t = Nt, and for filtered GR qualification.
    final_endpoint = work_energy_theorem or matched_gr
    loop_error = None
    try:
        if comm is not None:
            if comm.rank == 0:
                for name in ("fields", "particles"):
                    if plotting_parameters["plot_openpmd_" + name]:
                        setup_pmd_files(os.path.join(output_dir, "data"), name, ".h5")
            comm.Barrier()
        iterations = Nt + int(final_endpoint)
        for t in tqdm(range(iterations), disable=jax.process_index() != 0):
            if t % plotting_parameters["plotting_interval"] == 0 or (final_endpoint and t == Nt):
                if dynamic_metric:
                    from PyPIC3D.diagnostics.static_metric import bssn_constraint_norms
                    norms = bssn_constraint_norms(particles, species_config, fields,
                                                 static_parameters, dynamic_parameters)
                    for name, value in norms.items():
                        write_data(f"{output_dir}/data/{name}.txt",
                                   dynamic_parameters.bssn_parameters.time, value)
                    if work_energy_theorem and 0 < t < Nt:
                        write_work_energy_tracker(dynamic_parameters.work_energy_tracker, output_dir)

                if not gr_metric:
                    E, B, J, rho, phi, external_fields, *rest = fields
                    total_E, total_B = add_external_fields(E, B, external_fields)
                    e_energy, b_energy, kinetic_energy = compute_energy(
                        particles,
                        total_E,
                        total_B,
                        static_parameters,
                        dynamic_parameters,
                        species_config=species_config,
                    )
                    dark_energy = (compute_dark_energy(fields[7], static_parameters, dynamic_parameters,
                                                          None if fields[6] is None else fields[6][1])
                                   if static_parameters.solver == "dark_matter_yee" else 0.0)
                    total_energy = e_energy + b_energy + kinetic_energy + dark_energy
                    if static_parameters.solver == "dark_matter_yee":
                        write_data(f"{output_dir}/data/dark_field_energy.txt", t * dt, dark_energy)
                    write_data(f"{output_dir}/data/total_energy.txt", t * dt, total_energy)
                    write_data(
                        f"{output_dir}/data/energy_error.txt",
                        t * dt,
                        abs(initial_energy - total_energy) / max(initial_energy, 1e-10),
                    )
                    write_data(f"{output_dir}/data/electric_field_energy.txt", t * dt, e_energy)
                    write_data(f"{output_dir}/data/magnetic_field_energy.txt", t * dt, b_energy)
                    write_data(f"{output_dir}/data/kinetic_energy.txt", t * dt, kinetic_energy)

                    total_momentum = compute_total_momentum(particles, species_config=species_config)
                    write_data(f"{output_dir}/data/total_momentum.txt", t * dt, total_momentum)

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
                    if comm is None:
                        enqueue_openpmd_particle_output(
                            particle_writer, output_particles, dynamic_parameters, t, t,
                            species_config=species_config, species_names=particle_species_names,
                            gamma=gamma,
                        )
                    else:
                        snapshot = make_tiled_particle_snapshot(
                            output_particles, step=t, time=float(dynamic_parameters.bssn_parameters.time),
                            species_names=particle_species_names, species_config=species_config, gamma=gamma,
                        )
                        write_tiled_particle_snapshot_openpmd(
                            snapshot, output_dir=os.path.join(output_dir, "data"), filename="particles",
                            static_parameters=static_parameters, dynamic_parameters=dynamic_parameters,
                            file_extension=".h5", comm=comm,
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
                    if comm is None:
                        enqueue_openpmd_field_output(field_writer, field_map, dynamic_parameters, t, t)
                    else:
                        snapshot = make_tiled_field_snapshot(
                            field_map, step=t, time=float(dynamic_parameters.bssn_parameters.time),
                            current_time_offset=0.0 if t == 0 else -float(dt)/4,
                        )
                        write_tiled_field_snapshot_openpmd(
                            snapshot, output_dir=os.path.join(output_dir, "data"), filename="fields",
                            dynamic_parameters=dynamic_parameters, layout=layout, file_extension=".h5", comm=comm,
                        )

            if t == Nt:
                break

            if static_parameters.benchmark and t == 0:
                start_compile = time.perf_counter()
                jit_loop = jit_loop.lower(particles, species_config, fields,
                                          dynamic_parameters, injection_key, t).compile()
                compile_elapsed = time.perf_counter()-start_compile
                if comm is not None:
                    from mpi4py import MPI
                    compile_elapsed = comm.allreduce(compile_elapsed, op=MPI.MAX)
                if jax.process_index() == 0:
                    print(f"Step compilation (slowest process): {compile_elapsed:.3f} s")
                    print(f"Compiled step memory per device: {jit_loop.memory_analysis()}")
            if static_parameters.benchmark and comm is not None:
                # Exclude another rank's compilation or diagnostics from the
                # timed step; all ranks start after the same completed work.
                comm.Barrier()
            start_step = time.perf_counter() if static_parameters.benchmark else None
            result = jit_loop(
                particles,
                species_config,
                fields,
                dynamic_parameters,
                injection_key,
                t,
            )
            if static_parameters.benchmark:
                block_until_ready(result)
                elapsed = time.perf_counter()-start_step
                if comm is not None:
                    from mpi4py import MPI
                    elapsed = comm.allreduce(elapsed, op=MPI.MAX)
                if jax.process_index() == 0:
                    print(f"Step {t} elapsed (slowest process): {elapsed:.6f} s")
            if dynamic_metric:
                particles, fields, dynamic_parameters = result
            elif pair_injector is None:
                particles, fields = result
            else:
                particles, fields, injection_key, counts = result
                requested, inserted = jax.device_get(counts)
                rejected = requested - inserted
                if rejected.any():
                    raise RuntimeError(
                        f"Pair injection capacity/species failure at step {t}: "
                        f"requested={requested}, inserted={inserted}, "
                        f"rejected={rejected}")
            _raise_if_tiled_particles_overflowed(fields)
            if work_energy_theorem and t == 0:
                # The first step establishes its baseline after applying any
                # prescribed geometry callbacks and aligning the PIC history.
                from PyPIC3D.diagnostics.gr_work_energy_theorem import baseline_tracker
                write_work_energy_tracker(
                    baseline_tracker(dynamic_parameters.work_energy_tracker), output_dir)

        if work_energy_theorem:
            # The loop writes before advancing; its final endpoint has not
            # appeared at any cadence. This also writes the sole Nt=0 sample.
            write_work_energy_tracker(dynamic_parameters.work_energy_tracker, output_dir)

    except BaseException as exc:
        loop_error = exc
        if comm is not None:
            traceback.print_exc()
            comm.Abort(1)
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

    return (
        static_parameters,
        dynamic_parameters,
        plotting_parameters,
        plasma_parameters,
        particles,
        fields,
        species_config,
    )


def _configure_jax_runtime():
    """Enable the requested precision and report JAX's selected backend."""
    jax.config.update("jax_enable_x64", True)

    devices = jax.devices()
    backend = jax.default_backend()
    print(f"Using JAX backend: {backend}")
    print(f"JAX devices ({len(devices)}): {devices}")
    return backend, devices


def main():
    _configure_jax_runtime()

    toml_file = load_config_file()

    start = time.time()
    (
        static_parameters,
        dynamic_parameters,
        plotting_parameters,
        plasma_parameters,
        particles,
        fields,
        species_config,
    ) = block_until_ready(run_PyPIC3D(toml_file))
    end = time.time()

    if static_parameters.solver in ("static_metric", "dynamic_metric"):
        print("Skipping final flat-space energy diagnostics for GR fields and covariant particle u_i\n")
    else:
        E, B, J, rho, phi, external_fields, *rest = fields
        total_E, total_B = add_external_fields(E, B, external_fields)
        e_energy, b_energy, kinetic_energy = compute_energy(
            particles,
            total_E,
            total_B,
            static_parameters,
            dynamic_parameters,
            species_config=species_config,
        )
        print(f"Final Electric Field Energy: {e_energy}")
        print(f"Final Magnetic Field Energy: {b_energy}")
        print(f"Final Kinetic Energy: {kinetic_energy}")
        dark_energy = (compute_dark_energy(fields[7], static_parameters, dynamic_parameters,
                                              None if fields[6] is None else fields[6][1])
                       if static_parameters.solver == "dark_matter_yee" else 0.0)
        if static_parameters.solver == "dark_matter_yee":
            print(f"Final Dark Field Energy: {dark_energy}")
        print(f"Total Final Energy: {e_energy + b_energy + kinetic_energy + dark_energy}\n")

    duration = end - start
    Nt = static_parameters.Nt
    simulation_stats = {
        "total_time": duration,
        "total_iterations": Nt,
        "time_per_iteration": duration / Nt if Nt else 0.0,
    }

    dump_parameters_to_toml(
        simulation_stats, static_parameters, dynamic_parameters, plasma_parameters,
        plotting_parameters, particles)

    print("\nSimulation Complete")
    print(f"Total Simulation Time: {duration} s")
    print(f"Time Per Iteration: {duration / Nt if Nt else 0.0} s")


if __name__ == "__main__":
    main()
