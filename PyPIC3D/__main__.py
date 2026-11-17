# Christopher Woolford July 2024
# 3D PIC code in Python using the JAX library

import time

import jax
from jax import block_until_ready
from tqdm import tqdm

from PyPIC3D.diagnostics.simulation_output import (
    open_simulation_output,
    write_simulation_output,
    write_scalar_diagnostics,
    write_work_energy_tracker,
    simulation_energy_components,
    print_energy,
)
from PyPIC3D.utilities.toml_helpers import (
    dump_parameters_to_toml,
    load_config_file,
)
from PyPIC3D.initialization import initialize_simulation


def _raise_if_tiled_particles_overflowed(fields):
    if len(fields) < 8:
        return

    overflow = fields[-1]
    if bool(jax.device_get(overflow)):
        raise RuntimeError("tiled particle tile capacity overflowed during periodic retile")


def run_PyPIC3D(config_file, *, pair_injector=None, injection_key=None,
                injection_interval_steps=1, bssn_state=None, bssn_evolution_parameters=None,
                field_initializers=None):
    """Run a simulation, optionally injecting neutral pairs in the static solver.

    The injector follows time_loop_static_metric's callable contract. Supply an
    explicit RNG key and reserve enough particle slots in the initial configuration.
    """
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

    Nt = static_parameters.Nt
    output_dir = static_parameters.output_dir
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
        initial_energy = sum(simulation_energy_components(
            particles, species_config, fields, static_parameters, dynamic_parameters))

    matched_gr = gr_metric and static_parameters.current_filter != "none"
    # A common-time final snapshot is needed when the work-energy budget writes its
    # endpoint at t = Nt, and for filtered GR qualification.
    final_endpoint = work_energy_theorem or matched_gr
    with open_simulation_output(static_parameters, dynamic_parameters, plotting_parameters) as writers:
        iterations = Nt + int(final_endpoint)
        for t in tqdm(range(iterations), disable=jax.process_index() != 0):
            if t % plotting_parameters["plotting_interval"] == 0 or (final_endpoint and t == Nt):
                write_scalar_diagnostics(particles, species_config, fields, static_parameters,
                                         dynamic_parameters, t, initial_energy)
                write_simulation_output(particles, species_config, fields, static_parameters,
                                        dynamic_parameters, plotting_parameters, t, writers)

            if t == Nt:
                break

            result = jit_loop(
                particles,
                species_config,
                fields,
                dynamic_parameters,
                injection_key,
                t,
            )
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

        if dynamic_metric:
            from PyPIC3D.diagnostics.static_metric import print_bssn_constraints
            print_bssn_constraints("Final", particles, species_config, fields,
                                   static_parameters, dynamic_parameters)

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

    if static_parameters.solver not in ("static_metric", "dynamic_metric"):
        energies = simulation_energy_components(particles, species_config, fields,
                                                static_parameters, dynamic_parameters)
        print_energy("Final", energies, dark_matter=static_parameters.solver == "dark_matter_yee")

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
