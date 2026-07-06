# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities

########################################## IMPORT LIBRARIES #############################################
from sys import path
import os
import time
import jax
from jax import block_until_ready
from tqdm import tqdm

#from memory_profiler import profile
# Importing relevant libraries

from PyPIC3D.diagnostics.plotting import (
    write_particles_phase_space, write_data
)

from PyPIC3D.diagnostics.async_writer import (
    create_async_tiled_openpmd_field_writer,
    enqueue_openpmd_field_output,
)

from PyPIC3D.diagnostics.openPMD import (
    write_openpmd_particles,
)

from PyPIC3D.utils import (
    dump_parameters_to_toml, load_config_file, compute_energy,
    setup_pmd_files, add_external_fields, compute_total_momentum
)

from PyPIC3D.initialization import (
    initialize_simulation
)


# Importing functions from the PyPIC3D package
############################################################################################################


def _raise_if_tiled_particles_overflowed(fields, simulation_parameters):
    if len(fields) < 8:
        return

    overflow = fields[-1]
    if bool(jax.device_get(overflow)):
        raise RuntimeError("tiled particle tile capacity overflowed during periodic retile")


def run_PyPIC3D(config_file):
    ##################################### INITIALIZE SIMULATION ################################################

    loop, particles, fields, world, simulation_parameters, constants, plotting_parameters, plasma_parameters, solver, electrostatic, verbose, GPUs, Nt, relativistic, particle_pusher, species_config = initialize_simulation(config_file)
    # initialize the simulation

    dt = world['dt']
    output_dir = simulation_parameters['output_dir']
    particle_species_names = simulation_parameters.get("particle_species_names")

    def loop_with_static_world(
        particles,
        species_config,
        fields,
        constants,
        solver,
        relativistic=True,
        particle_pusher="boris",
    ):
        if electrostatic:
            return loop(
                particles,
                species_config,
                fields,
                world,
                constants,
                solver,
                relativistic=relativistic,
                particle_pusher=particle_pusher,
            )
        return loop(
            particles,
            species_config,
            fields,
            world,
            constants,
            relativistic=relativistic,
            particle_pusher=particle_pusher,
        )

    jit_loop = jax.jit(
        loop_with_static_world,
        static_argnames=('solver', 'relativistic', 'particle_pusher'),
    )

    E, B, J, rho, phi, external_fields, *rest = fields
    # unpack the fields
    total_E, total_B = add_external_fields(E, B, external_fields)
    # energy diagnostics use the fields seen by the particle pusher
    e_energy, b_energy, kinetic_energy = compute_energy(particles, total_E, total_B, world, constants, species_config=species_config)
    # Compute the energy of the system
    initial_energy = e_energy + b_energy + kinetic_energy

    field_writer = None
    if plotting_parameters['plot_openpmd_fields']:
        setup_pmd_files( os.path.join(output_dir, "data"), "fields", ".h5")
        field_writer = create_async_tiled_openpmd_field_writer(
            world,
            os.path.join(output_dir, "data"),
            filename="fields",
            file_extension=".h5",
            queue_size=int(plotting_parameters.get("openpmd_field_queue_size", 2)),
        )
    if plotting_parameters['plot_openpmd_particles']: setup_pmd_files( os.path.join(output_dir, "data"), "particles", ".h5")
    # setup the openPMD files if needed

    ############################################################################################################

    ###################################################### SIMULATION LOOP #####################################

    loop_error = None
    try:
        for t in tqdm(range(Nt)):

            # plot the data
            if t % plotting_parameters['plotting_interval'] == 0:

                plot_num = t // plotting_parameters['plotting_interval']
                # determine the plot number

                E, B, J, rho, phi, external_fields, *rest = fields
                # unpack the fields

                total_E, total_B = add_external_fields(E, B, external_fields)
                # energy diagnostics use the fields seen by the particle pusher
                e_energy, b_energy, kinetic_energy = compute_energy(particles, total_E, total_B, world, constants, species_config=species_config)
                # Compute the energy of the system
                write_data(f"{output_dir}/data/total_energy.txt", t * dt, e_energy + b_energy + kinetic_energy)
                write_data(f"{output_dir}/data/energy_error.txt", t * dt, abs( initial_energy - (e_energy + b_energy + kinetic_energy)) / max(initial_energy, 1e-10))
                write_data(f"{output_dir}/data/electric_field_energy.txt", t * dt, e_energy)
                write_data(f"{output_dir}/data/magnetic_field_energy.txt", t * dt, b_energy)
                write_data(f"{output_dir}/data/kinetic_energy.txt", t * dt, kinetic_energy)
                # Write the total energy to a file
                total_momentum = compute_total_momentum(particles, species_config=species_config)
                # Total momentum of the particles
                write_data(f"{output_dir}/data/total_momentum.txt", t * dt, total_momentum)
                # Write the total momentum to a file

                # for species in particles:
                #     write_data(f"{output_dir}/data/{species.name}_kinetic_energy.txt", t * dt, species.kinetic_energy())


                if plotting_parameters['plot_phasespace']:
                    write_particles_phase_space(particles, t, output_dir, species_config=species_config, species_names=particle_species_names, world=world)



                if plotting_parameters['plot_openpmd_particles']:
                    write_openpmd_particles(particles, world, constants, os.path.join(output_dir, "data"), plot_num, t, "particles", ".h5", species_config=species_config, species_names=particle_species_names)
                # Write the particles in openPMD format

                if field_writer is not None:
                    enqueue_openpmd_field_output(field_writer, fields, world, plot_num, t)
                # Queue tiled field chunks for asynchronous openPMD output

            particles, fields = jit_loop(
                particles,
                species_config,
                fields,
                constants,
                solver,
                relativistic=relativistic,
                particle_pusher=particle_pusher,
            )
            # time loop to update the particles and fields
            _raise_if_tiled_particles_overflowed(fields, simulation_parameters)
            # fixed tile capacity overflow would silently drop particles; stop as
            # soon as it is detected.

    except BaseException as exc:
        loop_error = exc
        raise
    finally:
        if field_writer is not None:
            field_writer.close(raise_errors=loop_error is None)


    return Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, fields, world, species_config

def main():
    ###################### JAX SETTINGS ########################################################################
    jax.config.update("jax_enable_x64", True)
    # set Jax to use 64 bit precision
    # jax.config.update("jax_debug_nans", True)
    # debugging for nans
    jax.config.update('jax_platform_name', 'cpu')
    # set Jax to use CPUs
    #jax.config.update("jax_disable_jit", True)
    ############################################################################################################

    toml_file = load_config_file()
    # load the configuration file

    start = time.time()
    # start the timer

    Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, fields, world, species_config =  block_until_ready(run_PyPIC3D(toml_file))
    # run the PyPIC3D simulation

    end = time.time()
    # end the timer

    E, B, J, rho, phi, external_fields, *rest = fields
    # unpack the fields

    total_E, total_B = add_external_fields(E, B, external_fields)
    # energy diagnostics use the fields seen by the particle pusher
    e_energy, b_energy, kinetic_energy = compute_energy(particles, total_E, total_B, world, constants, species_config=species_config)
    # compute the energy of the system
    print(f"Final Electric Field Energy: {e_energy}")
    print(f"Final Magnetic Field Energy: {b_energy}")
    print(f"Final Kinetic Energy: {kinetic_energy}")
    print(f"Total Final Energy: {e_energy + b_energy + kinetic_energy}\n")


    duration = end - start
    # calculate the total simulation time

    simulation_stats = {
        "total_time": duration,
        "total_iterations": Nt,
        "time_per_iteration": duration / Nt
    }

    dump_parameters_to_toml(simulation_stats, simulation_parameters, plasma_parameters, plotting_parameters, constants, particles)
    # save the parameters to an output file

    print(f"\nSimulation Complete")
    print(f"Total Simulation Time: {duration} s")
    print(f"Time Per Iteration: {duration/Nt} s")


if __name__ == "__main__":
    main()
    # run the main function
