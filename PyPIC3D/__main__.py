# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities

########################################## IMPORT LIBRARIES #############################################
from sys import path
import time
import jax
from jax import block_until_ready
import jax.numpy as jnp
from tqdm import tqdm
#from memory_profiler import profile
# Importing relevant libraries

from PyPIC3D.plotting import (
    plotter, write_particles_phase_space, write_data, plot_vtk_particles, plot_field_slice_vtk
)

from PyPIC3D.utils import (
    dump_parameters_to_toml, load_config_file, compute_energy
)

from PyPIC3D.initialization import (
    initialize_simulation
)


# Importing functions from the PyPIC3D package
############################################################################################################


def run_PyPIC3D(config_file):
    ##################################### INITIALIZE SIMULATION ################################################

    loop, particles, fields, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, \
            plasma_parameters, solver, bc, electrostatic, verbose, GPUs, Nt, curl_func, J_func = initialize_simulation(config_file)
    # initialize the simulation

    jit_loop = jax.jit(loop, static_argnames=('curl_func', 'J_func','solver', 'bc'))

    dt = world['dt']
    output_dir = simulation_parameters['output_dir']

    field_names = ["E_magnitude", "B_magnitude"]

    ############################################################################################################

    ###################################################### SIMULATION LOOP #####################################

    for t in tqdm(range(Nt)):
        # plotter(t, particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, simulation_parameters)
        # plot the data
        if t % plotting_parameters['plotting_interval'] == 0:
            E, B, J, *rest = fields
            # unpack the fields
            e_energy, b_energy, kinetic_energy = compute_energy(particles, E, B, world, constants)
            # Compute the energy of the system
            write_data(f"{output_dir}/data/total_energy.txt", t * dt, e_energy + b_energy + kinetic_energy)
            write_data(f"{output_dir}/data/electric_field_energy.txt", t * dt, e_energy)
            write_data(f"{output_dir}/data/magnetic_field_energy.txt", t * dt, b_energy)
            write_data(f"{output_dir}/data/kinetic_energy.txt", t * dt, kinetic_energy)
            # Write the total energy to a file
            total_momentum = sum(particle_species.momentum() for particle_species in particles)
            # Total momentum of the particles
            write_data(f"{output_dir}/data/total_momentum.txt", t * dt, total_momentum)
            # Write the total momentum to a file

            write_particles_phase_space(particles, t, output_dir)

            E_magnitude = jnp.sqrt(E[0]**2 + E[1]**2 + E[2]**2)[:,:,world['Nz']//2]
            B_magnitude = jnp.sqrt(B[0]**2 + B[1]**2 + B[2]**2)[:,:,world['Nz']//2]
            fields_mag = [E_magnitude, B_magnitude]
            plot_field_slice_vtk(fields_mag, field_names, 2, E_grid, t, "fields", output_dir, world)
            # Plot the fields in VTK format

            # if plotting_parameters['plot_vtk_particles']:
                # plot_vtk_particles(particles, t, output_dir)
            # Plot the particles in VTK format

        particles, fields = jit_loop(particles, fields, E_grid, B_grid, world, constants, curl_func, J_func, solver, bc)
        # time loop to update the particles and fields

    return Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, fields, world

def main():
    ###################### JAX SETTINGS ########################################################################
    jax.config.update("jax_enable_x64", True)
    # set Jax to use 64 bit precision
    jax.config.update("jax_debug_nans", True)
    # debugging for nans
    jax.config.update('jax_platform_name', 'cpu')
    # set Jax to use CPUs
    #jax.config.update("jax_disable_jit", True)
    ############################################################################################################

    toml_file = load_config_file()
    # load the configuration file

    start = time.time()
    # start the timer

    Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, fields, world =  block_until_ready(run_PyPIC3D(toml_file))
    # run the PyPIC3D simulation

    end = time.time()
    # end the timer

    E, B, J, *rest = fields
    # unpack the fields

    e_energy, b_energy, kinetic_energy = compute_energy(particles, E, B, world, constants)
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